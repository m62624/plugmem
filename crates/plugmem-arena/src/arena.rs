//! Sharded sorted arena over one flat byte pool.
//!
//! This is the v1 layer of the design in `specs/01-arena.md`: **one page per
//! shard** (the semantics of the original test version, `reference/opaque-v1`),
//! wearing the final v2 interfaces (key-prefix [`Slot`]s, big-endian keys,
//! byte-denominated pages). The v2 mechanics — page chains with range splits,
//! a page free-list — will grow on top without changing this API.
//!
//! The structure is tuned for **wasm environments first**: `no_std + alloc`,
//! a single linear byte pool (snapshot = memcpy), no threads, and the one
//! measured `unsafe` (uninitialized page allocation) whose entire benefit is
//! on the wasm allocation path — see [`Arena::insert`] internals and the
//! crate-level documentation for the numbers.

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

#[cfg(feature = "counters")]
use core::cell::Cell;

use crate::slot::Slot;

/// Size of one arena page in bytes.
///
/// Fixed, not per-slot-count: whatever the slot size, a page is one
/// L1-friendly unit of work, and every operation touches exactly one page.
pub const PAGE_BYTES: usize = 4096;

/// Sentinel for "shard has no page yet".
const NONE: u32 = u32::MAX;

/// Fibonacci hashing multiplier (2^64 / phi), used by [`ShardMode::Uniform`].
const FIB: u64 = 0x9E37_79B9_7F4A_7C15;

/// `true` when the `counters` feature is enabled; lets hot loops keep their
/// counting code in one branch that constant-folds away otherwise.
const COUNT: bool = cfg!(feature = "counters");

/// Adds `$n` to counter field `$field` when the `counters` feature is on;
/// expands to nothing otherwise (zero cost, and the counter expression is
/// not even evaluated).
macro_rules! bump {
    ($self:ident, $field:ident, $n:expr) => {
        #[cfg(feature = "counters")]
        {
            let mut c = $self.counters.get();
            c.$field += $n as u64;
            $self.counters.set(c);
        }
    };
}

/// How keys are mapped to shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardMode {
    /// Shard = Fibonacci hash of the key's leading bytes. Spreads any key
    /// distribution (including sequential ids) evenly across shards. Global
    /// iteration order is *not* the key order. Use for lookup tables.
    Uniform,
    /// Shard = top bits of the key's leading bytes. Shard index order equals
    /// key order, so [`Arena::iter`] yields globally ascending keys. Use for
    /// range-oriented data (temporal indexes, edges). Keys arriving in a
    /// narrow value range will concentrate in few shards — that is the
    /// trade-off for ordering.
    Ordered,
}

/// Arena configuration.
///
/// Stored verbatim in snapshots later (`specs/03`), so everything that
/// affects byte interpretation lives here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaCfg {
    /// Number of shards; must be a non-zero power of two.
    pub shards: usize,
    /// Hard ceiling for the byte pool; exceeding it fails with
    /// [`Error::CapacityExceeded`] instead of growing.
    pub max_bytes: usize,
    /// Key-to-shard mapping mode.
    pub mode: ShardMode,
}

impl ArenaCfg {
    /// Creates a config with the given shard count and mode, and no byte
    /// limit (`max_bytes = usize::MAX`).
    pub const fn new(shards: usize, mode: ShardMode) -> Self {
        Self {
            shards,
            max_bytes: usize::MAX,
            mode,
        }
    }

    /// Returns the config with `max_bytes` replaced.
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }
}

impl Default for ArenaCfg {
    /// 1024 shards, [`ShardMode::Uniform`], unlimited bytes.
    fn default() -> Self {
        Self::new(1024, ShardMode::Uniform)
    }
}

/// Errors returned by arena operations. No arena operation panics on
/// resource exhaustion — capacity problems are always typed errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// Growing the pool would exceed [`ArenaCfg::max_bytes`].
    #[error("arena capacity exceeded: pool would grow past {max_bytes} bytes")]
    CapacityExceeded {
        /// The configured ceiling that would have been crossed.
        max_bytes: usize,
    },
    /// The target shard's page is full (v1 semantics: one page per shard;
    /// the v2 range-split mechanics will remove this error).
    #[error("shard {shard} is full ({slots} slots per page)")]
    ShardFull {
        /// Shard whose page had no free slot.
        shard: usize,
        /// Slots a page holds for this slot size.
        slots: usize,
    },
    /// The [`Slot`] layout constants are invalid.
    #[error(
        "invalid slot layout: size {size}, key_len {key_len} (require 1 <= key_len <= size <= {PAGE_BYTES})"
    )]
    BadSlot {
        /// Declared [`Slot::SIZE`].
        size: usize,
        /// Declared [`Slot::KEY_LEN`].
        key_len: usize,
    },
    /// [`ArenaCfg::shards`] is not a non-zero power of two.
    #[error("shard count must be a non-zero power of two, got {got}")]
    BadShardCount {
        /// The rejected shard count.
        got: usize,
    },
}

/// Deterministic work counters (feature `counters`).
///
/// These are the basis of the project's CI performance gates: unlike
/// wall-clock time they are identical on every machine, so a complexity
/// regression fails the same way everywhere.
#[cfg(feature = "counters")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Key comparisons performed by binary searches.
    pub cmp_ops: u64,
    /// Bytes moved by insert/remove shifts.
    pub bytes_shifted: u64,
    /// Pages allocated in the pool.
    pub pages_allocated: u64,
}

/// A sharded collection of fixed-size records in one contiguous byte pool,
/// sorted by key prefix within each shard.
///
/// See the [crate-level documentation](crate) for the philosophy and the
/// [module documentation](self) for the v1-layer scope. Highlights:
///
/// - all state is `pool` + three small arrays — persisting the arena is a
///   `memcpy` of defined bytes (`specs/03`);
/// - every operation is confined to one 4 KiB page: binary search over at
///   most `PAGE_BYTES / T::SIZE` slots, shifts of at most one page;
/// - `Arena` deliberately implements neither `Clone` nor `PartialEq`: pages
///   are allocated **uninitialized** (the measured wasm optimization), and a
///   byte-wise clone/compare would read the uninitialized tails.
pub struct Arena<T: Slot> {
    /// Page pool; length is always a multiple of [`PAGE_BYTES`]. Bytes of a
    /// page beyond `counts[shard] * T::SIZE` are uninitialized and must
    /// never be read.
    pool: Vec<u8>,
    /// Shard -> page index (`NONE` until first insert; pages are lazy).
    heads: Vec<u32>,
    /// Shard -> number of occupied slots in its page.
    counts: Vec<u16>,
    /// Total records across all shards.
    total: usize,
    /// Reusable serialization buffer for [`Arena::insert`] (`T::SIZE` bytes,
    /// allocated once) — inserts do not allocate after the first call.
    scratch: Vec<u8>,
    cfg: ArenaCfg,
    #[cfg(feature = "counters")]
    counters: Cell<Counters>,
    /// `fn() -> T` keeps the arena `Send`/`Sync`-neutral and avoids bounding
    /// auto-traits on `T` itself.
    _marker: PhantomData<fn() -> T>,
}

impl<T: Slot> Arena<T> {
    /// Creates an empty arena.
    ///
    /// # Errors
    ///
    /// - [`Error::BadSlot`] unless `1 <= T::KEY_LEN <= T::SIZE <= PAGE_BYTES`;
    /// - [`Error::BadShardCount`] unless `cfg.shards` is a non-zero power of
    ///   two.
    pub fn new(cfg: ArenaCfg) -> Result<Self, Error> {
        if T::SIZE == 0 || T::SIZE > PAGE_BYTES || T::KEY_LEN == 0 || T::KEY_LEN > T::SIZE {
            return Err(Error::BadSlot {
                size: T::SIZE,
                key_len: T::KEY_LEN,
            });
        }
        if cfg.shards == 0 || !cfg.shards.is_power_of_two() {
            return Err(Error::BadShardCount { got: cfg.shards });
        }
        Ok(Self {
            pool: Vec::new(),
            heads: alloc::vec![NONE; cfg.shards],
            counts: alloc::vec![0u16; cfg.shards],
            total: 0,
            scratch: Vec::new(),
            cfg,
            #[cfg(feature = "counters")]
            counters: Cell::new(Counters::default()),
            _marker: PhantomData,
        })
    }

    /// Number of slots one page holds for this slot size.
    pub const fn slots_per_page() -> usize {
        PAGE_BYTES / T::SIZE
    }

    /// Total number of records.
    pub fn len(&self) -> usize {
        self.total
    }

    /// `true` when the arena holds no records.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Bytes currently allocated in the page pool.
    pub fn pool_bytes(&self) -> usize {
        self.pool.len()
    }

    /// The configuration this arena was created with.
    pub fn cfg(&self) -> &ArenaCfg {
        &self.cfg
    }

    /// Inserts a record, keeping its shard sorted by key prefix.
    ///
    /// Returns `Ok(false)` if a record with the same key prefix already
    /// exists (the arena is a map keyed by prefix; existing payload is left
    /// untouched — use [`Arena::payload_mut`] to update in place).
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityExceeded`] if a needed page allocation would cross
    ///   [`ArenaCfg::max_bytes`];
    /// - [`Error::ShardFull`] if the shard's page has no free slot (v1
    ///   semantics; removed by the v2 split mechanics).
    pub fn insert(&mut self, value: &T) -> Result<bool, Error> {
        // Take the scratch buffer out of `self` to sidestep aliasing between
        // `&self.scratch` and `&mut self.pool` below.
        let mut buf = core::mem::take(&mut self.scratch);
        buf.resize(T::SIZE, 0);
        value.write(&mut buf);
        let result = self.insert_bytes(&buf);
        self.scratch = buf;
        result
    }

    /// Insert path operating on the serialized slot.
    fn insert_bytes(&mut self, slot: &[u8]) -> Result<bool, Error> {
        let key = &slot[..T::KEY_LEN];
        let shard = self.shard_of(key);
        let page_idx = self.ensure_page(shard)?;
        let count = self.counts[shard] as usize;
        let page_start = page_idx as usize * PAGE_BYTES;

        let mut cmps = 0u64;
        let pos = {
            let page = &self.pool[page_start..page_start + PAGE_BYTES];
            match search::<T>(page, count, key, &mut cmps) {
                Ok(_) => {
                    bump!(self, cmp_ops, cmps);
                    return Ok(false);
                }
                Err(pos) => pos,
            }
        };
        bump!(self, cmp_ops, cmps);

        if count == Self::slots_per_page() {
            return Err(Error::ShardFull {
                shard,
                slots: Self::slots_per_page(),
            });
        }

        let slot_start = page_start + pos * T::SIZE;
        let used_end = page_start + count * T::SIZE;
        let shifted = used_end - slot_start;
        if shifted > 0 {
            // Shift the sorted tail right by one slot; stays within the page.
            self.pool
                .copy_within(slot_start..used_end, slot_start + T::SIZE);
        }
        self.pool[slot_start..slot_start + T::SIZE].copy_from_slice(slot);
        bump!(self, bytes_shifted, shifted);

        self.counts[shard] += 1;
        self.total += 1;
        Ok(true)
    }

    /// Returns the record with the given key prefix, if present.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN` (a caller bug, not a data
    /// condition — mirrors slice indexing).
    pub fn get(&self, key: &[u8]) -> Option<T> {
        self.locate(key)
            .map(|off| T::read(&self.pool[off..off + T::SIZE]))
    }

    /// `true` if a record with the given key prefix exists.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN`.
    pub fn contains(&self, key: &[u8]) -> bool {
        self.locate(key).is_some()
    }

    /// Returns the raw bytes of the record with the given key prefix.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN`.
    pub fn get_slot(&self, key: &[u8]) -> Option<&[u8]> {
        self.locate(key).map(|off| &self.pool[off..off + T::SIZE])
    }

    /// Returns a mutable view of the record's **payload** (the bytes after
    /// the key prefix), for in-place updates.
    ///
    /// The key prefix itself is not reachable through this method — sorted
    /// order cannot be corrupted by construction, which is why this is safe
    /// to expose at all.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN`.
    pub fn payload_mut(&mut self, key: &[u8]) -> Option<&mut [u8]> {
        let off = self.locate(key)?;
        Some(&mut self.pool[off + T::KEY_LEN..off + T::SIZE])
    }

    /// Removes the record with the given key prefix. Returns `true` if it
    /// existed. The shard's page stays allocated (v1 semantics; the v2
    /// free-list reclaims empty pages).
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN`.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        let Some(slot_start) = self.locate(key) else {
            return false;
        };
        let shard = self.shard_of(key);
        let count = self.counts[shard] as usize;
        let page_start = self.heads[shard] as usize * PAGE_BYTES;
        let used_end = page_start + count * T::SIZE;
        let tail = used_end - (slot_start + T::SIZE);
        if tail > 0 {
            // Shift the tail left over the removed slot; stays within the page.
            self.pool
                .copy_within(slot_start + T::SIZE..used_end, slot_start);
        }
        bump!(self, bytes_shifted, tail);
        self.counts[shard] -= 1;
        self.total -= 1;
        true
    }

    /// Iterates all records shard by shard.
    ///
    /// In [`ShardMode::Ordered`] the shard index order equals key order, so
    /// this yields **globally ascending keys**. In [`ShardMode::Uniform`]
    /// the global order is unspecified (each shard is still internally
    /// sorted).
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            arena: self,
            shard: 0,
            idx: 0,
            remaining: self.total,
        }
    }

    /// A snapshot of the work counters (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn counters(&self) -> Counters {
        self.counters.get()
    }

    /// Resets all work counters to zero (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn reset_counters(&self) {
        self.counters.set(Counters::default());
    }

    /// Byte offset of the slot with the given key, if present.
    fn locate(&self, key: &[u8]) -> Option<usize> {
        assert_eq!(key.len(), T::KEY_LEN, "key length must equal Slot::KEY_LEN");
        let shard = self.shard_of(key);
        let head = self.heads[shard];
        if head == NONE {
            return None;
        }
        let count = self.counts[shard] as usize;
        let page_start = head as usize * PAGE_BYTES;
        let page = &self.pool[page_start..page_start + PAGE_BYTES];
        let mut cmps = 0u64;
        let found = search::<T>(page, count, key, &mut cmps).ok();
        bump!(self, cmp_ops, cmps);
        found.map(|pos| page_start + pos * T::SIZE)
    }

    /// Maps a key to its shard. Only the first 8 key bytes participate;
    /// longer keys sharing an 8-byte prefix land in the same shard (harmless
    /// for `Ordered`: order across shards is still by prefix).
    fn shard_of(&self, key: &[u8]) -> usize {
        let mut pad = [0u8; 8];
        let n = key.len().min(8);
        pad[..n].copy_from_slice(&key[..n]);
        let v = u64::from_be_bytes(pad);
        let bits = self.cfg.shards.trailing_zeros();
        if bits == 0 {
            return 0; // single shard; also avoids the undefined `v >> 64`
        }
        let h = match self.cfg.mode {
            ShardMode::Ordered => v,
            ShardMode::Uniform => v.wrapping_mul(FIB),
        };
        (h >> (64 - bits)) as usize
    }

    /// Returns the shard's page index, allocating the page on first use.
    fn ensure_page(&mut self, shard: usize) -> Result<u32, Error> {
        let head = self.heads[shard];
        if head != NONE {
            return Ok(head);
        }
        let old_len = self.pool.len();
        let new_len = old_len + PAGE_BYTES;
        if new_len > self.cfg.max_bytes {
            return Err(Error::CapacityExceeded {
                max_bytes: self.cfg.max_bytes,
            });
        }
        let page_idx = (old_len / PAGE_BYTES) as u32;
        self.pool.reserve(PAGE_BYTES);
        // SAFETY: the new page is left uninitialized on purpose — this is the
        // one measured unsafe of the crate. Zeroing fresh pages was benched
        // at 12x slower on the wasm allocation path (wasmtime, 32k pages:
        // 3889 us zeroed vs 316 us uninit; native: noise) — and wasm is this
        // structure's primary environment. The invariant making it sound:
        // *no byte of a page beyond `counts[shard] * T::SIZE` is ever read*.
        // Every read (search, get, iter, shifts) is bounded by the shard
        // count, and a slot's bytes are fully written before the count is
        // incremented. Consequently `Arena` exposes no whole-pool reads
        // (no `Clone`/`PartialEq`/`as_bytes`); the future snapshot writer
        // emits only the initialized prefixes of pages (`specs/03`).
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.pool.set_len(new_len);
        }
        self.heads[shard] = page_idx;
        bump!(self, pages_allocated, 1);
        Ok(page_idx)
    }
}

/// Binary search over a page's occupied slots, comparing key prefixes.
///
/// Free function (not a method) so the borrow of the page slice stays
/// independent from `&mut self` at call sites. Counting compiles away when
/// the `counters` feature is off (`COUNT` is a const `false`).
fn search<T: Slot>(page: &[u8], count: usize, key: &[u8], cmps: &mut u64) -> Result<usize, usize> {
    let mut lo = 0usize;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let off = mid * T::SIZE;
        // Branchless: adds 0 when the `counters` feature is off, which the
        // compiler folds away entirely.
        *cmps += COUNT as u64;
        match page[off..off + T::KEY_LEN].cmp(key) {
            core::cmp::Ordering::Less => lo = mid + 1,
            core::cmp::Ordering::Greater => hi = mid,
            core::cmp::Ordering::Equal => return Ok(mid),
        }
    }
    Err(lo)
}

impl<T: Slot> fmt::Debug for Arena<T> {
    /// Summary only — dumping the pool would both flood output and read
    /// uninitialized page tails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.total)
            .field("shards", &self.cfg.shards)
            .field("mode", &self.cfg.mode)
            .field("pages", &(self.pool.len() / PAGE_BYTES))
            .field("slot_size", &T::SIZE)
            .finish()
    }
}

/// Iterator over all records of an [`Arena`]; see [`Arena::iter`] for
/// ordering guarantees.
pub struct Iter<'a, T: Slot> {
    arena: &'a Arena<T>,
    shard: usize,
    idx: usize,
    remaining: usize,
}

impl<T: Slot> Iterator for Iter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while self.shard < self.arena.cfg.shards {
            if self.idx < self.arena.counts[self.shard] as usize {
                let page_start = self.arena.heads[self.shard] as usize * PAGE_BYTES;
                let off = page_start + self.idx * T::SIZE;
                self.idx += 1;
                self.remaining -= 1;
                return Some(T::read(&self.arena.pool[off..off + T::SIZE]));
            }
            self.shard += 1;
            self.idx = 0;
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Slot> ExactSizeIterator for Iter<'_, T> {}

impl<'a, T: Slot> IntoIterator for &'a Arena<T> {
    type Item = T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
