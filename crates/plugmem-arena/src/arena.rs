//! Sharded sorted arena over one flat byte pool.
//!
//! Full v2 mechanics of `specs/01-arena.md`: each shard owns a **chain of
//! range-partitioned pages** — every page holds a sorted run of keys, pages
//! in a chain follow each other in ascending key ranges (a B+-tree leaf
//! level without interior nodes; chains stay short because shard hashing
//! spreads the load). A full page **splits** in half instead of failing, and
//! a page emptied by removals is unlinked into a **free-list** for reuse, so
//! capacity is bounded only by [`ArenaCfg::max_bytes`].
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
/// L1-friendly unit of work, and every operation touches O(1) pages.
pub const PAGE_BYTES: usize = 4096;

/// Sentinel for "no page": an empty chain head, the end of a chain, or an
/// empty free-list.
const NONE: u32 = u32::MAX;

/// Fibonacci hashing multiplier (2^64 / phi), used by [`ShardMode::Uniform`].
const FIB: u64 = 0x9E37_79B9_7F4A_7C15;

/// `true` when the `counters` feature is enabled; lets hot loops keep their
/// counting code in one expression that constant-folds away otherwise.
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
    /// key order, so [`Arena::iter`] yields globally ascending keys and
    /// [`Arena::range`] scans work. Keys arriving in a narrow value range
    /// will concentrate in few shards — their chains simply grow longer;
    /// that is the trade-off for ordering.
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
    /// Bytes moved by insert/remove shifts and page splits.
    pub bytes_shifted: u64,
    /// Pages taken from the pool or the free-list.
    pub pages_allocated: u64,
    /// Steps taken walking page chains (each step peeks one page's first
    /// key — one cache line).
    pub chain_steps: u64,
    /// Page splits performed by inserts into full pages.
    pub splits: u64,
}

/// A sharded collection of fixed-size records in one contiguous byte pool,
/// sorted by key prefix within each shard.
///
/// See the [crate-level documentation](crate) for the philosophy and the
/// [module documentation](self) for the chain/split mechanics. Highlights:
///
/// - all state is `pool` + three small arrays — persisting the arena is a
///   `memcpy` of defined bytes (`specs/03`);
/// - every operation touches O(1) pages: a short chain walk (one first-key
///   peek per step), then binary search and shifts inside one 4 KiB page;
/// - capacity is bounded only by [`ArenaCfg::max_bytes`] — full pages split,
///   emptied pages are recycled through a free-list;
/// - `Arena` deliberately implements neither `Clone` nor `PartialEq`: pages
///   are allocated **uninitialized** (the measured wasm optimization), and a
///   byte-wise clone/compare would read the uninitialized tails.
pub struct Arena<T: Slot> {
    /// Page pool; length is always a multiple of [`PAGE_BYTES`]. Bytes of a
    /// page beyond `counts[page] * T::SIZE` are uninitialized and must
    /// never be read.
    pool: Vec<u8>,
    /// Shard -> first page of its chain (`NONE` = empty shard).
    heads: Vec<u32>,
    /// Page -> successor: the next page in a shard chain, or the next free
    /// page when the page sits in the free-list.
    next: Vec<u32>,
    /// Page -> number of occupied slots.
    counts: Vec<u16>,
    /// Head of the free-list of recycled pages (linked through `next`).
    free_head: u32,
    /// Total records across all shards.
    total: usize,
    /// Reusable serialization buffer for [`Arena::insert`] (`T::SIZE` bytes,
    /// allocated once) — inserts do not allocate after the first call except
    /// when growing the pool itself.
    scratch: Vec<u8>,
    cfg: ArenaCfg,
    #[cfg(feature = "counters")]
    counters: Cell<Counters>,
    /// `fn() -> T` keeps the arena `Send`/`Sync`-neutral and avoids bounding
    /// auto-traits on `T` itself.
    _marker: PhantomData<fn() -> T>,
}

/// Where a key's record lives (or would live).
struct Target {
    /// Page preceding `page` in its chain, `NONE` when `page` is the head.
    prev: u32,
    /// The chain page whose key range covers the key.
    page: u32,
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
            next: Vec::new(),
            counts: Vec::new(),
            free_head: NONE,
            total: 0,
            scratch: Vec::new(),
            cfg,
            #[cfg(feature = "counters")]
            counters: Cell::new(Counters::default()),
            _marker: PhantomData,
        })
    }

    /// Number of slots one page holds for this slot size (at least 1).
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

    /// Bytes currently allocated in the page pool (including recycled
    /// free-list pages — the pool never shrinks; emptied pages are reused).
    pub fn pool_bytes(&self) -> usize {
        self.pool.len()
    }

    /// The configuration this arena was created with.
    pub fn cfg(&self) -> &ArenaCfg {
        &self.cfg
    }

    /// Inserts a record, keeping its shard's chain sorted by key prefix.
    ///
    /// Returns `Ok(false)` if a record with the same key prefix already
    /// exists (the arena is a map keyed by prefix; existing payload is left
    /// untouched — use [`Arena::payload_mut`] to update in place). A full
    /// target page splits in half — inserts never fail on page capacity.
    ///
    /// # Errors
    ///
    /// [`Error::CapacityExceeded`] if a needed page allocation would cross
    /// [`ArenaCfg::max_bytes`].
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

        let target = match self.find_page(shard, key) {
            Some(t) => t,
            None => {
                // Empty shard: allocate its first chain page.
                let page = self.alloc_page()?;
                self.heads[shard] = page;
                Target { prev: NONE, page }
            }
        };

        let mut page = target.page;
        let mut count = self.counts[page as usize] as usize;
        let mut pos = {
            let mut cmps = 0u64;
            let found = self.search_in(page, count, key, &mut cmps);
            bump!(self, cmp_ops, cmps);
            match found {
                Ok(_) => return Ok(false),
                Err(pos) => pos,
            }
        };

        if count == Self::slots_per_page() {
            // Split the full page; afterwards (page, pos, count) address the
            // half that must receive the new record.
            (page, pos, count) = self.split(page, pos)?;
        }

        let page_start = page as usize * PAGE_BYTES;
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

        self.counts[page as usize] += 1;
        self.total += 1;
        Ok(true)
    }

    /// Splits full `page`, given the insert position `pos` inside it.
    /// Returns `(target_page, target_pos, target_count)` for the record that
    /// triggered the split.
    fn split(&mut self, page: u32, pos: usize) -> Result<(u32, usize, usize), Error> {
        let spp = Self::slots_per_page();
        let fresh = self.alloc_page()?;
        bump!(self, splits, 1);

        // Link the fresh page right after the split one.
        self.next[fresh as usize] = self.next[page as usize];
        self.next[page as usize] = fresh;

        if spp == 1 {
            // Degenerate single-slot pages (T::SIZE > PAGE_BYTES / 2): the
            // "upper half" is the whole record when the new key precedes it,
            // otherwise the fresh page simply receives the new record.
            return Ok(if pos == 0 {
                let src = page as usize * PAGE_BYTES;
                let dst = fresh as usize * PAGE_BYTES;
                self.pool.copy_within(src..src + T::SIZE, dst);
                bump!(self, bytes_shifted, T::SIZE);
                self.counts[page as usize] = 0;
                self.counts[fresh as usize] = 1;
                (page, 0, 0)
            } else {
                (fresh, 0, 0)
            });
        }

        // Move the upper half [half..spp) into the fresh page.
        let half = spp / 2;
        let moved = spp - half;
        let src = page as usize * PAGE_BYTES + half * T::SIZE;
        let dst = fresh as usize * PAGE_BYTES;
        self.pool.copy_within(src..src + moved * T::SIZE, dst);
        bump!(self, bytes_shifted, moved * T::SIZE);
        self.counts[page as usize] = half as u16;
        self.counts[fresh as usize] = moved as u16;

        // `pos` was computed against the full page; place the new record in
        // whichever half now owns that position (pos == half belongs at the
        // end of the lower page: the key sorts before the fresh page's first).
        Ok(if pos <= half {
            (page, pos, half)
        } else {
            (fresh, pos - half, moved)
        })
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
    /// existed. A page emptied by the removal is unlinked from its chain and
    /// recycled through the free-list.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN`.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        assert_eq!(key.len(), T::KEY_LEN, "key length must equal Slot::KEY_LEN");
        let shard = self.shard_of(key);
        let Some(Target { prev, page }) = self.find_page(shard, key) else {
            return false;
        };
        let count = self.counts[page as usize] as usize;
        let pos = {
            let mut cmps = 0u64;
            let found = self.search_in(page, count, key, &mut cmps);
            bump!(self, cmp_ops, cmps);
            match found {
                Ok(pos) => pos,
                Err(_) => return false,
            }
        };

        let page_start = page as usize * PAGE_BYTES;
        let slot_start = page_start + pos * T::SIZE;
        let used_end = page_start + count * T::SIZE;
        let tail = used_end - (slot_start + T::SIZE);
        if tail > 0 {
            // Shift the tail left over the removed slot; stays within the page.
            self.pool
                .copy_within(slot_start + T::SIZE..used_end, slot_start);
        }
        bump!(self, bytes_shifted, tail);
        self.counts[page as usize] -= 1;
        self.total -= 1;

        if self.counts[page as usize] == 0 {
            // Unlink the emptied page and recycle it.
            let successor = self.next[page as usize];
            if prev == NONE {
                self.heads[shard] = successor;
            } else {
                self.next[prev as usize] = successor;
            }
            self.next[page as usize] = self.free_head;
            self.free_head = page;
        }
        true
    }

    /// Iterates all records shard by shard, following each shard's chain.
    ///
    /// In [`ShardMode::Ordered`] the shard index order equals key order and
    /// chains are range-partitioned, so this yields **globally ascending
    /// keys**. In [`ShardMode::Uniform`] the global order is unspecified
    /// (each shard is still internally sorted).
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            arena: self,
            shard: 0,
            page: NONE,
            idx: 0,
            remaining: self.total,
        }
    }

    /// Iterates records whose key prefix lies in `[from, to)` — `from`
    /// inclusive, `to` exclusive — in ascending key order.
    ///
    /// Only meaningful when shard order equals key order, hence restricted
    /// to [`ShardMode::Ordered`].
    ///
    /// # Panics
    ///
    /// Panics if the arena is in [`ShardMode::Uniform`], or if either bound's
    /// length differs from `T::KEY_LEN`.
    pub fn range<'a>(&'a self, from: &[u8], to: &'a [u8]) -> Range<'a, T> {
        assert_eq!(
            self.cfg.mode,
            ShardMode::Ordered,
            "range scans require ShardMode::Ordered"
        );
        assert_eq!(
            from.len(),
            T::KEY_LEN,
            "key length must equal Slot::KEY_LEN"
        );
        assert_eq!(to.len(), T::KEY_LEN, "key length must equal Slot::KEY_LEN");

        // Position on the first record with key >= `from`.
        let shard = self.shard_of(from);
        let mut page = NONE;
        let mut idx = 0usize;
        if let Some(t) = self.find_page(shard, from) {
            let count = self.counts[t.page as usize] as usize;
            let mut cmps = 0u64;
            let pos = match self.search_in(t.page, count, from, &mut cmps) {
                Ok(p) | Err(p) => p,
            };
            bump!(self, cmp_ops, cmps);
            if pos < count {
                page = t.page;
                idx = pos;
            } else {
                // Past the last record of the covering page: continue with
                // the next page in this chain, or fall through to the next
                // shard.
                page = self.next[t.page as usize];
            }
        }
        Range {
            arena: self,
            // The shard the iterator moves to once the current chain (if
            // any) is exhausted.
            shard: shard + 1,
            page,
            idx,
            to,
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

    /// Walks the shard's chain to the page whose key range covers `key`.
    /// `None` when the shard is empty.
    fn find_page(&self, shard: usize, key: &[u8]) -> Option<Target> {
        let head = self.heads[shard];
        if head == NONE {
            return None;
        }
        let mut prev = NONE;
        let mut page = head;
        let mut steps = 0u64;
        loop {
            let nxt = self.next[page as usize];
            // Advance while the next page's range can still contain the key
            // (its first key is <= key). Peeking a first key touches one
            // cache line.
            if nxt == NONE || self.first_key(nxt) > key {
                break;
            }
            steps += COUNT as u64;
            prev = page;
            page = nxt;
        }
        bump!(self, chain_steps, steps);
        let _ = steps; // read only by the counters feature
        Some(Target { prev, page })
    }

    /// First key of a page. Caller guarantees the page is non-empty (every
    /// page in a chain holds at least one record — emptied pages are
    /// unlinked immediately).
    fn first_key(&self, page: u32) -> &[u8] {
        let start = page as usize * PAGE_BYTES;
        &self.pool[start..start + T::KEY_LEN]
    }

    /// Binary search inside a page; wraps the free-function search with the
    /// page slice resolution.
    fn search_in(
        &self,
        page: u32,
        count: usize,
        key: &[u8],
        cmps: &mut u64,
    ) -> Result<usize, usize> {
        let start = page as usize * PAGE_BYTES;
        search::<T>(&self.pool[start..start + PAGE_BYTES], count, key, cmps)
    }

    /// Byte offset of the slot with the given key, if present.
    fn locate(&self, key: &[u8]) -> Option<usize> {
        assert_eq!(key.len(), T::KEY_LEN, "key length must equal Slot::KEY_LEN");
        let shard = self.shard_of(key);
        let Target { page, .. } = self.find_page(shard, key)?;
        let count = self.counts[page as usize] as usize;
        let mut cmps = 0u64;
        let found = self.search_in(page, count, key, &mut cmps).ok();
        bump!(self, cmp_ops, cmps);
        found.map(|pos| page as usize * PAGE_BYTES + pos * T::SIZE)
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

    /// Takes a page from the free-list, or grows the pool by one page.
    fn alloc_page(&mut self) -> Result<u32, Error> {
        if self.free_head != NONE {
            let page = self.free_head;
            self.free_head = self.next[page as usize];
            self.next[page as usize] = NONE;
            self.counts[page as usize] = 0;
            bump!(self, pages_allocated, 1);
            return Ok(page);
        }

        let old_len = self.pool.len();
        let new_len = old_len + PAGE_BYTES;
        if new_len > self.cfg.max_bytes {
            return Err(Error::CapacityExceeded {
                max_bytes: self.cfg.max_bytes,
            });
        }
        let page = (old_len / PAGE_BYTES) as u32;
        self.pool.reserve(PAGE_BYTES);
        // SAFETY: the new page is left uninitialized on purpose — this is the
        // one measured unsafe of the crate. Zeroing fresh pages was benched
        // at 12x slower on the wasm allocation path (wasmtime, 32k pages:
        // 3889 us zeroed vs 316 us uninit; native: noise) — and wasm is this
        // structure's primary environment. The invariant making it sound:
        // *no byte of a page beyond `counts[page] * T::SIZE` is ever read*.
        // Every read (search, get, iter, range, first_key, shifts) is
        // bounded by the page count, and a slot's bytes are fully written
        // before the count is incremented. Consequently `Arena` exposes no
        // whole-pool reads (no `Clone`/`PartialEq`/`as_bytes`); the future
        // snapshot writer emits only the initialized prefixes of pages
        // (`specs/03`).
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.pool.set_len(new_len);
        }
        self.next.push(NONE);
        self.counts.push(0);
        bump!(self, pages_allocated, 1);
        Ok(page)
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
    /// Current chain page; `NONE` means "advance to the next shard's head".
    page: u32,
    idx: usize,
    remaining: usize,
}

impl<T: Slot> Iterator for Iter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if self.page == NONE {
                if self.shard >= self.arena.cfg.shards {
                    return None;
                }
                self.page = self.arena.heads[self.shard];
                self.idx = 0;
                self.shard += 1;
                continue;
            }
            if self.idx < self.arena.counts[self.page as usize] as usize {
                let off = self.page as usize * PAGE_BYTES + self.idx * T::SIZE;
                self.idx += 1;
                self.remaining -= 1;
                return Some(T::read(&self.arena.pool[off..off + T::SIZE]));
            }
            self.page = self.arena.next[self.page as usize];
            self.idx = 0;
        }
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

/// Iterator over a key range of an [`Arena`]; see [`Arena::range`].
pub struct Range<'a, T: Slot> {
    arena: &'a Arena<T>,
    shard: usize,
    /// Current chain page; `NONE` means "advance to the next shard's head".
    page: u32,
    idx: usize,
    /// Exclusive upper bound on key prefixes.
    to: &'a [u8],
}

impl<T: Slot> Iterator for Range<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if self.page == NONE {
                if self.shard >= self.arena.cfg.shards {
                    return None;
                }
                self.page = self.arena.heads[self.shard];
                self.idx = 0;
                self.shard += 1;
                continue;
            }
            if self.idx < self.arena.counts[self.page as usize] as usize {
                let off = self.page as usize * PAGE_BYTES + self.idx * T::SIZE;
                if &self.arena.pool[off..off + T::KEY_LEN] >= self.to {
                    // Keys only grow from here on — the scan is complete.
                    return None;
                }
                self.idx += 1;
                return Some(T::read(&self.arena.pool[off..off + T::SIZE]));
            }
            self.page = self.arena.next[self.page as usize];
            self.idx = 0;
        }
    }
}
