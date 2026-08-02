//! Sharded sorted arena over one flat byte pool.
//!
//! Full v2 mechanics of: each shard owns a **chain of
//! range-partitioned pages** — every page holds a sorted run of keys, pages
//! in a chain follow each other in ascending key ranges (a B+-tree leaf
//! level). A full page **splits** in half instead of failing, and
//! a page emptied by removals is unlinked into a **free-list** for reuse, so
//! capacity is bounded only by [`ArenaCfg::max_bytes`].
//!
//! Chains are *not* assumed to be short. Shard hashing spreads uniform keys,
//! but an [`ShardMode::Ordered`] arena whose keys share their leading bytes
//! concentrates everything in one shard — every edge of a hub entity, every
//! timestamp of a real clock — and that chain grows with the record count. So
//! the interior level is a flat runtime **page directory** ([`Arena::dir`]):
//! locating a page is a binary search over it, O(log pages), never a walk. The
//! directory is derived state, rebuilt by the load walk and absent from the
//! on-disk image.
//!
//! The structure is tuned for **wasm environments first**: `no_std + alloc`,
//! a single linear byte pool (snapshot = memcpy), no threads, and the one
//! measured `unsafe` (uninitialized page allocation) whose entire benefit is
//! on the wasm allocation path — see [`Arena::insert`] internals and the
//! crate-level documentation for the numbers.
//!
//! # Overlay opens
//!
//! [`Arena::load_overlay`] opens an arena whose existing pages are **borrowed**
//! from a longer-lived buffer — typically a memory-mapped file — while the
//! arena stays fully mutable. Because an arena mutates pages *in place* (slot
//! shifts, page splits), it cannot use the append-only tail of
//! [`BlobHeap`](crate::BlobHeap): instead the first write to a borrowed page
//! copies just *that* page into an owned overlay, and pages grown after open
//! live in an owned tail (per-page copy-on-write). The borrowed base is never
//! mutated and never cloned as a whole, so a multi-gigabyte arena can be
//! *written to* while resident only in the pages it actually touches. Staying
//! true to the crate's flat philosophy, the overlay is two flat `Vec`s (a
//! dense redirect and one contiguous copy pool) — no `Box`, no map. See
//! `examples/overlay.rs`.

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

#[cfg(feature = "counters")]
use core::cell::Cell;

use crate::error::Error;
use crate::paged::Paged;
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

/// Fixed prefix of the serialized metadata section (see
/// [`Arena::dump_meta`]).
const IMAGE_HEADER: usize = 24;

/// Serialized width of one page index entry (`heads`, `next` — little-endian
/// `u32`), used to size and offset those metadata arrays.
const PAGE_IDX_BYTES: usize = core::mem::size_of::<u32>();

/// Serialized width of one page's slot count (`counts` — little-endian `u16`).
const COUNT_BYTES: usize = core::mem::size_of::<u16>();

/// Serialized bytes of per-page metadata: one `next` index plus one `counts`
/// entry.
const PAGE_META_BYTES: usize = PAGE_IDX_BYTES + COUNT_BYTES;

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShardMode {
    /// Shard = Fibonacci hash of the key's leading bytes. Spreads any key
    /// distribution (including sequential ids) evenly across shards. Global
    /// iteration order is *not* the key order. Use for lookup tables.
    Uniform,
    /// Shard = top bits of the key's leading bytes. Shard index order equals
    /// key order, so [`Arena::iter`] yields globally ascending keys and
    /// [`Arena::range`] scans work. Keys arriving in a narrow value range
    /// concentrate in few shards — commonly in exactly one, since the top bits
    /// of a real timestamp or of a repeated id prefix are constant. Their
    /// chains then grow with the record count; the page directory keeps
    /// locating a page logarithmic anyway, so this costs ordering, not
    /// asymptotics.
    Ordered,
}

/// Arena configuration.
///
/// Stored verbatim in snapshots later, so everything that
/// affects byte interpretation lives here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// First-key peeks spent locating a page (one cache line each). With the
    /// page directory this is the binary search's probe count, so it grows
    /// with the *logarithm* of a shard's page count; a value that grows
    /// linearly with the record count means the directory stopped being used.
    pub chain_steps: u64,
    /// Page splits performed by inserts into full pages.
    pub splits: u64,
}

/// A sharded collection of fixed-size records in one contiguous byte pool,
/// sorted by key prefix within each shard.
///
/// See the [crate-level documentation](crate) for the philosophy and the
/// module documentation for the chain/split mechanics. Highlights:
///
/// - all state is `pool` + small metadata arrays — persisting the arena is a
///   `memcpy` of defined bytes;
/// - every operation touches O(1) pages: a binary search over the page
///   directory, then binary search and shifts inside one 4 KiB page — so a
///   skewed key distribution costs a logarithm, not a scan;
/// - capacity is bounded only by [`ArenaCfg::max_bytes`] — full pages split,
///   emptied pages are recycled through a free-list;
/// - `Arena` deliberately implements neither `Clone` nor `PartialEq`: pages
///   are allocated **uninitialized** (the measured wasm optimization), and a
///   byte-wise clone/compare would read the uninitialized tails.
pub struct Arena<'a, T: Slot> {
    /// Page pool; length is always a multiple of [`PAGE_BYTES`]. Bytes of a
    /// page beyond `counts[page] * T::SIZE` are uninitialized and must
    /// never be read.
    ///
    /// A [`Paged`] backing so the arena can either own the pool (`new`/`load` —
    /// the writable default, and the only shape on wasm32) or borrow a base
    /// from a longer-lived buffer such as a memory-mapped snapshot
    /// (`load_borrowed`/`load_overlay` —). The overlay open borrows
    /// the base read-only and copies just the touched pages up on first write
    /// (per-page copy-on-write), so a mapped database is mutated without
    /// cloning it. The backing is `alloc`-only (no `std`/`Arc`), so the crate
    /// stays `no_std`, and the lifetime `'a` lets the compiler prove the arena
    /// never outlives its base. The uninitialized-tail invariant is untouched:
    /// base pages are a fully-initialized dumped image, overlay copies are
    /// taken from them, and freshly grown pages keep the same
    /// read-only-up-to-`count` contract as before.
    pool: Paged<'a, PAGE_BYTES>,
    /// Shard -> first page of its chain (`NONE` = empty shard).
    heads: Vec<u32>,
    /// Page -> successor: the next page in a shard chain, or the next free
    /// page when the page sits in the free-list.
    next: Vec<u32>,
    /// Page -> predecessor in a shard chain. Rebuilt from `next` on load and
    /// kept in memory so ordered arenas can scan a range newest-first without
    /// changing the on-disk metadata format.
    prev: Vec<u32>,
    /// Page -> number of occupied slots.
    counts: Vec<u16>,
    /// Shard -> last page of its chain (`NONE` when empty). Runtime-only
    /// metadata paired with `heads` for reverse ordered scans.
    tails: Vec<u32>,
    /// Every chain page, shard after shard, each shard's run in chain (=
    /// ascending key) order — the **page directory** that turns locating a
    /// page into a binary search instead of a walk. See [`Arena::find_page`].
    ///
    /// Runtime-only, like `prev`/`tails`: rebuilt by the load walk, absent
    /// from the on-disk image. Flat by design — one `u32` per 4 KiB page
    /// (≈0.1% overhead), no `Box`, no map.
    dir: Vec<u32>,
    /// Shard -> start of that shard's run in `dir`; `shards + 1` entries, the
    /// last one being `dir.len()`, so shard `s` owns `dir[dir_at[s]..
    /// dir_at[s + 1]]`.
    dir_at: Vec<u32>,
    /// Head of the free-list of recycled pages (linked through `next`).
    free_head: u32,
    /// Total records across all shards.
    total: usize,
    /// Reusable serialization buffer for [`Arena::insert`] (`T::SIZE` bytes,
    /// allocated once) — inserts do not allocate after the first call except
    /// when growing the pool itself.
    scratch: Vec<u8>,
    /// Reusable buffer for the cross-page slot moves in [`Arena::split`]. A
    /// split reads a run of slots out of one page and writes it into another;
    /// under the overlay backing those may be distinct owned pages, so the
    /// bytes route through this buffer instead of a single in-pool
    /// `copy_within`. Grown once; splits are amortized and off the recall path.
    split_buf: Vec<u8>,
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
    /// Index of `page` in [`Arena::dir`] — the mutation paths (split, page
    /// recycling) update the directory there without searching for it again.
    at: usize,
}

impl<'a, T: Slot> Arena<'a, T> {
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
            pool: Paged::owned_empty(),
            heads: alloc::vec![NONE; cfg.shards],
            next: Vec::new(),
            prev: Vec::new(),
            counts: Vec::new(),
            tails: alloc::vec![NONE; cfg.shards],
            dir: Vec::new(),
            dir_at: alloc::vec![0; cfg.shards + 1],
            free_head: NONE,
            total: 0,
            scratch: Vec::new(),
            split_buf: Vec::new(),
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
                self.tails[shard] = page;
                let at = self.dir_at[shard] as usize;
                self.dir_insert(shard, at, page);
                Target {
                    prev: NONE,
                    page,
                    at,
                }
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
            (page, pos, count) = self.split(shard, target.at, pos)?;
        }

        let slot_start = pos * T::SIZE;
        let used_end = count * T::SIZE;
        let shifted = used_end - slot_start;
        let bytes = self.pool.page_mut(page);
        if shifted > 0 {
            // Shift the sorted tail right by one slot; stays within the page.
            bytes.copy_within(slot_start..used_end, slot_start + T::SIZE);
        }
        bytes[slot_start..slot_start + T::SIZE].copy_from_slice(slot);
        bump!(self, bytes_shifted, shifted);

        self.counts[page as usize] += 1;
        self.total += 1;
        Ok(true)
    }

    /// Splits full `page` — the chain page at directory index `at` of `shard`
    /// — given the insert position `pos` inside it. Returns
    /// `(target_page, target_pos, target_count)` for the record that triggered
    /// the split.
    fn split(&mut self, shard: usize, at: usize, pos: usize) -> Result<(u32, usize, usize), Error> {
        let spp = Self::slots_per_page();
        let page = self.dir[at];
        let fresh = self.alloc_page()?;
        bump!(self, splits, 1);

        // Link the fresh page right after the split one, in the chain and in
        // the directory alike.
        let successor = self.next[page as usize];
        self.next[fresh as usize] = successor;
        self.prev[fresh as usize] = page;
        self.next[page as usize] = fresh;
        if successor == NONE {
            self.tails[shard] = fresh;
        } else {
            self.prev[successor as usize] = fresh;
        }
        self.dir_insert(shard, at + 1, fresh);

        if pos == spp {
            // Appending past the last key: splitting in half would leave both
            // pages half empty forever, because nothing will ever sort into
            // the lower one again. Every id and timestamp this crate stores
            // arrives ascending, so this is the common case — hand the record
            // a fresh page and leave the full one full. Pages stay 100% packed
            // under a monotonic load instead of 50%.
            return Ok((fresh, 0, 0));
        }

        if spp == 1 {
            // Degenerate single-slot pages (T::SIZE > PAGE_BYTES / 2): `pos`
            // can only be 0 here (`pos == spp` returned above), so the record
            // sorts before the existing one, which moves to the fresh page.
            self.move_slots(page, 0, fresh, T::SIZE);
            bump!(self, bytes_shifted, T::SIZE);
            self.counts[page as usize] = 0;
            self.counts[fresh as usize] = 1;
            return Ok((page, 0, 0));
        }

        // Move the upper half [half..spp) into the fresh page.
        let half = spp / 2;
        let moved = spp - half;
        self.move_slots(page, half * T::SIZE, fresh, moved * T::SIZE);
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

    /// Copies `len` bytes from `src_page` (starting at in-page offset
    /// `src_off`) to the start of `dst_page`. The two pages may be distinct
    /// owned pages under the overlay backing, so the bytes route through the
    /// reusable [`Arena::split_buf`] rather than a single in-pool
    /// `copy_within` — one immutable read of the source, then one write of the
    /// destination (which copies-up if it is still a borrowed base page).
    fn move_slots(&mut self, src_page: u32, src_off: usize, dst_page: u32, len: usize) {
        let mut buf = core::mem::take(&mut self.split_buf);
        buf.clear();
        buf.extend_from_slice(&self.pool.page(src_page)[src_off..src_off + len]);
        self.pool.page_mut(dst_page)[..len].copy_from_slice(&buf);
        self.split_buf = buf;
    }

    /// Returns the record with the given key prefix, if present.
    ///
    /// # Panics
    ///
    /// Panics if `key.len() != T::KEY_LEN` (a caller bug, not a data
    /// condition — mirrors slice indexing).
    pub fn get(&self, key: &[u8]) -> Option<T> {
        self.locate(key).map(|off| {
            let (page, rel) = (off / PAGE_BYTES, off % PAGE_BYTES);
            T::read(&self.pool.page(page as u32)[rel..rel + T::SIZE])
        })
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
        self.locate(key).map(|off| {
            let (page, rel) = (off / PAGE_BYTES, off % PAGE_BYTES);
            &self.pool.page(page as u32)[rel..rel + T::SIZE]
        })
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
        let (page, rel) = (off / PAGE_BYTES, off % PAGE_BYTES);
        Some(&mut self.pool.page_mut(page as u32)[rel + T::KEY_LEN..rel + T::SIZE])
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
        let Some(Target { prev, page, at }) = self.find_page(shard, key) else {
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

        let slot_start = pos * T::SIZE;
        let used_end = count * T::SIZE;
        let tail = used_end - (slot_start + T::SIZE);
        if tail > 0 {
            // Shift the tail left over the removed slot; stays within the page.
            self.pool
                .page_mut(page)
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
            if successor == NONE {
                self.tails[shard] = prev;
            } else {
                self.prev[successor as usize] = prev;
            }
            self.next[page as usize] = self.free_head;
            self.prev[page as usize] = NONE;
            self.free_head = page;
            self.dir_remove(shard, at);
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
    pub fn range<'s>(&'s self, from: &[u8], to: &'s [u8]) -> Range<'s, T> {
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

    /// Iterates records whose key prefix lies in `[from, to)` — `from`
    /// inclusive, `to` exclusive — in descending key order.
    ///
    /// This is the bounded newest-first counterpart to [`Arena::range`]. It
    /// is restricted to ordered arenas and keeps the predecessor links in
    /// runtime metadata, so the on-disk arena format remains unchanged.
    pub fn range_rev<'s>(&'s self, from: &'s [u8], to: &'s [u8]) -> RangeRev<'s, T> {
        assert_eq!(
            self.cfg.mode,
            ShardMode::Ordered,
            "reverse range scans require ShardMode::Ordered"
        );
        assert_eq!(
            from.len(),
            T::KEY_LEN,
            "key length must equal Slot::KEY_LEN"
        );
        assert_eq!(to.len(), T::KEY_LEN, "key length must equal Slot::KEY_LEN");

        if from >= to {
            return RangeRev {
                arena: self,
                shard: 0,
                page: NONE,
                idx: 0,
                from,
            };
        }

        // Position immediately before `to`: an exact key is excluded, while
        // an insertion point after the page's last key starts at that last
        // key. If the covering page has no key below `to`, walk to its
        // predecessor.
        let shard = self.shard_of(to);
        let mut page = NONE;
        let mut idx = 0usize;
        if let Some(t) = self.find_page(shard, to) {
            let count = self.counts[t.page as usize] as usize;
            let mut cmps = 0u64;
            let pos = match self.search_in(t.page, count, to, &mut cmps) {
                Ok(p) | Err(p) => p,
            };
            bump!(self, cmp_ops, cmps);
            if pos > 0 {
                page = t.page;
                idx = pos;
            } else {
                page = self.prev[t.page as usize];
                if page != NONE {
                    idx = self.counts[page as usize] as usize;
                }
            }
        }

        RangeRev {
            arena: self,
            // The shard the iterator moves to once the current chain (if
            // any) is exhausted.
            shard,
            page,
            idx,
            from,
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

    /// Appends the arena's metadata section to `out`.
    ///
    /// Layout (all little-endian): `[shards u32][pages u32][free_head u32]
    /// [total u64][mode u8][reserved 3]`, then `heads` (`shards × u32`),
    /// `next` (`pages × u32`), `counts` (`pages × u16`). Together with
    /// [`Arena::dump_pool`] this is the complete state; both dumps are
    /// canonical — dump → [`Arena::load`] → dump reproduces identical
    /// bytes.
    pub fn dump_meta(&self, out: &mut Vec<u8>) {
        let pages = self.pool.len() / PAGE_BYTES;
        debug_assert!(self.cfg.shards <= u32::MAX as usize && pages < u32::MAX as usize);
        out.reserve(IMAGE_HEADER + self.heads.len() * PAGE_IDX_BYTES + pages * PAGE_META_BYTES);
        out.extend_from_slice(&(self.cfg.shards as u32).to_le_bytes());
        out.extend_from_slice(&(pages as u32).to_le_bytes());
        out.extend_from_slice(&self.free_head.to_le_bytes());
        out.extend_from_slice(&(self.total as u64).to_le_bytes());
        out.push(match self.cfg.mode {
            ShardMode::Uniform => 0,
            ShardMode::Ordered => 1,
        });
        out.extend_from_slice(&[0u8; 3]);
        for &head in &self.heads {
            out.extend_from_slice(&head.to_le_bytes());
        }
        for &next in &self.next {
            out.extend_from_slice(&next.to_le_bytes());
        }
        for &count in &self.counts {
            out.extend_from_slice(&count.to_le_bytes());
        }
    }

    /// Appends the arena's pool section to `out`.
    ///
    /// Each page contributes its initialized prefix (`counts[page] ×
    /// SIZE` bytes) followed by zero padding to [`PAGE_BYTES`]: the
    /// uninitialized page tails (see [`Arena::insert`] internals) are
    /// never read, and zero-filling them makes the image canonical and
    /// leak-free.
    pub fn dump_pool(&self, out: &mut Vec<u8>) {
        out.reserve(self.pool.len());
        for (page, &count) in self.counts.iter().enumerate() {
            let used = count as usize * T::SIZE;
            out.extend_from_slice(&self.pool.page(page as u32)[..used]);
            out.resize(out.len() + (PAGE_BYTES - used), 0);
        }
    }

    /// Rebuilds an arena from its two dumped sections.
    ///
    /// The input is **untrusted**: every metadata invariant is checked
    /// before adoption and any inconsistency returns
    /// [`Error::Corrupt`] — this method never panics on arbitrary bytes.
    /// Cost is O(pages) on top of the pool copy:
    ///
    /// - exact section lengths; `cfg` agreement (shards, mode);
    /// - per-page slot counts within `PAGE_BYTES / SIZE`;
    /// - every page reachable exactly once — shard chains and the
    ///   free-list are walked with a visited bitmap (no cycles, no shared
    ///   or orphan pages); chain pages are non-empty, free pages empty;
    /// - chain pages ascend by key range and sit in the shard their first
    ///   key maps to; the record total matches the page counts.
    ///
    /// Slot *content* is not validated beyond the per-page first/last key
    /// reads — record payloads are semantically validated lazily by the
    /// owning engine.
    ///
    /// # Errors
    ///
    /// [`Error::BadSlot`] / [`Error::BadShardCount`] for an invalid `cfg`
    /// (same gates as [`Arena::new`]); [`Error::Corrupt`] for any image
    /// inconsistency.
    pub fn load(cfg: ArenaCfg, meta: &[u8], pool: &[u8]) -> Result<Self, Error> {
        Self::load_impl(cfg, meta, pool, Paged::owned_from(pool.to_vec()))
    }

    /// Rebuilds an arena that **borrows** its page pool from a longer-lived
    /// buffer (a memory-mapped snapshot) instead of copying it.
    /// Validation is identical to [`Arena::load`] — it reads each page's
    /// first and last keys, so it touches one OS page per arena page (the
    /// key-order check needs them), but no pool byte is copied.
    ///
    /// # Errors
    ///
    /// Same as [`Arena::load`].
    pub fn load_borrowed(cfg: ArenaCfg, meta: &[u8], pool: &'a [u8]) -> Result<Self, Error> {
        Self::load_impl(cfg, meta, pool, Paged::borrowed(pool))
    }

    /// Opens an arena over a borrowed base for the **overlay** write path: the
    /// base pages are mapped read-only, and the first write to any base page
    /// copies just that page into owned storage (per-page copy-on-write, see
    /// [`Paged`](crate::paged)), while pages grown after open live in an owned
    /// tail. Unlike [`Arena::load_borrowed`] the returned arena is fully
    /// mutable — inserts, removals and splits work — yet the borrowed base is
    /// never cloned as a whole and never mutated, so a memory-mapped database
    /// can be written to while resident only in the pages it actually touches.
    /// Validation is identical to [`Arena::load`].
    ///
    /// # Errors
    ///
    /// Same as [`Arena::load`].
    pub fn load_overlay(cfg: ArenaCfg, meta: &[u8], pool: &'a [u8]) -> Result<Self, Error> {
        Self::load_impl(cfg, meta, pool, Paged::borrowed(pool))
    }

    /// Shared body of [`Arena::load`] / [`Arena::load_borrowed`] /
    /// [`Arena::load_overlay`]: validates the untrusted `meta`/`pool` image and
    /// adopts `backing` as the pool.
    fn load_impl(
        cfg: ArenaCfg,
        meta: &[u8],
        pool: &[u8],
        backing: Paged<'a, PAGE_BYTES>,
    ) -> Result<Self, Error> {
        let mut arena = Self::new(cfg)?;
        if meta.len() < IMAGE_HEADER {
            return Err(Error::Corrupt("arena meta shorter than its header"));
        }
        let shards = u32::from_le_bytes(meta[0..4].try_into().unwrap()) as usize;
        let pages = u32::from_le_bytes(meta[4..8].try_into().unwrap()) as usize;
        let free_head = u32::from_le_bytes(meta[8..12].try_into().unwrap());
        let total = u64::from_le_bytes(meta[12..20].try_into().unwrap());
        let mode = meta[20];
        if meta[21..24] != [0u8; 3] {
            return Err(Error::Corrupt("arena meta reserved bytes must be zero"));
        }
        if shards != cfg.shards {
            return Err(Error::Corrupt("arena meta shard count disagrees with cfg"));
        }
        let want_mode = match cfg.mode {
            ShardMode::Uniform => 0u8,
            ShardMode::Ordered => 1,
        };
        if mode != want_mode {
            return Err(Error::Corrupt("arena meta shard mode disagrees with cfg"));
        }
        if pages as u64 >= u64::from(NONE) {
            return Err(Error::Corrupt("arena page count overflows the index space"));
        }
        let want_meta = IMAGE_HEADER as u64
            + shards as u64 * PAGE_IDX_BYTES as u64
            + pages as u64 * PAGE_META_BYTES as u64;
        if meta.len() as u64 != want_meta {
            return Err(Error::Corrupt("arena meta length mismatch"));
        }
        if pool.len() as u64 != pages as u64 * PAGE_BYTES as u64 {
            return Err(Error::Corrupt("arena pool length mismatch"));
        }

        let mut heads = Vec::with_capacity(shards);
        for i in 0..shards {
            let at = IMAGE_HEADER + i * PAGE_IDX_BYTES;
            heads.push(u32::from_le_bytes(
                meta[at..at + PAGE_IDX_BYTES].try_into().unwrap(),
            ));
        }
        let next_base = IMAGE_HEADER + shards * PAGE_IDX_BYTES;
        let mut next = Vec::with_capacity(pages);
        for i in 0..pages {
            let at = next_base + i * PAGE_IDX_BYTES;
            next.push(u32::from_le_bytes(
                meta[at..at + PAGE_IDX_BYTES].try_into().unwrap(),
            ));
        }
        let counts_base = next_base + pages * PAGE_IDX_BYTES;
        let mut counts = Vec::with_capacity(pages);
        for i in 0..pages {
            let at = counts_base + i * COUNT_BYTES;
            counts.push(u16::from_le_bytes(
                meta[at..at + COUNT_BYTES].try_into().unwrap(),
            ));
        }
        let mut prev_links = alloc::vec![NONE; pages];
        let mut tails = alloc::vec![NONE; shards];

        let spp = Self::slots_per_page();
        if counts.iter().any(|&c| c as usize > spp) {
            return Err(Error::Corrupt("arena page count exceeds slots per page"));
        }

        // Reachability walk: every page must appear exactly once across all
        // shard chains and the free-list; the bitmap catches cycles, shared
        // pages and (after the walks) orphans.
        let mut seen = alloc::vec![false; pages];
        let mut live = 0u64;
        // The walk already visits every chain page in shard-then-key order —
        // exactly the page directory's contents, so it is built here for free.
        let mut dir = Vec::with_capacity(pages);
        let mut dir_at = alloc::vec![0u32; shards + 1];
        for (shard, &head) in heads.iter().enumerate() {
            dir_at[shard] = dir.len() as u32;
            let mut page = head;
            let mut predecessor = NONE;
            while page != NONE {
                let p = page as usize;
                if p >= pages {
                    return Err(Error::Corrupt("arena chain page out of bounds"));
                }
                if core::mem::replace(&mut seen[p], true) {
                    return Err(Error::Corrupt("arena page linked more than once"));
                }
                let count = counts[p] as usize;
                if count == 0 {
                    return Err(Error::Corrupt("arena chain contains an empty page"));
                }
                live += count as u64;
                let first = &pool[p * PAGE_BYTES..p * PAGE_BYTES + T::KEY_LEN];
                if arena.shard_of(first) != shard {
                    return Err(Error::Corrupt("arena page sits in the wrong shard"));
                }
                if predecessor != NONE {
                    let pr = predecessor as usize;
                    let last_at = pr * PAGE_BYTES + (counts[pr] as usize - 1) * T::SIZE;
                    if pool[last_at..last_at + T::KEY_LEN] >= *first {
                        return Err(Error::Corrupt("arena chain pages out of key order"));
                    }
                }
                prev_links[p] = predecessor;
                if next[p] == NONE {
                    tails[shard] = page;
                }
                dir.push(page);
                predecessor = page;
                page = next[p];
            }
        }
        dir_at[shards] = dir.len() as u32;
        let mut page = free_head;
        while page != NONE {
            let p = page as usize;
            if p >= pages {
                return Err(Error::Corrupt("arena free page out of bounds"));
            }
            if core::mem::replace(&mut seen[p], true) {
                return Err(Error::Corrupt("arena page linked more than once"));
            }
            if counts[p] != 0 {
                return Err(Error::Corrupt("arena free page has a nonzero count"));
            }
            page = next[p];
        }
        if seen.iter().any(|&s| !s) {
            return Err(Error::Corrupt("arena has an orphan page"));
        }
        if live != total {
            return Err(Error::Corrupt(
                "arena record total disagrees with page counts",
            ));
        }

        arena.pool = backing;
        arena.heads = heads;
        arena.next = next;
        arena.prev = prev_links;
        arena.counts = counts;
        arena.tails = tails;
        arena.dir = dir;
        arena.dir_at = dir_at;
        arena.free_head = free_head;
        arena.total = total as usize;
        Ok(arena)
    }

    /// Locates the chain page whose key range covers `key`. `None` when the
    /// shard is empty.
    ///
    /// A shard's pages are range-partitioned and ascending, so the covering
    /// page is the **last** one whose first key is `<= key` (or the head, when
    /// `key` sorts below every page). The directory holds exactly that
    /// sequence, so this is a binary search over `dir`: O(log pages) first-key
    /// peeks instead of one per page.
    ///
    /// That difference is the whole point of the directory. A chain is short
    /// only while the shard hash spreads the load; an `Ordered` arena whose
    /// keys share their leading 8 bytes (every edge of one hub entity, every
    /// timestamp of a real clock) concentrates in a single shard, and a walk
    /// there costs O(records) per operation — quadratic over a load.
    fn find_page(&self, shard: usize, key: &[u8]) -> Option<Target> {
        let (lo, hi) = self.dir_run(shard);
        if lo == hi {
            debug_assert_eq!(self.heads[shard], NONE, "empty directory run, live chain");
            return None;
        }
        // Partition point of `first_key(page) <= key` over the run *after* the
        // head: the head covers everything below the second page's first key,
        // so it never needs a probe — and a single-page shard costs nothing.
        let mut steps = 0u64;
        let (mut left, mut right) = (lo + 1, hi);
        while left < right {
            let mid = left + (right - left) / 2;
            steps += COUNT as u64;
            if self.first_key(self.dir[mid]) <= key {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        bump!(self, chain_steps, steps);
        let _ = steps; // read only by the counters feature
        let at = left - 1;
        Some(Target {
            prev: if at > lo { self.dir[at - 1] } else { NONE },
            page: self.dir[at],
            at,
        })
    }

    /// The half-open `dir` range owned by `shard`.
    fn dir_run(&self, shard: usize) -> (usize, usize) {
        (self.dir_at[shard] as usize, self.dir_at[shard + 1] as usize)
    }

    /// Records `page` at directory index `at`, which must be inside (or at the
    /// end of) `shard`'s run. Shifts the following runs by one.
    fn dir_insert(&mut self, shard: usize, at: usize, page: u32) {
        debug_assert!(self.dir_at[shard] as usize <= at && at <= self.dir_at[shard + 1] as usize);
        self.dir.insert(at, page);
        for start in &mut self.dir_at[shard + 1..] {
            *start += 1;
        }
    }

    /// Drops directory index `at` from `shard`'s run.
    fn dir_remove(&mut self, shard: usize, at: usize) {
        debug_assert!(self.dir_at[shard] as usize <= at && at < self.dir_at[shard + 1] as usize);
        self.dir.remove(at);
        for start in &mut self.dir_at[shard + 1..] {
            *start -= 1;
        }
    }

    /// First key of a page. Caller guarantees the page is non-empty (every
    /// page in a chain holds at least one record — emptied pages are
    /// unlinked immediately).
    fn first_key(&self, page: u32) -> &[u8] {
        &self.pool.page(page)[..T::KEY_LEN]
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
        search::<T>(self.pool.page(page), count, key, cmps)
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
            self.prev[page as usize] = NONE;
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
        // Grow the owned tail by one page: the whole vector when owned, the
        // overlay's grown tail when borrowing (the borrowed base is never
        // resized). `grown_tail_mut` hands back that owned `Vec` directly — no
        // clone of the base — and the new page's global index is `old_len /
        // PAGE_BYTES` regardless of which tail holds it.
        let tail = self.pool.grown_tail_mut();
        let tail_len = tail.len() + PAGE_BYTES;
        tail.reserve(PAGE_BYTES);
        // SAFETY: the new page is left uninitialized on purpose — this is the
        // one measured unsafe of the crate. Zeroing fresh pages was benched
        // at 12x slower on the wasm allocation path (wasmtime, 32k pages:
        // 3889 us zeroed vs 316 us uninit; native: noise) — and wasm is this
        // structure's primary environment. The invariant making it sound:
        // *no byte of a page beyond `counts[page] * T::SIZE` is ever read*.
        // Every read (search, get, iter, range, first_key, shifts) is
        // bounded by the page count, and a slot's bytes are fully written
        // before the count is incremented. Consequently `Arena` exposes no
        // whole-pool reads (no `Clone`/`PartialEq`/`as_bytes`); the snapshot
        // writer emits only the initialized prefixes of pages.
        // Routing the same reserve+set_len at the grown tail (not the base)
        // keeps this the crate's sole `unsafe` — the overlay adds none.
        #[allow(clippy::uninit_vec)]
        unsafe {
            tail.set_len(tail_len);
        }
        self.next.push(NONE);
        self.prev.push(NONE);
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

impl<T: Slot> fmt::Debug for Arena<'_, T> {
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
    arena: &'a Arena<'a, T>,
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
                let rel = self.idx * T::SIZE;
                self.idx += 1;
                self.remaining -= 1;
                return Some(T::read(
                    &self.arena.pool.page(self.page)[rel..rel + T::SIZE],
                ));
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

impl<'a, T: Slot> IntoIterator for &'a Arena<'a, T> {
    type Item = T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over a key range of an [`Arena`]; see [`Arena::range`].
pub struct Range<'a, T: Slot> {
    arena: &'a Arena<'a, T>,
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
                let rel = self.idx * T::SIZE;
                let page = self.arena.pool.page(self.page);
                if page[rel..rel + T::KEY_LEN] >= *self.to {
                    // Keys only grow from here on — the scan is complete.
                    return None;
                }
                self.idx += 1;
                return Some(T::read(&page[rel..rel + T::SIZE]));
            }
            self.page = self.arena.next[self.page as usize];
            self.idx = 0;
        }
    }
}

/// Iterator over a key range of an [`Arena`] in descending order; see
/// [`Arena::range_rev`].
pub struct RangeRev<'a, T: Slot> {
    arena: &'a Arena<'a, T>,
    /// The current shard. Lower shards are visited after the current chain.
    shard: usize,
    /// Current chain page; `NONE` means "advance to the previous shard's
    /// tail".
    page: u32,
    /// Exclusive upper slot index in the current page.
    idx: usize,
    /// Inclusive lower key bound.
    from: &'a [u8],
}

impl<T: Slot> Iterator for RangeRev<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if self.page == NONE {
                if self.shard == 0 {
                    return None;
                }
                self.shard -= 1;
                self.page = self.arena.tails[self.shard];
                self.idx = if self.page == NONE {
                    0
                } else {
                    self.arena.counts[self.page as usize] as usize
                };
                continue;
            }

            if self.idx == 0 {
                self.page = self.arena.prev[self.page as usize];
                self.idx = if self.page == NONE {
                    0
                } else {
                    self.arena.counts[self.page as usize] as usize
                };
                continue;
            }

            self.idx -= 1;
            let rel = self.idx * T::SIZE;
            let page = self.arena.pool.page(self.page);
            if page[rel..rel + T::KEY_LEN] < *self.from {
                return None;
            }
            return Some(T::read(&page[rel..rel + T::SIZE]));
        }
    }
}
