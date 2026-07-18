//! Many small growable lists over one flat chunk pool.
//!
//! Inverted-index posting lists and graph adjacency lists share a shape:
//! *lots* of independent lists, most tiny, a few huge, all append-mostly.
//! Giving each a `Vec` means a heap allocation per list and pointer-chasing
//! on iteration. [`ChunkPool`] instead carves one byte pool into fixed
//! 64-byte chunks and threads each list through a chain of them; the pool
//! never allocates per list, freed chains recycle through a free-list, and
//! the whole state is two flat sections (pool + per-chunk fill counts) —
//! the same snapshot-is-memcpy contract as the other structures here.
//!
//! A list is addressed by a [`ListHandle`] that the *owner* stores (it is
//! 12 bytes — designed to sit inside a fixed-size arena
//! [`Slot`](crate::Slot)); the pool itself keeps no list directory.
//!
//! Values are opaque byte runs. The pool guarantees one property the
//! consumers rely on: **a value never straddles a chunk boundary** — a value
//! that does not fit in the current tail chunk starts a fresh chunk instead
//! (the skipped tail bytes are excluded from iteration, so the concatenated
//! stream stays exactly the pushed bytes). Self-delimiting encodings
//! (varints) can therefore decode chunk by chunk without a reassembly
//! buffer.

use alloc::vec::Vec;
use core::fmt;

use crate::error::Error;

/// Size of one chunk in bytes: a 4-byte chain link plus the payload.
pub const CHUNK_BYTES: usize = 64;

/// Payload bytes per chunk ([`CHUNK_BYTES`] minus the 4-byte chain link).
/// Also the maximum length of a single pushed value.
pub const CHUNK_PAYLOAD: usize = CHUNK_BYTES - 4;

/// Sentinel for "no chunk": an empty list, the end of a chain, or an empty
/// free-list.
const NONE: u32 = u32::MAX;

/// [`ChunkPool`] configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPoolCfg {
    /// Hard ceiling for the chunk pool; allocating a chunk past it fails
    /// with [`Error::CapacityExceeded`].
    pub max_bytes: usize,
}

impl ChunkPoolCfg {
    /// Creates a config with no byte limit.
    pub const fn new() -> Self {
        Self {
            max_bytes: usize::MAX,
        }
    }

    /// Returns the config with `max_bytes` replaced.
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }
}

impl Default for ChunkPoolCfg {
    /// Same as [`ChunkPoolCfg::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Owner-held handle to one list inside a [`ChunkPool`].
///
/// 12 bytes, `Copy` — meant to be embedded in a fixed-size record. The
/// handle *is* the list: the pool keeps no directory, so losing the handle
/// leaks the chain until the owner's compaction pass rebuilds the pool.
/// A handle is only meaningful with the pool that filled it; passing it to
/// another pool reads unrelated (but initialized) chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListHandle {
    /// First chunk of the chain (`NONE` = empty list).
    head: u32,
    /// Last chunk of the chain — O(1) appends and O(1) frees.
    tail: u32,
    /// Number of values pushed (bookkeeping for the owner; iteration yields
    /// chunk slices, not values).
    len: u32,
}

impl ListHandle {
    /// An empty list.
    pub const EMPTY: Self = Self {
        head: NONE,
        tail: NONE,
        len: 0,
    };

    /// Number of values pushed into the list.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// `true` when nothing has been pushed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Serializes the handle into 12 bytes: `[head BE | tail BE | len BE]`.
    ///
    /// This is the *stable* wire form — owners embed handles inside their
    /// own fixed-size records (and, later, snapshots), so the encoding is
    /// part of the crate's format contract and is fixed by tests.
    pub fn to_bytes(self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.head.to_be_bytes());
        out[4..8].copy_from_slice(&self.tail.to_be_bytes());
        out[8..12].copy_from_slice(&self.len.to_be_bytes());
        out
    }

    /// Inverse of [`ListHandle::to_bytes`].
    ///
    /// The bytes are trusted bookkeeping, not validated content: a handle
    /// is only meaningful with the pool that produced it (same rule as the
    /// in-memory value — see the type docs).
    pub fn from_bytes(bytes: [u8; 12]) -> Self {
        Self {
            head: u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
            tail: u32::from_be_bytes(bytes[4..8].try_into().unwrap()),
            len: u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
        }
    }
}

impl Default for ListHandle {
    /// Same as [`ListHandle::EMPTY`].
    fn default() -> Self {
        Self::EMPTY
    }
}

/// A pool of 64-byte chunks backing many independent append-only lists.
///
/// ```
/// use plugmem_arena::{ChunkPool, ChunkPoolCfg, ListHandle};
///
/// let mut pool = ChunkPool::new(ChunkPoolCfg::new());
/// let mut evens = ListHandle::EMPTY;
/// let mut odds = ListHandle::EMPTY;
/// for n in 0u32..10 {
///     let list = if n % 2 == 0 { &mut evens } else { &mut odds };
///     pool.push(list, &n.to_be_bytes()).unwrap();
/// }
/// let bytes: Vec<u8> = pool.iter(&evens).flatten().copied().collect();
/// let evens_back: Vec<u32> = bytes
///     .chunks_exact(4)
///     .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
///     .collect();
/// assert_eq!(evens_back, [0, 2, 4, 6, 8]);
/// pool.free(&mut odds); // chunks recycle; `evens` is untouched
/// ```
pub struct ChunkPool {
    /// Chunk storage; length is always a multiple of [`CHUNK_BYTES`]. The
    /// first 4 bytes of a chunk are its chain link (little-endian `u32`,
    /// internal metadata — not a sort key, so no big-endian mandate), the
    /// remaining [`CHUNK_PAYLOAD`] bytes hold values.
    pool: Vec<u8>,
    /// Chunk -> payload bytes occupied. Payload bytes beyond `used[chunk]`
    /// are zero-filled, never exposed. (Chunks are grown zeroed on purpose:
    /// the arena's measured uninit-page `unsafe` pays off when zeroing
    /// megabytes of pages at once; here growth is one 64-byte chunk at a
    /// time and zeroing it is noise, so the safe path costs nothing.)
    used: Vec<u8>,
    /// Head of the free-list of recycled chunks (linked through the chunks'
    /// own chain links).
    free_head: u32,
    cfg: ChunkPoolCfg,
}

impl ChunkPool {
    /// Creates an empty pool. Allocates nothing until the first push.
    pub const fn new(cfg: ChunkPoolCfg) -> Self {
        Self {
            pool: Vec::new(),
            used: Vec::new(),
            free_head: NONE,
            cfg,
        }
    }

    /// Appends one value to a list. Zero-length values only bump the
    /// handle's [`len`](ListHandle::len).
    ///
    /// # Errors
    ///
    /// - [`Error::ValueTooLarge`] if `value` is longer than
    ///   [`CHUNK_PAYLOAD`] bytes (it could never fit in one chunk);
    /// - [`Error::CapacityExceeded`] if a needed fresh chunk would grow the
    ///   pool past [`ChunkPoolCfg::max_bytes`].
    pub fn push(&mut self, list: &mut ListHandle, value: &[u8]) -> Result<(), Error> {
        if value.len() > CHUNK_PAYLOAD {
            return Err(Error::ValueTooLarge { len: value.len() });
        }
        if !value.is_empty() {
            let tail_fits = list.tail != NONE
                && self.used[list.tail as usize] as usize + value.len() <= CHUNK_PAYLOAD;
            if !tail_fits {
                let chunk = self.alloc_chunk()?;
                if list.tail == NONE {
                    list.head = chunk;
                } else {
                    self.set_link(list.tail, chunk);
                }
                list.tail = chunk;
            }
            let tail = list.tail as usize;
            let at = tail * CHUNK_BYTES + 4 + self.used[tail] as usize;
            self.pool[at..at + value.len()].copy_from_slice(value);
            self.used[tail] += value.len() as u8;
        }
        list.len += 1;
        Ok(())
    }

    /// Returns the whole chain of a list to the free-list and resets the
    /// handle to [`ListHandle::EMPTY`]. O(1): the chain is spliced onto the
    /// free-list as-is, chunk by chunk bookkeeping happens on reuse.
    pub fn free(&mut self, list: &mut ListHandle) {
        if list.head != NONE {
            self.set_link(list.tail, self.free_head);
            self.free_head = list.head;
        }
        *list = ListHandle::EMPTY;
    }

    /// Iterates a list's bytes as one `&[u8]` slice per chunk, in push
    /// order. Slices contain exactly the pushed bytes (skipped tail bytes of
    /// earlier chunks are excluded), and no pushed value ever spans two
    /// slices.
    pub fn iter<'a>(&'a self, list: &ListHandle) -> ChunkIter<'a> {
        ChunkIter {
            pool: self,
            chunk: list.head,
        }
    }

    /// Total bytes held by the pool (allocated chunks, including free ones).
    pub fn pool_bytes(&self) -> usize {
        self.pool.len()
    }

    /// Pops a free chunk or grows the pool by one chunk; returns it reset
    /// (`used = 0`, link = `NONE`).
    fn alloc_chunk(&mut self) -> Result<u32, Error> {
        let chunk = if self.free_head != NONE {
            let chunk = self.free_head;
            self.free_head = self.link(chunk);
            self.used[chunk as usize] = 0;
            chunk
        } else {
            let capacity_exceeded = Error::CapacityExceeded {
                max_bytes: self.cfg.max_bytes,
            };
            let new_len = self
                .pool
                .len()
                .checked_add(CHUNK_BYTES)
                .ok_or(capacity_exceeded)?;
            if new_len > self.cfg.max_bytes {
                return Err(capacity_exceeded);
            }
            let chunk = self.used.len();
            // Chunk indices are u32 with NONE reserved; unreachable before
            // max_bytes in any real configuration (would need a 256 GiB pool).
            let chunk = u32::try_from(chunk)
                .ok()
                .filter(|&c| c != NONE)
                .ok_or(capacity_exceeded)?;
            self.pool.resize(new_len, 0);
            self.used.push(0);
            chunk
        };
        self.set_link(chunk, NONE);
        Ok(chunk)
    }

    /// Reads a chunk's chain link.
    fn link(&self, chunk: u32) -> u32 {
        let at = chunk as usize * CHUNK_BYTES;
        u32::from_le_bytes(self.pool[at..at + 4].try_into().unwrap())
    }

    /// Writes a chunk's chain link.
    fn set_link(&mut self, chunk: u32, to: u32) {
        let at = chunk as usize * CHUNK_BYTES;
        self.pool[at..at + 4].copy_from_slice(&to.to_le_bytes());
    }

    /// Marks every chunk currently sitting in the free-list. Free chunks
    /// carry stale `used` values and payload bytes ([`ChunkPool::free`] is
    /// an O(1) splice, cleanup happens on reuse), so dumps consult this map
    /// to canonicalize them.
    fn free_map(&self) -> Vec<bool> {
        let mut free = alloc::vec![false; self.used.len()];
        let mut chunk = self.free_head;
        while chunk != NONE {
            free[chunk as usize] = true;
            chunk = self.link(chunk);
        }
        free
    }

    /// Appends the pool's metadata section to `out` (`specs/03`).
    ///
    /// Layout (little-endian): `[chunks u32][free_head u32]` then one
    /// `used u8` per chunk. Free chunks are written with `used = 0`
    /// regardless of their stale in-memory value, making the dump
    /// canonical. Together with [`ChunkPool::dump_pool`] this is the
    /// complete pool-side state; the list handles live with their owners.
    pub fn dump_meta(&self, out: &mut Vec<u8>) {
        let free = self.free_map();
        out.reserve(8 + self.used.len());
        out.extend_from_slice(&(self.used.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.free_head.to_le_bytes());
        for (chunk, &used) in self.used.iter().enumerate() {
            out.push(if free[chunk] { 0 } else { used });
        }
    }

    /// Appends the pool section to `out` (`specs/03`).
    ///
    /// Each chunk contributes its 4-byte link, its used payload prefix and
    /// zero padding to [`CHUNK_BYTES`]; free chunks contribute link plus
    /// zeros. This canonicalizes the stale bytes recycling leaves behind
    /// (see [`ChunkPool::free`]) — identical logical state, identical
    /// bytes.
    pub fn dump_pool(&self, out: &mut Vec<u8>) {
        let free = self.free_map();
        out.reserve(self.pool.len());
        for (chunk, &used) in self.used.iter().enumerate() {
            let at = chunk * CHUNK_BYTES;
            let used = if free[chunk] { 0 } else { used as usize };
            out.extend_from_slice(&self.pool[at..at + 4 + used]);
            out.resize(out.len() + (CHUNK_PAYLOAD - used), 0);
        }
    }

    /// Rebuilds a pool from its two dumped sections.
    ///
    /// The input is **untrusted** — validation is O(chunks), never panics
    /// on arbitrary bytes: exact section lengths (within `cfg.max_bytes`),
    /// per-chunk `used` within [`CHUNK_PAYLOAD`], every chain link in
    /// bounds (or the `NONE` sentinel — this is what keeps a later
    /// [`ChunkPool::iter`] from indexing out of bounds), and a free-list
    /// that is acyclic (visited bitmap) with `used = 0` on every member.
    ///
    /// What this method *cannot* check: cycles among chunks referenced by
    /// the owners' [`ListHandle`]s — the handles live in the owner's
    /// records, so the owning engine walks its handles with a shared
    /// visited bitmap as part of its own load (`specs/03`).
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency.
    pub fn load(cfg: ChunkPoolCfg, meta: &[u8], pool: &[u8]) -> Result<Self, Error> {
        if meta.len() < 8 {
            return Err(Error::Corrupt("chunk meta shorter than its header"));
        }
        let chunks = u32::from_le_bytes(meta[0..4].try_into().unwrap());
        let free_head = u32::from_le_bytes(meta[4..8].try_into().unwrap());
        if chunks == NONE {
            return Err(Error::Corrupt("chunk count overflows the index space"));
        }
        if meta.len() as u64 != 8 + u64::from(chunks) {
            return Err(Error::Corrupt("chunk meta length mismatch"));
        }
        if pool.len() as u64 != u64::from(chunks) * CHUNK_BYTES as u64 {
            return Err(Error::Corrupt("chunk pool length mismatch"));
        }
        if pool.len() > cfg.max_bytes {
            return Err(Error::Corrupt("chunk pool exceeds the configured ceiling"));
        }
        let used: Vec<u8> = meta[8..].to_vec();
        if used.iter().any(|&u| u as usize > CHUNK_PAYLOAD) {
            return Err(Error::Corrupt("chunk used bytes exceed the payload size"));
        }
        let link_of = |chunk: usize| {
            let at = chunk * CHUNK_BYTES;
            u32::from_le_bytes(pool[at..at + 4].try_into().unwrap())
        };
        for chunk in 0..chunks as usize {
            let link = link_of(chunk);
            if link != NONE && link >= chunks {
                return Err(Error::Corrupt("chunk link out of bounds"));
            }
        }
        let mut seen = alloc::vec![false; chunks as usize];
        let mut chunk = free_head;
        while chunk != NONE {
            let c = chunk as usize;
            if c >= chunks as usize {
                return Err(Error::Corrupt("chunk free-list head out of bounds"));
            }
            if core::mem::replace(&mut seen[c], true) {
                return Err(Error::Corrupt("chunk free-list contains a cycle"));
            }
            if used[c] != 0 {
                return Err(Error::Corrupt("free chunk has nonzero used bytes"));
            }
            chunk = link_of(c);
        }
        Ok(Self {
            pool: pool.to_vec(),
            used,
            free_head,
            cfg,
        })
    }
}

impl fmt::Debug for ChunkPool {
    /// Summary only — chunk contents are the owners' data, not ours to dump.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkPool")
            .field("chunks", &self.used.len())
            .field("pool_bytes", &self.pool.len())
            .finish()
    }
}

/// Iterator over one list's chunks; see [`ChunkPool::iter`].
pub struct ChunkIter<'a> {
    pool: &'a ChunkPool,
    chunk: u32,
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.chunk == NONE {
            return None;
        }
        let chunk = self.chunk as usize;
        self.chunk = self.pool.link(self.chunk);
        let at = chunk * CHUNK_BYTES + 4;
        Some(&self.pool.pool[at..at + self.pool.used[chunk] as usize])
    }
}
