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
