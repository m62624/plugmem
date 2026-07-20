//! Append-only heap for variable-length byte blobs.
//!
//! The counterpart to [`Arena`](crate::Arena)'s fixed-size slots: fact
//! texts, entity names, raw vectors — anything whose length varies — lives
//! here as a contiguous run of bytes, addressed by a dense [`BlobId`].
//!
//! State is two flat sections (the byte pool and the `(offset, len)` index),
//! so persisting a heap is a `memcpy` of both — the same snapshot contract
//! as the arena. There is **no removal**: blobs stay until the owner runs a
//! compaction pass (rewrite live blobs into a fresh heap and remap ids),
//! which keeps `push`/`get` trivially O(1) and the memory image dense in the
//! common append-mostly workload.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use crate::error::Error;

/// Handle to one blob in a [`BlobHeap`]: a dense index assigned in push
/// order, starting at 0.
///
/// Ids are plain `u32`s so owners can store them inside fixed-size
/// [`Slot`](crate::Slot)s; the newtype only guards against mixing them with
/// other integer ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlobId(pub u32);

/// [`BlobHeap`] configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobHeapCfg {
    /// Hard ceiling for the byte pool; a push that would grow past it fails
    /// with [`Error::CapacityExceeded`]. The pool is also inherently capped
    /// at 4 GiB because offsets are stored as `u32` (a deliberate fit for
    /// the 32-bit wasm address space).
    pub max_bytes: usize,
    /// Per-blob length ceiling; a longer blob fails with
    /// [`Error::BlobTooLarge`]. Guards against one runaway value swallowing
    /// the pool.
    pub max_blob: usize,
}

impl BlobHeapCfg {
    /// Creates a config with no pool limit (beyond the inherent 4 GiB) and
    /// no per-blob limit.
    pub const fn new() -> Self {
        Self {
            max_bytes: usize::MAX,
            max_blob: usize::MAX,
        }
    }

    /// Returns the config with `max_bytes` replaced.
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Returns the config with `max_blob` replaced.
    pub const fn with_max_blob(mut self, max_blob: usize) -> Self {
        self.max_blob = max_blob;
        self
    }
}

impl Default for BlobHeapCfg {
    /// Same as [`BlobHeapCfg::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Append-only byte heap addressed by dense [`BlobId`]s.
///
/// ```
/// use plugmem_arena::{BlobHeap, BlobHeapCfg};
///
/// let mut heap = BlobHeap::new(BlobHeapCfg::new());
/// let hello = heap.push(b"hello").unwrap();
/// let world = heap.push(b"world").unwrap();
/// assert_eq!(heap.get(hello), b"hello");
/// assert_eq!(heap.get(world), b"world");
/// assert_eq!(heap.len(), 2);
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct BlobHeap<'a> {
    /// All blob bytes, back to back in push order. Every byte is written
    /// exactly once by `push` — no uninitialized regions, hence `Clone` and
    /// `PartialEq` are safe to derive (unlike the arena).
    ///
    /// `Cow` so the heap can either own its pool (`new`/`load` — the
    /// writable path) or borrow it from a longer-lived buffer such as a
    /// memory-mapped snapshot (`load_borrowed` — the zero-copy read-only
    /// path, specs/16). The borrow is `alloc`-only (`Cow`, no `std`/`Arc`),
    /// so the crate stays `no_std`, and the lifetime `'a` lets the compiler
    /// prove the heap never outlives its backing buffer. The first mutation,
    /// if any, copies up to owned via `to_mut`.
    pool: Cow<'a, [u8]>,
    /// `BlobId` -> `(offset, len)` into `pool`.
    index: Vec<(u32, u32)>,
    cfg: BlobHeapCfg,
}

impl<'a> BlobHeap<'a> {
    /// Creates an empty heap. Allocates nothing until the first push.
    pub const fn new(cfg: BlobHeapCfg) -> Self {
        Self {
            pool: Cow::Owned(Vec::new()),
            index: Vec::new(),
            cfg,
        }
    }

    /// Appends a blob and returns its id. Zero-length blobs are valid.
    ///
    /// # Errors
    ///
    /// - [`Error::BlobTooLarge`] if `bytes` is longer than
    ///   [`BlobHeapCfg::max_blob`];
    /// - [`Error::CapacityExceeded`] if the pool would grow past
    ///   [`BlobHeapCfg::max_bytes`] (or past the inherent 4 GiB `u32`
    ///   offset ceiling).
    pub fn push(&mut self, bytes: &[u8]) -> Result<BlobId, Error> {
        if bytes.len() > self.cfg.max_blob {
            return Err(Error::BlobTooLarge {
                len: bytes.len(),
                max_blob: self.cfg.max_blob,
            });
        }
        let capacity_exceeded = Error::CapacityExceeded {
            max_bytes: self.cfg.max_bytes,
        };
        let offset = self.pool.len();
        let end = offset.checked_add(bytes.len()).ok_or(capacity_exceeded)?;
        if end > self.cfg.max_bytes || end > u32::MAX as usize {
            return Err(capacity_exceeded);
        }
        // Ids are u32 with `u32::MAX` excluded so `id + 1` always fits (the
        // interner's table relies on that); unreachable in practice before
        // the offset ceiling unless the heap holds billions of zero-length
        // blobs.
        let id = u32::try_from(self.index.len())
            .ok()
            .filter(|&i| i != u32::MAX)
            .ok_or(capacity_exceeded)?;
        self.pool.to_mut().extend_from_slice(bytes);
        self.index.push((offset as u32, bytes.len() as u32));
        Ok(BlobId(id))
    }

    /// Returns the bytes of a blob.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not returned by this heap's [`BlobHeap::push`] —
    /// a dangling id is a caller bug, not a runtime condition.
    pub fn get(&self, id: BlobId) -> &[u8] {
        let (offset, len) = self.index[id.0 as usize];
        &self.pool[offset as usize..(offset + len) as usize]
    }

    /// Number of blobs stored.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `true` when no blob has been pushed.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Total bytes of blob content (the pool size).
    pub fn pool_bytes(&self) -> usize {
        self.pool.len()
    }

    /// Iterates over all blobs in id order as `(id, bytes)` pairs.
    ///
    /// This is the substrate for the owner-driven compaction pass: walk the
    /// blobs, copy the live ones into a fresh heap, record the id remapping.
    pub fn iter(&self) -> impl Iterator<Item = (BlobId, &[u8])> {
        self.index.iter().enumerate().map(|(i, &(offset, len))| {
            (
                BlobId(i as u32),
                &self.pool[offset as usize..(offset + len) as usize],
            )
        })
    }

    /// Appends the heap's index section to `out` (`specs/03`).
    ///
    /// Layout (little-endian): `[blobs u32]` then one `len u32` per blob.
    /// Offsets are not stored — blobs are contiguous in push order, so the
    /// lengths alone reconstruct the index (one redundancy less to
    /// validate). Together with [`BlobHeap::dump_pool`] this is the
    /// complete state; dumps are canonical.
    pub fn dump_index(&self, out: &mut Vec<u8>) {
        out.reserve(4 + self.index.len() * 4);
        out.extend_from_slice(&(self.index.len() as u32).to_le_bytes());
        for &(_, len) in &self.index {
            out.extend_from_slice(&len.to_le_bytes());
        }
    }

    /// Appends the heap's pool section to `out` (`specs/03`) — a straight
    /// copy: every pool byte is initialized blob content.
    pub fn dump_pool(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.pool);
    }

    /// Rebuilds a heap from its two dumped sections.
    ///
    /// The input is **untrusted** — validation is O(blobs), never panics
    /// on arbitrary bytes: exact index length, per-blob length within
    /// `cfg.max_blob`, and the lengths summing exactly to the pool size
    /// (which itself must fit `cfg.max_bytes` and the `u32` offset
    /// ceiling). Blob *content* is the owner's data and is not
    /// interpreted.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency.
    pub fn load(cfg: BlobHeapCfg, index: &[u8], pool: &[u8]) -> Result<Self, Error> {
        let rebuilt = validate_index(cfg, index, pool.len())?;
        Ok(Self {
            pool: Cow::Owned(pool.to_vec()),
            index: rebuilt,
            cfg,
        })
    }

    /// Rebuilds a heap that **borrows** its pool from a longer-lived buffer
    /// (a memory-mapped snapshot, specs/16) instead of copying it. The
    /// index is validated and rebuilt exactly as in [`BlobHeap::load`]; no
    /// pool byte is copied, so opening an 8 GiB heap this way touches only
    /// the pages the reader dereferences. The heap is read-only in practice
    /// (a mutation would copy the pool up to owned via `Cow::to_mut`).
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency (same gates as `load`).
    pub fn load_borrowed(cfg: BlobHeapCfg, index: &[u8], pool: &'a [u8]) -> Result<Self, Error> {
        let rebuilt = validate_index(cfg, index, pool.len())?;
        Ok(Self {
            pool: Cow::Borrowed(pool),
            index: rebuilt,
            cfg,
        })
    }
}

/// Validates a dumped blob index against the pool length and rebuilds the
/// `(offset, len)` table. Shared by [`BlobHeap::load`] and
/// [`BlobHeap::load_borrowed`]; the input is untrusted (see `load`).
fn validate_index(
    cfg: BlobHeapCfg,
    index: &[u8],
    pool_len: usize,
) -> Result<Vec<(u32, u32)>, Error> {
    if index.len() < 4 {
        return Err(Error::Corrupt("blob index shorter than its header"));
    }
    let blobs = u32::from_le_bytes(index[0..4].try_into().unwrap());
    if blobs == u32::MAX {
        return Err(Error::Corrupt("blob count overflows the id space"));
    }
    if index.len() as u64 != 4 + u64::from(blobs) * 4 {
        return Err(Error::Corrupt("blob index length mismatch"));
    }
    if pool_len > cfg.max_bytes || pool_len > u32::MAX as usize {
        return Err(Error::Corrupt("blob pool exceeds the configured ceiling"));
    }
    let mut rebuilt = Vec::with_capacity(blobs as usize);
    let mut offset = 0u64;
    for i in 0..blobs as usize {
        let at = 4 + i * 4;
        let len = u32::from_le_bytes(index[at..at + 4].try_into().unwrap());
        if len as usize > cfg.max_blob {
            return Err(Error::Corrupt(
                "blob length exceeds the configured max_blob",
            ));
        }
        rebuilt.push((offset as u32, len));
        offset += u64::from(len);
        if offset > pool_len as u64 {
            return Err(Error::Corrupt("blob lengths overrun the pool"));
        }
    }
    if offset != pool_len as u64 {
        return Err(Error::Corrupt("blob lengths do not cover the pool"));
    }
    Ok(rebuilt)
}

impl fmt::Debug for BlobHeap<'_> {
    /// Summary only — blob contents are the owner's data, not ours to dump.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlobHeap")
            .field("blobs", &self.index.len())
            .field("pool_bytes", &self.pool.len())
            .finish()
    }
}
