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
//!
//! # Overlay opens
//!
//! [`BlobHeap::load_borrowed`] (and its alias [`BlobHeap::load_overlay`])
//! open a heap whose existing bytes are **borrowed** from a longer-lived
//! buffer — typically a memory-mapped file — while any later [`push`] lands
//! in a small **owned tail**. Because the append-only heap never rewrites an
//! existing blob, this needs no per-page copy-on-write: a blob is wholly in
//! the borrowed base or wholly in the tail, so reads dispatch on a single
//! length comparison and the base is never cloned. That is what lets a
//! multi-gigabyte heap be opened and *appended to* without loading it into
//! RAM (see `examples/overlay.rs`).
//!
//! [`push`]: BlobHeap::push

use alloc::vec::Vec;
use core::fmt;

use crate::error::Error;

/// Serialized metadata header of the dumped index: `[blobs u32]`
/// (see [`BlobHeap::dump_index`]).
const INDEX_HEADER: usize = core::mem::size_of::<u32>();

/// Serialized width of one blob's length entry (little-endian `u32`).
const LEN_BYTES: usize = core::mem::size_of::<u32>();

/// Handle to one blob in a [`BlobHeap`]: a dense index assigned in push
/// order, starting at 0.
///
/// Ids are plain `u32`s so owners can store them inside fixed-size
/// [`Slot`](crate::Slot)s; the newtype only guards against mixing them with
/// other integer ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlobId(pub u32);

/// [`BlobHeap`] configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlobHeapCfg {
    /// Hard ceiling for the byte pool; a push that would grow past it fails
    /// with [`Error::CapacityExceeded`]. The pool is otherwise bounded only by
    /// `usize`: 4 GiB on a 32-bit target (e.g. wasm32 linear memory), the host
    /// address space / RAM on a 64-bit one.
    pub max_bytes: usize,
    /// Per-blob length ceiling; a longer blob fails with
    /// [`Error::BlobTooLarge`]. Guards against one runaway value swallowing
    /// the pool.
    pub max_blob: usize,
}

impl BlobHeapCfg {
    /// Creates a config with no pool limit (beyond `usize`: 4 GiB on 32-bit,
    /// the address space on 64-bit) and no per-blob limit.
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
#[derive(Clone)]
pub struct BlobHeap<'a> {
    /// Borrowed base bytes: the blobs present at open time, e.g. an mmap'd
    /// snapshot section (`load_borrowed`). Empty (`&[]`) on the fully-owned
    /// path (`new`/`load`), where every byte lives in `tail`. Never mutated
    /// after open — that is what keeps the borrow zero-copy.
    base: &'a [u8],
    /// Owned append tail: bytes pushed after open. A [`BlobHeap::push`]
    /// always extends this and never touches `base`, so a heap opened over a
    /// borrowed base grows without cloning it. The logical pool is
    /// `base` followed by `tail`; blob offsets are cumulative over that
    /// concatenation, so each blob is wholly in one segment.
    tail: Vec<u8>,
    /// `BlobId` -> `(offset, len)` into the logical `base ++ tail` pool. The
    /// offset is the cumulative logical position (a `usize`, so a 64-bit host
    /// addresses a pool past 4 GiB; a 32-bit host caps at its address space),
    /// independent of where the base/tail boundary falls. The length stays a
    /// `u32` — a single blob is bounded by `max_blob` — which keeps the entry
    /// compact and the dumped index format unchanged.
    index: Vec<(usize, u32)>,
    cfg: BlobHeapCfg,
}

impl<'a> BlobHeap<'a> {
    /// Creates an empty heap. Allocates nothing until the first push.
    pub const fn new(cfg: BlobHeapCfg) -> Self {
        Self {
            base: &[],
            tail: Vec::new(),
            index: Vec::new(),
            cfg,
        }
    }

    /// Total bytes of the logical pool (`base` + `tail`).
    fn pool_len(&self) -> usize {
        self.base.len() + self.tail.len()
    }

    /// Bytes of one blob given its logical `(offset, len)`. A blob never
    /// straddles the base/tail boundary — its offset is either below
    /// `base.len()` (wholly in `base`) or at/above it (wholly in `tail`) —
    /// so this dispatches on a single comparison and returns a contiguous
    /// slice.
    fn slice(&self, offset: usize, len: usize) -> &[u8] {
        let base_len = self.base.len();
        if offset < base_len {
            &self.base[offset..offset + len]
        } else {
            let at = offset - base_len;
            &self.tail[at..at + len]
        }
    }

    /// Appends a blob and returns its id. Zero-length blobs are valid.
    ///
    /// The bytes always land in the owned `tail`; a heap opened over a
    /// borrowed base (`load_borrowed`) is appended to without cloning that
    /// base.
    ///
    /// # Errors
    ///
    /// - [`Error::BlobTooLarge`] if `bytes` is longer than
    ///   [`BlobHeapCfg::max_blob`];
    /// - [`Error::CapacityExceeded`] if the pool would grow past
    ///   [`BlobHeapCfg::max_bytes`] or the `usize` pool ceiling (4 GiB on a
    ///   32-bit target).
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
        let offset = self.pool_len();
        // `checked_add` bounds the cumulative offset to `usize`, which is the
        // pool ceiling on a 32-bit host (4 GiB); on 64-bit only `max_bytes`
        // and RAM bind. The blob length itself stays within `u32` via the
        // `max_blob` check above.
        let end = offset.checked_add(bytes.len()).ok_or(capacity_exceeded)?;
        if end > self.cfg.max_bytes {
            return Err(capacity_exceeded);
        }
        // Ids are u32 with `u32::MAX` excluded so `id + 1` always fits (the
        // interner's table relies on that); unreachable in practice before
        // the pool ceiling unless the heap holds billions of zero-length
        // blobs.
        let id = u32::try_from(self.index.len())
            .ok()
            .filter(|&i| i != u32::MAX)
            .ok_or(capacity_exceeded)?;
        self.tail.extend_from_slice(bytes);
        self.index.push((offset, bytes.len() as u32));
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
        self.slice(offset, len as usize)
    }

    /// Number of blobs stored.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `true` when no blob has been pushed.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Total bytes of blob content (the logical pool size).
    pub fn pool_bytes(&self) -> usize {
        self.pool_len()
    }

    /// Iterates over all blobs in id order as `(id, bytes)` pairs.
    ///
    /// This is the substrate for the owner-driven compaction pass: walk the
    /// blobs, copy the live ones into a fresh heap, record the id remapping.
    pub fn iter(&self) -> impl Iterator<Item = (BlobId, &[u8])> {
        self.index
            .iter()
            .enumerate()
            .map(|(i, &(offset, len))| (BlobId(i as u32), self.slice(offset, len as usize)))
    }

    /// Appends the heap's index section to `out` (`specs/03`).
    ///
    /// Layout (little-endian): `[blobs u32]` then one `len u32` per blob.
    /// Offsets are not stored — blobs are contiguous in push order, so the
    /// lengths alone reconstruct the index (one redundancy less to
    /// validate). Together with [`BlobHeap::dump_pool`] this is the
    /// complete state; dumps are canonical.
    pub fn dump_index(&self, out: &mut Vec<u8>) {
        out.reserve(INDEX_HEADER + self.index.len() * LEN_BYTES);
        out.extend_from_slice(&(self.index.len() as u32).to_le_bytes());
        for &(_, len) in &self.index {
            out.extend_from_slice(&len.to_le_bytes());
        }
    }

    /// Appends the heap's pool section to `out` (`specs/03`) — a straight
    /// copy of the logical pool (`base` then `tail`): every pool byte is
    /// initialized blob content, so an overlay heap dumps byte-identically
    /// to the owned heap holding the same blobs.
    pub fn dump_pool(&self, out: &mut Vec<u8>) {
        out.reserve(self.pool_len());
        out.extend_from_slice(self.base);
        out.extend_from_slice(&self.tail);
    }

    /// Rebuilds a heap from its two dumped sections.
    ///
    /// The input is **untrusted** — validation is O(blobs), never panics
    /// on arbitrary bytes: exact index length, per-blob length within
    /// `cfg.max_blob`, and the lengths summing exactly to the pool size
    /// (which itself must fit `cfg.max_bytes` and the `usize` pool ceiling).
    /// Blob *content* is the owner's data and is not interpreted.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency.
    pub fn load(cfg: BlobHeapCfg, index: &[u8], pool: &[u8]) -> Result<Self, Error> {
        let rebuilt = validate_index(cfg, index, pool.len())?;
        Ok(Self {
            base: &[],
            tail: pool.to_vec(),
            index: rebuilt,
            cfg,
        })
    }

    /// Rebuilds a heap that **borrows** its base pool from a longer-lived
    /// buffer (a memory-mapped snapshot, specs/16) instead of copying it.
    /// The index is validated and rebuilt exactly as in [`BlobHeap::load`];
    /// no base byte is copied, so opening an 8 GiB heap this way touches
    /// only the pages the reader dereferences.
    ///
    /// Unlike the old whole-pool copy-on-write, this heap **stays borrowed
    /// even under mutation**: a later [`BlobHeap::push`] appends to an owned
    /// tail, never cloning the base. A read-only caller simply never pushes.
    /// See also [`BlobHeap::load_overlay`].
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency (same gates as `load`).
    pub fn load_borrowed(cfg: BlobHeapCfg, index: &[u8], pool: &'a [u8]) -> Result<Self, Error> {
        let rebuilt = validate_index(cfg, index, pool.len())?;
        Ok(Self {
            base: pool,
            tail: Vec::new(),
            index: rebuilt,
            cfg,
        })
    }

    /// Opens a heap over a borrowed base for the **overlay** write path: the
    /// base is mapped read-only and appends accumulate in an owned tail. For
    /// an append-only heap this is exactly [`BlobHeap::load_borrowed`] — the
    /// alias exists so overlay-mode callers read uniformly across the flat
    /// structures (the in-place [`Arena`](crate::Arena) and
    /// [`ChunkPool`](crate::ChunkPool) take a distinct page-copy-on-write
    /// `load_overlay`).
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency (same gates as `load`).
    pub fn load_overlay(cfg: BlobHeapCfg, index: &[u8], pool: &'a [u8]) -> Result<Self, Error> {
        Self::load_borrowed(cfg, index, pool)
    }
}

/// The **index side** of a [`BlobHeap`], built without holding the pool.
///
/// A [`BlobHeap`] keeps two things: the concatenated blob bytes (the pool) and
/// a per-blob length table (the index). When the pool is streamed somewhere
/// else — a file, a staging area — and never held in RAM, this builder tracks
/// just the index: [`push_len`](BlobHeapBuilder::push_len) records one blob of a
/// given length (validating exactly as [`BlobHeap::push`] would) and hands back
/// its [`BlobId`], and [`dump_index`](BlobHeapBuilder::dump_index) emits the
/// section **byte-identically** to [`BlobHeap::dump_index`]. Pair its index with
/// the separately-streamed pool bytes and [`BlobHeap::load_borrowed`]
/// reconstructs the exact same heap.
///
/// It is flat and allocation-lean — one `u32` per blob — so the index of a
/// multi-gigabyte heap stays small while the pool lives on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobHeapBuilder {
    /// Per-blob length, in push order — the whole state of the index.
    lens: Vec<u32>,
    /// Running total of the pool the lengths describe.
    pool_len: usize,
    cfg: BlobHeapCfg,
}

impl BlobHeapBuilder {
    /// An empty index builder for a pool with the given config.
    pub const fn new(cfg: BlobHeapCfg) -> Self {
        Self {
            lens: Vec::new(),
            pool_len: 0,
            cfg,
        }
    }

    /// Records a blob of `len` bytes and returns its id — the streaming
    /// counterpart of [`BlobHeap::push`], validating identically so a builder
    /// and a heap fed the same lengths accept and reject in lockstep.
    ///
    /// # Errors
    ///
    /// - [`Error::BlobTooLarge`] if `len` exceeds [`BlobHeapCfg::max_blob`];
    /// - [`Error::CapacityExceeded`] if the pool would grow past
    ///   [`BlobHeapCfg::max_bytes`] or the `usize` pool ceiling (4 GiB on a
    ///   32-bit target), or if the blob count would reach `u32::MAX`.
    pub fn push_len(&mut self, len: usize) -> Result<BlobId, Error> {
        if len > self.cfg.max_blob {
            return Err(Error::BlobTooLarge {
                len,
                max_blob: self.cfg.max_blob,
            });
        }
        let capacity_exceeded = Error::CapacityExceeded {
            max_bytes: self.cfg.max_bytes,
        };
        // Same ceiling as `BlobHeap::push`: `checked_add` caps the pool at
        // `usize` (4 GiB on a 32-bit host), `max_bytes` binds on 64-bit.
        let end = self.pool_len.checked_add(len).ok_or(capacity_exceeded)?;
        if end > self.cfg.max_bytes {
            return Err(capacity_exceeded);
        }
        let id = u32::try_from(self.lens.len())
            .ok()
            .filter(|&i| i != u32::MAX)
            .ok_or(capacity_exceeded)?;
        self.lens.push(len as u32);
        self.pool_len = end;
        Ok(BlobId(id))
    }

    /// Number of blobs recorded.
    pub fn len(&self) -> usize {
        self.lens.len()
    }

    /// `true` when no blob has been recorded.
    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    /// Total bytes of the pool the recorded lengths describe.
    pub fn pool_bytes(&self) -> usize {
        self.pool_len
    }

    /// Appends the index section — `[blobs u32]` then one `len u32` per blob —
    /// byte-identically to [`BlobHeap::dump_index`].
    pub fn dump_index(&self, out: &mut Vec<u8>) {
        out.reserve(INDEX_HEADER + self.lens.len() * LEN_BYTES);
        out.extend_from_slice(&(self.lens.len() as u32).to_le_bytes());
        for &len in &self.lens {
            out.extend_from_slice(&len.to_le_bytes());
        }
    }
}

/// Two heaps are equal when they hold the same blobs — compared over the
/// **logical** pool, so an owned heap and an overlay heap (borrowed base +
/// tail) built from the same pushes compare equal despite the different
/// base/tail split.
impl PartialEq for BlobHeap<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cfg == other.cfg
            && self.index == other.index
            && self
                .base
                .iter()
                .chain(&self.tail)
                .eq(other.base.iter().chain(&other.tail))
    }
}

impl Eq for BlobHeap<'_> {}

/// Validates a dumped blob index against the pool length and rebuilds the
/// `(offset, len)` table. Shared by [`BlobHeap::load`] and
/// [`BlobHeap::load_borrowed`]; the input is untrusted (see `load`).
fn validate_index(
    cfg: BlobHeapCfg,
    index: &[u8],
    pool_len: usize,
) -> Result<Vec<(usize, u32)>, Error> {
    if index.len() < INDEX_HEADER {
        return Err(Error::Corrupt("blob index shorter than its header"));
    }
    let blobs = u32::from_le_bytes(index[0..4].try_into().unwrap());
    if blobs == u32::MAX {
        return Err(Error::Corrupt("blob count overflows the id space"));
    }
    if index.len() as u64 != INDEX_HEADER as u64 + u64::from(blobs) * LEN_BYTES as u64 {
        return Err(Error::Corrupt("blob index length mismatch"));
    }
    // `pool_len` is a real `&[u8]` length, so on a 32-bit host it can never
    // exceed the address space — the 4 GiB cap is enforced by `usize` itself.
    // On 64-bit only `max_bytes` binds.
    if pool_len > cfg.max_bytes {
        return Err(Error::Corrupt("blob pool exceeds the configured ceiling"));
    }
    let mut rebuilt = Vec::with_capacity(blobs as usize);
    let mut offset: usize = 0;
    for i in 0..blobs as usize {
        let at = INDEX_HEADER + i * LEN_BYTES;
        let len = u32::from_le_bytes(index[at..at + LEN_BYTES].try_into().unwrap());
        if len as usize > cfg.max_blob {
            return Err(Error::Corrupt(
                "blob length exceeds the configured max_blob",
            ));
        }
        rebuilt.push((offset, len));
        // `checked_add` keeps the cumulative offset within `usize`; the
        // `> pool_len` gate then rejects any lengths overrunning the pool
        // (and, on 32-bit, an overflow that would wrap).
        offset = offset
            .checked_add(len as usize)
            .filter(|&o| o <= pool_len)
            .ok_or(Error::Corrupt("blob lengths overrun the pool"))?;
    }
    if offset != pool_len {
        return Err(Error::Corrupt("blob lengths do not cover the pool"));
    }
    Ok(rebuilt)
}

impl fmt::Debug for BlobHeap<'_> {
    /// Summary only — blob contents are the owner's data, not ours to dump.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlobHeap")
            .field("blobs", &self.index.len())
            .field("pool_bytes", &self.pool_len())
            .finish()
    }
}
