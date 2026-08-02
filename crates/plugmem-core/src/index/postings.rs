//! Posting storage shared by every id-list index (–3).
//!
//! One shape serves BM25 postings, tag → facts and entity → facts: a
//! [`plugmem_arena::Arena`] of per-key handle slots plus one
//! [`ChunkPool`] holding the lists. Entries are appended in ascending
//! fact-id order (ids are assigned monotonically, so appends keep lists
//! sorted for free) and encoded as varint deltas — `[delta]` for plain id
//! lists, `[delta][tf u8]` for BM25 (`TF = true`).
//!
//! Each entry is pushed as one `ChunkPool` value, so entries never
//! straddle chunk boundaries and decoding walks whole entries per chunk
//! slice.

use plugmem_arena::{
    Arena, ArenaCfg, ChunkIter, ChunkPool, ChunkPoolCfg, ListHandle, ShardMode, Slot, key,
};

use crate::error::Error;
use crate::id::FactId;
use crate::index::varint::{MAX_VARINT, decode_u32, encode_u32};

/// Per-key slot: `[key 4 | ListHandle 12 | count u32 | last u32]`, 24
/// bytes, Uniform arena. `count` doubles as the document frequency for
/// BM25; `last` is the highest id in the list (the delta base).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IdListSlot {
    /// The indexed key (a `TermId`, a tag's `TermId`, an `EntityId` — the
    /// owner decides; raw here to serve them all).
    pub key: u32,
    /// The list's chunk chain.
    pub handle: ListHandle,
    /// Number of entries in the list.
    pub count: u32,
    /// Highest id appended so far (delta base for the next append).
    pub last: u32,
}

impl Slot for IdListSlot {
    const SIZE: usize = 24;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.key);
        out[4..16].copy_from_slice(&self.handle.to_bytes());
        key::write_u32(&mut out[16..], self.count);
        key::write_u32(&mut out[20..], self.last);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            key: key::read_u32(bytes),
            handle: ListHandle::from_bytes(bytes[4..16].try_into().unwrap()),
            count: key::read_u32(&bytes[16..]),
            last: key::read_u32(&bytes[20..]),
        }
    }
}

/// Key → sorted id list storage; `TF` adds a one-byte term frequency to
/// every entry (the BM25 flavor).
#[derive(Debug)]
pub struct PostingStore<'a, const TF: bool> {
    handles: Arena<'a, IdListSlot>,
    pool: ChunkPool<'a>,
}

impl<'a, const TF: bool> PostingStore<'a, TF> {
    /// Creates an empty store. `shards` follows the owning index's config
    /// (power of two); `max_bytes` bounds the chunk pool.
    pub fn new(shards: usize, max_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            handles: Arena::new(
                ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(max_bytes),
            )?,
            pool: ChunkPool::new(ChunkPoolCfg::new().with_max_bytes(max_bytes)),
        })
    }

    /// Appends `(id, tf)` to `key`'s list. Ids must arrive in strictly
    /// ascending order per key — the caller indexes facts in id order, so
    /// a violation is a caller bug and panics (debug) / corrupts the one
    /// list (release), never the store.
    ///
    /// # Errors
    ///
    /// [`Error::Arena`] when a pool hits its byte ceiling.
    pub fn push(&mut self, raw_key: u32, id: FactId, tf: u8) -> Result<(), Error> {
        let mut kb = [0u8; 4];
        key::write_u32(&mut kb, raw_key);
        let slot = self.handles.get(&kb);
        let (mut handle, count, last) = match slot {
            Some(s) => (s.handle, s.count, s.last),
            None => (ListHandle::EMPTY, 0, 0),
        };
        debug_assert!(
            count == 0 || id.0 > last,
            "posting ids must be appended in ascending order"
        );
        let delta = id.0.wrapping_sub(if count == 0 { 0 } else { last });
        let mut buf = [0u8; MAX_VARINT + 1];
        let mut head = [0u8; MAX_VARINT];
        let mut n = encode_u32(delta, &mut head);
        buf[..n].copy_from_slice(&head[..n]);
        if TF {
            buf[n] = tf;
            n += 1;
        }
        self.pool.push(&mut handle, &buf[..n])?;
        let updated = IdListSlot {
            key: raw_key,
            handle,
            count: count + 1,
            last: id.0,
        };
        if slot.is_some() {
            let payload = self
                .handles
                .payload_mut(&kb)
                .expect("slot existed just above");
            let mut full = [0u8; IdListSlot::SIZE];
            updated.write(&mut full);
            payload.copy_from_slice(&full[IdListSlot::KEY_LEN..]);
        } else {
            self.handles.insert(&updated)?;
        }
        Ok(())
    }

    /// Number of entries in `key`'s list (0 when absent). This is the
    /// document frequency for BM25 keys.
    pub fn count(&self, raw_key: u32) -> u32 {
        let mut kb = [0u8; 4];
        key::write_u32(&mut kb, raw_key);
        self.handles.get(&kb).map_or(0, |s| s.count)
    }

    /// Iterates `key`'s list as `(id, tf)` pairs in ascending id order
    /// (`tf` is 1 for `TF = false` stores).
    pub fn entries(&self, raw_key: u32) -> Entries<'_, TF> {
        let mut kb = [0u8; 4];
        key::write_u32(&mut kb, raw_key);
        let handle = self
            .handles
            .get(&kb)
            .map_or(ListHandle::EMPTY, |s| s.handle);
        Entries {
            chunks: self.pool.iter(&handle),
            cur: &[],
            prev: 0,
            first: true,
        }
    }

    /// Total bytes held by the two underlying pools.
    pub fn pool_bytes(&self) -> usize {
        self.handles.pool_bytes() + self.pool.pool_bytes()
    }

    /// Number of distinct keys.
    pub fn keys(&self) -> usize {
        self.handles.len()
    }

    /// Iterates the per-key posting-list headers. This is an internal
    /// maintenance hook: callers can rebuild/compact a store by walking each
    /// list in place, without knowing the arena layout.
    pub(crate) fn slots(&self) -> impl Iterator<Item = IdListSlot> + '_ {
        self.handles.iter()
    }

    /// Section dumps: handle-arena meta.
    pub(crate) fn handles_meta(&self) -> alloc::vec::Vec<u8> {
        let mut out = alloc::vec::Vec::new();
        self.handles.dump_meta(&mut out);
        out
    }

    /// Section dumps: handle-arena pool.
    pub(crate) fn handles_pool(&self) -> alloc::vec::Vec<u8> {
        let mut out = alloc::vec::Vec::new();
        self.handles.dump_pool(&mut out);
        out
    }

    /// Section dumps: chunk-pool meta.
    pub(crate) fn chunks_meta(&self) -> alloc::vec::Vec<u8> {
        let mut out = alloc::vec::Vec::new();
        self.pool.dump_meta(&mut out);
        out
    }

    /// Section dumps: chunk-pool bytes.
    pub(crate) fn chunks_pool(&self) -> alloc::vec::Vec<u8> {
        let mut out = alloc::vec::Vec::new();
        self.pool.dump_pool(&mut out);
        out
    }

    /// Assembles a store from already-validated parts (the load path).
    pub(crate) fn from_parts(handles: Arena<'a, IdListSlot>, pool: ChunkPool<'a>) -> Self {
        Self { handles, pool }
    }
}

/// Iterator over one list's `(id, tf)` entries; see
/// [`PostingStore::entries`].
pub struct Entries<'a, const TF: bool> {
    chunks: ChunkIter<'a>,
    cur: &'a [u8],
    prev: u32,
    first: bool,
}

impl<const TF: bool> Iterator for Entries<'_, TF> {
    type Item = (FactId, u8);

    fn next(&mut self) -> Option<(FactId, u8)> {
        while self.cur.is_empty() {
            self.cur = self.chunks.next()?;
        }
        // Entries never straddle chunks, so a non-empty slice starts a
        // whole entry; the store only ever appends well-formed varints.
        let (delta, used) = decode_u32(self.cur).expect("posting entry is well-formed");
        let mut at = used;
        let tf = if TF {
            let tf = self.cur[at];
            at += 1;
            tf
        } else {
            1
        };
        self.cur = &self.cur[at..];
        let id = if self.first {
            self.first = false;
            delta
        } else {
            self.prev + delta
        };
        self.prev = id;
        Some((FactId(id), tf))
    }
}
