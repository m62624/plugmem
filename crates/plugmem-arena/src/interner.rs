//! String interner: `&str` -> dense [`TermId`].
//!
//! Tokenized terms, tags and entity names repeat constantly; the engine
//! stores them once and passes 4-byte ids around. The interner is a
//! [`BlobHeap`] (the string bytes; `BlobId` doubles as [`TermId`]) plus one
//! flat open-addressing hash table — no per-string allocations, and both
//! sections snapshot as-is (the table is stored, not rebuilt: cold-start
//! time matters more than 4 bytes per slot).
//!
//! Interning is **never-forget**: terms are not removed. Vocabulary grows
//! slowly in practice (Zipf), and dictionary compaction is explicitly out of
//! scope for v1 (`specs/01`).

use alloc::vec::Vec;
use core::fmt;

use xxhash_rust::xxh3::xxh3_64;

use crate::blob::{BlobHeap, BlobHeapCfg, BlobId};
use crate::error::Error;

/// Handle to one interned string: dense, assigned in first-seen order,
/// starting at 0. Numerically equal to the underlying heap's `BlobId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermId(pub u32);

/// Initial hash-table size (slots); must be a power of two.
const INITIAL_SLOTS: usize = 16;

/// A deduplicating string store over a [`BlobHeap`] and a flat hash table.
///
/// ```
/// use plugmem_arena::{BlobHeapCfg, Interner};
///
/// let mut terms = Interner::new(BlobHeapCfg::new());
/// let apple = terms.intern("apple").unwrap();
/// let banana = terms.intern("banana").unwrap();
/// assert_eq!(terms.intern("apple").unwrap(), apple); // stable id
/// assert_ne!(apple, banana);
/// assert_eq!(terms.resolve(apple), "apple");
/// assert_eq!(terms.len(), 2);
/// ```
///
/// `Clone` is cheap-ish (two flat memcpys) and safe: unlike the arena, every
/// stored byte is initialized.
#[derive(Clone)]
pub struct Interner {
    /// String bytes; `BlobId` values equal `TermId` values.
    heap: BlobHeap,
    /// Open-addressing table: `0` = empty slot, otherwise `TermId + 1`.
    /// Length is always a power of two; load factor is kept <= 0.7.
    table: Vec<u32>,
    /// Number of interned strings.
    len: u32,
    /// Table probes performed (feature `counters`): every slot inspection
    /// during intern and rehash. The deterministic health metric of the
    /// hash function + load factor combination.
    #[cfg(feature = "counters")]
    probes: u64,
}

impl Interner {
    /// Creates an empty interner; `cfg` bounds the underlying string heap
    /// ([`BlobHeapCfg::max_blob`] caps a single term's byte length). The
    /// hash table itself is small (4 bytes per slot) and not counted against
    /// `cfg.max_bytes`.
    pub fn new(cfg: BlobHeapCfg) -> Self {
        Self {
            heap: BlobHeap::new(cfg),
            table: alloc::vec![0; INITIAL_SLOTS],
            len: 0,
            #[cfg(feature = "counters")]
            probes: 0,
        }
    }

    /// Returns the id for `s`, storing it on first sight.
    ///
    /// # Errors
    ///
    /// - [`Error::BlobTooLarge`] if `s` is longer than the configured
    ///   [`BlobHeapCfg::max_blob`];
    /// - [`Error::CapacityExceeded`] if storing a new term would grow the
    ///   heap past [`BlobHeapCfg::max_bytes`].
    ///
    /// A failed intern leaves the interner unchanged.
    pub fn intern(&mut self, s: &str) -> Result<TermId, Error> {
        // Grow before probing so the insert below always finds an empty slot
        // within the load-factor bound.
        if (self.len as usize + 1) * 10 > self.table.len() * 7 {
            self.rehash();
        }
        let bytes = s.as_bytes();
        let mask = self.table.len() - 1;
        let mut idx = xxh3_64(bytes) as usize & mask;
        loop {
            #[cfg(feature = "counters")]
            {
                self.probes += 1;
            }
            match self.table[idx] {
                0 => {
                    let id = self.heap.push(bytes)?;
                    self.table[idx] = id.0 + 1;
                    self.len += 1;
                    return Ok(TermId(id.0));
                }
                entry if self.heap.get(BlobId(entry - 1)) == bytes => {
                    return Ok(TermId(entry - 1));
                }
                _ => idx = (idx + 1) & mask,
            }
        }
    }

    /// Returns the string behind an id. O(1).
    ///
    /// # Panics
    ///
    /// Panics if `id` was not returned by this interner's
    /// [`Interner::intern`] — a dangling id is a caller bug.
    pub fn resolve(&self, id: TermId) -> &str {
        core::str::from_utf8(self.heap.get(BlobId(id.0)))
            .expect("interner heap holds only pushed &str bytes")
    }

    /// Number of distinct strings interned.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` when nothing has been interned.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Doubles the table and reinserts all entries. Amortized: one table
    /// allocation, no per-string work beyond rehashing their bytes.
    fn rehash(&mut self) {
        let mask = self.table.len() * 2 - 1;
        let mut table = alloc::vec![0u32; mask + 1];
        for &entry in self.table.iter().filter(|&&e| e != 0) {
            let mut idx = xxh3_64(self.heap.get(BlobId(entry - 1))) as usize & mask;
            #[cfg(feature = "counters")]
            {
                self.probes += 1;
            }
            while table[idx] != 0 {
                idx = (idx + 1) & mask;
                #[cfg(feature = "counters")]
                {
                    self.probes += 1;
                }
            }
            table[idx] = entry;
        }
        self.table = table;
    }

    /// Table probes performed so far (see the field docs). Feature
    /// `counters` only.
    #[cfg(feature = "counters")]
    pub fn probes(&self) -> u64 {
        self.probes
    }

    /// Resets the probe counter to zero. Feature `counters` only.
    #[cfg(feature = "counters")]
    pub fn reset_probes(&mut self) {
        self.probes = 0;
    }
}

impl fmt::Debug for Interner {
    /// Summary only — the vocabulary is the owner's data, not ours to dump.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interner")
            .field("terms", &self.len)
            .field("table_slots", &self.table.len())
            .field("heap_bytes", &self.heap.pool_bytes())
            .finish()
    }
}
