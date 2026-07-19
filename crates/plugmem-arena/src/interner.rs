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

    /// Returns the id of an already-interned string, without creating it.
    ///
    /// The read-only sibling of [`Interner::intern`] — query paths must
    /// not grow the vocabulary (a query is not a mutation).
    pub fn lookup(&self, s: &str) -> Option<TermId> {
        let bytes = s.as_bytes();
        let mask = self.table.len() - 1;
        let mut idx = xxh3_64(bytes) as usize & mask;
        loop {
            match self.table[idx] {
                0 => return None,
                entry if self.heap.get(BlobId(entry - 1)) == bytes => {
                    return Some(TermId(entry - 1));
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

    /// Total bytes of interned string content (the underlying heap's pool
    /// size; the hash table's few bytes per slot are not counted).
    pub fn pool_bytes(&self) -> usize {
        self.heap.pool_bytes()
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

    /// Appends the string heap's index section to `out` (`specs/03`);
    /// see [`BlobHeap::dump_index`].
    pub fn dump_index(&self, out: &mut Vec<u8>) {
        self.heap.dump_index(out);
    }

    /// Appends the string heap's pool section to `out` (`specs/03`);
    /// see [`BlobHeap::dump_pool`].
    pub fn dump_pool(&self, out: &mut Vec<u8>) {
        self.heap.dump_pool(out);
    }

    /// Appends the hash-table section to `out` (`specs/03`).
    ///
    /// Layout (little-endian): `[slots u32][len u32]` then `slots × u32`
    /// entries (`0` = empty, else `TermId + 1`). The table is persisted
    /// rather than rebuilt on load: rebuilding is O(terms × hash) and the
    /// cold-start budget is tighter than the 4 bytes per slot.
    pub fn dump_table(&self, out: &mut Vec<u8>) {
        out.reserve(8 + self.table.len() * 4);
        out.extend_from_slice(&(self.table.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
        for &entry in &self.table {
            out.extend_from_slice(&entry.to_le_bytes());
        }
    }

    /// Rebuilds an interner from its three dumped sections.
    ///
    /// The input is **untrusted** — no panic on arbitrary bytes. On top of
    /// [`BlobHeap::load`], validation covers:
    ///
    /// - every stored blob is valid UTF-8 (O(pool bytes), SIMD-speed) —
    ///   this is what keeps [`Interner::resolve`] infallible;
    /// - table shape: power-of-two slot count `>= 16`, exact section
    ///   length, the ≤ 0.7 load factor, `len` equal to the heap's blob
    ///   count;
    /// - table entries: in bounds, no id stored twice, and exactly `len`
    ///   of them (checked with a visited bitmap).
    ///
    /// Entry *placement* (that each id sits on its hash's probe path) is
    /// not re-verified — that would cost the full rebuild the stored table
    /// exists to avoid. A well-formed but misplaced table cannot cause
    /// memory unsafety or a panic; it degrades `intern` to assigning a
    /// duplicate id for the affected terms. Section checksums (`specs/03`)
    /// cover accidental corruption.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for any inconsistency.
    pub fn load(cfg: BlobHeapCfg, index: &[u8], pool: &[u8], table: &[u8]) -> Result<Self, Error> {
        let heap = BlobHeap::load(cfg, index, pool)?;
        for (_, blob) in heap.iter() {
            if core::str::from_utf8(blob).is_err() {
                return Err(Error::Corrupt("interned term is not valid UTF-8"));
            }
        }
        if table.len() < 8 {
            return Err(Error::Corrupt("interner table shorter than its header"));
        }
        let slots = u32::from_le_bytes(table[0..4].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(table[4..8].try_into().unwrap());
        if slots < INITIAL_SLOTS || !slots.is_power_of_two() {
            return Err(Error::Corrupt("interner table size is not a power of two"));
        }
        if table.len() as u64 != 8 + slots as u64 * 4 {
            return Err(Error::Corrupt("interner table length mismatch"));
        }
        if len as usize != heap.len() {
            return Err(Error::Corrupt("interner length disagrees with its heap"));
        }
        if len as u64 * 10 > slots as u64 * 7 {
            return Err(Error::Corrupt("interner table over the load factor"));
        }
        let mut entries = Vec::with_capacity(slots);
        let mut seen = alloc::vec![false; heap.len()];
        let mut filled = 0u64;
        for i in 0..slots {
            let at = 8 + i * 4;
            let entry = u32::from_le_bytes(table[at..at + 4].try_into().unwrap());
            if entry != 0 {
                let id = (entry - 1) as usize;
                if id >= heap.len() {
                    return Err(Error::Corrupt("interner table entry out of bounds"));
                }
                if core::mem::replace(&mut seen[id], true) {
                    return Err(Error::Corrupt("interner table stores an id twice"));
                }
                filled += 1;
            }
            entries.push(entry);
        }
        if filled != u64::from(len) {
            return Err(Error::Corrupt(
                "interner table entry count disagrees with len",
            ));
        }
        Ok(Self {
            heap,
            table: entries,
            len,
            #[cfg(feature = "counters")]
            probes: 0,
        })
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
