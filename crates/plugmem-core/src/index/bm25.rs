//! The lexical index: classic BM25 over delta-encoded postings
//!
//! Scoring is the standard formula with the Robertson idf:
//!
//! ```text
//! idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
//! tf_norm(d)  = tf · (k1 + 1) / (tf + k1 · (1 - b + b · len(d) / avg_len))
//! score(d, q) = Σ_t idf(t) · tf_norm(d, t)
//! ```
//!
//! Query cost is O(Σ df) — a full decode of every query term's postings
//! with accumulation in a reusable scratch map. No WAND-style pruning in
//! v1: at the capacity passport's scale decoding is microseconds, and the
//! deterministic `postings_decoded` counter gates it in CI.
//!
//! Deletions never touch the postings: tombstoned facts are filtered per
//! candidate by the caller's `live` predicate and fall out physically
//! when `maintain` rebuilds the index.

use alloc::vec::Vec;

#[cfg(feature = "counters")]
use core::cell::Cell;

use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, key};

use crate::error::Error;
use crate::id::FactId;
use crate::index::postings::PostingStore;

/// Per-document length record: `[fact 4 | len u16 | pad 2]`, Uniform
/// arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DocLenSlot {
    /// The document (fact) id — the key.
    pub fact: FactId,
    /// Token count of the document, saturated at `u16::MAX`.
    pub len: u16,
}

impl Slot for DocLenSlot {
    const SIZE: usize = 8;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.fact.0);
        out[4..6].copy_from_slice(&self.len.to_be_bytes());
        out[6..8].copy_from_slice(&[0, 0]);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            fact: FactId(key::read_u32(bytes)),
            len: u16::from_be_bytes(bytes[4..6].try_into().unwrap()),
        }
    }
}

/// Reusable query scratch: accumulator and top-k selection buffer. One
/// per engine; after warm-up a query allocates nothing (the zero-alloc
/// recall invariant).
#[derive(Debug, Default)]
pub struct Bm25Scratch {
    /// fact id → accumulated score. The xxh3 hasher is explicit:
    /// hashbrown's default hasher is behind a feature we do not enable,
    /// and a fixed hasher keeps scratch behavior deterministic.
    acc: hashbrown::HashMap<u32, f32, xxhash_rust::xxh3::Xxh3Builder>,
    /// Selection buffer for the top-k extraction.
    top: Vec<(f32, u32)>,
}

impl Bm25Scratch {
    /// Empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }
}

/// The BM25 index: postings with term frequencies plus per-document
/// lengths and corpus statistics.
#[derive(Debug)]
pub struct Bm25Index<'a> {
    postings: PostingStore<'a, true>,
    doc_len: Arena<'a, DocLenSlot>,
    total_docs: u64,
    total_len: u64,
    /// Posting entries decoded by queries (feature `counters`) — the
    /// deterministic cost metric of the lexical source.
    #[cfg(feature = "counters")]
    decoded: Cell<u64>,
}

impl<'a> Bm25Index<'a> {
    /// Creates an empty index; `shards` per the engine config
    /// (`shards_postings`), `max_bytes` bounds each underlying pool.
    pub fn new(shards: usize, max_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            postings: PostingStore::new(shards, max_bytes)?,
            doc_len: Arena::new(
                ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(max_bytes),
            )?,
            total_docs: 0,
            total_len: 0,
            #[cfg(feature = "counters")]
            decoded: Cell::new(0),
        })
    }

    /// Indexes one document given its `(term, tf)` pairs (the caller
    /// tokenizes and counts; pairs may arrive in any order, terms must be
    /// unique). Documents must arrive in ascending fact-id order.
    ///
    /// # Errors
    ///
    /// [`Error::Arena`] when a pool hits its byte ceiling; the index may
    /// then hold the document partially — the engine treats that as fatal
    /// for the whole operation (journal replay rebuilds consistently).
    pub fn index_doc(&mut self, fact: FactId, term_tfs: &[(u32, u8)]) -> Result<(), Error> {
        let mut len = 0u32;
        for &(term, tf) in term_tfs {
            self.postings.push(term, fact, tf)?;
            len += u32::from(tf);
        }
        self.doc_len.insert(&DocLenSlot {
            fact,
            len: u16::try_from(len).unwrap_or(u16::MAX),
        })?;
        self.total_docs += 1;
        self.total_len += u64::from(len);
        Ok(())
    }

    /// Document frequency of a term.
    pub fn df(&self, term: u32) -> u32 {
        self.postings.count(term)
    }

    /// Number of indexed documents.
    pub fn docs(&self) -> u64 {
        self.total_docs
    }

    /// Robertson idf for a term with document frequency `df` in this
    /// corpus (monotonically decreasing in `df`, always positive).
    pub fn idf(&self, df: u32) -> f32 {
        let n = self.total_docs as f32;
        let df = df as f32;
        libm::logf(1.0 + (n - df + 0.5) / (df + 0.5))
    }

    /// Scores `terms` against the corpus and writes the top `k` live
    /// documents into `out` (descending score, ties by ascending id).
    /// `live` filters candidates (tombstones, as-of, tag allow-sets) —
    /// filtered documents cost their posting decode but never rank.
    ///
    /// Duplicate query terms are the caller's choice: each occurrence
    /// accumulates again (a term repeated in the query weighs more).
    pub fn search(
        &self,
        (k1, b): (f32, f32),
        terms: &[u32],
        k: usize,
        live: &mut dyn FnMut(FactId) -> bool,
        scratch: &mut Bm25Scratch,
        out: &mut Vec<(FactId, f32)>,
    ) {
        out.clear();
        if self.total_docs == 0 || k == 0 {
            return;
        }
        scratch.acc.clear();
        let avg_len = self.total_len as f32 / self.total_docs as f32;
        #[cfg(feature = "counters")]
        let mut decoded = 0u64;
        for &term in terms {
            let df = self.postings.count(term);
            if df == 0 {
                continue;
            }
            let idf = self.idf(df);
            for (fact, tf) in self.postings.entries(term) {
                #[cfg(feature = "counters")]
                {
                    decoded += 1;
                }
                let Some(doc) = self.doc_len.get(&fact.0.to_be_bytes()) else {
                    continue;
                };
                let tf = f32::from(tf);
                let norm =
                    tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * f32::from(doc.len) / avg_len));
                *scratch.acc.entry(fact.0).or_insert(0.0) += idf * norm;
            }
        }
        #[cfg(feature = "counters")]
        self.decoded.set(self.decoded.get() + decoded);

        // Top-k: collect survivors, sort the (small) buffer. k is ≤ 64 in
        // the engine; a heap would not buy anything at these sizes.
        scratch.top.clear();
        for (&id, &score) in &scratch.acc {
            if live(FactId(id)) {
                scratch.top.push((score, id));
            }
        }
        scratch
            .top
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(score, id) in scratch.top.iter().take(k) {
            out.push((FactId(id), score));
        }
    }

    /// Bytes held by the underlying pools.
    pub fn pool_bytes(&self) -> usize {
        self.postings.pool_bytes() + self.doc_len.pool_bytes()
    }

    /// Total token count across the corpus (persisted in the engine
    /// state).
    pub fn total_len(&self) -> u64 {
        self.total_len
    }

    /// The underlying posting store (the persistence composer dumps it).
    pub(crate) fn postings(&self) -> &PostingStore<'a, true> {
        &self.postings
    }

    /// The per-document length arena (the persistence composer dumps it).
    pub(crate) fn doc_len_arena(&self) -> &Arena<'a, DocLenSlot> {
        &self.doc_len
    }

    /// Assembles an index from already-validated parts (the load path).
    pub(crate) fn from_parts(
        postings: PostingStore<'a, true>,
        doc_len: Arena<'a, DocLenSlot>,
        total_docs: u64,
        total_len: u64,
    ) -> Self {
        Self {
            postings,
            doc_len,
            total_docs,
            total_len,
            #[cfg(feature = "counters")]
            decoded: Cell::new(0),
        }
    }

    /// Posting entries decoded so far (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn decoded(&self) -> u64 {
        self.decoded.get()
    }

    /// Resets the decode counter (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn reset_decoded(&self) {
        self.decoded.set(0);
    }
}
