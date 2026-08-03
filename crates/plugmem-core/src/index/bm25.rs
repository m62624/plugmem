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

/// Byte layout of [`DocLenSlot`]. Every offset is the previous field's offset
/// plus its width, so a field cannot be moved by editing one number.
mod doclen_at {
    use core::mem::size_of;

    pub(super) const FACT: usize = 0;
    pub(super) const KEY_LEN: usize = FACT + size_of::<u32>();
    pub(super) const LEN: usize = KEY_LEN;
    pub(super) const DISTINCT: usize = LEN + size_of::<u16>();
    pub(super) const SIG: usize = DISTINCT + size_of::<u16>();
    pub(super) const SIZE: usize = SIG + size_of::<u64>();
}

/// Width of a [`DocLenSlot::sig`] signature in bits.
const SIG_BITS: u32 = 64;
/// Fibonacci hashing multiplier — the same constant the arena shards with, so
/// term ids scatter over the signature's bits without a second hash family.
const SIG_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The signature bit a term claims. Distinct terms may collide; that is what
/// makes [`DocLenSlot::sig`] an over-approximation and never an
/// under-approximation of a document's term set.
pub(crate) fn sig_bit(term: u32) -> u64 {
    // The top `log2(SIG_BITS)` bits of the multiplied word, so the index is in
    // range by construction.
    let index = u64::from(term).wrapping_mul(SIG_MULT) >> (64 - SIG_BITS.trailing_zeros());
    1u64 << index
}

/// Per-document record: `[fact 4 | len u16 | distinct u16 | sig u64]`,
/// Uniform arena.
///
/// Beyond the length BM25 scores with, the slot carries a summary of the
/// document's *term set*: how many distinct terms it has, and one bit per
/// term hashed into a 64-bit word. That summary is what lets the write path
/// bound the term-set overlap of two facts without re-reading and
/// re-tokenizing their texts (see `Memory::find_similar`). It is written when
/// the document is indexed, where the term set is already in hand, so it
/// costs nothing to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DocLenSlot {
    /// The document (fact) id — the key.
    pub fact: FactId,
    /// Token count of the document, saturated at `u16::MAX`.
    pub len: u16,
    /// Number of distinct terms, saturated at `u16::MAX`. Zero means
    /// "unknown": a document indexed before the signature existed, read
    /// through the legacy migration.
    pub distinct: u16,
    /// Union of [`sig_bit`] over the document's distinct terms. A term absent
    /// from this word is definitely absent from the document; a term present
    /// may still be absent (bits collide). Zero alongside `distinct == 0`
    /// means "unknown".
    pub sig: u64,
}

impl DocLenSlot {
    /// Whether the term-set summary is present. A legacy document carries
    /// none, and callers must fall back to reading its text.
    pub fn has_signature(&self) -> bool {
        self.distinct != 0
    }

    /// An upper bound on how many of `terms` this document also holds.
    ///
    /// Exact in the direction that matters: a term whose bit is clear cannot
    /// be in the document, so the true intersection is never larger than the
    /// count returned here. Callers use it to rule overlap *out*.
    pub fn overlap_bound(&self, terms: &[u32]) -> usize {
        terms
            .iter()
            .filter(|&&term| self.sig & sig_bit(term) != 0)
            .count()
    }
}

impl Slot for DocLenSlot {
    const SIZE: usize = doclen_at::SIZE;
    const KEY_LEN: usize = doclen_at::KEY_LEN;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(&mut out[doclen_at::FACT..], self.fact.0);
        out[doclen_at::LEN..doclen_at::DISTINCT].copy_from_slice(&self.len.to_be_bytes());
        out[doclen_at::DISTINCT..doclen_at::SIG].copy_from_slice(&self.distinct.to_be_bytes());
        out[doclen_at::SIG..doclen_at::SIZE].copy_from_slice(&self.sig.to_be_bytes());
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            fact: FactId(key::read_u32(&bytes[doclen_at::FACT..])),
            len: u16::from_be_bytes(
                bytes[doclen_at::LEN..doclen_at::DISTINCT]
                    .try_into()
                    .unwrap(),
            ),
            distinct: u16::from_be_bytes(
                bytes[doclen_at::DISTINCT..doclen_at::SIG]
                    .try_into()
                    .unwrap(),
            ),
            sig: u64::from_be_bytes(bytes[doclen_at::SIG..doclen_at::SIZE].try_into().unwrap()),
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
    /// Documents whose BM25 contribution was actually evaluated — a document
    /// length fetched and `tf_norm` computed (feature `counters`).
    ///
    /// Decoding a posting entry is a few nanoseconds of varint; *scoring* it
    /// costs a document-length lookup, which is a random probe into an arena
    /// that grows with the corpus. The two counters therefore measure
    /// different things, and this is the one that dominates a large query.
    #[cfg(feature = "counters")]
    scored: Cell<u64>,
    /// Calls to the caller's `live` predicate (feature `counters`).
    ///
    /// Every call is a fact-record lookup on the engine side — the second
    /// random probe per candidate. A candidate that cannot reach the top `k`
    /// should never cost one.
    #[cfg(feature = "counters")]
    admitted: Cell<u64>,
    /// Some documents arrived without a term-set summary — this index came out
    /// of a pre-signature image. Derived at load, never persisted: the next
    /// compaction fills the summaries in and clears it.
    unsummarized: bool,
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
            #[cfg(feature = "counters")]
            scored: Cell::new(0),
            #[cfg(feature = "counters")]
            admitted: Cell::new(0),
            unsummarized: false,
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
        let mut sig = 0u64;
        for &(term, tf) in term_tfs {
            self.postings.push(term, fact, tf)?;
            len += u32::from(tf);
            sig |= sig_bit(term);
        }
        self.doc_len.insert(&DocLenSlot {
            fact,
            len: u16::try_from(len).unwrap_or(u16::MAX),
            // The caller's pairs are already unique per term, so their count
            // is the distinct-term count.
            distinct: u16::try_from(term_tfs.len()).unwrap_or(u16::MAX),
            sig,
        })?;
        self.total_docs += 1;
        self.total_len += u64::from(len);
        Ok(())
    }

    /// The per-document record of `fact`, or `None` when the document is not
    /// indexed. Carries the term-set summary the write path bounds overlap
    /// with — see [`DocLenSlot`].
    pub fn doc(&self, fact: FactId) -> Option<DocLenSlot> {
        self.doc_len.get(&fact.0.to_be_bytes())
    }

    /// Whether this index holds documents with no term-set summary, which
    /// compaction can fill in from the postings. True only after opening a
    /// pre-signature image.
    pub(crate) fn needs_resummarize(&self) -> bool {
        self.unsummarized
    }

    /// Marks the index as holding unsummarized documents (the load path, after
    /// a legacy migration).
    pub(crate) fn mark_unsummarized(&mut self) {
        self.unsummarized = true;
    }

    /// Builds a compacted BM25 index by filtering this index's existing
    /// postings and document lengths through `live`.
    ///
    /// This is the ordinary-maintenance path: it preserves the exact term ids
    /// and term frequencies already indexed, so compaction does not have to
    /// read and tokenize every live document again. A tokenizer migration must
    /// use the text reindex path instead.
    pub(crate) fn compact_live(
        &self,
        shards: usize,
        max_bytes: usize,
        mut live: impl FnMut(FactId) -> bool,
    ) -> Result<Bm25Index<'static>, Error> {
        let mut out = Bm25Index::new(shards, max_bytes)?;
        // Documents carried over from a pre-signature image, and the id range
        // they span. Compaction is the cheapest place to fill their term-set
        // summaries in: it already walks every posting, which is the transpose
        // of what a signature needs, so no text is read and nothing is
        // tokenized.
        let mut legacy = 0usize;
        let mut max_fact = 0u32;
        for doc in self.doc_len.iter() {
            if live(doc.fact) {
                out.doc_len.insert(&doc)?;
                out.total_docs += 1;
                out.total_len += u64::from(doc.len);
                if !doc.has_signature() {
                    legacy += 1;
                    max_fact = max_fact.max(doc.fact.0);
                }
            }
        }
        // `(sig, distinct)` per fact id, dense because the transpose visits
        // facts in posting order rather than document order. Allocated only
        // when there is something to fill, and dropped with this call.
        //
        // The length is computed through `checked_add` because `usize` is 32
        // bits on wasm32: a fact id near `u32::MAX` would wrap there. Failing
        // that check skips the rebuild, which costs speed and nothing else —
        // an unsummarized document simply keeps reading its text.
        let mut rebuilt = match usize::try_from(max_fact)
            .ok()
            .and_then(|m| m.checked_add(1))
        {
            Some(len) if legacy > 0 => alloc::vec![(0u64, 0u16); len],
            _ => Vec::new(),
        };
        for slot in self.postings.slots() {
            for (fact, tf) in self.postings.entries(slot.key) {
                if !live(fact) {
                    continue;
                }
                out.postings.push(slot.key, fact, tf)?;
                if let Some(entry) = rebuilt.get_mut(fact.0 as usize) {
                    entry.0 |= sig_bit(slot.key);
                    entry.1 = entry.1.saturating_add(1);
                }
            }
        }
        if !rebuilt.is_empty() {
            out.fill_missing_signatures(&rebuilt);
        }
        Ok(out)
    }

    /// Writes the recomputed term-set summaries of [`Self::compact_live`]
    /// into the documents that arrived without one. Documents that already
    /// carry a signature keep it: it was written from the exact term set the
    /// indexer saw, and the transpose can only reproduce it.
    ///
    /// The compacted index never inherits [`Self::unsummarized`]. A document
    /// still unsummarized after this pass has no indexed terms at all, so no
    /// later pass could summarize it either — carrying the flag forward would
    /// make every future `maintain` recompact for work that cannot be done.
    fn fill_missing_signatures(&mut self, rebuilt: &[(u64, u16)]) {
        let stale: Vec<DocLenSlot> = self
            .doc_len
            .iter()
            .filter(|doc| !doc.has_signature())
            .collect();
        for mut doc in stale {
            let Some(&(sig, distinct)) = rebuilt.get(doc.fact.0 as usize) else {
                continue;
            };
            if distinct == 0 {
                continue; // a document with no indexed terms has nothing to summarize
            }
            doc.sig = sig;
            doc.distinct = distinct;
            let Some(payload) = self.doc_len.payload_mut(&doc.fact.0.to_be_bytes()) else {
                continue;
            };
            let mut full = [0u8; DocLenSlot::SIZE];
            doc.write(&mut full);
            payload.copy_from_slice(&full[DocLenSlot::KEY_LEN..]);
        }
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
        #[cfg(feature = "counters")]
        let mut scored = 0u64;
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
                #[cfg(feature = "counters")]
                {
                    scored += 1;
                }
                let tf = f32::from(tf);
                let norm =
                    tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * f32::from(doc.len) / avg_len));
                *scratch.acc.entry(fact.0).or_insert(0.0) += idf * norm;
            }
        }
        #[cfg(feature = "counters")]
        {
            self.decoded.set(self.decoded.get() + decoded);
            self.scored.set(self.scored.get() + scored);
            self.admitted
                .set(self.admitted.get() + scratch.acc.len() as u64);
        }

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
            #[cfg(feature = "counters")]
            scored: Cell::new(0),
            #[cfg(feature = "counters")]
            admitted: Cell::new(0),
            unsummarized: false,
        }
    }

    /// Posting entries decoded so far (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn decoded(&self) -> u64 {
        self.decoded.get()
    }

    /// Documents scored so far — document-length fetches (feature
    /// `counters`). See the [`Bm25Index::scored`] field docs for why this is
    /// tracked apart from [`Bm25Index::decoded`].
    #[cfg(feature = "counters")]
    pub fn scored(&self) -> u64 {
        self.scored.get()
    }

    /// `live` predicate calls so far (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn admitted(&self) -> u64 {
        self.admitted.get()
    }

    /// Resets the query work counters (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn reset_query_counters(&self) {
        self.decoded.set(0);
        self.scored.set(0);
        self.admitted.set(0);
    }
}
