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
//! A query decodes every query term's postings — O(Σ df), and there is no
//! WAND-style pruning to avoid it. That is affordable because a decode is a
//! few nanoseconds of varint over a contiguous chunk chain; what is not
//! affordable is a *random* lookup per posting, and the scan is built to have
//! none:
//!
//! - document lengths come from a flat array indexed by fact id, not from the
//!   stored arena (see [`Bm25Index::doc_len_dense`]);
//! - partial scores accumulate by merging sorted runs, because the postings
//!   are already sorted by fact id, so no map is probed;
//! - the caller's `live` predicate — a fact-record lookup on the engine side —
//!   is asked only about documents in contention for the top `k`, since a
//!   filter can remove entries from a ranking but never reorder it.
//!
//! The `decoded`, `scored` and `admitted` counters gate exactly this split in
//! CI, and `bm25_probe_work_is_bounded` pins the last one at `k`.
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

/// Reusable query scratch: score accumulator and top-k selection buffer. One
/// per concurrent reader; after warm-up a query allocates nothing (the
/// zero-alloc recall invariant).
#[derive(Debug, Default)]
pub struct Bm25Scratch {
    /// Partial scores as `(fact, score)` **sorted by fact id**.
    ///
    /// A map would be the obvious shape, and was the original one, but the
    /// postings are already sorted by fact id: accumulating a term is then a
    /// linear merge of two sorted runs instead of one hash probe per posting.
    /// The difference is not the hashing — it is that a map big enough to hold
    /// a frequent term's postings misses cache on essentially every probe,
    /// while a merge walks three arrays forwards.
    acc: Vec<(u32, f32)>,
    /// Merge target, swapped with `acc` after each term past the first.
    merge: Vec<(u32, f32)>,
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
    /// Document length by fact id, [`DOC_LEN_ABSENT`] where the id names no
    /// indexed document.
    ///
    /// Scoring needs the length of every document a query term's postings
    /// name, which is one lookup per posting entry — the single hottest read
    /// in the engine, and a random probe into an arena that deepens as the
    /// corpus grows. Fact ids are dense and monotone, so a flat array answers
    /// it by indexing, at four bytes per id.
    ///
    /// Runtime-only, like the arena's own page directory: rebuilt from
    /// `doc_len` on load, never written to the snapshot. `doc_len` remains the
    /// stored form and the only place the term-set summary lives.
    doc_len_dense: Vec<u32>,
}

/// [`Bm25Index::doc_len_dense`] entry for a fact id with no indexed document.
/// Lengths saturate at `u16::MAX`, so this cannot collide with a real one.
const DOC_LEN_ABSENT: u32 = u32::MAX;

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
            doc_len_dense: Vec::new(),
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
        let doc = DocLenSlot {
            fact,
            len: u16::try_from(len).unwrap_or(u16::MAX),
            // The caller's pairs are already unique per term, so their count
            // is the distinct-term count.
            distinct: u16::try_from(term_tfs.len()).unwrap_or(u16::MAX),
            sig,
        };
        self.doc_len.insert(&doc)?;
        self.note_dense(&doc);
        self.total_docs += 1;
        self.total_len += u64::from(len);
        Ok(())
    }

    /// Records a document's length in the flat index, growing it to reach the
    /// id. Fact ids are dense and monotone, so the growth is amortized.
    ///
    /// `usize` is 32 bits on wasm32, so an id that cannot be indexed there is
    /// simply left out: [`Bm25Index::doc_len_of`] then reports the document as
    /// absent, which scores it as the stored arena would for a missing record.
    fn note_dense(&mut self, doc: &DocLenSlot) {
        let Some(at) = usize::try_from(doc.fact.0).ok() else {
            return;
        };
        if at >= self.doc_len_dense.len() {
            let Some(len) = at.checked_add(1) else { return };
            self.doc_len_dense.resize(len, DOC_LEN_ABSENT);
        }
        self.doc_len_dense[at] = u32::from(doc.len);
    }

    /// Rebuilds the flat length index from the stored records (the load path).
    fn rebuild_dense(&mut self) {
        self.doc_len_dense.clear();
        let docs: Vec<DocLenSlot> = self.doc_len.iter().collect();
        for doc in docs {
            self.note_dense(&doc);
        }
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
                out.note_dense(&doc);
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
    ///
    /// Cost is O(Σ df) in *decodes*, which is the honest price of a lexical
    /// scan, but the expensive part of a candidate is not its decode — it is
    /// the two random lookups that used to follow it, one for the document
    /// length and one for the `live` predicate. Neither is paid per candidate
    /// any more: lengths come from a flat array, and `live` is asked only
    /// about documents that are actually in contention for the top `k`.
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
        let Bm25Scratch { acc, merge, top } = scratch;
        acc.clear();
        let avg_len = self.total_len as f32 / self.total_docs as f32;
        #[cfg(feature = "counters")]
        let (mut decoded, mut scored) = (0u64, 0u64);

        for &term in terms {
            let df = self.postings.count(term);
            if df == 0 {
                continue;
            }
            let idf = self.idf(df);
            // A term's postings are already ascending by fact id and `acc`
            // holds the same order, so accumulating is a merge of two sorted
            // runs. The first term has nothing to merge against and fills
            // `acc` directly.
            let mut ahead = 0usize;
            merge.clear();
            for (fact, tf) in self.postings.entries(term) {
                #[cfg(feature = "counters")]
                {
                    decoded += 1;
                }
                // Everything in `acc` below this posting keeps its score.
                while let Some(&entry) = acc.get(ahead)
                    && entry.0 < fact.0
                {
                    merge.push(entry);
                    ahead += 1;
                }
                let carried = match acc.get(ahead) {
                    Some(&entry) if entry.0 == fact.0 => {
                        ahead += 1;
                        Some(entry.1)
                    }
                    _ => None,
                };
                // A posting naming a document with no length record scores
                // nothing — and must not create a candidate either.
                let Some(len) = self.doc_len_of(fact) else {
                    if let Some(score) = carried {
                        merge.push((fact.0, score));
                    }
                    continue;
                };
                #[cfg(feature = "counters")]
                {
                    scored += 1;
                }
                let tf = f32::from(tf);
                let norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * f32::from(len) / avg_len));
                // Terms are summed in query order, the order the accumulating
                // map used, so the float result is bit-for-bit the same.
                merge.push((fact.0, carried.unwrap_or(0.0) + idf * norm));
            }
            merge.extend_from_slice(&acc[ahead.min(acc.len())..]);
            core::mem::swap(acc, merge);
        }
        #[cfg(feature = "counters")]
        {
            self.decoded.set(self.decoded.get() + decoded);
            self.scored.set(self.scored.get() + scored);
        }

        // Top-k. Ranking is by score alone, so `live` cannot change the order
        // — only remove entries from it. Asking it about every candidate is
        // therefore wasted work: rank first, then walk the ranking and ask
        // only until `k` survivors are found. The result is the same set in
        // the same order an exhaustive filter produces.
        top.clear();
        top.extend(acc.iter().map(|&(id, score)| (score, id)));
        let order = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        #[cfg(feature = "counters")]
        let mut admitted = 0u64;
        let mut consume = |band: &[(f32, u32)], out: &mut Vec<(FactId, f32)>| {
            for &(score, id) in band {
                if out.len() == k {
                    return;
                }
                #[cfg(feature = "counters")]
                {
                    admitted += 1;
                }
                if live(FactId(id)) {
                    out.push((FactId(id), score));
                }
            }
        };

        // The usual case: partition the `k` highest scores to the front,
        // order them, and take the survivors. Linear, and it asks `live`
        // about `k` documents rather than every candidate.
        let band = k.min(top.len());
        if band > 0 {
            if band < top.len() {
                top.select_nth_unstable_by(band - 1, order);
            }
            top[..band].sort_unstable_by(order);
            consume(&top[..band], out);
        }
        // The band was thinned by tombstones or a filter. Order what is left
        // in one pass and continue down it — the same total cost the
        // exhaustive filter used to pay on every query, now only on a query
        // that needs it.
        if out.len() < k && band < top.len() {
            top[band..].sort_unstable_by(order);
            let (_, rest) = top.split_at(band);
            consume(rest, out);
        }
        #[cfg(feature = "counters")]
        self.admitted.set(self.admitted.get() + admitted);
    }

    /// Length of the document `fact` names, or `None` when it names none.
    fn doc_len_of(&self, fact: FactId) -> Option<u16> {
        match self.doc_len_dense.get(fact.0 as usize).copied() {
            Some(DOC_LEN_ABSENT) | None => None,
            Some(len) => Some(len as u16),
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
        let mut index = Self {
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
            doc_len_dense: Vec::new(),
        };
        // One sequential pass over the stored records; the flat index is
        // derived state and is never part of the image.
        index.rebuild_dense();
        index
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
