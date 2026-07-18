//! The engine: data model state, verbs, journaling and replay
//! (specs/02, /03, /05).
//!
//! All state lives in the flat structures of `plugmem-arena`; every
//! mutating verb journals itself through the caller's
//! [`Storage`](crate::storage::Storage) and is replayed through the same
//! internal `apply_*` path on open — the journal's ids are authoritative,
//! so replay is deterministic and idempotent over a reapplied tail.
//!
//! This stage covers the structural engine (facts, entities, edges, tags,
//! BM25, temporal). The vector layer lands in stage 4 — a config with
//! `dim != 0` is rejected until then, so nobody can hold vectors the
//! engine would silently drop. Recall and similar-detection land with the
//! fusion layer; `maintain` with compaction.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, ChunkPool, ChunkPoolCfg, Interner, ListHandle,
    ShardMode, TermId, key,
};

use crate::config::Config;
use crate::error::Error;
use crate::id::{EntityId, FactId, NONE_U32};
use crate::index::IdListIndex;
use crate::index::bm25::Bm25Index;
use crate::journal::{JournalScan, Op, scan};
use crate::model::{
    EdgeSlot, EntityByName, EntityRecord, FactAux, FactRecord, TemporalSlot, VALID_TO_OPEN,
    fact_flags,
};
use crate::storage::Storage;
use crate::tokenizer::Tokenizer;

/// Most recent same-entity facts examined by similar-detection.
const SIMILAR_CANDIDATE_CAP: usize = 32;

mod recall;

pub use recall::{RecallQuery, RecallResult, RecalledEdge, RecalledFact, source};

/// Input of `remember` and `revise` (specs/05).
#[derive(Clone, Copy, Debug)]
pub struct RememberInput<'a> {
    /// Host timestamp, unix milliseconds.
    pub now: u64,
    /// Fact text, ≤ `Config::max_text` bytes.
    pub text: &'a str,
    /// Subject entity name; created lazily on first mention.
    pub entity: Option<&'a str>,
    /// Tags, verbatim strings, ≤ 32.
    pub tags: &'a [&'a str],
    /// `(rel, target_entity)` pairs, ≤ 16; edges go subject → target.
    pub links: &'a [(&'a str, &'a str)],
    /// Validity start; defaults to `now`.
    pub valid_from: Option<u64>,
}

impl<'a> RememberInput<'a> {
    /// A minimal input: text only.
    pub fn text(now: u64, text: &'a str) -> Self {
        Self {
            now,
            text,
            entity: None,
            tags: &[],
            links: &[],
            valid_from: None,
        }
    }
}

/// Input of `link` (specs/05).
#[derive(Clone, Copy, Debug)]
pub struct LinkInput<'a> {
    /// Host timestamp, unix milliseconds.
    pub now: u64,
    /// Source entity name (created lazily).
    pub src: &'a str,
    /// Relation term, verbatim (`"works_at"`, …).
    pub rel: &'a str,
    /// Destination entity name (created lazily).
    pub dst: &'a str,
    /// Optional provenance fact.
    pub provenance: Option<FactId>,
}

/// Result of `remember`/`revise` (specs/05).
#[derive(Clone, Debug, PartialEq)]
pub struct RememberOutcome {
    /// The new fact's id.
    pub id: FactId,
    /// The subject entity (resolved or created), if one was named.
    pub entity: Option<EntityId>,
    /// Similar / potentially conflicting **live** facts, best match
    /// first (≤ 8). The engine never revises on its own — it surfaces
    /// the candidates, the agent judges: `revise`, keep both, or
    /// `forget` (specs/05).
    pub similar: Vec<Similar>,
}

/// One similar-fact hint (specs/05).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similar {
    /// The existing fact.
    pub id: FactId,
    /// Match strength (for the lexical detector: the Jaccard overlap of
    /// the term sets, in `(similar_jaccard, 1]`).
    pub score: f32,
    /// What triggered the hint.
    pub reason: SimilarReason,
}

/// Why a fact was flagged as similar (specs/05). The vector variant
/// arrives with the vector layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimilarReason {
    /// Same subject entity and a term-set overlap above the configured
    /// Jaccard threshold.
    LexicalOverlap,
}

/// A read view of one fact.
#[derive(Clone, Copy, Debug)]
pub struct FactView<'a> {
    /// The raw record (temporality, flags, references).
    pub record: FactRecord,
    /// The fact text.
    pub text: &'a str,
}

/// Report of an `open`: what the journal replay found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenReport {
    /// Journal records applied.
    pub replayed: usize,
    /// Journal records skipped as already contained in the snapshot.
    pub skipped: usize,
    /// A torn tail record was dropped (crash between appends).
    pub truncated_tail: bool,
}

/// The memory engine (specs/05). See the module docs for staging.
pub struct Memory {
    cfg: Config,
    // -- data model --
    facts: Arena<FactRecord>,
    fact_aux: Arena<FactAux>,
    entities: Arena<EntityRecord>,
    by_name: Arena<EntityByName>,
    edges_out: Arena<EdgeSlot>,
    edges_in: Arena<EdgeSlot>,
    temporal: Arena<TemporalSlot>,
    /// Fact texts and canonical entity names.
    texts: BlobHeap,
    /// Terms: tokens, tags (verbatim), relation names, normalized entity
    /// names.
    terms: Interner,
    /// Per-fact tag lists (`TermId` values), handles in [`FactAux`].
    tag_lists: ChunkPool,
    // -- indexes --
    bm25: Bm25Index,
    tags_idx: IdListIndex,
    entity_facts: IdListIndex,
    // -- id allocation (derived from the arenas on load) --
    next_fact: u32,
    next_entity: u32,
    // -- reusable scratches --
    tokenizer: Tokenizer,
    tf_scratch: Vec<(u32, u8)>,
    name_scratch: String,
    recall_scratch: recall::RecallScratch,
}

impl Memory {
    /// Creates an empty database.
    ///
    /// # Errors
    ///
    /// [`Error::ConfigMismatch`] for an invalid config, including
    /// `dim != 0` while the vector layer is not built.
    pub fn new(cfg: Config) -> Result<Self, Error> {
        cfg.validate()?;
        if cfg.dim != 0 {
            return Err(Error::ConfigMismatch(
                "vector layer lands in stage 4: dim must be 0",
            ));
        }
        let uni =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(cfg.max_bytes);
        let ord =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Ordered).with_max_bytes(cfg.max_bytes);
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);
        Ok(Self {
            facts: Arena::new(uni(cfg.shards_facts))?,
            fact_aux: Arena::new(uni(cfg.shards_facts))?,
            entities: Arena::new(uni(cfg.shards_entities))?,
            by_name: Arena::new(ord(cfg.shards_entities))?,
            edges_out: Arena::new(ord(cfg.shards_edges))?,
            edges_in: Arena::new(ord(cfg.shards_edges))?,
            temporal: Arena::new(ord(cfg.shards_temporal))?,
            texts: BlobHeap::new(blob),
            terms: Interner::new(blob),
            tag_lists: ChunkPool::new(ChunkPoolCfg::new().with_max_bytes(cfg.max_bytes)),
            bm25: Bm25Index::new(cfg.shards_postings, cfg.max_bytes)?,
            tags_idx: IdListIndex::new(cfg.shards_postings, cfg.max_bytes)?,
            entity_facts: IdListIndex::new(cfg.shards_entities, cfg.max_bytes)?,
            next_fact: 0,
            next_entity: 0,
            tokenizer: Tokenizer::new(),
            tf_scratch: Vec::new(),
            name_scratch: String::new(),
            recall_scratch: recall::RecallScratch::default(),
            cfg,
        })
    }

    /// Opens a database from a storage: journal replay over an empty
    /// state (the snapshot fast-path lands with the section composition;
    /// a non-empty snapshot is rejected rather than half-read).
    pub fn open<S: Storage>(store: &mut S, cfg: Config) -> Result<(Self, OpenReport), Error> {
        let snapshot = store
            .read_snapshot()
            .map_err(|e| Error::Storage(format!("{e:?}")))?;
        let journal = store
            .read_journal()
            .map_err(|e| Error::Storage(format!("{e:?}")))?;
        Self::from_bytes(snapshot.as_deref(), &journal, cfg)
    }

    /// Opens from raw bytes (the wasm path: the host already read them).
    pub fn from_bytes(
        snapshot: Option<&[u8]>,
        journal: &[u8],
        cfg: Config,
    ) -> Result<(Self, OpenReport), Error> {
        if snapshot.is_some() {
            return Err(Error::Corrupt(
                "snapshot loading lands with the section composition",
            ));
        }
        let mut mem = Self::new(cfg)?;
        let report = mem.replay(journal)?;
        Ok((mem, report))
    }

    /// Applies every journal record on top of the current state (specs/03
    /// replay rules: assigned ids are authoritative; records whose
    /// assigned id is already present are skipped, which makes reapplying
    /// a tail idempotent).
    fn replay(&mut self, journal: &[u8]) -> Result<OpenReport, Error> {
        let JournalScan {
            entries,
            truncated_tail,
        } = scan(journal)?;
        let mut report = OpenReport {
            truncated_tail,
            ..OpenReport::default()
        };
        for entry in entries {
            let op = Op::decode(entry.op, entry.payload)?;
            match op {
                Op::Remember {
                    now,
                    valid_from,
                    entity,
                    text,
                    ref tags,
                    ref links,
                    revises,
                    assigned,
                } => {
                    if assigned.0 < self.next_fact {
                        report.skipped += 1;
                        continue;
                    }
                    if assigned.0 != self.next_fact {
                        return Err(Error::Corrupt("journal fact ids are not contiguous"));
                    }
                    if let Some(target) = revises.some() {
                        self.check_revisable(target)
                            .map_err(|_| Error::Corrupt("journal revises an unrevisable fact"))?;
                    }
                    self.apply_remember(
                        &RememberInput {
                            now,
                            text,
                            entity,
                            tags: &tags.to_vec(),
                            links: &links.to_vec(),
                            valid_from: Some(valid_from),
                        },
                        revises,
                    )?;
                    if let Some(target) = revises.some() {
                        self.close_target(target, valid_from);
                    }
                    report.replayed += 1;
                }
                Op::Forget { fact, .. } => {
                    // Idempotent; a missing fact in a valid journal is
                    // corruption, but re-tombstoning is fine.
                    match self.apply_forget(fact) {
                        Ok(_) => report.replayed += 1,
                        Err(Error::NotFound(_)) => {
                            return Err(Error::Corrupt("journal forgets an unknown fact"));
                        }
                        Err(e) => return Err(e),
                    }
                }
                Op::Link {
                    now,
                    src,
                    rel,
                    dst,
                    provenance,
                } => {
                    self.apply_link(now, src, rel, dst, provenance)?;
                    report.replayed += 1;
                }
                Op::Maintain { .. } => {
                    // v1 replay: the marker carries no state to reapply.
                    report.replayed += 1;
                }
            }
        }
        Ok(report)
    }

    /// Remembers a new fact (specs/05).
    pub fn remember<S: Storage>(
        &mut self,
        store: &mut S,
        input: RememberInput<'_>,
    ) -> Result<RememberOutcome, Error> {
        self.validate_input(&input)?;
        let mut outcome = self.apply_remember(&input, FactId::NONE)?;
        self.find_similar(&mut outcome);
        self.journal_remember(store, &input, FactId::NONE, outcome.id)?;
        Ok(outcome)
    }

    /// Batch import: a sequence of remembers in one call, journaled
    /// individually (specs/05). `skip_similar` turns off the
    /// similar-detection pass — imports don't need hints, and skipping
    /// them makes a bulk load cheaper.
    ///
    /// Stops at the first error; already-applied inputs stay applied and
    /// journaled (replay reproduces exactly the applied prefix).
    pub fn remember_batch<S: Storage>(
        &mut self,
        store: &mut S,
        inputs: &[RememberInput<'_>],
        skip_similar: bool,
    ) -> Result<Vec<RememberOutcome>, Error> {
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            self.validate_input(input)?;
            let mut outcome = self.apply_remember(input, FactId::NONE)?;
            if !skip_similar {
                self.find_similar(&mut outcome);
            }
            self.journal_remember(store, input, FactId::NONE, outcome.id)?;
            out.push(outcome);
        }
        Ok(out)
    }

    /// Lexical similar-detection (specs/05): live facts of the same
    /// entity whose term sets overlap the new fact's above the
    /// `similar_jaccard` threshold. Bounded: the entity's most recent
    /// [`SIMILAR_CANDIDATE_CAP`] facts are compared (a hub's full list is
    /// not re-tokenized), candidate texts are tokenized against the
    /// term-frequency scratch the new fact just filled.
    fn find_similar(&mut self, outcome: &mut RememberOutcome) {
        let Some(entity) = outcome.entity else { return };
        // The new fact's term set is still in tf_scratch (apply_remember
        // filled it); snapshot the term ids.
        let new_terms: Vec<u32> = self.tf_scratch.iter().map(|&(t, _)| t).collect();
        if new_terms.is_empty() {
            return;
        }
        // Most recent candidates of the entity (ring over the list).
        let mut ring = [FactId::NONE; SIMILAR_CANDIDATE_CAP];
        let mut n = 0usize;
        for (fact, _) in self.entity_facts.entries(entity.0) {
            if fact != outcome.id {
                ring[n % SIMILAR_CANDIDATE_CAP] = fact;
                n += 1;
            }
        }
        let mut cand_terms: Vec<u32> = Vec::new();
        for &fact in ring.iter().take(n.min(SIMILAR_CANDIDATE_CAP)) {
            let Some(record) = self.fact(fact) else {
                continue;
            };
            if record.is_tombstone() || record.is_closed() {
                continue;
            }
            let text = core::str::from_utf8(self.texts.get(record.text))
                .expect("fact texts are written from &str");
            cand_terms.clear();
            let terms = &self.terms;
            let cand = &mut cand_terms;
            self.tokenizer.tokenize(text, &mut |token| {
                if let Some(term) = terms.lookup(token)
                    && !cand.contains(&term.0)
                {
                    cand.push(term.0);
                }
            });
            if cand_terms.is_empty() {
                continue;
            }
            let both = cand_terms.iter().filter(|t| new_terms.contains(t)).count();
            let union = cand_terms.len() + new_terms.len() - both;
            let jaccard = both as f32 / union as f32;
            if jaccard > self.cfg.similar_jaccard {
                outcome.similar.push(Similar {
                    id: fact,
                    score: jaccard,
                    reason: SimilarReason::LexicalOverlap,
                });
            }
        }
        outcome
            .similar
            .sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        outcome.similar.truncate(8);
    }

    /// Revises `target`: closes its validity at the new fact's
    /// `valid_from` and records the new fact with `revises = target`
    /// (specs/02 rule 2).
    pub fn revise<S: Storage>(
        &mut self,
        store: &mut S,
        target: FactId,
        input: RememberInput<'_>,
    ) -> Result<RememberOutcome, Error> {
        self.validate_input(&input)?;
        // Check first, mutate last: the target closes only after the new
        // fact exists, so a mid-operation failure (capacity) never leaves
        // a closed fact without its successor.
        self.check_revisable(target)?;
        let outcome = self.apply_remember(&input, target)?;
        let valid_from = input.valid_from.unwrap_or(input.now);
        self.close_target(target, valid_from);
        self.journal_remember(store, &input, target, outcome.id)?;
        Ok(outcome)
    }

    /// Tombstones a fact: it disappears from every query immediately, the
    /// bytes go with the next `maintain` (specs/02 rule 3). `Ok(false)`
    /// when it was already tombstoned.
    pub fn forget<S: Storage>(
        &mut self,
        store: &mut S,
        now: u64,
        id: FactId,
    ) -> Result<bool, Error> {
        let fresh = self.apply_forget(id)?;
        let mut entry = Vec::new();
        Op::Forget { now, fact: id }.encode(&mut entry);
        store
            .append_journal(&entry)
            .map_err(|e| Error::Storage(format!("{e:?}")))?;
        Ok(fresh)
    }

    /// Upserts a typed edge between two entities (created lazily);
    /// re-linking an existing `(src, rel, dst)` updates its provenance.
    pub fn link<S: Storage>(&mut self, store: &mut S, input: LinkInput<'_>) -> Result<(), Error> {
        self.apply_link(
            input.now,
            input.src,
            input.rel,
            input.dst,
            FactId::from_opt(input.provenance),
        )?;
        let mut entry = Vec::new();
        Op::Link {
            now: input.now,
            src: input.src,
            rel: input.rel,
            dst: input.dst,
            provenance: FactId::from_opt(input.provenance),
        }
        .encode(&mut entry);
        store
            .append_journal(&entry)
            .map_err(|e| Error::Storage(format!("{e:?}")))?;
        Ok(())
    }

    /// Returns a fact unless it is tombstoned (closed facts are
    /// returned — their interval says so).
    pub fn get(&self, id: FactId) -> Option<FactView<'_>> {
        let record = self.fact(id)?;
        if record.is_tombstone() {
            return None;
        }
        let text = core::str::from_utf8(self.texts.get(record.text))
            .expect("fact texts are written from &str");
        Some(FactView { record, text })
    }

    /// Appends the tag terms of a fact to `out`. A tombstoned or unknown
    /// fact contributes nothing.
    pub fn tags_of(&self, id: FactId, out: &mut Vec<TermId>) {
        let Some(record) = self.fact(id) else { return };
        if record.is_tombstone() {
            return;
        }
        let Some(aux) = self.fact_aux.get(&id.0.to_be_bytes()) else {
            return;
        };
        for chunk in self.tag_lists.iter(&aux.tags) {
            for raw in chunk.chunks_exact(4) {
                out.push(TermId(u32::from_be_bytes(raw.try_into().unwrap())));
            }
        }
    }

    /// Resolves an entity by name (normalized), without creating it.
    pub fn entity(&mut self, name: &str) -> Option<EntityId> {
        let mut norm = core::mem::take(&mut self.name_scratch);
        normalize_name(&mut self.tokenizer, name, &mut norm);
        let found = if norm.is_empty() {
            None
        } else {
            // Only resolve an *existing* term: interning would create it.
            self.lookup_entity_by_norm(&norm)
        };
        self.name_scratch = norm;
        found
    }

    /// Resolves the string behind a term id (tags, relations).
    pub fn term(&self, id: TermId) -> &str {
        self.terms.resolve(id)
    }

    /// Number of facts ever recorded (including tombstoned, until
    /// `maintain` purges them).
    pub fn facts_len(&self) -> usize {
        self.facts.len()
    }

    /// Number of entities.
    pub fn entities_len(&self) -> usize {
        self.entities.len()
    }

    /// The engine configuration.
    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    // ---- internals ----

    fn fact(&self, id: FactId) -> Option<FactRecord> {
        self.facts.get(&id.0.to_be_bytes())
    }

    /// Size-limit checks shared by remember and revise.
    fn validate_input(&self, input: &RememberInput<'_>) -> Result<(), Error> {
        if input.text.len() > self.cfg.max_text {
            return Err(Error::TooLarge {
                what: "text",
                len: input.text.len(),
                max: self.cfg.max_text,
            });
        }
        if input.tags.len() > 32 {
            return Err(Error::TooLarge {
                what: "tags",
                len: input.tags.len(),
                max: 32,
            });
        }
        if input.links.len() > 16 {
            return Err(Error::TooLarge {
                what: "links",
                len: input.links.len(),
                max: 16,
            });
        }
        if input.tags.iter().any(|t| t.is_empty()) {
            return Err(Error::Invalid("empty tag"));
        }
        if !input.links.is_empty() && input.entity.is_none() {
            return Err(Error::Invalid("links require a subject entity"));
        }
        Ok(())
    }

    /// `Ok` when `target` exists, is not tombstoned and not yet closed.
    fn check_revisable(&self, target: FactId) -> Result<(), Error> {
        let record = self.fact(target).ok_or(Error::NotFound(target))?;
        if record.is_tombstone() {
            return Err(Error::NotFound(target));
        }
        if record.is_closed() {
            return Err(Error::AlreadyClosed(target));
        }
        Ok(())
    }

    /// Closes a revision target (validity ends where the successor
    /// starts). The caller checked [`Memory::check_revisable`] first.
    fn close_target(&mut self, target: FactId, valid_to: u64) {
        let record = self.fact(target).expect("checked revisable");
        let payload = self
            .facts
            .payload_mut(&target.0.to_be_bytes())
            .expect("record fetched above");
        // Payload offsets = slot offsets - KEY_LEN(4): flags at 4..6,
        // valid_to at 36..44.
        let flags = record.flags | fact_flags::CLOSED;
        payload[4..6].copy_from_slice(&flags.to_be_bytes());
        payload[36..44].copy_from_slice(&valid_to.to_be_bytes());
    }

    /// The one execution path of a new fact — called by the public verbs
    /// and by replay with identical effects.
    fn apply_remember(
        &mut self,
        input: &RememberInput<'_>,
        revises: FactId,
    ) -> Result<RememberOutcome, Error> {
        let id = FactId(self.next_fact);
        let entity = match input.entity {
            Some(name) => Some(self.resolve_or_create_entity(name, input.now)?),
            None => None,
        };
        let text_id = self.texts.push(input.text.as_bytes())?;

        // Tokenize into (term, tf) pairs in the reusable scratch.
        let mut tfs = core::mem::take(&mut self.tf_scratch);
        tfs.clear();
        let terms = &mut self.terms;
        let mut intern_err = None;
        self.tokenizer.tokenize(input.text, &mut |token| {
            if intern_err.is_some() {
                return;
            }
            match terms.intern(token) {
                Ok(term) => match tfs.iter_mut().find(|(t, _)| *t == term.0) {
                    Some((_, tf)) => *tf = tf.saturating_add(1),
                    None => tfs.push((term.0, 1)),
                },
                Err(e) => intern_err = Some(e),
            }
        });
        if let Some(e) = intern_err {
            self.tf_scratch = tfs;
            return Err(Error::Arena(e));
        }
        self.bm25.index_doc(id, &tfs)?;
        self.tf_scratch = tfs;

        // Tags: interned verbatim, deduplicated, listed on the fact and
        // inverted.
        let mut aux = FactAux {
            id,
            tags: ListHandle::EMPTY,
        };
        let mut seen_tags: [u32; 32] = [NONE_U32; 32];
        let mut seen_cnt = 0usize;
        for tag in input.tags {
            let term = self.terms.intern(tag)?;
            if seen_tags[..seen_cnt].contains(&term.0) {
                continue;
            }
            seen_tags[seen_cnt] = term.0;
            seen_cnt += 1;
            self.tag_lists.push(&mut aux.tags, &term.0.to_be_bytes())?;
            self.tags_idx.push(term.0, id, 0)?;
        }
        self.fact_aux.insert(&aux)?;

        // Links: subject → target edges with this fact as provenance.
        if let Some(src) = entity {
            for &(rel, dst_name) in input.links {
                let dst = self.resolve_or_create_entity(dst_name, input.now)?;
                let rel = self.terms.intern(rel)?;
                self.upsert_edge(src, rel, dst, id)?;
            }
            self.entity_facts.push(src.0, id, 0)?;
        }

        let recorded_at = input.now;
        let valid_from = input.valid_from.unwrap_or(input.now);
        self.facts.insert(&FactRecord {
            id,
            entity: EntityId::from_opt(entity),
            flags: 0,
            kind: 0,
            text: text_id,
            vector: NONE_U32,
            revises,
            recorded_at,
            valid_from,
            valid_to: VALID_TO_OPEN,
        })?;
        self.temporal.insert(&TemporalSlot {
            recorded_at,
            fact: id,
        })?;
        self.next_fact += 1;
        Ok(RememberOutcome {
            id,
            entity,
            similar: Vec::new(),
        })
    }

    fn apply_forget(&mut self, id: FactId) -> Result<bool, Error> {
        let record = self.fact(id).ok_or(Error::NotFound(id))?;
        if record.is_tombstone() {
            return Ok(false);
        }
        let payload = self
            .facts
            .payload_mut(&id.0.to_be_bytes())
            .expect("record fetched above");
        let flags = record.flags | fact_flags::TOMBSTONE;
        payload[4..6].copy_from_slice(&flags.to_be_bytes());
        Ok(true)
    }

    fn apply_link(
        &mut self,
        now: u64,
        src: &str,
        rel: &str,
        dst: &str,
        provenance: FactId,
    ) -> Result<(), Error> {
        let src = self.resolve_or_create_entity(src, now)?;
        let dst = self.resolve_or_create_entity(dst, now)?;
        let rel = self.terms.intern(rel)?;
        self.upsert_edge(src, rel, dst, provenance)
    }

    fn upsert_edge(
        &mut self,
        src: EntityId,
        rel: TermId,
        dst: EntityId,
        fact: FactId,
    ) -> Result<(), Error> {
        for (arena, a, b) in [
            (&mut self.edges_out, src, dst),
            (&mut self.edges_in, dst, src),
        ] {
            let slot = EdgeSlot { a, rel, b, fact };
            if !arena.insert(&slot)? {
                let mut kb = [0u8; 12];
                key::write_u32(&mut kb, a.0);
                key::write_u32(&mut kb[4..], rel.0);
                key::write_u32(&mut kb[8..], b.0);
                let payload = arena.payload_mut(&kb).expect("insert reported a duplicate");
                payload.copy_from_slice(&fact.0.to_be_bytes());
            }
        }
        Ok(())
    }

    /// Looks an entity up by its already-normalized name (read-only:
    /// neither the vocabulary nor the arenas change on a miss).
    fn lookup_entity_by_norm(&self, norm: &str) -> Option<EntityId> {
        let term = self.terms.lookup(norm)?;
        let mut from = [0u8; 8];
        key::write_u32(&mut from, term.0);
        let mut to = [0u8; 8];
        key::write_u32(&mut to, term.0);
        to[4..].copy_from_slice(&u32::MAX.to_be_bytes());
        self.by_name.range(&from, &to).next().map(|e| e.id)
    }

    fn resolve_or_create_entity(&mut self, name: &str, now: u64) -> Result<EntityId, Error> {
        let mut norm = core::mem::take(&mut self.name_scratch);
        normalize_name(&mut self.tokenizer, name, &mut norm);
        if norm.is_empty() {
            self.name_scratch = norm;
            return Err(Error::Invalid("entity name has no indexable characters"));
        }
        let result = (|| {
            if let Some(found) = self.lookup_entity_by_norm(&norm) {
                return Ok(found);
            }
            let term = self.terms.intern(&norm)?;
            let id = EntityId(self.next_entity);
            let name_id = self.texts.push(name.as_bytes())?;
            self.entities.insert(&EntityRecord {
                id,
                name: name_id,
                name_term: term,
                created_at: now,
                flags: 0,
            })?;
            self.by_name.insert(&EntityByName {
                name_term: term,
                id,
            })?;
            self.next_entity += 1;
            Ok(id)
        })();
        self.name_scratch = norm;
        result
    }

    fn journal_remember<S: Storage>(
        &mut self,
        store: &mut S,
        input: &RememberInput<'_>,
        revises: FactId,
        assigned: FactId,
    ) -> Result<(), Error> {
        let mut entry = Vec::new();
        Op::Remember {
            now: input.now,
            valid_from: input.valid_from.unwrap_or(input.now),
            entity: input.entity,
            text: input.text,
            tags: input.tags.to_vec(),
            links: input.links.to_vec(),
            revises,
            assigned,
        }
        .encode(&mut entry);
        store
            .append_journal(&entry)
            .map_err(|e| Error::Storage(format!("{e:?}")))
    }
}

impl core::fmt::Debug for Memory {
    /// Summary only — the contents are the user's memory, not ours to
    /// print.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Memory")
            .field("facts", &self.facts.len())
            .field("entities", &self.entities.len())
            .field("terms", &self.terms.len())
            .finish()
    }
}

/// Normalizes an entity name: its tokens joined by single spaces
/// ("  Проект   PlugMem " → "проект plugmem"). Deterministic and aligned
/// with search-time tokenization.
fn normalize_name(tokenizer: &mut Tokenizer, name: &str, out: &mut String) {
    out.clear();
    tokenizer.tokenize(name, &mut |token| {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(token);
    });
}
