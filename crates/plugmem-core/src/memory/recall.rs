//! Hybrid recall: source ranking, RRF fusion, budgeted selection and the
//! rendered prompt block (–7).
//!
//! The pipeline (scratch buffers reused, the zero-alloc invariant):
//!
//! 1. the tag filter builds a sorted allow-set (intersection of tag
//!    lists); an unknown tag empties it — and the result;
//! 2. every source admits a candidate only through the shared rule:
//!    not tombstoned, `recorded_at ≤ as_of`, inside its validity
//!    interval (`include_closed` drops the upper bound), in the
//!    allow-set when tags are present;
//! 3. sources produce ranked lists of ≤ 128: BM25 over the query text,
//!    graph expansion from entity anchors (breadth-first over the edge
//!    arenas, weight `decay^depth`, hard caps on entities, edges and
//!    candidates), temporal range scan ranked by recency;
//! 4. **RRF**: `score(f) = Σ_s w_s / (rrf_k + rank_s(f))` — rank-based,
//!    so sources need no score calibration against each other;
//! 5. recency boost `× (1 + w_rec · 2^(-age / half_life))`;
//! 6. greedy selection by fused score under `k` and the token budget
//!    (`len(text)/4 + 8` tokens per fact);
//! 7. rendering into the compact prompt block (format fixed by golden
//!    tests).
//!
//! Revision chains need no extra dedup here: closing a fact bounds its
//! validity at the successor's start, so the `as_of` rule keeps at most
//! one live version of a chain (with `include_closed` the whole chain is
//! shown by design, intervals marking who is who).

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use plugmem_arena::TermId;

use crate::error::Error;
use crate::id::{EntityId, FactId};
use crate::index::bm25::Bm25Scratch;
use crate::index::hnsw::HnswScratch;
use crate::index::vecpool::{VecScratch, dot_i8};
use crate::index::{IntersectScratch, intersect};
use crate::model::{
    FactRecord, VALID_TO_OPEN, edge_end, edge_floor, edge_history_ceiling, edge_history_floor,
};
use crate::tokenizer::Tokenizer;

use super::Memory;

/// Source bits of [`RecalledFact::sources`].
pub mod source {
    /// The lexical (BM25) source.
    pub const BM25: u8 = 1;
    /// The graph-expansion source.
    pub const GRAPH: u8 = 1 << 1;
    /// The temporal-range source.
    pub const TIME: u8 = 1 << 2;
    /// The vector (quantized flat) source.
    pub const VEC: u8 = 1 << 3;
}

/// Per-source candidate cap.
const SOURCE_CAP: usize = 128;
/// Tag allow-sets up to this size are cheaper to inspect directly than to
/// discover through a broad temporal scan. Larger sets keep the temporal-first
/// path, which avoids materializing a large tag-side candidate list.
const TEMPORAL_TAG_FIRST_MAX: usize = SOURCE_CAP * 64;

/// Graph expansion caps.
const GRAPH_ENTITY_CAP: usize = 64;
const GRAPH_FACT_CAP: usize = 256;
const GRAPH_EDGE_CAP: usize = 128;
/// Hard budget on posting entries the graph source may *examine* — a hub
/// entity with tens of thousands of facts must not turn expansion into a
/// full decode of its list (the "hub super-node" guard applies
/// to work, not only to the candidate count).
const GRAPH_EXAMINE_CAP: usize = 2048;

/// Stop-frequency guard of the lexical source: a query term present in
/// more than 1/8 of the corpus (and in over [`STOP_DF_FLOOR`] documents)
/// is dropped from the query — its idf makes it nearly rank-neutral
/// while its posting list dominates the decode cost (querying "the" must
/// not cost O(corpus)). When *every* term is stop-frequent the least
/// frequent one is kept, so such a query still answers.
const STOP_DF_DIVISOR: u64 = 8;
/// Below this document frequency a term is never considered
/// stop-frequent (small corpora skip nothing).
const STOP_DF_FLOOR: u64 = 1024;

/// A recall request. `Default`-like construction via
/// [`RecallQuery::text`] plus field overrides.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RecallQuery<'a> {
    /// Host timestamp, unix milliseconds.
    pub now: u64,
    /// Free-text query for the lexical source.
    pub text: Option<&'a str>,
    /// Query embedding for the vector source (`len == Config::dim`).
    pub vector: Option<&'a [f32]>,
    /// Tag filter: a fact must carry *all* of these.
    pub tags: &'a [&'a str],
    /// Entity anchors for the graph source.
    pub entities: &'a [&'a str],
    /// Validity instant; defaults to `now`.
    pub as_of: Option<u64>,
    /// `recorded_at` window `[from, to)` for the temporal source.
    pub range: Option<(u64, u64)>,
    /// Result cap; `0` means the default 8, hard ceiling 64.
    pub k: usize,
    /// Token budget of the rendered block; defaults to 512.
    pub token_budget: Option<usize>,
    /// Show closed revisions too (whole chains, marked by intervals).
    pub include_closed: bool,
    /// HNSW beam-width override for the vector source; defaults to
    /// `Config::hnsw_ef_search`. Ignored while the engine is in the flat
    /// regime (below `Config::flat_to_hnsw`).
    pub ef: Option<usize>,
}

impl<'a> RecallQuery<'a> {
    /// A plain text query with every other knob at its default.
    pub fn text(now: u64, text: &'a str) -> Self {
        Self {
            now,
            text: Some(text),
            vector: None,
            tags: &[],
            entities: &[],
            as_of: None,
            range: None,
            k: 0,
            token_budget: None,
            include_closed: false,
            ef: None,
        }
    }
}

/// One recalled fact.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecalledFact {
    /// The fact.
    pub id: FactId,
    /// Fused score (RRF + recency boost).
    pub score: f32,
    /// Which sources surfaced it (see [`source`]).
    pub sources: u8,
    /// Subject entity or [`EntityId::NONE`].
    pub entity: EntityId,
    /// Knowledge axis.
    pub recorded_at: u64,
    /// Truth axis, start.
    pub valid_from: u64,
    /// Truth axis, end ([`VALID_TO_OPEN`] = open).
    pub valid_to: u64,
}

/// One edge the graph source walked (: agents want the relations,
/// not only the facts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecalledEdge {
    /// Source entity.
    pub src: EntityId,
    /// Relation term.
    pub rel: TermId,
    /// Destination entity.
    pub dst: EntityId,
    /// Provenance fact or [`FactId::NONE`].
    pub provenance: FactId,
}

/// A recall response. Reusable: pass to
/// [`Memory::recall_into`] repeatedly and the buffers are recycled.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecallResult {
    /// Selected facts, descending fused score.
    pub facts: Vec<RecalledFact>,
    /// Edges walked by the graph source (deduplicated).
    pub edges: Vec<RecalledEdge>,
    /// The compact prompt block (empty string when nothing was found).
    pub rendered: String,
    /// `true` when selection stopped at `k` or the token budget with
    /// candidates left over.
    pub truncated: bool,
}

/// Reusable recall scratch — **caller-owned**, so [`Memory::recall`] and
/// [`Memory::recall_into`] take `&self`: many readers can recall the same
/// engine at once, each threading its own scratch (the host wraps one per
/// thread). Carries every buffer a recall mutates — the query-side term/score
/// vectors, the fusion map, *and* its own tokenizer and name-normalization
/// buffer — so a recall never touches the engine's write-side scratches. Reused
/// across calls it upholds the zero-alloc invariant.
///
/// Opaque: construct with [`RecallScratch::new`] (or `Default`) and pass by
/// `&mut`; the fields are engine-internal.
#[derive(Debug, Default)]
pub struct RecallScratch {
    /// Read-path tokenizer (query text + entity-name normalization). Kept here,
    /// not in [`Memory`], so recall stays `&self`; writers use the engine's own.
    tokenizer: Tokenizer,
    /// Scratch for one normalized entity name during graph-anchor resolution.
    name_scratch: String,
    bm25: Bm25Scratch,
    intersect: IntersectScratch,
    allow: Vec<FactId>,
    allow_bits: AllowFilter,
    tag_terms: Vec<u32>,
    query_terms: Vec<u32>,
    bm25_out: Vec<(FactId, f32)>,
    vec: VecScratch,
    vec_out: Vec<(FactId, f32)>,
    hnsw: HnswScratch,
    hnsw_out: Vec<(u32, f32)>,
    graph_out: Vec<(FactId, f32)>,
    time_out: Vec<(FactId, f32)>,
    time_tag: Vec<(FactId, u64)>,
    visited: Vec<(EntityId, f32)>,
    fused: hashbrown::HashMap<u32, (f32, u8), xxhash_rust::xxh3::Xxh3Builder>,
    ranked: Vec<(FactId, f32, u8)>,
    tags_tmp: Vec<TermId>,
}

impl RecallScratch {
    /// An empty recall scratch (all buffers grow on first use). One per
    /// concurrent reader; reused across that reader's calls for zero-alloc.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Memory<'_> {
    /// Runs a recall, allocating a fresh [`RecallScratch`] and
    /// [`RecallResult`]. Convenience over [`Memory::recall_into`] for one-shot
    /// callers; a hot loop should own a [`RecallScratch`] and call
    /// `recall_into` to stay zero-alloc.
    pub fn recall(&self, q: RecallQuery<'_>) -> Result<RecallResult, Error> {
        let mut scratch = RecallScratch::default();
        let mut out = RecallResult::default();
        self.recall_into(q, &mut scratch, &mut out)?;
        Ok(out)
    }

    /// Runs a recall into a reused result and caller-owned scratch (the
    /// zero-alloc path: after warm-up neither `s` nor `out` allocate).
    ///
    /// Takes `&self` — recall never mutates engine data; every mutable buffer
    /// it needs lives in `s`. This is what lets many readers recall
    /// one engine concurrently, each with its own [`RecallScratch`].
    pub fn recall_into(
        &self,
        q: RecallQuery<'_>,
        s: &mut RecallScratch,
        out: &mut RecallResult,
    ) -> Result<(), Error> {
        out.facts.clear();
        out.edges.clear();
        out.rendered.clear();
        out.truncated = false;

        let k = if q.k == 0 { 8 } else { q.k.min(64) };
        let budget = q.token_budget.unwrap_or(512);
        let as_of = q.as_of.unwrap_or(q.now);

        // 1. Tag allow-set. An unknown tag can match nothing.
        s.allow.clear();
        s.tag_terms.clear();
        let mut dead_tag = false;
        for tag in q.tags {
            match self.terms.lookup(tag) {
                Some(term) => s.tag_terms.push(term.0),
                None => dead_tag = true,
            }
        }
        if !dead_tag && !s.tag_terms.is_empty() {
            intersect(&self.tags_idx, &s.tag_terms, &mut s.intersect, &mut s.allow);
        }
        let filtered = !q.tags.is_empty();
        if filtered && (dead_tag || s.allow.is_empty()) {
            return Ok(());
        }
        // Membership filter over the set the sources will test against, built
        // once per query rather than searched once per candidate.
        if filtered {
            s.allow_bits.fill(&s.allow);
        } else {
            s.allow_bits.clear();
        }

        // 2–3. Sources (each admits through the shared rule).
        s.bm25_out.clear();
        if let Some(text) = q.text {
            s.query_terms.clear();
            let terms = &self.terms;
            // Disjoint field borrows of `s`: the tokenizer writes into
            // `query_terms`, both live in the caller's scratch.
            let query_terms = &mut s.query_terms;
            s.tokenizer.tokenize(text, &mut |token| {
                if let Some(term) = terms.lookup(token) {
                    query_terms.push(term.0);
                }
            });
            // Stop-frequency filter (see the constants above).
            let docs = self.bm25.docs();
            let is_stop = |df: u64| df > STOP_DF_FLOOR && df * STOP_DF_DIVISOR > docs;
            if s.query_terms
                .iter()
                .any(|&t| !is_stop(u64::from(self.bm25.df(t))))
            {
                let bm25 = &self.bm25;
                s.query_terms.retain(|&t| !is_stop(u64::from(bm25.df(t))));
            } else if let Some(&least) = s.query_terms.iter().min_by_key(|&&t| self.bm25.df(t)) {
                s.query_terms.clear();
                s.query_terms.push(least);
            }
            let facts = &self.facts;
            let allow = &s.allow;
            let allow_bits = &s.allow_bits;
            self.bm25.search(
                (self.cfg.bm25_k1, self.cfg.bm25_b),
                &s.query_terms,
                SOURCE_CAP,
                &mut |id| {
                    admit(
                        facts,
                        allow,
                        allow_bits,
                        filtered,
                        as_of,
                        q.include_closed,
                        id,
                    )
                    .is_some()
                },
                &mut s.bm25,
                &mut s.bm25_out,
            );
        }

        // Vector source: flat quantized search below the HNSW threshold,
        // graph search plus a flat-tail scan above it.
        s.vec_out.clear();
        if let Some(v) = q.vector
            && self.cfg.dim > 0
        {
            let res = if self.hnsw.indexed() == 0 {
                let facts = &self.facts;
                let allow = &s.allow;
                let allow_bits = &s.allow_bits;
                self.vecs.search(
                    v,
                    SOURCE_CAP,
                    &mut |id| {
                        admit(
                            facts,
                            allow,
                            allow_bits,
                            filtered,
                            as_of,
                            q.include_closed,
                            id,
                        )
                        .is_some()
                    },
                    &mut s.vec,
                    &mut s.vec_out,
                )
            } else {
                self.vec_graph_source(v, &q, as_of, filtered, s)
            };
            res?;
        }

        // Graph anchors resolve here (name normalization needs the tokenizer
        // and name buffer — both in the caller's scratch); expansion is
        // read-only. `tokenizer` and `name_scratch` are disjoint fields of `s`.
        s.visited.clear();
        for name in q.entities {
            super::normalize_name(&mut s.tokenizer, name, &mut s.name_scratch);
            let found = self.lookup_entity_by_norm(&s.name_scratch);
            if let Some(id) = found
                && !s.visited.iter().any(|&(e, _)| e == id)
            {
                s.visited.push((id, 1.0));
            }
        }
        self.graph_source(&q, as_of, filtered, s, out);
        self.time_source(&q, as_of, filtered, s);

        // 4. RRF fusion.
        s.fused.clear();
        for (list, weight, bit) in [
            (&s.bm25_out, self.cfg.w_bm25, source::BM25),
            (&s.vec_out, self.cfg.w_vec, source::VEC),
            (&s.graph_out, self.cfg.w_graph, source::GRAPH),
            (&s.time_out, self.cfg.w_time, source::TIME),
        ] {
            for (rank, &(fact, _)) in list.iter().enumerate() {
                let contribution = weight / (self.cfg.rrf_k as f32 + rank as f32 + 1.0);
                let entry = s.fused.entry(fact.0).or_insert((0.0, 0));
                entry.0 += contribution;
                entry.1 |= bit;
            }
        }

        // 5. Recency boost.
        let half_life_ms = self.cfg.half_life_days as f32 * 86_400_000.0;
        s.ranked.clear();
        for (&id, &(score, bits)) in &s.fused {
            let record = self.facts.get(&id.to_be_bytes()).expect("fused ids exist");
            let age = q.now.saturating_sub(record.recorded_at) as f32;
            let boost = 1.0 + self.cfg.w_recency * libm::exp2f(-age / half_life_ms);
            s.ranked.push((FactId(id), score * boost, bits));
        }
        s.ranked
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        // 6. Budgeted selection.
        let mut spent = 0usize;
        for &(id, score, bits) in &s.ranked {
            if out.facts.len() == k {
                out.truncated = true;
                break;
            }
            let record = self
                .facts
                .get(&id.0.to_be_bytes())
                .expect("ranked ids exist");
            let cost = self.texts.get(record.text).len() / 4 + 8;
            if spent + cost > budget {
                out.truncated = true;
                break;
            }
            spent += cost;
            out.facts.push(RecalledFact {
                id,
                score,
                sources: bits,
                entity: record.entity,
                recorded_at: record.recorded_at,
                valid_from: record.valid_from,
                valid_to: record.valid_to,
            });
        }

        // 7. Render.
        self.render(out, &mut s.tags_tmp);
        Ok(())
    }

    /// The above-threshold vector source (phase 2): an HNSW
    /// beam search over the graph plus an exact scan of the flat tail
    /// (vectors appended since the last `maintain` build), merged,
    /// admission-filtered and capped like every other source.
    fn vec_graph_source(
        &self,
        v: &[f32],
        q: &RecallQuery<'_>,
        as_of: u64,
        filtered: bool,
        s: &mut RecallScratch,
    ) -> Result<(), Error> {
        let RecallScratch {
            vec,
            hnsw,
            hnsw_out,
            vec_out,
            allow,
            allow_bits,
            ..
        } = s;
        self.vecs.quantize_query(v, vec)?;
        let (q_scale, q_q) = self.vecs.quantized(vec);
        let ef = q.ef.unwrap_or(self.cfg.hnsw_ef_search).max(1);
        self.hnsw
            .search_quantized(&self.vecs, (q_scale, q_q), ef, hnsw, hnsw_out);
        for slot in self.hnsw.indexed()..self.vecs.len() as u32 {
            let (s_scale, s_q) = self.vecs.quant(slot as usize);
            hnsw_out.push((slot, q_scale * s_scale * dot_i8(q_q, s_q) as f32));
        }
        vec_out.clear();
        for &(slot, sim) in hnsw_out.iter() {
            let fact = FactId(self.vecs.slot_fact(slot as usize));
            if admit(
                &self.facts,
                allow,
                allow_bits,
                filtered,
                as_of,
                q.include_closed,
                fact,
            )
            .is_some()
            {
                vec_out.push((fact, sim));
            }
        }
        vec_out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        vec_out.truncate(SOURCE_CAP);
        Ok(())
    }

    /// Graph expansion: anchors → neighbors (≤ depth), candidate facts of
    /// every visited entity plus edge provenance, ranked by hop weight.
    fn graph_source(
        &self,
        q: &RecallQuery<'_>,
        as_of: u64,
        filtered: bool,
        s: &mut RecallScratch,
        out: &mut RecallResult,
    ) {
        let RecallScratch {
            allow,
            allow_bits,
            graph_out,
            visited,
            ..
        } = s;
        graph_out.clear();
        if visited.is_empty() {
            return;
        }

        // Breadth-first: `frontier` marks where the current depth starts.
        //
        // Both caps full means every further edge is a no-op — it can neither
        // enter `out.edges` nor `visited` — so expansion stops there instead of
        // decoding the rest of a hub's edge list. The visited prefix, and with
        // it the result, is exactly what an exhaustive walk produced.
        let full = |edges: &Vec<RecalledEdge>, visited: &Vec<(EntityId, f32)>| {
            edges.len() >= GRAPH_EDGE_CAP && visited.len() >= GRAPH_ENTITY_CAP
        };
        let mut frontier = 0usize;
        let mut weight = 1.0f32;
        'expand: for _ in 0..self.cfg.graph_depth {
            let depth_end = visited.len();
            weight *= self.cfg.graph_decay;
            for at in frontier..depth_end {
                if full(&out.edges, visited) {
                    break 'expand;
                }
                let (entity, _) = visited[at];
                self.neighbors(
                    entity,
                    as_of,
                    q.as_of.is_some(),
                    &mut |neighbor, rel, this_side_src, provenance| {
                        let (src, dst) = if this_side_src {
                            (entity, neighbor)
                        } else {
                            (neighbor, entity)
                        };
                        let edge = RecalledEdge {
                            src,
                            rel,
                            dst,
                            provenance,
                        };
                        if out.edges.len() < GRAPH_EDGE_CAP && !out.edges.contains(&edge) {
                            out.edges.push(edge);
                        }
                        if visited.len() < GRAPH_ENTITY_CAP
                            && !visited.iter().any(|&(e, _)| e == neighbor)
                        {
                            visited.push((neighbor, weight));
                        }
                        !full(&out.edges, visited)
                    },
                );
            }
            frontier = depth_end;
        }

        // Candidate facts: every visited entity's facts at that entity's
        // weight, plus provenance facts at their edge's weight. Both the
        // candidate count and the *examined* entries are budgeted.
        let mut examined = 0usize;
        'entities: for &(entity, weight) in visited.iter() {
            for (fact, _) in self.entity_facts.entries(entity.0) {
                examined += 1;
                if graph_out.len() >= GRAPH_FACT_CAP || examined > GRAPH_EXAMINE_CAP {
                    break 'entities;
                }
                if admit(
                    &self.facts,
                    allow,
                    allow_bits,
                    filtered,
                    as_of,
                    q.include_closed,
                    fact,
                )
                .is_some()
                {
                    graph_out.push((fact, weight));
                }
            }
        }
        for edge in out.edges.iter() {
            if graph_out.len() >= GRAPH_FACT_CAP {
                break;
            }
            if let Some(fact) = edge.provenance.some()
                && !graph_out.iter().any(|&(f, _)| f == fact)
                && admit(
                    &self.facts,
                    allow,
                    allow_bits,
                    filtered,
                    as_of,
                    q.include_closed,
                    fact,
                )
                .is_some()
            {
                graph_out.push((fact, self.cfg.graph_decay));
            }
        }
        graph_out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        graph_out.truncate(SOURCE_CAP);
        graph_out.dedup_by_key(|&mut (f, _)| f);
    }

    /// Temporal range source: facts recorded in `[from, to)`, most recent
    /// first.
    fn time_source(&self, q: &RecallQuery<'_>, as_of: u64, filtered: bool, s: &mut RecallScratch) {
        s.time_out.clear();
        let Some((from, to)) = q.range else { return };
        if filtered && !s.allow.is_empty() && s.allow.len() <= TEMPORAL_TAG_FIRST_MAX {
            self.time_source_from_tags(from, to, as_of, q.include_closed, s);
            return;
        }
        let RecallScratch {
            allow,
            allow_bits,
            time_out,
            ..
        } = s;
        let mut from_key = [0u8; 12];
        plugmem_arena::key::write_pair(&mut from_key, from, 0);
        let mut to_key = [0u8; 12];
        plugmem_arena::key::write_pair(&mut to_key, to, 0);
        for slot in self.temporal.range_rev(&from_key, &to_key) {
            if admit(
                &self.facts,
                allow,
                allow_bits,
                filtered,
                as_of,
                q.include_closed,
                slot.fact,
            )
            .is_some()
            {
                time_out.push((slot.fact, slot.recorded_at as f32));
                // The reverse range starts at the newest record, so once the
                // source cap is full the remaining entries cannot outrank
                // these candidates by recency.
                if time_out.len() == SOURCE_CAP {
                    break;
                }
            }
        }
    }

    /// Tag-first temporal source used when the tag allow-set is smaller than
    /// the recent temporal window. It preserves the temporal source's exact
    /// newest-first ordering while avoiding a broad temporal scan.
    fn time_source_from_tags(
        &self,
        from: u64,
        to: u64,
        as_of: u64,
        include_closed: bool,
        s: &mut RecallScratch,
    ) {
        let RecallScratch {
            allow,
            time_tag,
            time_out,
            ..
        } = s;
        time_tag.clear();
        for &fact in allow.iter() {
            let Some(record) = admit(
                &self.facts,
                &[],
                &AllowFilter::default(),
                false,
                as_of,
                include_closed,
                fact,
            ) else {
                continue;
            };
            if record.recorded_at >= from && record.recorded_at < to {
                time_tag.push((fact, record.recorded_at));
            }
        }
        time_tag.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        time_out.extend(
            time_tag
                .iter()
                .take(SOURCE_CAP)
                .map(|&(fact, recorded_at)| (fact, recorded_at as f32)),
        );
    }

    /// Visits the edges touching `entity` in both mirrored arenas as
    /// `(neighbor, rel, entity_is_src, provenance)` — outgoing in ascending
    /// key order, then incoming.
    ///
    /// The walk is **lazy and interruptible**: `visit` returning `false` stops
    /// it. A hub entity holds as many edges as the corpus has records, and
    /// expansion consumes at most a fixed cap of them, so materializing the
    /// whole list would make graph recall O(edges of the hub) for a bounded
    /// answer. Stopping is a pure prefix of the same deterministic order, so
    /// the caller's result is unchanged.
    fn neighbors(
        &self,
        entity: EntityId,
        as_of: u64,
        historical: bool,
        visit: &mut impl FnMut(EntityId, TermId, bool, FactId) -> bool,
    ) {
        if historical {
            // History is keyed `[a | valid_from | edge]`, so walking backwards
            // from `as_of` yields the entity's versions newest-first: those
            // that most recently became true, and so the ones most likely to
            // still be valid. The range already enforces `valid_from <= as_of`;
            // `as_of < valid_to` is the other half of the validity test.
            //
            // The walk stops as soon as the caller has enough valid edges, so
            // a deep history costs nothing extra for the usual question. The
            // exception is an instant at which the entity had *no* valid edge
            // at all: there is nothing to stop at, so proving the absence
            // reads every version that had already begun. Answering that in
            // sublinear time needs an interval index, not an ordering.
            let from = edge_history_floor(entity);
            let to = edge_history_ceiling(entity, as_of);
            for (arena, entity_is_src) in
                [(&self.edges_hist_out, true), (&self.edges_hist_in, false)]
            {
                for e in arena.range_rev(&from, &to) {
                    if as_of < e.valid_to && !visit(e.b, e.rel, entity_is_src, e.fact) {
                        return;
                    }
                }
            }
        } else {
            let from = edge_floor(entity);
            let to = edge_end(entity);
            for (arena, entity_is_src) in [(&self.edges_out, true), (&self.edges_in, false)] {
                for e in arena.range(&from, &to) {
                    if !visit(e.b, e.rel, entity_is_src, e.fact) {
                        return;
                    }
                }
            }
        }
    }

    /// Renders the compact prompt block (format fixed by golden tests).
    fn render(&self, out: &mut RecallResult, tags_tmp: &mut Vec<TermId>) {
        if out.facts.is_empty() && out.edges.is_empty() {
            return; // empty string: don't spend tokens saying "nothing"
        }
        out.rendered.push_str("## memory\n");
        for fact in &out.facts {
            let record = self
                .facts
                .get(&fact.id.0.to_be_bytes())
                .expect("selected ids exist");
            // Deferred validation: tolerate invalid text bytes —
            // an unreadable fact renders with an empty body, never a panic.
            let text = core::str::from_utf8(self.texts.get(record.text)).unwrap_or("");
            let _ = write!(out.rendered, "- [f{}] ", fact.id.0);
            // A corrupt subject name (deferred validation)
            // renders without the subject prefix rather than panicking.
            if let Some(entity) = fact.entity.some()
                && let Some(name) = self.entity_name(entity)
            {
                let _ = write!(out.rendered, "{name}: ");
            }
            out.rendered.push_str(text);
            out.rendered.push_str(" (");
            render_ym(&mut out.rendered, fact.valid_from);
            if fact.valid_to == VALID_TO_OPEN {
                out.rendered.push_str("; active)");
            } else {
                out.rendered.push_str(" → ");
                render_ym(&mut out.rendered, fact.valid_to);
                out.rendered.push_str("; closed)");
            }
            tags_tmp.clear();
            self.tags_of(fact.id, tags_tmp);
            for &tag in tags_tmp.iter() {
                let _ = write!(out.rendered, " #{}", self.terms.resolve(tag));
            }
            out.rendered.push('\n');
        }
        for edge in &out.edges {
            // Deferred validation, as for a fact's text and subject name: an
            // edge whose endpoints do not resolve is rendered as nothing
            // rather than as a panic. Only a corrupt image reaches this, and
            // `verify` reports it explicitly.
            let (Some(src), Some(dst)) = (self.entity_name(edge.src), self.entity_name(edge.dst))
            else {
                continue;
            };
            let _ = writeln!(
                out.rendered,
                "- links: {src} —{}→ {dst}",
                self.terms.resolve(edge.rel),
            );
        }
    }
}

/// The tag allow-set, as the sources actually use it.
///
/// The set is a sorted vector because [`Memory::time_source_from_tags`]
/// *enumerates* it. Every other source asks a different question — "is this
/// one fact in it?" — and asks it once per candidate, which for a graph
/// expansion under a tag filter is thousands of times. A binary search
/// answers that in a dozen unpredictable branches over a list sized like the
/// tag; the filter below answers "no" in one word read, and only "maybe"
/// costs the search.
///
/// The filter is a Bloom filter over the member ids: a member's bit is always
/// set, so a clear bit is proof of absence and the exact answer is unchanged.
#[derive(Debug, Default)]
struct AllowFilter {
    /// Bit per hashed id; length is a power of two so the hash needs no
    /// modulo. Empty when no tag filter is active.
    bits: Vec<u64>,
    /// `log2(bits.len() * 64)`, the shift that maps a hash onto a bit index.
    shift: u32,
}

/// Bits per member. Eight keeps the false-positive rate near 12% while the
/// whole filter stays small enough to sit in L1 for a realistic tag.
const ALLOW_BITS_PER_MEMBER: usize = 8;
/// Smallest and largest filter, in 64-bit words: 64 bytes to 128 KiB.
const ALLOW_MIN_WORDS: usize = 8;
const ALLOW_MAX_WORDS: usize = 1 << 14;

impl AllowFilter {
    /// Rebuilds the filter over `allow`. Reuses the buffer, so a warm scratch
    /// does not allocate.
    fn fill(&mut self, allow: &[FactId]) {
        // Clamped before rounding up: `next_power_of_two` overflows rather
        // than saturates, and on wasm32 `usize` is 32 bits, so a large
        // allow-set could reach that edge.
        let words = allow
            .len()
            .saturating_mul(ALLOW_BITS_PER_MEMBER)
            .div_ceil(64)
            .clamp(ALLOW_MIN_WORDS, ALLOW_MAX_WORDS)
            .next_power_of_two();
        self.bits.clear();
        self.bits.resize(words, 0);
        self.shift = 64 - (words * 64).trailing_zeros();
        for &id in allow {
            let at = self.index(id);
            self.bits[at / 64] |= 1u64 << (at % 64);
        }
    }

    /// Drops the filter (no tag filter on this query).
    fn clear(&mut self) {
        self.bits.clear();
    }

    /// Bit index of `id`. Fibonacci hashing, the same mixing the arena shards
    /// with, so ids that differ in their low bits do not collide in runs.
    fn index(&self, id: FactId) -> usize {
        (u64::from(id.0).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// `false` proves `id` is not in the allow-set; `true` means the caller
    /// must confirm against the set itself.
    fn maybe_contains(&self, id: FactId) -> bool {
        let at = self.index(id);
        self.bits[at / 64] & (1u64 << (at % 64)) != 0
    }
}

/// The shared admission rule of every source. Returns the record so
/// callers can reuse it.
///
/// The tag test comes first on purpose. Reading the fact record is an arena
/// lookup — the most expensive step here — and a candidate outside the
/// allow-set is rejected whatever the record says, so fetching it would be
/// work thrown away. The rule itself is unchanged: a candidate is admitted
/// exactly when it was before.
fn admit(
    facts: &plugmem_arena::Arena<'_, FactRecord>,
    allow: &[FactId],
    filter: &AllowFilter,
    filtered: bool,
    as_of: u64,
    include_closed: bool,
    id: FactId,
) -> Option<FactRecord> {
    if filtered && (!filter.maybe_contains(id) || allow.binary_search(&id).is_err()) {
        return None;
    }
    let record = facts.get(&id.0.to_be_bytes())?;
    if record.is_tombstone() || record.recorded_at > as_of || record.valid_from > as_of {
        return None;
    }
    if !include_closed && as_of >= record.valid_to {
        return None;
    }
    Some(record)
}

/// Writes `year-month` (`2025-11`) of a unix-millisecond timestamp,
/// proleptic Gregorian (civil-from-days, Hinnant's algorithm).
fn render_ym(out: &mut String, ms: u64) {
    let days = (ms / 86_400_000) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let _ = write!(out, "{year:04}-{month:02}");
}
