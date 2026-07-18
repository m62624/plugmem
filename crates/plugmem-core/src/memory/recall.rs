//! Hybrid recall: source ranking, RRF fusion, budgeted selection and the
//! rendered prompt block (specs/04 §6–7, specs/05).
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
use crate::index::{IntersectScratch, intersect};
use crate::model::{FactRecord, VALID_TO_OPEN};

use super::Memory;

/// Source bits of [`RecalledFact::sources`].
pub mod source {
    /// The lexical (BM25) source.
    pub const BM25: u8 = 1;
    /// The graph-expansion source.
    pub const GRAPH: u8 = 1 << 1;
    /// The temporal-range source.
    pub const TIME: u8 = 1 << 2;
}

/// Per-source candidate cap (specs/04 §7).
const SOURCE_CAP: usize = 128;

/// Graph expansion caps (specs/04 §6).
const GRAPH_ENTITY_CAP: usize = 64;
const GRAPH_FACT_CAP: usize = 256;
const GRAPH_EDGE_CAP: usize = 128;
/// Hard budget on posting entries the graph source may *examine* — a hub
/// entity with tens of thousands of facts must not turn expansion into a
/// full decode of its list (the specs/04 "hub super-node" guard applies
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

/// A recall request (specs/05). `Default`-like construction via
/// [`RecallQuery::text`] plus field overrides.
#[derive(Clone, Copy, Debug)]
pub struct RecallQuery<'a> {
    /// Host timestamp, unix milliseconds.
    pub now: u64,
    /// Free-text query for the lexical source.
    pub text: Option<&'a str>,
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
}

impl<'a> RecallQuery<'a> {
    /// A plain text query with every other knob at its default.
    pub fn text(now: u64, text: &'a str) -> Self {
        Self {
            now,
            text: Some(text),
            tags: &[],
            entities: &[],
            as_of: None,
            range: None,
            k: 0,
            token_budget: None,
            include_closed: false,
        }
    }
}

/// One recalled fact (specs/05).
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// One edge the graph source walked (specs/05: agents want the relations,
/// not only the facts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Reusable recall scratch, owned by the engine.
#[derive(Debug, Default)]
pub(super) struct RecallScratch {
    bm25: Bm25Scratch,
    intersect: IntersectScratch,
    allow: Vec<FactId>,
    tag_terms: Vec<u32>,
    query_terms: Vec<u32>,
    bm25_out: Vec<(FactId, f32)>,
    graph_out: Vec<(FactId, f32)>,
    time_out: Vec<(FactId, f32)>,
    visited: Vec<(EntityId, f32)>,
    edges_tmp: Vec<(EntityId, TermId, bool, FactId)>,
    fused: hashbrown::HashMap<u32, (f32, u8), xxhash_rust::xxh3::Xxh3Builder>,
    ranked: Vec<(FactId, f32, u8)>,
    tags_tmp: Vec<TermId>,
}

impl Memory {
    /// Runs a recall, allocating a fresh [`RecallResult`]. Convenience
    /// wrapper over [`Memory::recall_into`].
    pub fn recall(&mut self, q: RecallQuery<'_>) -> Result<RecallResult, Error> {
        let mut out = RecallResult::default();
        self.recall_into(q, &mut out)?;
        Ok(out)
    }

    /// Runs a recall into a reused result (the zero-alloc path: after
    /// warm-up neither the engine scratches nor `out` allocate).
    ///
    /// `&mut self` is for the scratch buffers only — recall never mutates
    /// data (specs/05).
    pub fn recall_into(&mut self, q: RecallQuery<'_>, out: &mut RecallResult) -> Result<(), Error> {
        out.facts.clear();
        out.edges.clear();
        out.rendered.clear();
        out.truncated = false;

        let k = if q.k == 0 { 8 } else { q.k.min(64) };
        let budget = q.token_budget.unwrap_or(512);
        let as_of = q.as_of.unwrap_or(q.now);
        let mut s = core::mem::take(&mut self.recall_scratch);

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
            self.recall_scratch = s;
            return Ok(());
        }

        // 2–3. Sources (each admits through the shared rule).
        s.bm25_out.clear();
        if let Some(text) = q.text {
            s.query_terms.clear();
            let terms = &self.terms;
            let query_terms = &mut s.query_terms;
            self.tokenizer.tokenize(text, &mut |token| {
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
            self.bm25.search(
                (self.cfg.bm25_k1, self.cfg.bm25_b),
                &s.query_terms,
                SOURCE_CAP,
                &mut |id| admit(facts, allow, filtered, as_of, q.include_closed, id).is_some(),
                &mut s.bm25,
                &mut s.bm25_out,
            );
        }

        // Graph anchors resolve here (name normalization needs the
        // mutable tokenizer scratch); expansion itself is read-only.
        s.visited.clear();
        for name in q.entities {
            let mut norm = core::mem::take(&mut self.name_scratch);
            super::normalize_name(&mut self.tokenizer, name, &mut norm);
            let found = self.lookup_entity_by_norm(&norm);
            self.name_scratch = norm;
            if let Some(id) = found
                && !s.visited.iter().any(|&(e, _)| e == id)
            {
                s.visited.push((id, 1.0));
            }
        }
        self.graph_source(&q, as_of, filtered, &mut s, out);
        self.time_source(&q, as_of, filtered, &mut s);

        // 4. RRF fusion.
        s.fused.clear();
        for (list, weight, bit) in [
            (&s.bm25_out, self.cfg.w_bm25, source::BM25),
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
        self.recall_scratch = s;
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
            graph_out,
            visited,
            edges_tmp,
            ..
        } = s;
        graph_out.clear();
        if visited.is_empty() {
            return;
        }

        // Breadth-first: `frontier` marks where the current depth starts.
        let mut frontier = 0usize;
        let mut weight = 1.0f32;
        for _ in 0..self.cfg.graph_depth {
            let depth_end = visited.len();
            weight *= self.cfg.graph_decay;
            for at in frontier..depth_end {
                let (entity, _) = visited[at];
                self.neighbors(entity, edges_tmp);
                let batch = core::mem::take(edges_tmp);
                for &(neighbor, rel, this_side_src, provenance) in &batch {
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
                }
                *edges_tmp = batch;
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
                if admit(&self.facts, allow, filtered, as_of, q.include_closed, fact).is_some() {
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
                && admit(&self.facts, allow, filtered, as_of, q.include_closed, fact).is_some()
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
        let RecallScratch {
            allow, time_out, ..
        } = s;
        time_out.clear();
        let Some((from, to)) = q.range else { return };
        let mut from_key = [0u8; 12];
        plugmem_arena::key::write_pair(&mut from_key, from, 0);
        let mut to_key = [0u8; 12];
        plugmem_arena::key::write_pair(&mut to_key, to, 0);
        for slot in self.temporal.range(&from_key, &to_key) {
            if admit(
                &self.facts,
                allow,
                filtered,
                as_of,
                q.include_closed,
                slot.fact,
            )
            .is_some()
            {
                time_out.push((slot.fact, slot.recorded_at as f32));
                // Keep only the most recent SOURCE_CAP without unbounded
                // growth on huge windows.
                if time_out.len() > SOURCE_CAP * 2 {
                    time_out.drain(..SOURCE_CAP);
                }
            }
        }
        time_out.reverse();
        time_out.truncate(SOURCE_CAP);
    }

    /// Collects the edges touching `entity` from both mirrored arenas
    /// into `out` as `(neighbor, rel, entity_is_src, provenance)`.
    fn neighbors(&self, entity: EntityId, out: &mut Vec<(EntityId, TermId, bool, FactId)>) {
        out.clear();
        let mut from = [0u8; 12];
        plugmem_arena::key::write_u32(&mut from, entity.0);
        let mut to = [0u8; 12];
        plugmem_arena::key::write_u32(&mut to, entity.0 + 1);
        out.extend(
            self.edges_out
                .range(&from, &to)
                .map(|e| (e.b, e.rel, true, e.fact)),
        );
        out.extend(
            self.edges_in
                .range(&from, &to)
                .map(|e| (e.b, e.rel, false, e.fact)),
        );
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
            let text = core::str::from_utf8(self.texts.get(record.text))
                .expect("fact texts are written from &str");
            let _ = write!(out.rendered, "- [f{}] ", fact.id.0);
            if let Some(entity) = fact.entity.some() {
                let _ = write!(out.rendered, "{}: ", self.entity_name(entity));
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
            let _ = writeln!(
                out.rendered,
                "- links: {} —{}→ {}",
                self.entity_name(edge.src),
                self.terms.resolve(edge.rel),
                self.entity_name(edge.dst),
            );
        }
    }

    /// Canonical display name of an entity.
    fn entity_name(&self, id: EntityId) -> &str {
        let record = self
            .entities
            .get(&id.0.to_be_bytes())
            .expect("edges reference existing entities");
        core::str::from_utf8(self.texts.get(record.name)).expect("names are written from &str")
    }
}

/// The shared admission rule of every source. Returns the record so
/// callers can reuse it.
fn admit(
    facts: &plugmem_arena::Arena<FactRecord>,
    allow: &[FactId],
    filtered: bool,
    as_of: u64,
    include_closed: bool,
    id: FactId,
) -> Option<FactRecord> {
    let record = facts.get(&id.0.to_be_bytes())?;
    if record.is_tombstone() || record.recorded_at > as_of || record.valid_from > as_of {
        return None;
    }
    if !include_closed && as_of >= record.valid_to {
        return None;
    }
    if filtered && allow.binary_search(&id).is_err() {
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
