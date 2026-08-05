//! Typed result surface: `#[napi(object)]` mirrors of the host result types, so
//! napi emits precise TypeScript interfaces (a TS host like Pi gets autocomplete
//! and checking on what a verb returns, not `any`).
//!
//! Most structs also derive `Deserialize` with **host field names**, so a result
//! is converted by a serde round-trip ([`to_typed`]) — no hand-written per-field
//! mapping, no second source of truth to drift. The hot export path maps its
//! owned strings directly instead. napi renders Rust snake_case fields as
//! camelCase on the TS side (`recorded_at` → `recordedAt`).
//!
//! Numeric policy: every count/id/timestamp is a JS `number` (`f64`). Ids and
//! counts are exact; unix-millisecond timestamps are exact well past any real
//! date. The one caveat is `validTo`: an **open** fact carries the sentinel
//! `VALID_TO_OPEN` (`u64::MAX`), which as a JS number is a very large value, not
//! an exact instant — treat "validTo is astronomically large" as "still open".
//! `Stats::db_uuid` (a 128-bit lineage id) is intentionally omitted: it exceeds
//! JS number precision and is engine bookkeeping, not agent-facing.

use std::collections::BTreeMap;

use napi_derive::napi;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// One similar / potentially-conflicting live fact surfaced by `remember`.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct Similar {
    /// The existing fact's id.
    pub id: f64,
    /// Match strength (higher = closer).
    pub score: f64,
    /// What triggered the hint: `"LexicalOverlap"` or `"VectorCosine"`.
    pub reason: String,
}

/// The result of `remember` / `revise`.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct RememberOutcome {
    /// The new fact's id.
    pub id: f64,
    /// The subject entity id, if one was named.
    pub entity: Option<f64>,
    /// Similar / potentially-conflicting live facts (best first; the engine
    /// never merges on its own — the caller decides).
    pub similar: Vec<Similar>,
}

impl From<plugmem_host::Similar> for Similar {
    fn from(similar: plugmem_host::Similar) -> Self {
        let reason = match similar.reason {
            plugmem_host::SimilarReason::LexicalOverlap => "LexicalOverlap",
            plugmem_host::SimilarReason::VectorCosine => "VectorCosine",
        };
        Self {
            id: f64::from(similar.id.0),
            score: f64::from(similar.score),
            reason: reason.to_string(),
        }
    }
}

impl From<plugmem_host::RememberOutcome> for RememberOutcome {
    fn from(outcome: plugmem_host::RememberOutcome) -> Self {
        Self {
            id: f64::from(outcome.id.0),
            entity: outcome.entity.map(|entity| f64::from(entity.0)),
            similar: outcome.similar.into_iter().map(Similar::from).collect(),
        }
    }
}

/// One recalled fact.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct RecalledFact {
    /// The fact id.
    pub id: f64,
    /// Fused score (reciprocal-rank fusion + recency).
    pub score: f64,
    /// Bit set of the sources that surfaced it.
    pub sources: f64,
    /// Subject entity id (a sentinel when none).
    pub entity: f64,
    /// Knowledge axis: when the memory learned it (unix ms).
    pub recorded_at: f64,
    /// Truth axis start (unix ms).
    pub valid_from: f64,
    /// Truth axis end (unix ms), or the open sentinel — see the module note.
    pub valid_to: f64,
}

/// One edge walked by the graph source.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct RecalledEdge {
    /// Source entity id.
    pub src: f64,
    /// Relation term id.
    pub rel: f64,
    /// Destination entity id.
    pub dst: f64,
    /// Provenance fact id (a sentinel when none).
    pub provenance: f64,
}

/// A recall response: the structured hits plus the prompt-ready block.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct RecallResult {
    /// Selected facts, descending fused score.
    pub facts: Vec<RecalledFact>,
    /// Edges the graph source walked (deduplicated).
    pub edges: Vec<RecalledEdge>,
    /// The compact prompt block (empty when nothing was found).
    pub rendered: String,
    /// `true` when selection stopped at `k`/the token budget with more left.
    pub truncated: bool,
}

/// Engine size counters.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct Stats {
    /// Fact records stored (live, closed and tombstoned-awaiting-maintain).
    pub facts: f64,
    /// Entities.
    pub entities: f64,
    /// Interned terms (tokens, tags, relations, names).
    pub terms: f64,
    /// Directed edges.
    pub edges: f64,
    /// Historical edge versions, including closed versions.
    pub edge_versions: f64,
    /// Quantized vector slots.
    pub vectors: f64,
    /// Tombstoned fact records awaiting physical purge.
    pub tombstones: f64,
    /// Vector slots covered by HNSW.
    pub hnsw_indexed: f64,
    /// The next fact id to be assigned.
    pub next_fact: f64,
    /// The next entity id to be assigned.
    pub next_entity: f64,
    /// The next edge-version id to be assigned.
    pub next_edge: f64,
    /// Total bytes held by the engine's pools.
    pub pool_bytes: f64,
}

/// The raw record behind a [`FactSnapshot`] — temporality and flags. (Internal
/// pointers — the blob/vector slots and the reserved `kind` — are omitted.)
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct FactRecord {
    /// The fact id.
    pub id: f64,
    /// Subject entity id (a sentinel when none).
    pub entity: f64,
    /// Bit set of fact flags (tombstone / closed / has-vector).
    pub flags: f64,
    /// Predecessor in the revision chain (a sentinel when none).
    pub revises: f64,
    /// Knowledge axis: when the memory learned it (unix ms).
    pub recorded_at: f64,
    /// Truth axis start (unix ms).
    pub valid_from: f64,
    /// Truth axis end (unix ms), or the open sentinel — see the module note.
    pub valid_to: f64,
}

/// One fact's full card (from `get`).
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct FactSnapshot {
    /// The raw record (temporality, flags, references).
    pub record: FactRecord,
    /// The fact text.
    pub text: String,
    /// The fact's metadata as a key→value map (empty when it has none). Opaque
    /// to the engine — a URI to the real payload, a mime type, an external key.
    pub metadata: BTreeMap<String, String>,
}

/// One exported fact — the id-free, import-ready shape.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct ExportedFact {
    /// The fact's id in the database it came from. Informational: an import
    /// assigns fresh ids. Present because edges name their provenance fact by
    /// id, so a dump carrying edges needs something for them to point at.
    pub id: f64,
    /// The fact text.
    pub text: String,
    /// Subject entity name, if any.
    pub entity: Option<String>,
    /// Tag strings.
    pub tags: Vec<String>,
    /// Metadata as a key→value map (empty when none); preserved on import.
    pub metadata: BTreeMap<String, String>,
    /// When the memory learned it (unix ms; informational).
    pub recorded_at: f64,
    /// Validity start (unix ms; preserved on import).
    pub valid_from: f64,
}

impl From<plugmem_host::ExportedFact> for ExportedFact {
    fn from(fact: plugmem_host::ExportedFact) -> Self {
        Self {
            id: f64::from(fact.id),
            text: fact.text,
            entity: fact.entity,
            tags: fact.tags,
            metadata: fact.metadata,
            recorded_at: fact.recorded_at as f64,
            valid_from: fact.valid_from as f64,
        }
    }
}

/// One bounded page returned by `exportPage`.
#[napi(object)]
pub struct ExportPage {
    /// Open facts in fact-id order; never longer than the native page bound.
    pub facts: Vec<ExportedFact>,
    /// Pass this opaque cursor to the next call; absent when the scan is done.
    pub next_cursor: Option<f64>,
}

impl From<plugmem_host::ExportPage> for ExportPage {
    fn from(page: plugmem_host::ExportPage) -> Self {
        Self {
            facts: page.facts.into_iter().map(ExportedFact::from).collect(),
            next_cursor: page.next_cursor.map(f64::from),
        }
    }
}

/// The report of a `maintain` pass.
#[napi(object)]
#[derive(serde::Deserialize)]
pub struct MaintainReport {
    /// Tombstoned facts physically removed by this pass.
    pub purged: f64,
    /// On-disk image bytes before the pass.
    pub bytes_before: f64,
    /// On-disk image bytes after the pass.
    pub bytes_after: f64,
    /// No storage/index rewrite was needed.
    pub no_op: bool,
    /// Tombstones present before the pass.
    pub tombstones_before: f64,
    /// Fact records before the pass.
    pub facts_before: f64,
    /// Fact records after the pass.
    pub facts_after: f64,
    /// Vector slots before the pass.
    pub vectors_before: f64,
    /// Vector slots after the pass.
    pub vectors_after: f64,
    /// HNSW coverage before the pass.
    pub hnsw_indexed_before: f64,
    /// HNSW coverage after the pass.
    pub hnsw_indexed_after: f64,
    /// Physical compaction ran.
    pub structural_compacted: bool,
    /// BM25 was compacted from existing postings.
    pub bm25_compacted: bool,
    /// BM25 was rebuilt from text.
    pub bm25_reindexed: bool,
    /// HNSW was rebuilt from empty.
    pub hnsw_rebuilt: bool,
    /// HNSW was carried/remapped.
    pub hnsw_remapped: bool,
    /// Vector slots inserted into HNSW.
    pub hnsw_inserted: f64,
    /// The edge arenas were rewritten page-dense (`full` only). No edge
    /// version is ever dropped, so the version count is unchanged — only the
    /// bytes shrink.
    pub edges_compacted: bool,
    /// Current edges before the pass.
    pub edges_before: f64,
    /// Historical edge versions before the pass.
    pub edge_versions_before: f64,
}

/// How much work a `maintain` pass should do.
#[napi(string_enum = "kebab-case")]
pub enum MaintainMode {
    /// Only pending work: purge tombstones, refresh a stale text index, and
    /// advance the vector graph within a bounded budget. Cheap to run often.
    Auto,
    /// Physically purge tombstoned facts and compact storage and indexes.
    Compact,
    /// Rebuild the text index by re-reading and re-tokenizing every fact.
    ReindexText,
    /// Build or advance the vector graph without compacting anything else.
    OptimizeVectors,
    /// Rebuild every rebuildable structure, fully optimize vectors and repack
    /// the edge arenas. O(database) work; no history is ever dropped.
    Full,
}

impl From<MaintainMode> for plugmem_host::MaintenanceOptions {
    fn from(mode: MaintainMode) -> Self {
        use plugmem_host::MaintenanceMode as Engine;
        let mode = match mode {
            MaintainMode::Auto => return Self::auto(),
            MaintainMode::Full => return Self::full(),
            MaintainMode::Compact => Engine::Compact,
            MaintainMode::ReindexText => Engine::ReindexText,
            MaintainMode::OptimizeVectors => Engine::OptimizeVectors,
        };
        // An explicitly named mode keeps `auto`'s bounded vector budget.
        Self {
            mode,
            ..Self::auto()
        }
    }
}

/// One memory as the workspace registry knows it.
///
/// Hand-mapped rather than serde-round-tripped like the rest of this file:
/// `DbEntry` carries a `DbName`, which is a validated newtype with no serde
/// derive, and giving one to a public host type only so a wrapper could
/// round-trip it would be the wrong direction of dependency.
#[napi(object)]
pub struct DbEntry {
    /// The memory's name — its identity, and what `Workspace.open` takes.
    pub db: String,
    /// What it is for.
    pub description: String,
    /// Its tags.
    pub tags: Vec<String>,
    /// Its owner, if recorded.
    pub owner: Option<String>,
    /// Whether it is labelled archived.
    pub archived: bool,
}

/// What a `Workspace.reindex()` pass did.
#[napi(object)]
pub struct ReindexReport {
    /// Memories whose own description was copied into the registry.
    pub indexed: Vec<String>,
    /// Memories nobody has described. Not a fault — a memory works without a
    /// description; it just cannot be found by one.
    pub undescribed: Vec<String>,
    /// Memories another process holds open, so this pass could not read them.
    /// Named rather than skipped silently: the registry is knowingly incomplete.
    pub busy: Vec<String>,
}

/// Something `Workspace.verify()` found. Reported, never repaired.
#[napi(object)]
pub struct WorkspaceProblem {
    /// The memory it concerns (empty for a kind this binding does not know).
    pub db: String,
    /// What kind: `"missing"`, `"undescribed"`, `"stale"`, `"unreadable"`,
    /// `"ambiguous-self"`. The same vocabulary the CLI prints in `--json`.
    pub issue: String,
    /// More detail, where the kind carries any.
    pub detail: Option<String>,
}

/// Convert a host result into its typed JS mirror via a serde round-trip: the
/// mirror's fields match the host type's serialized names, so serde maps them
/// with no hand-written per-field code. Unmapped host fields (e.g. `Stats`'s
/// `db_uuid`) are dropped.
pub(crate) fn to_typed<T: DeserializeOwned>(v: &impl Serialize) -> crate::error::Result<T> {
    let value = serde_json::to_value(v)
        .map_err(|e| crate::error::internal(format!("serialization error: {e}")))?;
    serde_json::from_value(value)
        .map_err(|e| crate::error::internal(format!("result shape error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use plugmem_host::{EntityId, FactId, SimilarReason};

    #[test]
    fn direct_remember_mapping_preserves_ids_scores_and_reason_names() {
        let mapped = RememberOutcome::from(plugmem_host::RememberOutcome {
            id: FactId(7),
            entity: Some(EntityId(3)),
            similar: vec![
                plugmem_host::Similar {
                    id: FactId(1),
                    score: 0.75,
                    reason: SimilarReason::LexicalOverlap,
                },
                plugmem_host::Similar {
                    id: FactId(2),
                    score: 0.875,
                    reason: SimilarReason::VectorCosine,
                },
            ],
        });

        assert_eq!(mapped.id, 7.0);
        assert_eq!(mapped.entity, Some(3.0));
        assert_eq!(mapped.similar[0].reason, "LexicalOverlap");
        assert_eq!(mapped.similar[1].reason, "VectorCosine");
        assert_eq!(mapped.similar[1].score, 0.875);
    }
}
