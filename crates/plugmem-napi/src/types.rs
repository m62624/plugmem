//! Typed result surface: `#[napi(object)]` mirrors of the host result types, so
//! napi emits precise TypeScript interfaces (a TS host like Pi gets autocomplete
//! and checking on what a verb returns, not `any`).
//!
//! Each struct also derives `Deserialize` with **host field names**, so a result
//! is converted by a serde round-trip ([`to_typed`]) — no hand-written per-field
//! mapping, no second source of truth to drift. napi renders the Rust snake_case
//! fields as camelCase on the TS side (`recorded_at` → `recordedAt`).
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
    /// Quantized vector slots.
    pub vectors: f64,
    /// The next fact id to be assigned.
    pub next_fact: f64,
    /// The next entity id to be assigned.
    pub next_entity: f64,
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
}

/// Convert a host result into its typed JS mirror via a serde round-trip: the
/// mirror's fields match the host type's serialized names, so serde maps them
/// with no hand-written per-field code. Unmapped host fields (e.g. `Stats`'s
/// `db_uuid`) are dropped.
pub(crate) fn to_typed<T: DeserializeOwned>(v: &impl Serialize) -> napi::Result<T> {
    let value = serde_json::to_value(v)
        .map_err(|e| napi::Error::from_reason(format!("serialization error: {e}")))?;
    serde_json::from_value(value)
        .map_err(|e| napi::Error::from_reason(format!("result shape error: {e}")))
}
