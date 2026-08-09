//! Typed result surface: `#[pyclass]` mirrors of the host result types, so a
//! verb returns an attribute-addressable object rather than a dict a caller has
//! to know the keys of.
//!
//! Every mirror is `frozen` and `get_all`: results are values, and a caller who
//! mutates one is mutating a copy of what the engine said, which is never what
//! they meant. Mapping is written out by hand rather than round-tripped through
//! serde, which keeps `serde`/`serde_json` out of the wheel and keeps every
//! integer an integer.
//!
//! **Every conversion destructures the host type exhaustively.** Written as
//! `fact.text`, a mapping keeps compiling when the engine grows a field and
//! simply fails to carry it — drift with no symptom until someone notices the
//! value is missing. Written as `let Host { text, .. }` with no `..`, a new
//! field is a compile error here, and a field we deliberately do not carry is
//! named with `_` and is therefore a decision on the page rather than an
//! omission nobody recorded.
//!
//! **Numbers are exact here, unlike in the Node binding.** napi renders every
//! count, id and timestamp as a JavaScript `number` (an `f64`) because JS has
//! nothing else; Python has real integers, so `valid_to`'s open sentinel
//! (`u64::MAX`) arrives as that exact value rather than as "astronomically
//! large". The field *names* are napi's, one for one.
//!
//! Two `Stats` fields the host has are omitted here, and only because napi
//! omits them: `db_uuid` (a 128-bit lineage id, engine bookkeeping) and
//! `shards`. Python could carry both. Carrying them would make this the third
//! surface with its own idea of what `stats()` returns, which is the thing the
//! parity rule exists to prevent.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Generate a `__repr__` that prints the class name and the listed fields.
///
/// Every mirror gets one: a result object whose `repr` is `<builtins.Stats
/// object at 0x…>` is useless in a REPL, and a REPL is where most of these are
/// first met.
///
/// Each value is converted to a Python object and printed by Python's own
/// `repr`, rather than by Rust's `{:?}`. That is not fussiness: `{:?}` renders
/// an `Option<u32>` as `Some(0)` and a `bool` as `false`, so a Python user
/// would be reading Rust in their REPL and could not paste the output back.
macro_rules! repr {
    ($ty:ty, $name:literal, $($field:ident),+ $(,)?) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $ty {
            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let mut out = String::from(concat!($name, "("));
                let mut first = true;
                $(
                    if !first {
                        out.push_str(", ");
                    }
                    first = false;
                    out.push_str(concat!(stringify!($field), "="));
                    let value = self.$field.clone().into_pyobject(py)?;
                    out.push_str(&value.repr()?.to_string());
                )+
                let _ = first;
                out.push(')');
                Ok(out)
            }
        }
    };
}

/// One similar / potentially-conflicting live fact surfaced by `remember`.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct Similar {
    /// The existing fact's id.
    pub id: u32,
    /// Match strength (higher = closer).
    pub score: f32,
    /// What triggered the hint: `"LexicalOverlap"` or `"VectorCosine"`.
    pub reason: String,
}
repr!(Similar, "Similar", id, score, reason);

impl From<plugmem_host::Similar> for Similar {
    fn from(similar: plugmem_host::Similar) -> Self {
        let plugmem_host::Similar { id, score, reason } = similar;
        let reason = match reason {
            plugmem_host::SimilarReason::LexicalOverlap => "LexicalOverlap",
            plugmem_host::SimilarReason::VectorCosine => "VectorCosine",
        };
        Self {
            id: id.0,
            score,
            reason: reason.to_string(),
        }
    }
}

/// The result of `remember` / `revise`.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct RememberOutcome {
    /// The new fact's id.
    pub id: u32,
    /// The subject entity id, if one was named.
    pub entity: Option<u32>,
    /// Similar / potentially-conflicting live facts (best first; the engine
    /// never merges on its own — the caller decides).
    pub similar: Vec<Py<Similar>>,
}
repr!(RememberOutcome, "RememberOutcome", id, entity);

impl RememberOutcome {
    /// Build the mirror, allocating each `Similar` on the Python heap.
    pub fn build(py: Python<'_>, outcome: plugmem_host::RememberOutcome) -> PyResult<Self> {
        let plugmem_host::RememberOutcome {
            id,
            entity,
            similar,
        } = outcome;
        let similar = similar
            .into_iter()
            .map(|s| Py::new(py, Similar::from(s)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            id: id.0,
            entity: entity.map(|entity| entity.0),
            similar,
        })
    }
}

/// One recalled fact.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct RecalledFact {
    /// The fact id.
    pub id: u32,
    /// Fused score (reciprocal-rank fusion + recency).
    pub score: f32,
    /// Bit set of the sources that surfaced it.
    pub sources: u8,
    /// Subject entity id (a sentinel when none).
    pub entity: u32,
    /// Knowledge axis: when the memory learned it (unix ms).
    pub recorded_at: u64,
    /// Truth axis start (unix ms).
    pub valid_from: u64,
    /// Truth axis end (unix ms), or `VALID_TO_OPEN` while the fact is open.
    pub valid_to: u64,
}
repr!(RecalledFact, "RecalledFact", id, score, sources, entity);

impl From<plugmem_host::RecalledFact> for RecalledFact {
    fn from(fact: plugmem_host::RecalledFact) -> Self {
        let plugmem_host::RecalledFact {
            id,
            score,
            sources,
            entity,
            recorded_at,
            valid_from,
            valid_to,
        } = fact;
        Self {
            id: id.0,
            score,
            sources,
            entity: entity.0,
            recorded_at,
            valid_from,
            valid_to,
        }
    }
}

/// One edge walked by the graph source.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct RecalledEdge {
    /// Source entity id.
    pub src: u32,
    /// Relation term id.
    pub rel: u32,
    /// Destination entity id.
    pub dst: u32,
    /// Provenance fact id (a sentinel when none).
    pub provenance: u32,
}
repr!(RecalledEdge, "RecalledEdge", src, rel, dst, provenance);

impl From<plugmem_host::RecalledEdge> for RecalledEdge {
    fn from(edge: plugmem_host::RecalledEdge) -> Self {
        let plugmem_host::RecalledEdge {
            src,
            rel,
            dst,
            provenance,
        } = edge;
        Self {
            src: src.0,
            rel: rel.0,
            dst: dst.0,
            provenance: provenance.0,
        }
    }
}

/// A recall response: the structured hits plus the bounded rendered block.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct RecallResult {
    /// Selected facts, descending fused score.
    pub facts: Vec<Py<RecalledFact>>,
    /// Edges the graph source walked (deduplicated).
    pub edges: Vec<Py<RecalledEdge>>,
    /// The compact rendered block (empty when nothing was found).
    pub rendered: String,
    /// `true` when selection stopped at `k`/the token budget with more left.
    pub truncated: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl RecallResult {
    fn __repr__(&self) -> String {
        format!(
            "RecallResult(facts={}, edges={}, truncated={}, rendered={} chars)",
            self.facts.len(),
            self.edges.len(),
            // Python spells its booleans with a capital, and this repr is read
            // in a Python REPL.
            if self.truncated { "True" } else { "False" },
            self.rendered.len()
        )
    }
}

impl RecallResult {
    /// Build the mirror, allocating the hits on the Python heap.
    pub fn build(py: Python<'_>, result: plugmem_host::RecallResult) -> PyResult<Self> {
        let plugmem_host::RecallResult {
            facts,
            edges,
            rendered,
            truncated,
        } = result;
        let facts = facts
            .into_iter()
            .map(|f| Py::new(py, RecalledFact::from(f)))
            .collect::<PyResult<Vec<_>>>()?;
        let edges = edges
            .into_iter()
            .map(|e| Py::new(py, RecalledEdge::from(e)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            facts,
            edges,
            rendered,
            truncated,
        })
    }
}

/// Engine size counters.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct Stats {
    /// Fact records stored (live, closed and tombstoned-awaiting-maintain).
    pub facts: usize,
    /// Entities.
    pub entities: usize,
    /// Interned terms (tokens, tags, relations, names).
    pub terms: usize,
    /// Directed edges.
    pub edges: usize,
    /// Historical edge versions, including closed versions.
    pub edge_versions: usize,
    /// Quantized vector slots.
    pub vectors: usize,
    /// Tombstoned fact records awaiting physical purge.
    pub tombstones: usize,
    /// Vector slots covered by HNSW.
    pub hnsw_indexed: u32,
    /// The next fact id to be assigned.
    pub next_fact: u32,
    /// The next entity id to be assigned.
    pub next_entity: u32,
    /// The next edge-version id to be assigned.
    pub next_edge: u32,
    /// Total bytes held by the engine's pools.
    pub pool_bytes: usize,
}
repr!(Stats, "Stats", facts, entities, edges, vectors, pool_bytes);

impl From<plugmem_host::Stats> for Stats {
    fn from(stats: plugmem_host::Stats) -> Self {
        let plugmem_host::Stats {
            facts,
            entities,
            terms,
            edges,
            edge_versions,
            vectors,
            tombstones,
            hnsw_indexed,
            next_fact,
            next_entity,
            next_edge,
            pool_bytes,
            // `..` covers `db_uuid` and `shards`, both dropped on purpose and
            // only because napi drops them: a 128-bit lineage id exceeds a
            // JavaScript number, and the shard layout is engine bookkeeping.
            // Python could carry either — carrying them here alone would make
            // `stats()` mean something different per language.
            //
            // Naming them as `_` alongside the `..` would read as documentation
            // but buy nothing, which is what `clippy::unneeded_wildcard_pattern`
            // says. The rest pattern is not optional here: `Stats` is the one
            // host result marked `#[non_exhaustive]`, so this is also the one
            // conversion in this file without the compile-error-on-a-new-field
            // guarantee. That is the host reserving the right to add counters,
            // and the cost of it is that this list has to be reviewed by hand.
            ..
        } = stats;
        Self {
            facts,
            entities,
            terms,
            edges,
            edge_versions,
            vectors,
            tombstones,
            hnsw_indexed,
            next_fact,
            next_entity,
            next_edge,
            pool_bytes,
        }
    }
}

/// The raw record behind a [`FactSnapshot`] — temporality and flags. (Internal
/// pointers — the blob/vector slots and the reserved `kind` — are omitted.)
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct FactRecord {
    /// The fact id.
    pub id: u32,
    /// Subject entity id (a sentinel when none).
    pub entity: u32,
    /// Bit set of fact flags (tombstone / closed / has-vector).
    pub flags: u16,
    /// Predecessor in the revision chain (a sentinel when none).
    pub revises: u32,
    /// Knowledge axis: when the memory learned it (unix ms).
    pub recorded_at: u64,
    /// Truth axis start (unix ms).
    pub valid_from: u64,
    /// Truth axis end (unix ms), or `VALID_TO_OPEN` while the fact is open.
    pub valid_to: u64,
}
repr!(FactRecord, "FactRecord", id, entity, flags, valid_to);

impl From<plugmem_host::FactRecord> for FactRecord {
    fn from(record: plugmem_host::FactRecord) -> Self {
        let plugmem_host::FactRecord {
            id,
            entity,
            flags,
            revises,
            recorded_at,
            valid_from,
            valid_to,
            // Internal pointers into the pools and a field reserved in v1.
            // Meaningless outside the engine, and napi omits them too.
            kind: _,
            text: _,
            vector: _,
        } = record;
        Self {
            id: id.0,
            entity: entity.0,
            flags,
            revises: revises.0,
            recorded_at,
            valid_from,
            valid_to,
        }
    }
}

/// One fact's full card (from `get`).
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct FactSnapshot {
    /// The raw record (temporality, flags, references).
    pub record: Py<FactRecord>,
    /// The fact text.
    pub text: String,
    /// The fact's metadata as a key→value map (empty when it has none). Opaque
    /// to the engine — a URI to the real payload, a mime type, an external key.
    pub metadata: BTreeMap<String, String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl FactSnapshot {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "FactSnapshot(id={}, text={:?})",
            self.record.borrow(py).id,
            self.text
        )
    }
}

impl FactSnapshot {
    /// Build the mirror, allocating the nested record on the Python heap.
    pub fn build(py: Python<'_>, snapshot: plugmem_host::FactSnapshot) -> PyResult<Self> {
        let plugmem_host::FactSnapshot {
            record,
            text,
            metadata,
        } = snapshot;
        Ok(Self {
            record: Py::new(py, FactRecord::from(record))?,
            text,
            metadata,
        })
    }
}

/// One exported fact — the id-free, import-ready shape.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct ExportedFact {
    /// The fact's id in the database it came from. Informational: an import
    /// assigns fresh ids. Present because edges name their provenance fact by
    /// id, so a dump carrying edges needs something for them to point at.
    pub id: u32,
    /// The fact text.
    pub text: String,
    /// Subject entity name, if any.
    pub entity: Option<String>,
    /// Tag strings.
    pub tags: Vec<String>,
    /// Metadata as a key→value map (empty when none); preserved on import.
    pub metadata: BTreeMap<String, String>,
    /// When the memory learned it (unix ms; informational).
    pub recorded_at: u64,
    /// Validity start (unix ms; preserved on import).
    pub valid_from: u64,
}
repr!(ExportedFact, "ExportedFact", id, text, entity, tags);

impl From<plugmem_host::ExportedFact> for ExportedFact {
    fn from(fact: plugmem_host::ExportedFact) -> Self {
        let plugmem_host::ExportedFact {
            id,
            text,
            entity,
            tags,
            metadata,
            recorded_at,
            valid_from,
        } = fact;
        Self {
            id,
            text,
            entity,
            tags,
            metadata,
            recorded_at,
            valid_from,
        }
    }
}

/// One exported edge — the shape `export_edges` streams, and the same fields
/// the CLI's JSONL dump writes for an edge.
///
/// Edges are not part of a fact's dump: a fact names its tags and metadata, but
/// an edge is a statement *between* two entities and outlives any single fact.
/// That is why a complete backup is the two streams together.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct ExportedEdge {
    /// Source entity name.
    pub src: String,
    /// The relation, verbatim.
    pub rel: String,
    /// Destination entity name.
    pub dst: String,
    /// The fact this edge follows from, if it was recorded with one. `None`
    /// rather than a sentinel, so "no provenance" cannot be mistaken for fact 0.
    pub provenance: Option<u32>,
}
repr!(ExportedEdge, "ExportedEdge", src, rel, dst, provenance);

/// One bounded page returned by `export_page`.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct ExportPage {
    /// Open facts in fact-id order; never longer than the native page bound.
    pub facts: Vec<Py<ExportedFact>>,
    /// Pass this opaque cursor to the next call; `None` when the scan is done.
    pub next_cursor: Option<u32>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ExportPage {
    fn __repr__(&self) -> String {
        format!(
            "ExportPage(facts={}, next_cursor={:?})",
            self.facts.len(),
            self.next_cursor
        )
    }
}

impl ExportPage {
    /// Build the mirror, allocating each fact on the Python heap.
    pub fn build(py: Python<'_>, page: plugmem_host::ExportPage) -> PyResult<Self> {
        let plugmem_host::ExportPage { facts, next_cursor } = page;
        let facts = facts
            .into_iter()
            .map(|f| Py::new(py, ExportedFact::from(f)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { facts, next_cursor })
    }
}

/// What a `recover` salvaged, and what it had to leave behind.
///
/// The three `dropped` counts are the damage: each is a fact the source could
/// not produce intact, so a non-zero total means the recovered copy is smaller
/// than the original claimed to be. Zeroes across the board mean the image was
/// content-clean and this was a compaction.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct RecoverReport {
    /// Facts written to the destination — the survivors.
    pub kept: usize,
    /// Dropped: the stored text was not valid UTF-8.
    pub dropped_text: usize,
    /// Dropped: the vector slot was out of range or disagreed with the fact.
    pub dropped_vector: usize,
    /// Dropped: the metadata blob did not decode to a well-formed map.
    pub dropped_metadata: usize,
}
repr!(
    RecoverReport,
    "RecoverReport",
    kept,
    dropped_text,
    dropped_vector,
    dropped_metadata,
);

impl From<plugmem_host::RecoverReport> for RecoverReport {
    fn from(report: plugmem_host::RecoverReport) -> Self {
        let plugmem_host::RecoverReport {
            kept,
            dropped_text,
            dropped_vector,
            dropped_metadata,
        } = report;
        Self {
            kept,
            dropped_text,
            dropped_vector,
            dropped_metadata,
        }
    }
}

/// How far a `Scrub` has got.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct ScrubProgress {
    /// Bytes checksummed so far; equals `total_bytes` on the last step.
    pub done_bytes: u64,
    /// The generation file's length — the total to checksum.
    pub total_bytes: u64,
}
repr!(ScrubProgress, "ScrubProgress", done_bytes, total_bytes);

impl From<plugmem_host::ScrubProgress> for ScrubProgress {
    fn from(progress: plugmem_host::ScrubProgress) -> Self {
        let plugmem_host::ScrubProgress {
            done_bytes,
            total_bytes,
        } = progress;
        Self {
            done_bytes,
            total_bytes,
        }
    }
}

/// The report of a `maintain` pass.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct MaintainReport {
    /// Tombstoned facts physically removed by this pass.
    pub purged: usize,
    /// On-disk image bytes before the pass.
    pub bytes_before: usize,
    /// On-disk image bytes after the pass.
    pub bytes_after: usize,
    /// No storage/index rewrite was needed.
    pub no_op: bool,
    /// Tombstones present before the pass.
    pub tombstones_before: usize,
    /// Fact records before the pass.
    pub facts_before: usize,
    /// Fact records after the pass.
    pub facts_after: usize,
    /// Vector slots before the pass.
    pub vectors_before: usize,
    /// Vector slots after the pass.
    pub vectors_after: usize,
    /// HNSW coverage before the pass.
    pub hnsw_indexed_before: u32,
    /// HNSW coverage after the pass.
    pub hnsw_indexed_after: u32,
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
    pub hnsw_inserted: u32,
    /// The edge arenas were rewritten page-dense (`full` only). No edge version
    /// is ever dropped, so the version count is unchanged — only the bytes
    /// shrink.
    pub edges_compacted: bool,
    /// Current edges before the pass.
    pub edges_before: usize,
    /// Historical edge versions before the pass.
    pub edge_versions_before: usize,
}
repr!(
    MaintainReport,
    "MaintainReport",
    purged,
    bytes_before,
    bytes_after,
    no_op,
);

impl From<plugmem_host::MaintainReport> for MaintainReport {
    fn from(report: plugmem_host::MaintainReport) -> Self {
        let plugmem_host::MaintainReport {
            purged,
            bytes_before,
            bytes_after,
            no_op,
            tombstones_before,
            facts_before,
            facts_after,
            vectors_before,
            vectors_after,
            hnsw_indexed_before,
            hnsw_indexed_after,
            structural_compacted,
            bm25_compacted,
            bm25_reindexed,
            hnsw_rebuilt,
            hnsw_remapped,
            hnsw_inserted,
            edges_compacted,
            edges_before,
            edge_versions_before,
            // The shard layout the engine moved to. Observable through
            // `stats()` in the host, and dropped from both wrappers for the
            // same reason: it is a plan the engine chose, not an outcome the
            // caller acts on.
            shards_before: _,
            shards_after: _,
        } = report;
        Self {
            purged,
            bytes_before,
            bytes_after,
            no_op,
            tombstones_before,
            facts_before,
            facts_after,
            vectors_before,
            vectors_after,
            hnsw_indexed_before,
            hnsw_indexed_after,
            structural_compacted,
            bm25_compacted,
            bm25_reindexed,
            hnsw_rebuilt,
            hnsw_remapped,
            hnsw_inserted,
            edges_compacted,
            edges_before,
            edge_versions_before,
        }
    }
}

/// One memory as the workspace registry knows it.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct DbEntry {
    /// The memory's name — its identity, and what `Workspace.memory` takes.
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
repr!(DbEntry, "DbEntry", db, description, tags, archived);

/// What a `Workspace.reindex()` pass did.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
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
repr!(ReindexReport, "ReindexReport", indexed, undescribed, busy);

/// Something `Workspace.verify()` found. Reported, never repaired.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct WorkspaceProblem {
    /// The memory it concerns (empty for a kind this binding does not know).
    pub db: String,
    /// What kind: `"missing"`, `"undescribed"`, `"stale"`, `"unreadable"`,
    /// `"ambiguous-self"`. The same vocabulary the CLI prints in `--json`.
    pub issue: String,
    /// More detail, where the kind carries any.
    pub detail: Option<String>,
}
repr!(WorkspaceProblem, "WorkspaceProblem", db, issue, detail);

/// One `config.toml` setting returned by `settings_help`.
///
/// `Clone` because it is read through a getter on [`SettingsHelpResult`], and a
/// getter hands out a copy rather than a borrow into a frozen object.
/// `skip_from_py_object` because it only ever travels outward: nothing in this
/// binding accepts one as an argument.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem", skip_from_py_object)]
#[derive(Clone)]
pub struct SettingHelpItem {
    /// TOML section name.
    pub section: String,
    /// TOML key name.
    pub key: String,
    /// Human-readable value type.
    pub value_type: String,
    /// Displayed default value.
    pub default_value: String,
    /// What the setting does.
    pub description: String,
    /// Owning surface: shared, CLI or MCP.
    pub scope: String,
}
repr!(SettingHelpItem, "SettingHelpItem", section, key, value_type);

/// Complete `config.toml` help returned by `settings_help`.
#[gen_stub_pyclass]
#[pyclass(frozen, get_all, module = "plugmem._plugmem")]
pub struct SettingsHelpResult {
    /// Config discovery order from highest to lowest precedence.
    pub config_path_precedence: Vec<String>,
    /// Resolved platform default config path, if the OS exposes a user home.
    pub default_config_path: Option<String>,
    /// Every supported `config.toml` setting.
    pub settings: Vec<SettingHelpItem>,
}

#[gen_stub_pymethods]
#[pymethods]
impl SettingsHelpResult {
    fn __repr__(&self) -> String {
        format!("SettingsHelpResult(settings={})", self.settings.len())
    }
}

impl SettingsHelpResult {
    /// Read the host's settings catalogue into the mirror.
    pub fn collect() -> Self {
        let help = plugmem_host::settings_help();
        Self {
            config_path_precedence: help
                .config_path_precedence()
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            default_config_path: plugmem_host::default_config_path()
                .map(|path| path.display().to_string()),
            settings: help
                .docs()
                .iter()
                .map(|doc| SettingHelpItem {
                    section: doc.section.to_owned(),
                    key: doc.key.to_owned(),
                    value_type: doc.value_type.to_owned(),
                    default_value: doc.default.to_owned(),
                    description: doc.description.to_owned(),
                    scope: doc.scope.as_str().to_owned(),
                })
                .collect(),
        }
    }
}

/// Add every result mirror to the module, so `isinstance` and the generated
/// stubs can name them.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Similar>()?;
    module.add_class::<RememberOutcome>()?;
    module.add_class::<RecalledFact>()?;
    module.add_class::<RecalledEdge>()?;
    module.add_class::<RecallResult>()?;
    module.add_class::<Stats>()?;
    module.add_class::<FactRecord>()?;
    module.add_class::<FactSnapshot>()?;
    module.add_class::<ExportedFact>()?;
    module.add_class::<ExportedEdge>()?;
    module.add_class::<ExportPage>()?;
    module.add_class::<RecoverReport>()?;
    module.add_class::<ScrubProgress>()?;
    module.add_class::<MaintainReport>()?;
    module.add_class::<DbEntry>()?;
    module.add_class::<ReindexReport>()?;
    module.add_class::<WorkspaceProblem>()?;
    module.add_class::<SettingHelpItem>()?;
    module.add_class::<SettingsHelpResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use plugmem_host::{EntityId, FactId, SimilarReason};

    #[test]
    fn ids_and_scores_survive_the_mapping_without_a_float_detour() {
        let mapped = Similar::from(plugmem_host::Similar {
            id: FactId(1),
            score: 0.75,
            reason: SimilarReason::LexicalOverlap,
        });
        assert_eq!(mapped.id, 1);
        assert_eq!(mapped.score, 0.75);
        assert_eq!(mapped.reason, "LexicalOverlap");
    }

    #[test]
    fn the_open_sentinel_arrives_exact_rather_than_approximate() {
        // The whole reason this binding does not go through `f64` like napi: an
        // open fact's `valid_to` is `u64::MAX`, which no `f64` can represent.
        let mapped = RecalledFact::from(plugmem_host::RecalledFact {
            id: FactId(1),
            score: 0.5,
            sources: 3,
            entity: EntityId(2),
            recorded_at: 1_784_000_000_000,
            valid_from: 1_784_000_000_000,
            valid_to: plugmem_host::VALID_TO_OPEN,
        });
        assert_eq!(mapped.valid_to, plugmem_host::VALID_TO_OPEN);
        assert_eq!(mapped.valid_to, u64::MAX);
    }

    #[test]
    fn settings_help_exposes_the_shared_database_setting() {
        let help = SettingsHelpResult::collect();
        assert!(
            help.settings
                .iter()
                .any(|setting| setting.section == "database" && setting.key == "path")
        );
        // The recall knobs opened up in 0.6.0 are part of the same catalogue.
        assert!(
            help.settings
                .iter()
                .any(|setting| setting.section == "recall" && setting.key == "w_bm25")
        );
    }
}
