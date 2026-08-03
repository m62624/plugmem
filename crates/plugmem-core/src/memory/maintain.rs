//! Maintenance: tombstone purge and satellite compaction (
//! B).
//!
//! `maintain` is policy-driven: the default mode is a cheap no-op when no
//! work is pending, while compaction, text reindexing and full vector-graph
//! rebuilds remain explicit O(base) maintenance work.
//! It reclaims the space held by forgotten facts without ever renumbering
//! ids: `FactId`/`EntityId`/`TermId` are stable forever, so external
//! references, revision chains and edges stay valid across a compaction.
//!
//! Tombstoned facts are purged **physically**: their `FactRecord` and
//! `FactAux` are simply not carried into the rebuilt arenas. The id itself
//! is *burned*, never reissued — id allocation runs on the persisted
//! `next_fact` counter, not on record presence, so replay determinism and
//! the "ids are never reused" invariant survive removal (
//! allows numbering holes explicitly). A burned id behaves
//! exactly like a tombstoned one did: `get` returns `None`, verbs return
//! `NotFound`. References *to* a purged fact (a successor's `revises`, an
//! edge's provenance) keep the burned id rather than being rewritten:
//! resolving it yields `None` either way, which is what makes a maintained
//! and an unmaintained run observation-equivalent.
//!
//! Every satellite structure (the blob heap, the tag pool, the three
//! posting stores, the temporal arena, the vector pool) is rebuilt from
//! the live facts alone. The interner is not rebuilt (term ids are
//! stable; leaked terms are a documented v2 concern), and edges and the
//! by-name index carry only stable ids, so they ride through untouched.
//!
//! Determinism is the load-bearing property: the rebuild walks entities
//! and facts in id order and re-derives each index the same way every
//! time, so a snapshot taken after a live `maintain` is byte-identical to
//! one taken after replaying the journal (which re-executes the `Maintain`
//! marker). The commit order is check-first: the whole new state is built
//! (fallible) and the journal marker is appended (fallible) before
//! anything is swapped in (infallible).

use alloc::format;
use alloc::vec::Vec;

use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapBuilder, BlobHeapCfg, BlobId, ChunkPool, ChunkPoolCfg,
    ListHandle, ShardMode,
};

use crate::error::Error;
use crate::id::{FactId, NONE_U32};
use crate::index::IdListIndex;
use crate::index::bm25::Bm25Index;
use crate::index::hnsw::{HnswGraph, HnswScratch};
use crate::index::vecpool::VecPool;
use crate::journal::Op;
use crate::memory::persist::Sections;
use crate::model::{EdgeHistorySlot, EdgeSlot, EntityRecord, FactAux, FactRecord, TemporalSlot};
use crate::snapshot::SnapshotSink;
use crate::storage::{Scratch, Storage};
use crate::tokenizer::Tokenizer;

use super::Memory;

/// Version of the tokenizer semantics used by the BM25 index.
///
/// Bump this when a tokenizer change intentionally changes indexed tokens.
/// Snapshots persist the value; a future mismatch can trigger explicit
/// reindexing instead of silently compacting an index with stale semantics.
pub(crate) const TOKENIZER_INDEX_VERSION: u32 = 1;

const AUTO_HNSW_INSERT_BUDGET: usize = 4096;
const NO_HNSW_INSERT_LIMIT: u32 = u32::MAX;

/// The maintenance work requested by a caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MaintenanceMode {
    /// Do the minimum necessary work: purge tombstones, refresh stale text
    /// indexes, and advance the vector graph within a bounded budget.
    #[default]
    Auto,
    /// Physically purge tombstones and compact storage/indexes.
    Compact,
    /// Rebuild BM25 by reading and tokenizing live text.
    ReindexText,
    /// Advance or rebuild the vector graph without compacting text/facts.
    OptimizeVectors,
    /// Rebuild every rebuildable structure and fully optimize vectors.
    Full,
}

/// Options for a maintenance pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaintenanceOptions {
    /// The requested maintenance mode.
    pub mode: MaintenanceMode,
    /// Maximum HNSW tail slots to insert in this pass. `None` means no limit.
    pub max_hnsw_inserts: Option<usize>,
}

impl Default for MaintenanceOptions {
    fn default() -> Self {
        Self::auto()
    }
}

impl MaintenanceOptions {
    /// Bounded production-oriented maintenance.
    pub const fn auto() -> Self {
        Self {
            mode: MaintenanceMode::Auto,
            max_hnsw_inserts: Some(AUTO_HNSW_INSERT_BUDGET),
        }
    }

    /// Full offline rebuild.
    pub const fn full() -> Self {
        Self {
            mode: MaintenanceMode::Full,
            max_hnsw_inserts: None,
        }
    }

    pub(crate) fn from_journal(mode: u8, max_hnsw_inserts: u32) -> Result<Self, Error> {
        let mode = match mode {
            0 => MaintenanceMode::Auto,
            1 => MaintenanceMode::Compact,
            2 => MaintenanceMode::ReindexText,
            3 => MaintenanceMode::OptimizeVectors,
            4 => MaintenanceMode::Full,
            _ => return Err(Error::Corrupt("journal maintenance mode is invalid")),
        };
        let max_hnsw_inserts = if max_hnsw_inserts == NO_HNSW_INSERT_LIMIT {
            None
        } else {
            Some(max_hnsw_inserts as usize)
        };
        Ok(Self {
            mode,
            max_hnsw_inserts,
        })
    }

    pub(crate) fn journal_mode(self) -> u8 {
        match self.mode {
            MaintenanceMode::Auto => 0,
            MaintenanceMode::Compact => 1,
            MaintenanceMode::ReindexText => 2,
            MaintenanceMode::OptimizeVectors => 3,
            MaintenanceMode::Full => 4,
        }
    }

    pub(crate) fn journal_max_hnsw_inserts(self) -> u32 {
        self.max_hnsw_inserts
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX - 1))
            .unwrap_or(NO_HNSW_INSERT_LIMIT)
    }
}

/// Maps a [`Scratch`] error into the engine's storage-error variant.
fn scratch_err<E: core::fmt::Debug>(e: E) -> Error {
    Error::Storage(format!("{e:?}"))
}

/// Sink for the two dominant pools (text, vectors) during a rebuild. The in-RAM
/// path ([`OwnedPools`]) builds them owned; the disk-first path
/// ([`StreamPools`]) streams them into a [`Scratch`] and never holds them
/// Everything else a rebuild produces is metadata — small enough
/// (∝ record count) to build in RAM on either path.
trait PoolSink {
    /// Records a text blob, returning its new dense id.
    fn push_text(&mut self, bytes: &[u8]) -> Result<BlobId, Error>;
    /// Copies vector slot `slot` of `src`, returning its new dense slot.
    fn push_vector(&mut self, src: &VecPool<'_>, slot: u32) -> Result<u32, Error>;
}

/// In-RAM pools: the classic owned rebuild.
struct OwnedPools {
    texts: BlobHeap<'static>,
    vecs: VecPool<'static>,
}

impl PoolSink for OwnedPools {
    fn push_text(&mut self, bytes: &[u8]) -> Result<BlobId, Error> {
        Ok(self.texts.push(bytes)?)
    }

    fn push_vector(&mut self, src: &VecPool<'_>, slot: u32) -> Result<u32, Error> {
        Ok(self.vecs.copy_slot(src, slot))
    }
}

/// Disk-first pools: text bytes and vector slots stream into two `Scratch`es;
/// only the flat text index ([`BlobHeapBuilder`]) and the slot counter stay in
/// RAM (both ∝ record count).
struct StreamPools<'s, T: Scratch, V: Scratch> {
    text_scratch: &'s mut T,
    text_index: BlobHeapBuilder,
    vec_scratch: &'s mut V,
    vec_count: u32,
}

impl<T: Scratch, V: Scratch> PoolSink for StreamPools<'_, T, V> {
    fn push_text(&mut self, bytes: &[u8]) -> Result<BlobId, Error> {
        self.text_scratch.write(bytes).map_err(scratch_err)?;
        Ok(self.text_index.push_len(bytes.len())?)
    }

    fn push_vector(&mut self, src: &VecPool<'_>, slot: u32) -> Result<u32, Error> {
        self.vec_scratch
            .write(src.slot_bytes(slot as usize))
            .map_err(scratch_err)?;
        let new = self.vec_count;
        self.vec_count += 1;
        Ok(new)
    }
}

/// The rebuildable metadata (everything but the two big pools and the graph):
/// produced by [`Memory::rebuild_parts`] and shared by the in-RAM and
/// disk-first paths.
struct RebuildMeta {
    facts: Arena<'static, FactRecord>,
    fact_aux: Arena<'static, FactAux>,
    entities: Arena<'static, EntityRecord>,
    temporal: Arena<'static, TemporalSlot>,
    tag_lists: ChunkPool<'static>,
    /// Compacted metadata blobs of the live facts (built owned in RAM on both
    /// paths — metadata is pointers/attributes, ∝ record count, not a big pool).
    metas: BlobHeap<'static>,
    bm25: Bm25Index<'static>,
    tags_idx: IdListIndex<'static>,
    entity_facts: IdListIndex<'static>,
}

/// Report of a `maintain` pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaintainReport {
    /// Tombstoned facts physically removed by this pass (their ids stay
    /// burned; a second pass over the same state purges nothing).
    pub purged: usize,
    /// Bytes across the rebuilt pools before the pass.
    pub bytes_before: usize,
    /// Bytes across the rebuilt pools after the pass.
    pub bytes_after: usize,
    /// `true` when the selected mode had no work to do and did not rewrite
    /// storage or append a maintenance marker.
    pub no_op: bool,
    /// Tombstoned records present before the pass.
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
    /// Physical fact/text/vector compaction ran.
    pub structural_compacted: bool,
    /// BM25 was compacted from existing postings without re-tokenizing text.
    pub bm25_compacted: bool,
    /// BM25 was rebuilt by reading and tokenizing live text.
    pub bm25_reindexed: bool,
    /// HNSW was rebuilt from an empty graph.
    pub hnsw_rebuilt: bool,
    /// HNSW was carried/remapped from the previous graph.
    pub hnsw_remapped: bool,
    /// Slots inserted into HNSW during this pass.
    pub hnsw_inserted: u32,
    /// The edge arenas were rewritten page-dense. No version is ever dropped,
    /// so `edge_versions_after` always equals `edge_versions_before`; what
    /// shrinks is the bytes they occupy.
    pub edges_compacted: bool,
    /// Current edges before the pass.
    pub edges_before: usize,
    /// Historical edge versions before the pass.
    pub edge_versions_before: usize,
}

/// The four edge arenas rebuilt page-dense; see [`Memory::repack_edges`].
struct RepackedEdges {
    out: Arena<'static, EdgeSlot>,
    inn: Arena<'static, EdgeSlot>,
    hist_out: Arena<'static, EdgeHistorySlot>,
    hist_in: Arena<'static, EdgeHistorySlot>,
}

impl RepackedEdges {
    fn pool_bytes(&self) -> usize {
        self.out.pool_bytes()
            + self.inn.pool_bytes()
            + self.hist_out.pool_bytes()
            + self.hist_in.pool_bytes()
    }
}

/// The freshly rebuilt structures, swapped in atomically once the journal
/// marker is durable.
struct Rebuilt {
    facts: Arena<'static, FactRecord>,
    entities: Arena<'static, EntityRecord>,
    fact_aux: Arena<'static, FactAux>,
    texts: BlobHeap<'static>,
    metas: BlobHeap<'static>,
    tag_lists: ChunkPool<'static>,
    bm25: Bm25Index<'static>,
    tags_idx: IdListIndex<'static>,
    entity_facts: IdListIndex<'static>,
    temporal: Arena<'static, TemporalSlot>,
    vecs: VecPool<'static>,
    hnsw: HnswGraph<'static>,
    edges: Option<RepackedEdges>,
    bm25_tokenizer_version: u32,
    report: MaintainReport,
}

#[derive(Clone, Copy)]
enum Bm25Policy {
    Compact,
    Reindex,
}

#[derive(Clone, Copy)]
struct WorkPlan {
    compact: bool,
    bm25_reindex: bool,
    optimize_vectors: bool,
    hnsw_full_rebuild: bool,
    repack_edges: bool,
    max_hnsw_inserts: Option<usize>,
}

#[derive(Clone, Copy, Default)]
struct GraphWork {
    rebuilt: bool,
    remapped: bool,
    inserted: u32,
}

impl WorkPlan {
    fn needs_work(self) -> bool {
        self.compact || self.bm25_reindex || self.optimize_vectors || self.repack_edges
    }
}

fn hnsw_target(start: u32, total: u32, max_hnsw_inserts: Option<usize>) -> u32 {
    let Some(max) = max_hnsw_inserts else {
        return total;
    };
    let max = u32::try_from(max).unwrap_or(u32::MAX);
    start.saturating_add(max).min(total)
}

impl Memory<'_> {
    /// Physically purges tombstoned facts and compacts every satellite
    /// structure. Ids of living facts are preserved; purged ids
    /// are burned (never reissued); observable state is unchanged; only
    /// bytes shrink. Journaled as a `Maintain` marker so replay reproduces
    /// the compaction exactly.
    ///
    /// # Errors
    ///
    /// [`Error::CapacityExceeded`] if a rebuilt pool hits its ceiling (it
    /// cannot, being a subset of the live data, but the path is honest),
    /// or an [`Error::Storage`] from the journal append — in either case
    /// nothing is swapped in and the engine is unchanged.
    pub fn maintain<S: Storage>(
        &mut self,
        store: &mut S,
        now: u64,
    ) -> Result<MaintainReport, Error> {
        self.maintain_with_options(store, now, MaintenanceOptions::auto())
    }

    /// Runs a maintenance pass with explicit policy.
    pub fn maintain_with_options<S: Storage>(
        &mut self,
        store: &mut S,
        now: u64,
        options: MaintenanceOptions,
    ) -> Result<MaintainReport, Error> {
        let plan = self.work_plan(options);
        let bytes_before = self.satellite_bytes(plan.repack_edges);
        let mut report = self.report_skeleton(bytes_before);
        if !plan.needs_work() {
            report.no_op = true;
            return Ok(report);
        }

        if !plan.compact {
            let mut bm25 = None;
            if plan.bm25_reindex {
                bm25 = Some(self.reindex_bm25_from_text()?);
                report.bm25_reindexed = true;
            }
            let mut hnsw = None;
            if plan.optimize_vectors {
                let (graph, work) =
                    self.optimize_graph(plan.hnsw_full_rebuild, plan.max_hnsw_inserts)?;
                hnsw = Some(graph);
                report.hnsw_rebuilt = work.rebuilt;
                report.hnsw_remapped = work.remapped;
                report.hnsw_inserted = work.inserted;
            }
            // Commit point: journal before swapping any rebuilt index in.
            let mut entry = Vec::new();
            Op::Maintain {
                now,
                mode: options.journal_mode(),
                max_hnsw_inserts: options.journal_max_hnsw_inserts(),
            }
            .encode(&mut entry);
            store
                .append_journal(&entry)
                .map_err(|e| Error::Storage(format!("{e:?}")))?;
            if let Some(bm25) = bm25 {
                self.bm25 = bm25;
                self.bm25_tokenizer_version = TOKENIZER_INDEX_VERSION;
            }
            if let Some(hnsw) = hnsw {
                self.hnsw = hnsw;
            }
            report.bytes_after = self.satellite_bytes(plan.repack_edges);
            report.hnsw_indexed_after = self.hnsw.indexed();
            return Ok(report);
        }

        let (rebuilt, _) = self.rebuild(plan)?;
        report = rebuilt.report.clone();
        // Commit point: the marker becomes durable before the swap, so a
        // replay of this journal reproduces the compacted image exactly.
        let mut entry = Vec::new();
        Op::Maintain {
            now,
            mode: options.journal_mode(),
            max_hnsw_inserts: options.journal_max_hnsw_inserts(),
        }
        .encode(&mut entry);
        store
            .append_journal(&entry)
            .map_err(|e| Error::Storage(format!("{e:?}")))?;
        self.install(rebuilt);
        Ok(report)
    }

    pub(super) fn replay_maintain_with_options(
        &mut self,
        options: MaintenanceOptions,
    ) -> Result<(), Error> {
        let plan = self.work_plan(options);
        if !plan.needs_work() {
            return Ok(());
        }
        if !plan.compact {
            if plan.bm25_reindex {
                self.bm25 = self.reindex_bm25_from_text()?;
                self.bm25_tokenizer_version = TOKENIZER_INDEX_VERSION;
            }
            if plan.optimize_vectors {
                let (hnsw, _) =
                    self.optimize_graph(plan.hnsw_full_rebuild, plan.max_hnsw_inserts)?;
                self.hnsw = hnsw;
            }
            return Ok(());
        }
        let (rebuilt, _) = self.rebuild(plan)?;
        self.install(rebuilt);
        Ok(())
    }

    /// Bytes across the pools a pass replaces. The interner and the by-name
    /// index always ride through; the edge arenas ride through unless the
    /// plan repacks them, in which case counting them is what makes
    /// `bytes_before`/`bytes_after` describe the same set of pools.
    fn satellite_bytes(&self, with_edges: bool) -> usize {
        let edges = if with_edges { self.edge_bytes() } else { 0 };
        edges
            + self.facts.pool_bytes()
            + self.fact_aux.pool_bytes()
            + self.entities.pool_bytes()
            + self.hnsw.pool_bytes()
            + self.texts.pool_bytes()
            + self.metas.pool_bytes()
            + self.tag_lists.pool_bytes()
            + self.bm25.pool_bytes()
            + self.tags_idx.pool_bytes()
            + self.entity_facts.pool_bytes()
            + self.temporal.pool_bytes()
            + self.vecs.pool_bytes()
    }

    fn edge_bytes(&self) -> usize {
        self.edges_out.pool_bytes()
            + self.edges_in.pool_bytes()
            + self.edges_hist_out.pool_bytes()
            + self.edges_hist_in.pool_bytes()
    }

    /// `true` if the selected maintenance policy would change engine state.
    pub fn maintenance_needed(&self, options: MaintenanceOptions) -> bool {
        self.work_plan(options).needs_work()
    }

    /// Builds a report for the current state without running maintenance.
    /// Hosts use this for cheap no-op returns before deciding to write a new
    /// snapshot.
    pub fn maintenance_preview(
        &self,
        options: MaintenanceOptions,
        bytes_before: usize,
    ) -> MaintainReport {
        let mut report = self.report_skeleton(bytes_before);
        if !self.maintenance_needed(options) {
            report.no_op = true;
        }
        report
    }

    fn report_skeleton(&self, bytes_before: usize) -> MaintainReport {
        MaintainReport {
            purged: 0,
            bytes_before,
            bytes_after: bytes_before,
            no_op: false,
            tombstones_before: self.tombstones,
            facts_before: self.facts.len(),
            facts_after: self.facts.len(),
            vectors_before: self.vecs.len(),
            vectors_after: self.vecs.len(),
            hnsw_indexed_before: self.hnsw.indexed(),
            hnsw_indexed_after: self.hnsw.indexed(),
            structural_compacted: false,
            bm25_compacted: false,
            bm25_reindexed: false,
            hnsw_rebuilt: false,
            hnsw_remapped: false,
            hnsw_inserted: 0,
            edges_compacted: false,
            edges_before: self.edges_out.len(),
            edge_versions_before: self.edges_hist_out.len(),
        }
    }

    fn work_plan(&self, options: MaintenanceOptions) -> WorkPlan {
        let tokenizer_stale = self.bm25_tokenizer_version != TOKENIZER_INDEX_VERSION;
        let has_tombstones = self.tombstones != 0;
        // A migrated image holds documents with no term-set summary. Filling
        // them in is compaction's job — it already transposes the postings —
        // so an image that needs it counts as pending work even with nothing
        // to purge. One pass settles it: the compacted index never asks again.
        //
        // Only the `compact` decision keys off this. Nothing is purged, so no
        // fact id moves, and the vector graph stays valid — remapping it here
        // would be work for a mapping that did not change.
        let compaction_due = has_tombstones || self.bm25.needs_resummarize();
        let graph_tail = self.cfg.dim != 0
            && self.vecs.len() >= self.cfg.flat_to_hnsw
            && self.hnsw.indexed() < self.vecs.len() as u32;
        match options.mode {
            MaintenanceMode::Auto => WorkPlan {
                compact: compaction_due,
                bm25_reindex: tokenizer_stale,
                optimize_vectors: graph_tail,
                hnsw_full_rebuild: false,
                repack_edges: false,
                max_hnsw_inserts: options.max_hnsw_inserts,
            },
            MaintenanceMode::Compact => WorkPlan {
                compact: compaction_due,
                bm25_reindex: tokenizer_stale,
                optimize_vectors: has_tombstones && self.hnsw.indexed() != 0,
                hnsw_full_rebuild: false,
                repack_edges: false,
                max_hnsw_inserts: options.max_hnsw_inserts,
            },
            MaintenanceMode::ReindexText => WorkPlan {
                compact: compaction_due,
                bm25_reindex: true,
                optimize_vectors: has_tombstones && self.hnsw.indexed() != 0,
                hnsw_full_rebuild: false,
                repack_edges: false,
                max_hnsw_inserts: options.max_hnsw_inserts,
            },
            MaintenanceMode::OptimizeVectors => WorkPlan {
                compact: false,
                bm25_reindex: false,
                optimize_vectors: self.cfg.dim != 0
                    && self.vecs.len() >= self.cfg.flat_to_hnsw
                    && (graph_tail || self.hnsw.indexed() == 0),
                hnsw_full_rebuild: self.hnsw.indexed() == 0,
                repack_edges: false,
                max_hnsw_inserts: options.max_hnsw_inserts,
            },
            MaintenanceMode::Full => WorkPlan {
                compact: true,
                bm25_reindex: true,
                optimize_vectors: self.cfg.dim != 0 && self.vecs.len() >= self.cfg.flat_to_hnsw,
                hnsw_full_rebuild: true,
                repack_edges: !self.edges_hist_out.is_empty(),
                max_hnsw_inserts: None,
            },
        }
    }

    /// Rewrites the four edge arenas by inserting their records in ascending
    /// key order, which packs every page.
    ///
    /// **No version is ever dropped.** History is the feature, there is no
    /// retention policy to apply, and an entity's past is not garbage.
    /// What this reclaims is page slack: an arena splits a full page in half
    /// unless the insert appends past its last key, and the *incoming* mirror
    /// is keyed by the far endpoint, so a workload that relinks many relations
    /// interleaves their runs and lands mid-page again and again. Measured on
    /// 200 relations relinked 1000 times, the edge arenas held 31.9 MB for
    /// 19.2 MB of versions; rebuilt in key order they hold the 19.2 MB.
    ///
    /// Only [`MaintenanceMode::Full`] asks for this: it is O(versions) work
    /// for a size win, not something a background pass should do.
    fn repack_edges(&self) -> Result<RepackedEdges, Error> {
        let ord = ArenaCfg::new(self.cfg.shards_edges, ShardMode::Ordered)
            .with_max_bytes(self.cfg.max_bytes);
        let mut out = Arena::new(ord)?;
        let mut inn = Arena::new(ord)?;
        let mut hist_out = Arena::new(ord)?;
        let mut hist_in = Arena::new(ord)?;
        // `iter` on an ordered arena yields ascending keys, so every insert
        // appends and every page fills.
        for edge in self.edges_out.iter() {
            out.insert(&edge)?;
        }
        for edge in self.edges_in.iter() {
            inn.insert(&edge)?;
        }
        for version in self.edges_hist_out.iter() {
            hist_out.insert(&version)?;
        }
        for version in self.edges_hist_in.iter() {
            hist_in.insert(&version)?;
        }
        Ok(RepackedEdges {
            out,
            inn,
            hist_out,
            hist_in,
        })
    }

    fn reindex_bm25_from_text(&self) -> Result<Bm25Index<'static>, Error> {
        let cfg = &self.cfg;
        let mut bm25 = Bm25Index::new(cfg.shards_postings, cfg.max_bytes)?;
        let mut tokenizer = Tokenizer::new();
        let mut tf: Vec<(u32, u8)> = Vec::new();
        for fid in 0..self.next_fact {
            let id = FactId(fid);
            let Some(rec) = self.facts.get(&fid.to_be_bytes()) else {
                continue;
            };
            if rec.is_tombstone() {
                continue;
            }
            let text = core::str::from_utf8(self.texts.get(rec.text))
                .map_err(|_| Error::Corrupt("maintain: fact text is not UTF-8"))?;
            tf.clear();
            let terms = &self.terms;
            let tf_ref = &mut tf;
            tokenizer.tokenize(text, &mut |token| {
                if let Some(term) = terms.lookup(token) {
                    match tf_ref.iter_mut().find(|(t, _)| *t == term.0) {
                        Some((_, c)) => *c = c.saturating_add(1),
                        None => tf_ref.push((term.0, 1)),
                    }
                }
            });
            bm25.index_doc(id, &tf)?;
        }
        Ok(bm25)
    }

    fn optimize_graph(
        &self,
        full_rebuild: bool,
        max_hnsw_inserts: Option<usize>,
    ) -> Result<(HnswGraph<'static>, GraphWork), Error> {
        let total = self.vecs.len() as u32;
        let mut graph = if full_rebuild || self.hnsw.indexed() == 0 {
            HnswGraph::new(self.cfg.hnsw_m, self.cfg.hnsw_m0, self.cfg.max_bytes)?
        } else {
            self.hnsw.to_owned(self.cfg.max_bytes)?
        };
        let start = graph.indexed();
        let target = hnsw_target(start, total, max_hnsw_inserts);
        let mut scratch = HnswScratch::default();
        graph.insert_bulk(
            &self.vecs,
            target,
            self.cfg.hnsw_ef_construction,
            &mut scratch,
        )?;
        Ok((
            graph,
            GraphWork {
                rebuilt: full_rebuild || self.hnsw.indexed() == 0,
                remapped: !full_rebuild && self.hnsw.indexed() != 0,
                inserted: target.saturating_sub(start),
            },
        ))
    }

    /// Builds the compacted state without touching `self` (so a failure
    /// leaves the engine intact). Returns the new structures and the count
    /// of purged tombstones.
    fn rebuild(&self, plan: WorkPlan) -> Result<(Rebuilt, usize), Error> {
        let cfg = &self.cfg;
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);
        let mut pools = OwnedPools {
            texts: BlobHeap::new(blob),
            vecs: VecPool::new(cfg.dim, cfg.max_bytes),
        };
        let bm25_policy = if plan.bm25_reindex {
            Bm25Policy::Reindex
        } else {
            Bm25Policy::Compact
        };
        let (m, vec_map, purged) = self.rebuild_parts(&mut pools, bm25_policy)?;
        let (hnsw, graph_work) = self.rebuild_graph(
            &vec_map,
            &pools.vecs,
            plan.hnsw_full_rebuild,
            plan.max_hnsw_inserts,
        )?;
        let edges = plan.repack_edges.then(|| self.repack_edges()).transpose()?;
        let mut report = self.report_skeleton(self.satellite_bytes(plan.repack_edges));
        report.purged = purged;
        report.bytes_after = edges.as_ref().map_or(0, RepackedEdges::pool_bytes)
            + m.facts.pool_bytes()
            + m.fact_aux.pool_bytes()
            + m.entities.pool_bytes()
            + hnsw.pool_bytes()
            + pools.texts.pool_bytes()
            + m.metas.pool_bytes()
            + m.tag_lists.pool_bytes()
            + m.bm25.pool_bytes()
            + m.tags_idx.pool_bytes()
            + m.entity_facts.pool_bytes()
            + m.temporal.pool_bytes()
            + pools.vecs.pool_bytes();
        report.facts_after = m.facts.len();
        report.vectors_after = pools.vecs.len();
        report.hnsw_indexed_after = hnsw.indexed();
        report.structural_compacted = true;
        report.bm25_compacted = matches!(bm25_policy, Bm25Policy::Compact);
        report.bm25_reindexed = matches!(bm25_policy, Bm25Policy::Reindex);
        report.hnsw_rebuilt = graph_work.rebuilt;
        report.hnsw_remapped = graph_work.remapped;
        report.hnsw_inserted = graph_work.inserted;
        report.edges_compacted = edges.is_some();
        Ok((
            Rebuilt {
                facts: m.facts,
                entities: m.entities,
                fact_aux: m.fact_aux,
                texts: pools.texts,
                metas: m.metas,
                tag_lists: m.tag_lists,
                bm25: m.bm25,
                tags_idx: m.tags_idx,
                entity_facts: m.entity_facts,
                temporal: m.temporal,
                vecs: pools.vecs,
                hnsw,
                edges,
                bm25_tokenizer_version: if plan.bm25_reindex {
                    TOKENIZER_INDEX_VERSION
                } else {
                    self.bm25_tokenizer_version
                },
                report,
            },
            purged,
        ))
    }

    /// Builds the compacted metadata and pushes the two big pools through
    /// `pools` — the walk shared by the in-RAM rebuild ([`OwnedPools`]) and the
    /// disk-first one ([`StreamPools`]). Ids are **not** renumbered;
    /// only text-blob ids and vector slots are re-densified, in fact-id order,
    /// so both paths produce byte-identical output. Returns the metadata, the
    /// old→new vector-slot map (for the graph) and the purge count.
    fn rebuild_parts<P: PoolSink>(
        &self,
        pools: &mut P,
        bm25_policy: Bm25Policy,
    ) -> Result<(RebuildMeta, alloc::vec::Vec<u32>, usize), Error> {
        let cfg = &self.cfg;
        let uni =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(cfg.max_bytes);
        let ord =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Ordered).with_max_bytes(cfg.max_bytes);

        let mut entities = Arena::new(uni(cfg.shards_entities))?;
        let mut facts = Arena::new(uni(cfg.shards_facts))?;
        let mut fact_aux = Arena::new(uni(cfg.shards_facts))?;
        let mut tag_lists = ChunkPool::new(ChunkPoolCfg::new().with_max_bytes(cfg.max_bytes));
        let mut live = alloc::vec![false; self.next_fact as usize];
        for rec in self.facts.iter() {
            if !rec.is_tombstone() {
                live[rec.id.0 as usize] = true;
            }
        }
        let mut bm25 = match bm25_policy {
            Bm25Policy::Compact => {
                self.bm25
                    .compact_live(cfg.shards_postings, cfg.max_bytes, |id| {
                        live.get(id.0 as usize).copied().unwrap_or(false)
                    })?
            }
            Bm25Policy::Reindex => Bm25Index::new(cfg.shards_postings, cfg.max_bytes)?,
        };
        let mut tags_idx = IdListIndex::new(cfg.shards_postings, cfg.max_bytes)?;
        let mut entity_facts = IdListIndex::new(cfg.shards_entities, cfg.max_bytes)?;
        let mut temporal = Arena::new(ord(cfg.shards_temporal))?;
        let mut metas = BlobHeap::new(
            BlobHeapCfg::new()
                .with_max_bytes(cfg.max_bytes)
                .with_max_blob(cfg.max_blob),
        );

        // Entities first (id order), each with its name pushed into the new
        // text pool. Entities are never purged, so a gap is corruption.
        for eid in 0..self.next_entity {
            let rec = self
                .entities
                .get(&eid.to_be_bytes())
                .ok_or(Error::Corrupt("maintain: entity id gap"))?;
            let name_id = pools.push_text(self.texts.get(rec.name))?;
            entities.insert(&EntityRecord {
                name: name_id,
                ..rec
            })?;
        }

        // Re-tokenization reuses the (unchanged) interner via read-only
        // lookup — every live token was interned at creation, so it
        // resolves; the tokenizer is a scratch, taken to satisfy borrows.
        // Constraint: this only holds while the tokenizer matches the one
        // the texts were indexed with. A future tokenizer change must not
        // ship through this lookup path (new tokens would silently drop
        // from BM25) — a reindex migration has to intern, not look up.
        let mut tokenizer = Tokenizer::new();
        let mut tf: Vec<(u32, u8)> = Vec::new();

        // old vector-slot id → new slot id (NONE for purged vectors);
        // carries the HNSW graph across the compaction.
        let mut vec_map = alloc::vec![NONE_U32; self.vecs.len()];

        let mut purged = 0usize;
        for fid in 0..self.next_fact {
            let id = FactId(fid);
            // A missing record is an id burned by an earlier pass — legal
            // (: numbering holes after a purge are the norm).
            let Some(rec) = self.facts.get(&fid.to_be_bytes()) else {
                continue;
            };

            if rec.is_tombstone() {
                // Physical purge: neither the record nor its aux is carried
                // over. The id stays burned via the untouched `next_fact`.
                purged += 1;
                continue;
            }

            // Live fact: push its text and re-derive every index.
            let text_bytes = self.texts.get(rec.text);
            let text_id = pools.push_text(text_bytes)?;
            if matches!(bm25_policy, Bm25Policy::Reindex) {
                let text = core::str::from_utf8(text_bytes)
                    .map_err(|_| Error::Corrupt("maintain: fact text is not UTF-8"))?;

                tf.clear();
                let terms = &self.terms;
                let tf_ref = &mut tf;
                tokenizer.tokenize(text, &mut |token| {
                    if let Some(term) = terms.lookup(token) {
                        match tf_ref.iter_mut().find(|(t, _)| *t == term.0) {
                            Some((_, c)) => *c = c.saturating_add(1),
                            None => tf_ref.push((term.0, 1)),
                        }
                    }
                });
                bm25.index_doc(id, &tf)?;
            }

            // Tags: re-read the old list, rebuild the fact's handle and the
            // inverted index. Every fact gets an aux record at creation, so
            // a gap here is corruption — same strictness as the fact gap
            // above, not a silent "no tags".
            let aux = self
                .fact_aux
                .get(&fid.to_be_bytes())
                .ok_or(Error::Corrupt("maintain: fact aux gap"))?;
            let mut tags = ListHandle::EMPTY;
            for chunk in self.tag_lists.iter(&aux.tags) {
                for raw in chunk.chunks_exact(4) {
                    let term = u32::from_be_bytes(raw.try_into().unwrap());
                    tag_lists.push(&mut tags, &term.to_be_bytes())?;
                    tags_idx.push(term, id, 0)?;
                }
            }
            // Metadata rides across the compaction verbatim: the stored blob is
            // already canonical, so it is copied byte for byte into the new heap.
            let meta = if aux.meta.0 == NONE_U32 {
                BlobId(NONE_U32)
            } else {
                metas.push(self.metas.get(aux.meta))?
            };
            fact_aux.insert(&FactAux { id, tags, meta })?;

            // Entity index and temporal index.
            if let Some(entity) = rec.entity.some() {
                entity_facts.push(entity.0, id, 0)?;
            }
            temporal.insert(&TemporalSlot {
                recorded_at: rec.recorded_at,
                fact: id,
            })?;

            // Vector: push the already-quantized slot verbatim.
            let vector = if rec.has_vector() {
                let new_slot = pools.push_vector(&self.vecs, rec.vector)?;
                vec_map[rec.vector as usize] = new_slot;
                new_slot
            } else {
                NONE_U32
            };
            facts.insert(&FactRecord {
                text: text_id,
                vector,
                ..rec
            })?;
        }

        Ok((
            RebuildMeta {
                facts,
                fact_aux,
                entities,
                temporal,
                tag_lists,
                metas,
                bm25,
                tags_idx,
                entity_facts,
            },
            vec_map,
            purged,
        ))
    }

    /// Disk-first compaction (milestone H): rebuilds the compacted
    /// image and writes it to `sink`, streaming the two big pools (text,
    /// vectors) through `text_scratch`/`vec_scratch` so peak RAM stays ∝ the
    /// record count (metadata + graph), never ∝ the content size. Byte-identical
    /// to a snapshot taken after an in-RAM [`Memory::maintain`] — it drives the
    /// same walk (`rebuild_parts`) and the same emit (`write_snapshot_with`),
    /// the pools merely borrowing the frozen scratch instead of RAM. Returns the
    /// purge count.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for a malformed source, [`Error::Storage`] from a
    /// scratch or the sink, or a pool ceiling error (a subset never exceeds it).
    pub fn snapshot_disk_first<T: Scratch, V: Scratch, Sk: SnapshotSink>(
        &self,
        created_at: u64,
        text_scratch: &mut T,
        vec_scratch: &mut V,
        sink: Sk,
    ) -> Result<usize, Error> {
        let options = MaintenanceOptions::auto();
        if !self.maintenance_needed(options) {
            self.write_snapshot_with(&self.sections(), created_at, sink)?;
            return Ok(0);
        }
        Ok(self
            .snapshot_disk_first_with_options(created_at, text_scratch, vec_scratch, sink, options)?
            .purged)
    }

    /// Options-aware disk-first sibling of [`Memory::maintain_with_options`].
    /// It emits a compacted/optimized snapshot and returns the same style of
    /// maintenance report without mutating `self`.
    pub fn snapshot_disk_first_with_options<T: Scratch, V: Scratch, Sk: SnapshotSink>(
        &self,
        created_at: u64,
        text_scratch: &mut T,
        vec_scratch: &mut V,
        sink: Sk,
        options: MaintenanceOptions,
    ) -> Result<MaintainReport, Error> {
        let plan = self.work_plan(options);
        let mut report = self.report_skeleton(self.satellite_bytes(plan.repack_edges));
        if !plan.needs_work() {
            report.no_op = true;
            return Ok(report);
        }
        let cfg = &self.cfg;
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);
        let mut pools = StreamPools {
            text_scratch,
            text_index: BlobHeapBuilder::new(blob),
            vec_scratch,
            vec_count: 0,
        };
        let bm25_policy = if plan.bm25_reindex {
            Bm25Policy::Reindex
        } else {
            Bm25Policy::Compact
        };
        let (m, vec_map, purged) = self.rebuild_parts(&mut pools, bm25_policy)?;

        // Freeze the staged pools and borrow them as the two big sections; the
        // metadata and graph are the only things in RAM.
        let StreamPools {
            text_scratch,
            text_index,
            vec_scratch,
            ..
        } = pools;
        let mut text_index_bytes = Vec::new();
        text_index.dump_index(&mut text_index_bytes);
        let text_pool = text_scratch.freeze().map_err(scratch_err)?;
        let vec_pool = vec_scratch.freeze().map_err(scratch_err)?;
        let texts = BlobHeap::load_borrowed(blob, &text_index_bytes, text_pool)?;
        let vecs = VecPool::from_parts_borrowed(cfg.dim, cfg.max_bytes, vec_pool)?;
        let (hnsw, graph_work) = self.rebuild_graph(
            &vec_map,
            &vecs,
            plan.hnsw_full_rebuild,
            plan.max_hnsw_inserts,
        )?;

        // Repacked edges are built owned like the other metadata: they are
        // records, not content, so they scale with the edge count and not with
        // the image size the disk-first path exists to keep out of RAM.
        let edges = plan.repack_edges.then(|| self.repack_edges()).transpose()?;
        let sections = Sections {
            facts: &m.facts,
            fact_aux: &m.fact_aux,
            entities: &m.entities,
            temporal: &m.temporal,
            texts: &texts,
            metas: &m.metas,
            tag_lists: &m.tag_lists,
            bm25: &m.bm25,
            tags_idx: &m.tags_idx,
            entity_facts: &m.entity_facts,
            vecs: &vecs,
            hnsw: &hnsw,
            edges_out: edges.as_ref().map_or(&self.edges_out, |e| &e.out),
            edges_in: edges.as_ref().map_or(&self.edges_in, |e| &e.inn),
            edges_hist_out: edges.as_ref().map_or(&self.edges_hist_out, |e| &e.hist_out),
            edges_hist_in: edges.as_ref().map_or(&self.edges_hist_in, |e| &e.hist_in),
        };
        self.write_snapshot_with(&sections, created_at, sink)?;
        report.purged = purged;
        report.edges_compacted = edges.is_some();
        report.bytes_after = edges.as_ref().map_or(0, RepackedEdges::pool_bytes)
            + m.facts.pool_bytes()
            + m.fact_aux.pool_bytes()
            + m.entities.pool_bytes()
            + hnsw.pool_bytes()
            + texts.pool_bytes()
            + m.metas.pool_bytes()
            + m.tag_lists.pool_bytes()
            + m.bm25.pool_bytes()
            + m.tags_idx.pool_bytes()
            + m.entity_facts.pool_bytes()
            + m.temporal.pool_bytes()
            + vecs.pool_bytes();
        report.facts_after = m.facts.len();
        report.vectors_after = vecs.len();
        report.hnsw_indexed_after = hnsw.indexed();
        report.structural_compacted = true;
        report.bm25_compacted = matches!(bm25_policy, Bm25Policy::Compact);
        report.bm25_reindexed = matches!(bm25_policy, Bm25Policy::Reindex);
        report.hnsw_rebuilt = graph_work.rebuilt;
        report.hnsw_remapped = graph_work.remapped;
        report.hnsw_inserted = graph_work.inserted;
        Ok(report)
    }

    /// The vector index's maintenance policy (phase 2), all
    /// deterministic:
    ///
    /// - below `flat_to_hnsw` the graph is empty (flat regime);
    /// - the first crossing (or > 10% of the graph's nodes dead) builds
    ///   the graph from scratch over the compacted pool;
    /// - otherwise the existing graph is *carried over*: neighbor lists
    ///   are remapped through the compaction map (dead nodes drop out)
    ///   and the flat tail is bulk-inserted — the cheap steady-state
    ///   path that keeps `maintain` inside its budget.
    fn rebuild_graph(
        &self,
        vec_map: &[u32],
        pool: &VecPool<'_>,
        full_rebuild: bool,
        max_hnsw_inserts: Option<usize>,
    ) -> Result<(HnswGraph<'static>, GraphWork), Error> {
        let cfg = &self.cfg;
        let mut graph: HnswGraph<'static> = HnswGraph::new(cfg.hnsw_m, cfg.hnsw_m0, cfg.max_bytes)?;
        let total = pool.len() as u32;
        if cfg.dim == 0 || (total as usize) < cfg.flat_to_hnsw {
            return Ok((graph, GraphWork::default()));
        }
        let old_indexed = self.hnsw.indexed() as usize;
        let dead = vec_map[..old_indexed]
            .iter()
            .filter(|&&m| m == NONE_U32)
            .count();
        let mut scratch = HnswScratch::default();
        let mut work = GraphWork::default();
        if old_indexed > 0 && !full_rebuild {
            graph = self.hnsw.remapped(vec_map, pool, cfg.max_bytes)?;
            work.remapped = true;
        } else if old_indexed > 0 || total > 0 {
            work.rebuilt = true;
        }
        if full_rebuild && dead * 10 > old_indexed {
            work.rebuilt = true;
        }
        let start = graph.indexed();
        let target = hnsw_target(start, total, max_hnsw_inserts);
        graph.insert_bulk(pool, target, cfg.hnsw_ef_construction, &mut scratch)?;
        work.inserted = target.saturating_sub(start);
        Ok((graph, work))
    }

    /// Swaps the rebuilt structures in (infallible). The interner, by-name
    /// index, edges and id counters are unchanged by design.
    fn install(&mut self, r: Rebuilt) {
        self.facts = r.facts;
        self.entities = r.entities;
        self.fact_aux = r.fact_aux;
        self.texts = r.texts;
        self.metas = r.metas;
        self.tag_lists = r.tag_lists;
        self.bm25 = r.bm25;
        self.tags_idx = r.tags_idx;
        self.entity_facts = r.entity_facts;
        self.temporal = r.temporal;
        self.vecs = r.vecs;
        self.hnsw = r.hnsw;
        if let Some(edges) = r.edges {
            self.edges_out = edges.out;
            self.edges_in = edges.inn;
            self.edges_hist_out = edges.hist_out;
            self.edges_hist_in = edges.hist_in;
        }
        self.tombstones = 0;
        self.bm25_tokenizer_version = r.bm25_tokenizer_version;
    }
}
