//! Maintenance: tombstone purge and satellite compaction (
//! B).
//!
//! `maintain` is the one O(base) verb — everything else is microseconds.
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
use crate::model::{EntityRecord, FactAux, FactRecord, TemporalSlot};
use crate::snapshot::SnapshotSink;
use crate::storage::{Scratch, Storage};
use crate::tokenizer::Tokenizer;

use super::Memory;

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
        let bytes_before = self.satellite_bytes();
        let (rebuilt, purged) = self.rebuild()?;
        // Commit point: the marker becomes durable before the swap, so a
        // replay of this journal reproduces the compacted image exactly.
        let mut entry = Vec::new();
        Op::Maintain { now }.encode(&mut entry);
        store
            .append_journal(&entry)
            .map_err(|e| Error::Storage(format!("{e:?}")))?;
        self.install(rebuilt);
        Ok(MaintainReport {
            purged,
            bytes_before,
            bytes_after: self.satellite_bytes(),
        })
    }

    /// Replay entry point: re-execute the compaction without journaling it
    /// again (the marker being replayed *is* the record of it).
    pub(super) fn replay_maintain(&mut self) -> Result<(), Error> {
        let (rebuilt, _) = self.rebuild()?;
        self.install(rebuilt);
        Ok(())
    }

    /// Bytes across the pools the rebuild replaces (everything except the
    /// interner, the by-name index and the edges — those ride through).
    fn satellite_bytes(&self) -> usize {
        self.facts.pool_bytes()
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

    /// Builds the compacted state without touching `self` (so a failure
    /// leaves the engine intact). Returns the new structures and the count
    /// of purged tombstones.
    fn rebuild(&self) -> Result<(Rebuilt, usize), Error> {
        let cfg = &self.cfg;
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);
        let mut pools = OwnedPools {
            texts: BlobHeap::new(blob),
            vecs: VecPool::new(cfg.dim, cfg.max_bytes),
        };
        let (m, vec_map, purged) = self.rebuild_parts(&mut pools)?;
        let hnsw = self.rebuild_graph(&vec_map, &pools.vecs)?;
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
        let mut bm25 = Bm25Index::new(cfg.shards_postings, cfg.max_bytes)?;
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
        let (m, vec_map, purged) = self.rebuild_parts(&mut pools)?;

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
        let hnsw = self.rebuild_graph(&vec_map, &vecs)?;

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
        };
        self.write_snapshot_with(&sections, created_at, sink)?;
        Ok(purged)
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
    ) -> Result<HnswGraph<'static>, Error> {
        let cfg = &self.cfg;
        let mut graph: HnswGraph<'static> = HnswGraph::new(cfg.hnsw_m, cfg.hnsw_m0, cfg.max_bytes)?;
        let total = pool.len() as u32;
        if cfg.dim == 0 || (total as usize) < cfg.flat_to_hnsw {
            return Ok(graph);
        }
        let old_indexed = self.hnsw.indexed() as usize;
        let dead = vec_map[..old_indexed]
            .iter()
            .filter(|&&m| m == NONE_U32)
            .count();
        let mut scratch = HnswScratch::default();
        if old_indexed > 0 && dead * 10 <= old_indexed {
            graph = self.hnsw.remapped(vec_map, pool, cfg.max_bytes)?;
        }
        graph.insert_bulk(pool, total, cfg.hnsw_ef_construction, &mut scratch)?;
        Ok(graph)
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
    }
}
