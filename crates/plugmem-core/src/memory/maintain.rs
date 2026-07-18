//! Maintenance: tombstone purge and satellite compaction (specs/05,
//! specs/11 B).
//!
//! `maintain` is the one O(base) verb — everything else is microseconds.
//! It reclaims the space held by forgotten facts without ever renumbering
//! ids: `FactId`/`EntityId`/`TermId` are stable forever, so external
//! references, revision chains and edges stay valid across a compaction.
//!
//! The design keeps the tombstoned fact *records* in the facts arena (48
//! bytes each, ids must stay occupied) but strips their payload — text
//! collapses to one shared empty blob, the vector slot is dropped, the tag
//! list is emptied — and rebuilds every satellite structure (the blob
//! heap, the tag pool, the three posting stores, the temporal arena, the
//! vector pool) from the live facts alone. The interner is not rebuilt
//! (term ids are stable; leaked terms are a documented v2 concern), and
//! edges and the by-name index carry only stable ids, so they ride through
//! untouched.
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
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, ChunkPool, ChunkPoolCfg, ListHandle, ShardMode,
};

use crate::error::Error;
use crate::id::{FactId, NONE_U32};
use crate::index::IdListIndex;
use crate::index::bm25::Bm25Index;
use crate::index::vecpool::VecPool;
use crate::journal::Op;
use crate::model::{EntityRecord, FactAux, FactRecord, TemporalSlot, fact_flags};
use crate::storage::Storage;
use crate::tokenizer::Tokenizer;

use super::Memory;

/// Report of a `maintain` pass (specs/05).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaintainReport {
    /// Tombstoned facts whose payload was reclaimed.
    pub purged: usize,
    /// Reclaimable satellite bytes before the pass.
    pub bytes_before: usize,
    /// Reclaimable satellite bytes after the pass.
    pub bytes_after: usize,
}

/// The freshly rebuilt structures, swapped in atomically once the journal
/// marker is durable.
struct Rebuilt {
    facts: Arena<FactRecord>,
    entities: Arena<EntityRecord>,
    fact_aux: Arena<FactAux>,
    texts: BlobHeap,
    tag_lists: ChunkPool,
    bm25: Bm25Index,
    tags_idx: IdListIndex,
    entity_facts: IdListIndex,
    temporal: Arena<TemporalSlot>,
    vecs: VecPool,
}

impl Memory {
    /// Purges tombstoned facts and compacts every satellite structure
    /// (specs/05). Ids are preserved; observable state is unchanged; only
    /// reclaimable bytes shrink. Journaled as a `Maintain` marker so
    /// replay reproduces the compaction exactly.
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

    /// Reclaimable bytes across the satellite pools (the facts/entities
    /// arenas are not reclaimed — their tombstone records stay).
    fn satellite_bytes(&self) -> usize {
        self.texts.pool_bytes()
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
        let uni =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(cfg.max_bytes);
        let ord =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Ordered).with_max_bytes(cfg.max_bytes);
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);

        let mut texts = BlobHeap::new(blob);
        let mut entities = Arena::new(uni(cfg.shards_entities))?;
        let mut facts = Arena::new(uni(cfg.shards_facts))?;
        let mut fact_aux = Arena::new(uni(cfg.shards_facts))?;
        let mut tag_lists = ChunkPool::new(ChunkPoolCfg::new().with_max_bytes(cfg.max_bytes));
        let mut bm25 = Bm25Index::new(cfg.shards_postings, cfg.max_bytes)?;
        let mut tags_idx = IdListIndex::new(cfg.shards_postings, cfg.max_bytes)?;
        let mut entity_facts = IdListIndex::new(cfg.shards_entities, cfg.max_bytes)?;
        let mut temporal = Arena::new(ord(cfg.shards_temporal))?;
        let mut vecs = VecPool::new(cfg.dim, cfg.max_bytes);

        // Entities first (id order), each with its name copied into the new
        // heap. Entities are never purged.
        for eid in 0..self.next_entity {
            let rec = self
                .entities
                .get(&eid.to_be_bytes())
                .ok_or(Error::Corrupt("maintain: entity id gap"))?;
            let name_id = texts.push(self.texts.get(rec.name))?;
            entities.insert(&EntityRecord {
                name: name_id,
                ..rec
            })?;
        }
        // One shared empty blob backs every tombstone's text.
        let empty = texts.push(b"")?;

        // Re-tokenization reuses the (unchanged) interner via read-only
        // lookup — every live token was interned at creation, so it
        // resolves; the tokenizer is a scratch, taken to satisfy borrows.
        let mut tokenizer = Tokenizer::new();
        let mut tf: Vec<(u32, u8)> = Vec::new();

        let mut purged = 0usize;
        for fid in 0..self.next_fact {
            let id = FactId(fid);
            let rec = self
                .facts
                .get(&fid.to_be_bytes())
                .ok_or(Error::Corrupt("maintain: fact id gap"))?;

            if rec.is_tombstone() {
                purged += 1;
                facts.insert(&FactRecord {
                    text: empty,
                    vector: NONE_U32,
                    flags: rec.flags & !fact_flags::HAS_VECTOR,
                    ..rec
                })?;
                fact_aux.insert(&FactAux {
                    id,
                    tags: ListHandle::EMPTY,
                })?;
                continue;
            }

            // Live fact: copy its text and re-derive every index.
            let text_bytes = self.texts.get(rec.text);
            let text_id = texts.push(text_bytes)?;
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
            // inverted index.
            let mut tags = ListHandle::EMPTY;
            if let Some(aux) = self.fact_aux.get(&fid.to_be_bytes()) {
                for chunk in self.tag_lists.iter(&aux.tags) {
                    for raw in chunk.chunks_exact(4) {
                        let term = u32::from_be_bytes(raw.try_into().unwrap());
                        tag_lists.push(&mut tags, &term.to_be_bytes())?;
                        tags_idx.push(term, id, 0)?;
                    }
                }
            }
            fact_aux.insert(&FactAux { id, tags })?;

            // Entity index and temporal index.
            if let Some(entity) = rec.entity.some() {
                entity_facts.push(entity.0, id, 0)?;
            }
            temporal.insert(&TemporalSlot {
                recorded_at: rec.recorded_at,
                fact: id,
            })?;

            // Vector: copy the already-quantized slot verbatim.
            let vector = if rec.has_vector() {
                vecs.copy_slot(&self.vecs, rec.vector)
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
            Rebuilt {
                facts,
                entities,
                fact_aux,
                texts,
                tag_lists,
                bm25,
                tags_idx,
                entity_facts,
                temporal,
                vecs,
            },
            purged,
        ))
    }

    /// Swaps the rebuilt structures in (infallible). The interner, by-name
    /// index, edges and id counters are unchanged by design.
    fn install(&mut self, r: Rebuilt) {
        self.facts = r.facts;
        self.entities = r.entities;
        self.fact_aux = r.fact_aux;
        self.texts = r.texts;
        self.tag_lists = r.tag_lists;
        self.bm25 = r.bm25;
        self.tags_idx = r.tags_idx;
        self.entity_facts = r.entity_facts;
        self.temporal = r.temporal;
        self.vecs = r.vecs;
    }
}
