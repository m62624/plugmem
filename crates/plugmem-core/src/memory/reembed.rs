//! Explicit replacement of the complete vector axis.
//!
//! The core never calls a model. A host supplies one bounded batch callback;
//! this module validates and quantizes its output, streams the new vector pool
//! through [`Scratch`], rebuilds HNSW, and emits a new snapshot whose other
//! indexes and temporal history are borrowed unchanged from the source.

use alloc::{format, string::String, vec::Vec};

use plugmem_arena::{Arena, ArenaCfg, ShardMode};

use crate::error::Error;
use crate::id::NONE_U32;
use crate::index::hnsw::{HnswGraph, HnswScratch};
use crate::index::vecpool::VecPool;
use crate::model::{FactRecord, fact_flags};
use crate::snapshot::SnapshotSink;
use crate::storage::Scratch;

use super::Memory;
use super::persist::Sections;
use super::shards::ShardLayout;

/// Result of replacing every retained fact's embedding.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReembedReport {
    /// Space recorded before the operation (`None` for a legacy/untracked DB).
    pub previous_space: Option<String>,
    /// Space recorded with the newly published vectors.
    pub new_space: String,
    /// Previous vector dimension.
    pub previous_dim: usize,
    /// New vector dimension, supplied by the target embedder.
    pub new_dim: usize,
    /// Non-tombstoned facts embedded, including closed historical revisions.
    pub embedded: usize,
    /// Tombstoned records kept but deliberately not sent to the provider.
    pub tombstones_skipped: usize,
    /// New quantized vector-pool bytes before snapshot framing.
    pub vector_bytes: u64,
    /// Slots inserted into the rebuilt HNSW graph (zero in the flat regime).
    pub hnsw_indexed: u32,
}

/// Failure from either the engine-side rewrite or the caller's embedder.
#[derive(Debug)]
pub enum ReembedError<E> {
    /// Validation, capacity, scratch, or snapshot failure.
    Engine(Error),
    /// Error returned by the host-provided embedding callback.
    Embedder(E),
}

fn engine<E>(error: Error) -> ReembedError<E> {
    ReembedError::Engine(error)
}

#[allow(clippy::too_many_arguments)]
fn flush_batch<V, E, F>(
    encoder: &VecPool<'_>,
    vec_scratch: &mut V,
    facts: &mut Arena<'static, FactRecord>,
    records: &mut Vec<FactRecord>,
    texts: &mut Vec<&str>,
    slot_buf: &mut Vec<u8>,
    next_slot: &mut u32,
    embed: &mut F,
) -> Result<(), ReembedError<E>>
where
    V: Scratch,
    F: FnMut(&[&str]) -> Result<Vec<Vec<f32>>, E>,
{
    if records.is_empty() {
        return Ok(());
    }
    let vectors = embed(texts).map_err(ReembedError::Embedder)?;
    if vectors.len() != records.len() {
        return Err(engine(Error::Invalid(
            "embedder returned the wrong number of vectors",
        )));
    }
    for (mut record, vector) in records.drain(..).zip(vectors) {
        encoder
            .encode_slot_into(record.id, &vector, slot_buf)
            .map_err(engine)?;
        vec_scratch
            .write(slot_buf)
            .map_err(|e| engine(Error::Storage(format!("{e:?}"))))?;
        record.vector = *next_slot;
        record.flags |= fact_flags::HAS_VECTOR;
        facts.insert(&record).map_err(Error::from).map_err(engine)?;
        *next_slot = next_slot.checked_add(1).ok_or_else(|| {
            engine(Error::CapacityExceeded {
                what: "vector slots",
            })
        })?;
    }
    texts.clear();
    Ok(())
}

impl Memory<'_> {
    /// Streams a snapshot with a completely new vector pool and vector-space
    /// identity. The source memory is not mutated.
    ///
    /// `embed` is invoked with at most `batch_size` borrowed fact texts. It is
    /// also called once with an empty probe string when the database has no
    /// retained fact, so the target embedder still validates its advertised
    /// dimension. Tombstones are never sent to it; closed revisions are,
    /// because historical/as-of recall must stay vector-searchable.
    ///
    /// # Errors
    ///
    /// [`ReembedError::Embedder`] preserves a callback failure. Engine errors
    /// cover an invalid space/dimension, malformed provider output, corrupt
    /// source text, capacity, scratch I/O, HNSW, and snapshot output.
    #[allow(clippy::too_many_arguments)]
    pub fn write_reembedded_snapshot<V, Sk, E, F>(
        &self,
        created_at: u64,
        target_dim: usize,
        target_space: &str,
        batch_size: usize,
        vec_scratch: &mut V,
        sink: Sk,
        mut embed: F,
    ) -> Result<ReembedReport, ReembedError<E>>
    where
        V: Scratch,
        Sk: SnapshotSink,
        F: FnMut(&[&str]) -> Result<Vec<Vec<f32>>, E>,
    {
        Self::validate_vector_space(target_space).map_err(engine)?;
        if target_dim == 0 {
            return Err(engine(Error::Invalid(
                "reembed target dimension must be nonzero",
            )));
        }
        if batch_size == 0 {
            return Err(engine(Error::Invalid("reembed batch size must be nonzero")));
        }
        let mut target_cfg = self.cfg.clone();
        target_cfg.dim = target_dim;
        target_cfg.validate().map_err(engine)?;

        let arena = ArenaCfg::new(target_cfg.shards_facts, ShardMode::Uniform)
            .with_max_bytes(target_cfg.max_bytes);
        let mut facts = Arena::new(arena).map_err(Error::from).map_err(engine)?;
        let encoder = VecPool::new(target_dim, target_cfg.max_bytes);
        let mut records = Vec::with_capacity(batch_size);
        let mut texts = Vec::with_capacity(batch_size);
        let mut slot_buf = Vec::with_capacity(encoder.stride());
        let mut next_slot = 0u32;
        let mut tombstones_skipped = 0usize;

        for fid in self.fact_ids_ascending() {
            let Some(mut record) = self.facts.get(&fid.to_be_bytes()) else {
                continue;
            };
            if record.is_tombstone() {
                record.flags &= !fact_flags::HAS_VECTOR;
                record.vector = NONE_U32;
                facts.insert(&record).map_err(Error::from).map_err(engine)?;
                tombstones_skipped += 1;
                continue;
            }
            let text = core::str::from_utf8(self.texts.get(record.text))
                .map_err(|_| engine(Error::Corrupt("reembed: fact text is not UTF-8")))?;
            records.push(record);
            texts.push(text);
            if records.len() == batch_size {
                flush_batch(
                    &encoder,
                    vec_scratch,
                    &mut facts,
                    &mut records,
                    &mut texts,
                    &mut slot_buf,
                    &mut next_slot,
                    &mut embed,
                )?;
            }
        }
        flush_batch(
            &encoder,
            vec_scratch,
            &mut facts,
            &mut records,
            &mut texts,
            &mut slot_buf,
            &mut next_slot,
            &mut embed,
        )?;

        if next_slot == 0 {
            let probe = embed(&[""]).map_err(ReembedError::Embedder)?;
            if probe.len() != 1 {
                return Err(engine(Error::Invalid(
                    "embedder returned the wrong number of probe vectors",
                )));
            }
            encoder
                .encode_slot_into(crate::FactId(0), &probe[0], &mut slot_buf)
                .map_err(engine)?;
        }

        let vector_bytes = vec_scratch.len();
        let vec_bytes = vec_scratch
            .freeze()
            .map_err(|e| engine(Error::Storage(format!("{e:?}"))))?;
        let vecs = VecPool::from_parts_borrowed(target_dim, target_cfg.max_bytes, vec_bytes)
            .map_err(engine)?;
        debug_assert_eq!(vecs.len(), next_slot as usize);

        let mut hnsw = HnswGraph::new(target_cfg.hnsw_m, target_cfg.hnsw_m0, target_cfg.max_bytes)
            .map_err(engine)?;
        if vecs.len() >= target_cfg.flat_to_hnsw {
            let mut scratch = HnswScratch::default();
            hnsw.insert_bulk(
                &vecs,
                next_slot,
                target_cfg.hnsw_ef_construction,
                &mut scratch,
            )
            .map_err(engine)?;
        }

        let sections = Sections {
            facts: &facts,
            fact_aux: &self.fact_aux,
            entities: &self.entities,
            by_name: &self.by_name,
            temporal: &self.temporal,
            texts: &self.texts,
            metas: &self.metas,
            tag_lists: &self.tag_lists,
            bm25: &self.bm25,
            tags_idx: &self.tags_idx,
            entity_facts: &self.entity_facts,
            vecs: &vecs,
            hnsw: &hnsw,
            edges_out: &self.edges_out,
            edges_in: &self.edges_in,
            edges_hist_out: &self.edges_hist_out,
            edges_hist_in: &self.edges_hist_in,
            layout: ShardLayout::of_config(&self.cfg),
        };
        self.write_snapshot_reconfigured(
            &sections,
            &target_cfg,
            Some(target_space),
            created_at,
            sink,
        )
        .map_err(engine)?;

        Ok(ReembedReport {
            previous_space: self.vector_space.clone(),
            new_space: target_space.into(),
            previous_dim: self.cfg.dim,
            new_dim: target_dim,
            embedded: next_slot as usize,
            tombstones_skipped,
            vector_bytes,
            hnsw_indexed: hnsw.indexed(),
        })
    }
}
