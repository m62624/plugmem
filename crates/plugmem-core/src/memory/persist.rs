//! Snapshot composition: the engine's state as container sections and the
//! validated load path (specs/03).
//!
//! Saving concatenates every structure's canonical dump into the
//! [`snapshot`](crate::snapshot) container. Loading is the untrusted-input
//! side: after the container's structural and checksum validation, every
//! structure validates its own image, chunk chains are walked with shared
//! visited maps (cycles, double-claims, orphans), posting lists are fully
//! decoded (well-formed varints, ascending ids, counts and last-id
//! agreement), text and term pools are UTF-8-checked, and **every stored
//! id is range-checked** — facts' blob/entity/revision references, edge
//! endpoints, temporal and by-name entries. That last pass is what keeps
//! the engine's panicking accessors (`get`, `resolve` — contract-violation
//! panics by design) sound on arbitrary input: after a successful load no
//! persisted id can violate a contract.

use alloc::vec::Vec;

use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, ChunkPool, ChunkPoolCfg, Interner, ShardMode, Slot,
};

use crate::config::Config;
use crate::error::Error;
use crate::id::NONE_U32;
use crate::index::IdListIndex;
use crate::index::bm25::Bm25Index;
use crate::index::postings::PostingStore;
use crate::index::varint::decode_u32;
use crate::snapshot::{Snapshot, SnapshotWriter};

use super::Memory;

/// Section kinds of the engine snapshot (`meta`/`index` before `pool` —
/// readers want the small section first).
mod kind {
    pub const FACTS_META: u16 = 1;
    pub const FACTS_POOL: u16 = 2;
    pub const AUX_META: u16 = 3;
    pub const AUX_POOL: u16 = 4;
    pub const ENTITIES_META: u16 = 5;
    pub const ENTITIES_POOL: u16 = 6;
    pub const BY_NAME_META: u16 = 7;
    pub const BY_NAME_POOL: u16 = 8;
    pub const EDGES_OUT_META: u16 = 9;
    pub const EDGES_OUT_POOL: u16 = 10;
    pub const EDGES_IN_META: u16 = 11;
    pub const EDGES_IN_POOL: u16 = 12;
    pub const TEMPORAL_META: u16 = 13;
    pub const TEMPORAL_POOL: u16 = 14;
    pub const TEXTS_INDEX: u16 = 15;
    pub const TEXTS_POOL: u16 = 16;
    pub const TERMS_INDEX: u16 = 17;
    pub const TERMS_POOL: u16 = 18;
    pub const TERMS_TABLE: u16 = 19;
    pub const TAG_LISTS_META: u16 = 20;
    pub const TAG_LISTS_POOL: u16 = 21;
    pub const BM25_HANDLES_META: u16 = 22;
    pub const BM25_HANDLES_POOL: u16 = 23;
    pub const BM25_CHUNKS_META: u16 = 24;
    pub const BM25_CHUNKS_POOL: u16 = 25;
    pub const BM25_DOCLEN_META: u16 = 26;
    pub const BM25_DOCLEN_POOL: u16 = 27;
    pub const TAGS_HANDLES_META: u16 = 28;
    pub const TAGS_HANDLES_POOL: u16 = 29;
    pub const TAGS_CHUNKS_META: u16 = 30;
    pub const TAGS_CHUNKS_POOL: u16 = 31;
    pub const ENTFACTS_HANDLES_META: u16 = 32;
    pub const ENTFACTS_HANDLES_POOL: u16 = 33;
    pub const ENTFACTS_CHUNKS_META: u16 = 34;
    pub const ENTFACTS_CHUNKS_POOL: u16 = 35;
    pub const ENGINE_STATE: u16 = 36;
}

/// Byte length of the engine-state section.
const STATE_LEN: usize = 24;

/// Dumps an arena as its `(meta, pool)` section pair.
fn arena_sections<T: Slot>(a: &Arena<T>) -> (Vec<u8>, Vec<u8>) {
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    a.dump_meta(&mut meta);
    a.dump_pool(&mut pool);
    (meta, pool)
}

/// Fetches a required section.
fn section<'a>(snap: &Snapshot<'a>, kind: u16) -> Result<&'a [u8], Error> {
    snap.section(kind)
        .ok_or(Error::Corrupt("snapshot is missing a required section"))
}

impl<const TF: bool> PostingStore<TF> {
    /// Dumps the store's four sections.
    pub(crate) fn dump_sections(&self) -> [Vec<u8>; 4] {
        let (hm, hp) = (self.handles_meta(), self.handles_pool());
        let (cm, cp) = (self.chunks_meta(), self.chunks_pool());
        [hm, hp, cm, cp]
    }

    /// Rebuilds a store from its sections and validates every list: chain
    /// walks over a shared visited map, full entry decode (well-formed
    /// varints, strictly ascending ids without overflow), `count`/`last`
    /// agreement, and no orphan chunks.
    pub(crate) fn load_sections(
        shards: usize,
        max_bytes: usize,
        hm: &[u8],
        hp: &[u8],
        cm: &[u8],
        cp: &[u8],
    ) -> Result<Self, Error> {
        let handles = Arena::<crate::index::postings::IdListSlot>::load(
            ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(max_bytes),
            hm,
            hp,
        )?;
        let pool = ChunkPool::load(ChunkPoolCfg::new().with_max_bytes(max_bytes), cm, cp)?;
        let mut visited = alloc::vec![false; pool.chunks()];
        for slot in handles.iter() {
            pool.validate_chain(&slot.handle, &mut visited)?;
            let mut count = 0u32;
            let mut last = 0u32;
            let mut first = true;
            for chunk in pool.iter(&slot.handle) {
                let mut cur = chunk;
                while !cur.is_empty() {
                    let Some((delta, used)) = decode_u32(cur) else {
                        return Err(Error::Corrupt("posting entry is malformed"));
                    };
                    let mut entry_len = used;
                    if TF {
                        if cur.len() < used + 1 {
                            return Err(Error::Corrupt("posting entry is malformed"));
                        }
                        entry_len += 1;
                    }
                    cur = &cur[entry_len..];
                    let id = if first {
                        first = false;
                        delta
                    } else {
                        if delta == 0 {
                            return Err(Error::Corrupt("posting ids are not ascending"));
                        }
                        last.checked_add(delta)
                            .ok_or(Error::Corrupt("posting id overflows"))?
                    };
                    last = id;
                    count += 1;
                }
            }
            if count != slot.count || (count > 0 && last != slot.last) {
                return Err(Error::Corrupt("posting list disagrees with its handle"));
            }
        }
        if pool.orphan_count(&visited) != 0 {
            return Err(Error::Corrupt("posting pool has orphan chunks"));
        }
        Ok(Self::from_parts(handles, pool))
    }
}

impl Bm25Index {
    fn dump_into(&self, w: &mut SnapshotWriter) -> Result<(), Error> {
        let [hm, hp, cm, cp] = self.postings().dump_sections();
        w.section(kind::BM25_HANDLES_META, hm)?;
        w.section(kind::BM25_HANDLES_POOL, hp)?;
        w.section(kind::BM25_CHUNKS_META, cm)?;
        w.section(kind::BM25_CHUNKS_POOL, cp)?;
        let (dm, dp) = arena_sections(self.doc_len_arena());
        w.section(kind::BM25_DOCLEN_META, dm)?;
        w.section(kind::BM25_DOCLEN_POOL, dp)?;
        Ok(())
    }

    fn load_from(snap: &Snapshot<'_>, cfg: &Config) -> Result<Self, Error> {
        let postings = PostingStore::<true>::load_sections(
            cfg.shards_postings,
            cfg.max_bytes,
            section(snap, kind::BM25_HANDLES_META)?,
            section(snap, kind::BM25_HANDLES_POOL)?,
            section(snap, kind::BM25_CHUNKS_META)?,
            section(snap, kind::BM25_CHUNKS_POOL)?,
        )?;
        let doc_len = Arena::load(
            ArenaCfg::new(cfg.shards_postings, ShardMode::Uniform).with_max_bytes(cfg.max_bytes),
            section(snap, kind::BM25_DOCLEN_META)?,
            section(snap, kind::BM25_DOCLEN_POOL)?,
        )?;
        let state = section(snap, kind::ENGINE_STATE)?;
        if state.len() != STATE_LEN {
            return Err(Error::Corrupt("engine state section has a wrong length"));
        }
        let total_docs = u64::from_le_bytes(state[8..16].try_into().unwrap());
        let total_len = u64::from_le_bytes(state[16..24].try_into().unwrap());
        if total_docs != doc_len.len() as u64 {
            return Err(Error::Corrupt("bm25 document total disagrees with doc_len"));
        }
        Ok(Self::from_parts(postings, doc_len, total_docs, total_len))
    }
}

impl Memory {
    /// Serializes the whole engine into snapshot-container bytes.
    /// Deterministic and canonical: save → load → save is byte-identical.
    pub fn snapshot_bytes(&self, created_at: u64) -> Vec<u8> {
        let mut w = SnapshotWriter::new();
        let mut push = |k: u16, bytes: Vec<u8>| {
            w.section(k, bytes)
                .expect("section kinds are unique consts");
        };
        let (m, p) = arena_sections(&self.facts);
        push(kind::FACTS_META, m);
        push(kind::FACTS_POOL, p);
        let (m, p) = arena_sections(&self.fact_aux);
        push(kind::AUX_META, m);
        push(kind::AUX_POOL, p);
        let (m, p) = arena_sections(&self.entities);
        push(kind::ENTITIES_META, m);
        push(kind::ENTITIES_POOL, p);
        let (m, p) = arena_sections(&self.by_name);
        push(kind::BY_NAME_META, m);
        push(kind::BY_NAME_POOL, p);
        let (m, p) = arena_sections(&self.edges_out);
        push(kind::EDGES_OUT_META, m);
        push(kind::EDGES_OUT_POOL, p);
        let (m, p) = arena_sections(&self.edges_in);
        push(kind::EDGES_IN_META, m);
        push(kind::EDGES_IN_POOL, p);
        let (m, p) = arena_sections(&self.temporal);
        push(kind::TEMPORAL_META, m);
        push(kind::TEMPORAL_POOL, p);
        let (mut i, mut p) = (Vec::new(), Vec::new());
        self.texts.dump_index(&mut i);
        self.texts.dump_pool(&mut p);
        push(kind::TEXTS_INDEX, i);
        push(kind::TEXTS_POOL, p);
        let (mut i, mut p, mut t) = (Vec::new(), Vec::new(), Vec::new());
        self.terms.dump_index(&mut i);
        self.terms.dump_pool(&mut p);
        self.terms.dump_table(&mut t);
        push(kind::TERMS_INDEX, i);
        push(kind::TERMS_POOL, p);
        push(kind::TERMS_TABLE, t);
        let (mut m, mut p) = (Vec::new(), Vec::new());
        self.tag_lists.dump_meta(&mut m);
        self.tag_lists.dump_pool(&mut p);
        push(kind::TAG_LISTS_META, m);
        push(kind::TAG_LISTS_POOL, p);
        self.bm25.dump_into(&mut w).expect("unique kinds");
        let [hm, hp, cm, cp] = self.tags_idx.dump_sections();
        let mut push = |k: u16, bytes: Vec<u8>| {
            w.section(k, bytes).expect("unique kinds");
        };
        push(kind::TAGS_HANDLES_META, hm);
        push(kind::TAGS_HANDLES_POOL, hp);
        push(kind::TAGS_CHUNKS_META, cm);
        push(kind::TAGS_CHUNKS_POOL, cp);
        let [hm, hp, cm, cp] = self.entity_facts.dump_sections();
        push(kind::ENTFACTS_HANDLES_META, hm);
        push(kind::ENTFACTS_HANDLES_POOL, hp);
        push(kind::ENTFACTS_CHUNKS_META, cm);
        push(kind::ENTFACTS_CHUNKS_POOL, cp);
        let mut state = Vec::with_capacity(STATE_LEN);
        state.extend_from_slice(&self.next_fact.to_le_bytes());
        state.extend_from_slice(&self.next_entity.to_le_bytes());
        state.extend_from_slice(&self.bm25.docs().to_le_bytes());
        state.extend_from_slice(&self.bm25.total_len().to_le_bytes());
        push(kind::ENGINE_STATE, state);

        let mut cfg_bytes = Vec::new();
        self.cfg.encode(&mut cfg_bytes);
        w.finish(&cfg_bytes, 0, created_at, env!("CARGO_PKG_VERSION"))
    }

    /// Writes a full snapshot and clears the journal (specs/05).
    pub fn snapshot<S: crate::storage::Storage>(
        &mut self,
        store: &mut S,
        now: u64,
    ) -> Result<(), Error> {
        let bytes = self.snapshot_bytes(now);
        store
            .write_snapshot(&bytes)
            .map_err(|e| Error::Storage(alloc::format!("{e:?}")))?;
        store
            .clear_journal()
            .map_err(|e| Error::Storage(alloc::format!("{e:?}")))?;
        Ok(())
    }

    /// Loads an engine from snapshot bytes (the untrusted path — see the
    /// module docs for the validation inventory).
    pub(super) fn load_snapshot(bytes: &[u8], cfg: Config) -> Result<Self, Error> {
        cfg.validate()?;
        let snap = Snapshot::parse(bytes, cfg.fast_load)?;
        let stored = Config::decode(snap.config())?;
        // Structural fields must match; tuning fields follow the caller.
        if stored.dim != cfg.dim {
            return Err(Error::ConfigMismatch("stored dim differs"));
        }
        if [
            (stored.shards_facts, cfg.shards_facts),
            (stored.shards_entities, cfg.shards_entities),
            (stored.shards_edges, cfg.shards_edges),
            (stored.shards_temporal, cfg.shards_temporal),
            (stored.shards_postings, cfg.shards_postings),
        ]
        .iter()
        .any(|&(a, b)| a != b)
        {
            return Err(Error::ConfigMismatch("stored shard counts differ"));
        }
        if stored.max_bytes != cfg.max_bytes
            || stored.max_text != cfg.max_text
            || stored.max_blob != cfg.max_blob
        {
            return Err(Error::ConfigMismatch("stored size limits differ"));
        }

        let mut mem = Self::new(cfg)?;
        let cfg = &mem.cfg;
        let uni =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(cfg.max_bytes);
        let ord =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Ordered).with_max_bytes(cfg.max_bytes);
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);
        mem.facts = Arena::load(
            uni(cfg.shards_facts),
            section(&snap, kind::FACTS_META)?,
            section(&snap, kind::FACTS_POOL)?,
        )?;
        mem.fact_aux = Arena::load(
            uni(cfg.shards_facts),
            section(&snap, kind::AUX_META)?,
            section(&snap, kind::AUX_POOL)?,
        )?;
        mem.entities = Arena::load(
            uni(cfg.shards_entities),
            section(&snap, kind::ENTITIES_META)?,
            section(&snap, kind::ENTITIES_POOL)?,
        )?;
        mem.by_name = Arena::load(
            ord(cfg.shards_entities),
            section(&snap, kind::BY_NAME_META)?,
            section(&snap, kind::BY_NAME_POOL)?,
        )?;
        mem.edges_out = Arena::load(
            ord(cfg.shards_edges),
            section(&snap, kind::EDGES_OUT_META)?,
            section(&snap, kind::EDGES_OUT_POOL)?,
        )?;
        mem.edges_in = Arena::load(
            ord(cfg.shards_edges),
            section(&snap, kind::EDGES_IN_META)?,
            section(&snap, kind::EDGES_IN_POOL)?,
        )?;
        mem.temporal = Arena::load(
            ord(cfg.shards_temporal),
            section(&snap, kind::TEMPORAL_META)?,
            section(&snap, kind::TEMPORAL_POOL)?,
        )?;
        mem.texts = BlobHeap::load(
            blob,
            section(&snap, kind::TEXTS_INDEX)?,
            section(&snap, kind::TEXTS_POOL)?,
        )?;
        for (_, bytes) in mem.texts.iter() {
            if core::str::from_utf8(bytes).is_err() {
                return Err(Error::Corrupt("stored text is not valid UTF-8"));
            }
        }
        mem.terms = Interner::load(
            blob,
            section(&snap, kind::TERMS_INDEX)?,
            section(&snap, kind::TERMS_POOL)?,
            section(&snap, kind::TERMS_TABLE)?,
        )?;
        mem.tag_lists = ChunkPool::load(
            ChunkPoolCfg::new().with_max_bytes(cfg.max_bytes),
            section(&snap, kind::TAG_LISTS_META)?,
            section(&snap, kind::TAG_LISTS_POOL)?,
        )?;
        mem.bm25 = Bm25Index::load_from(&snap, cfg)?;
        mem.tags_idx = IdListIndex::load_sections(
            cfg.shards_postings,
            cfg.max_bytes,
            section(&snap, kind::TAGS_HANDLES_META)?,
            section(&snap, kind::TAGS_HANDLES_POOL)?,
            section(&snap, kind::TAGS_CHUNKS_META)?,
            section(&snap, kind::TAGS_CHUNKS_POOL)?,
        )?;
        mem.entity_facts = IdListIndex::load_sections(
            cfg.shards_entities,
            cfg.max_bytes,
            section(&snap, kind::ENTFACTS_HANDLES_META)?,
            section(&snap, kind::ENTFACTS_HANDLES_POOL)?,
            section(&snap, kind::ENTFACTS_CHUNKS_META)?,
            section(&snap, kind::ENTFACTS_CHUNKS_POOL)?,
        )?;
        let state = section(&snap, kind::ENGINE_STATE)?;
        if state.len() != STATE_LEN {
            return Err(Error::Corrupt("engine state section has a wrong length"));
        }
        mem.next_fact = u32::from_le_bytes(state[0..4].try_into().unwrap());
        mem.next_entity = u32::from_le_bytes(state[4..8].try_into().unwrap());
        if (mem.next_fact as usize) < mem.facts.len()
            || (mem.next_entity as usize) < mem.entities.len()
        {
            return Err(Error::Corrupt("engine id counters below record counts"));
        }
        mem.validate_references()?;
        Ok(mem)
    }

    /// Range-checks every stored id so the engine's panicking accessors
    /// are sound on loaded data (module docs). O(records) — the price of
    /// panic-freedom on hostile input, linear and cache-friendly.
    fn validate_references(&self) -> Result<(), Error> {
        let texts = self.texts.len() as u32;
        let terms = self.terms.len() as u32;
        for fact in self.facts.iter() {
            if fact.id.0 >= self.next_fact
                || fact.text.0 >= texts
                || (fact.entity.0 != NONE_U32 && fact.entity.0 >= self.next_entity)
                || (fact.revises.0 != NONE_U32 && fact.revises.0 >= self.next_fact)
                || fact.vector != NONE_U32
                || fact.kind != 0
            {
                return Err(Error::Corrupt("fact record references out of range"));
            }
        }
        let mut visited = alloc::vec![false; self.tag_lists.chunks()];
        for aux in self.fact_aux.iter() {
            if aux.id.0 >= self.next_fact {
                return Err(Error::Corrupt("aux record references out of range"));
            }
            self.tag_lists.validate_chain(&aux.tags, &mut visited)?;
            for chunk in self.tag_lists.iter(&aux.tags) {
                if !chunk.len().is_multiple_of(4) {
                    return Err(Error::Corrupt("tag list is not a term-id sequence"));
                }
                for raw in chunk.chunks_exact(4) {
                    if u32::from_be_bytes(raw.try_into().unwrap()) >= terms {
                        return Err(Error::Corrupt("tag term out of range"));
                    }
                }
            }
        }
        if self.tag_lists.orphan_count(&visited) != 0 {
            return Err(Error::Corrupt("tag pool has orphan chunks"));
        }
        for entity in self.entities.iter() {
            if entity.id.0 >= self.next_entity
                || entity.name.0 >= texts
                || entity.name_term.0 >= terms
            {
                return Err(Error::Corrupt("entity record references out of range"));
            }
        }
        for by_name in self.by_name.iter() {
            if by_name.name_term.0 >= terms || !self.entities.contains(&by_name.id.0.to_be_bytes())
            {
                return Err(Error::Corrupt("by-name record references out of range"));
            }
        }
        for arena in [&self.edges_out, &self.edges_in] {
            for edge in arena.iter() {
                if edge.a.0 >= self.next_entity
                    || edge.b.0 >= self.next_entity
                    || edge.rel.0 >= terms
                    || (edge.fact.0 != NONE_U32 && edge.fact.0 >= self.next_fact)
                    || !self.entities.contains(&edge.a.0.to_be_bytes())
                    || !self.entities.contains(&edge.b.0.to_be_bytes())
                {
                    return Err(Error::Corrupt("edge record references out of range"));
                }
            }
        }
        for slot in self.temporal.iter() {
            if slot.fact.0 >= self.next_fact {
                return Err(Error::Corrupt("temporal record references out of range"));
            }
        }
        Ok(())
    }
}
