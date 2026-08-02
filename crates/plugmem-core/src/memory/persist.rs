//! Snapshot composition: the engine's state as container sections and the
//! validated load path.
//!
//! Saving concatenates every structure's canonical dump into the
//! [`snapshot`](crate::snapshot) container. Loading is the untrusted-input
//! side: after the container's structural validation (checksums are checked
//! on demand via [`Snapshot::scrub`](crate::snapshot::Snapshot::scrub), not
//! at load), every structure validates its own image, chunk chains are walked with shared
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
use crate::id::{FactId, NONE_U32};
use crate::index::IdListIndex;
use crate::index::bm25::Bm25Index;
use crate::index::hnsw::HnswGraph;
use crate::index::postings::PostingStore;
use crate::index::varint::decode_u32;
use crate::index::vecpool::VecPool;
use crate::memory::FactFault;
use crate::memory::migrations::{self, STATE_LEN};
use crate::model::{
    EntityRecord, FactAux, FactRecord, TemporalSlot, VALID_TO_OPEN, edge_history_key, edge_key,
};
use crate::snapshot::{Prefix, SectionMeta, Snapshot, SnapshotSink, build_prefix, pad_len};
use xxhash_rust::xxh3::Xxh3;

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
    // 9..=12 and 46..=49 were the edge sections before the time-ordered
    // history layout; see `migrations::legacy_kind`.
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
    pub const VEC_POOL: u16 = 37;
    pub const HNSW_META: u16 = 38;
    pub const HNSW_LEVEL0: u16 = 39;
    pub const HNSW_UPPER_META: u16 = 40;
    pub const HNSW_UPPER_POOL: u16 = 41;
    pub const HNSW_LISTS_META: u16 = 42;
    pub const HNSW_LISTS_POOL: u16 = 43;
    pub const METAS_INDEX: u16 = 44;
    pub const METAS_POOL: u16 = 45;
    /// Current edges carrying their open version's identity.
    pub const EDGES_OUT_META: u16 = 50;
    pub const EDGES_OUT_POOL: u16 = 51;
    pub const EDGES_IN_META: u16 = 52;
    pub const EDGES_IN_POOL: u16 = 53;
    /// Edge history keyed `[a | valid_from | edge]`.
    pub const EDGE_HIST_OUT_META: u16 = 54;
    pub const EDGE_HIST_OUT_POOL: u16 = 55;
    pub const EDGE_HIST_IN_META: u16 = 56;
    pub const EDGE_HIST_IN_POOL: u16 = 57;
}

/// The callback [`Memory::emit_sections_from`] drives once per snapshot
/// section: the section `kind` and the byte pieces whose concatenation is its
/// body.
type SectionFn<'f> = dyn FnMut(u16, &[&[u8]]) -> Result<(), Error> + 'f;

/// The engine structures a snapshot emit reads for the *rebuildable* sections —
/// everything `maintain` recompacts. Bundled behind references so one emit path
/// serves both the live engine (`self`'s own structures, [`Memory::sections`])
/// and the disk-first rebuild (freshly rebuilt metadata + graph, with the two
/// big pools borrowing a `Scratch`). The ride-through structures —
/// the interner, the by-name index, the edges, and the id counters — are read
/// straight from `self` in [`Memory::emit_sections_from`]; `maintain` never
/// touches them, so they are the same on both paths.
pub(crate) struct Sections<'r, 'a> {
    pub(crate) facts: &'r Arena<'a, FactRecord>,
    pub(crate) fact_aux: &'r Arena<'a, FactAux>,
    pub(crate) entities: &'r Arena<'a, EntityRecord>,
    pub(crate) temporal: &'r Arena<'a, TemporalSlot>,
    pub(crate) texts: &'r BlobHeap<'a>,
    pub(crate) metas: &'r BlobHeap<'a>,
    pub(crate) tag_lists: &'r ChunkPool<'a>,
    pub(crate) bm25: &'r Bm25Index<'a>,
    pub(crate) tags_idx: &'r IdListIndex<'a>,
    pub(crate) entity_facts: &'r IdListIndex<'a>,
    pub(crate) vecs: &'r VecPool<'a>,
    pub(crate) hnsw: &'r HnswGraph<'a>,
}

/// Dumps an arena as its `(meta, pool)` section pair.
fn arena_sections<T: Slot>(a: &Arena<'_, T>) -> (Vec<u8>, Vec<u8>) {
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

/// The eight current-format edge sections.
struct EdgeSections<'a> {
    out_meta: &'a [u8],
    out_pool: &'a [u8],
    in_meta: &'a [u8],
    in_pool: &'a [u8],
    hist_out_meta: &'a [u8],
    hist_out_pool: &'a [u8],
    hist_in_meta: &'a [u8],
    hist_in_pool: &'a [u8],
}

/// Collects the current-format edge sections, or `None` when the image has
/// none of them — an empty database, or one written before the time-ordered
/// history layout, which [`Memory::migrate_edges`] then rebuilds. Present but
/// incomplete is corruption: the eight sections are written together.
fn edge_sections<'a>(snap: &Snapshot<'a>) -> Result<Option<EdgeSections<'a>>, Error> {
    const KINDS: [u16; 8] = [
        kind::EDGES_OUT_META,
        kind::EDGES_OUT_POOL,
        kind::EDGES_IN_META,
        kind::EDGES_IN_POOL,
        kind::EDGE_HIST_OUT_META,
        kind::EDGE_HIST_OUT_POOL,
        kind::EDGE_HIST_IN_META,
        kind::EDGE_HIST_IN_POOL,
    ];
    let found = KINDS.map(|k| snap.section(k));
    if found.iter().all(Option::is_none) {
        return Ok(None);
    }
    let [
        out_meta,
        out_pool,
        in_meta,
        in_pool,
        hist_out_meta,
        hist_out_pool,
        hist_in_meta,
        hist_in_pool,
    ] = found.map(|s| s.ok_or(Error::Corrupt("snapshot has incomplete edge sections")));
    Ok(Some(EdgeSections {
        out_meta: out_meta?,
        out_pool: out_pool?,
        in_meta: in_meta?,
        in_pool: in_pool?,
        hist_out_meta: hist_out_meta?,
        hist_out_pool: hist_out_pool?,
        hist_in_meta: hist_in_meta?,
        hist_in_pool: hist_in_pool?,
    }))
}

impl<'a, const TF: bool> PostingStore<'a, TF> {
    /// Dumps the store's four sections.
    pub(crate) fn dump_sections(&self) -> [Vec<u8>; 4] {
        let (hm, hp) = (self.handles_meta(), self.handles_pool());
        let (cm, cp) = (self.chunks_meta(), self.chunks_pool());
        [hm, hp, cm, cp]
    }

    /// Rebuilds a store from its sections and validates every list: chain
    /// walks over a shared visited map, full entry decode (well-formed
    /// varints, strictly ascending ids without overflow), `count`/`last`
    /// agreement, and no orphan chunks. Owned path — the parts are copied
    /// (`'static`); see [`PostingStore::load_sections_borrowed`] for the
    /// zero-copy sibling.
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
        Self::validate_lists(&handles, &pool)?;
        Ok(Self::from_parts(handles, pool))
    }

    /// Zero-copy sibling of [`PostingStore::load_sections`]: the handle
    /// arena pool and the chunk pool borrow their mmap'd sections
    /// Same validation; the lifetime ties the store to `hp`
    /// and `cp`.
    pub(crate) fn load_sections_borrowed(
        shards: usize,
        max_bytes: usize,
        hm: &[u8],
        hp: &'a [u8],
        cm: &[u8],
        cp: &'a [u8],
    ) -> Result<Self, Error> {
        let handles = Arena::<crate::index::postings::IdListSlot>::load_borrowed(
            ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(max_bytes),
            hm,
            hp,
        )?;
        let pool = ChunkPool::load_borrowed(ChunkPoolCfg::new().with_max_bytes(max_bytes), cm, cp)?;
        Self::validate_lists(&handles, &pool)?;
        Ok(Self::from_parts(handles, pool))
    }

    /// The shared list validation, over already-built parts: chain walks,
    /// full entry decode, `count`/`last` agreement, no orphan chunks. The
    /// only difference between the owned and borrowed load paths is how
    /// `handles`/`pool` were constructed, so both funnel through here.
    fn validate_lists(
        handles: &Arena<'_, crate::index::postings::IdListSlot>,
        pool: &ChunkPool<'_>,
    ) -> Result<(), Error> {
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
        Ok(())
    }
}

impl<'a> Bm25Index<'a> {
    /// The six BM25 sections as `(kind, bytes)` pairs, in canonical order.
    fn dump_pairs(&self) -> [(u16, Vec<u8>); 6] {
        let [hm, hp, cm, cp] = self.postings().dump_sections();
        let (dm, dp) = arena_sections(self.doc_len_arena());
        [
            (kind::BM25_HANDLES_META, hm),
            (kind::BM25_HANDLES_POOL, hp),
            (kind::BM25_CHUNKS_META, cm),
            (kind::BM25_CHUNKS_POOL, cp),
            (kind::BM25_DOCLEN_META, dm),
            (kind::BM25_DOCLEN_POOL, dp),
        ]
    }

    /// Owned load: postings and per-document lengths are copied
    /// (`'static`).
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
        Self::assemble(postings, doc_len, snap)
    }

    /// Zero-copy sibling of [`Bm25Index::load_from`]: the postings and
    /// doc-length pools borrow their mmap'd sections.
    fn load_from_borrowed(snap: &Snapshot<'a>, cfg: &Config) -> Result<Self, Error> {
        let postings = PostingStore::<true>::load_sections_borrowed(
            cfg.shards_postings,
            cfg.max_bytes,
            section(snap, kind::BM25_HANDLES_META)?,
            section(snap, kind::BM25_HANDLES_POOL)?,
            section(snap, kind::BM25_CHUNKS_META)?,
            section(snap, kind::BM25_CHUNKS_POOL)?,
        )?;
        let doc_len = Arena::load_borrowed(
            ArenaCfg::new(cfg.shards_postings, ShardMode::Uniform).with_max_bytes(cfg.max_bytes),
            section(snap, kind::BM25_DOCLEN_META)?,
            section(snap, kind::BM25_DOCLEN_POOL)?,
        )?;
        Self::assemble(postings, doc_len, snap)
    }

    /// Reconciles the corpus totals from the engine-state section and
    /// assembles the index. Shared by both load paths (reads only the tiny
    /// state section, so it ties nothing).
    fn assemble(
        postings: PostingStore<'a, true>,
        doc_len: Arena<'a, crate::index::bm25::DocLenSlot>,
        snap: &Snapshot<'_>,
    ) -> Result<Self, Error> {
        let state = section(snap, kind::ENGINE_STATE)?;
        // Validates the width for every layout this crate has written; the
        // corpus totals sit in the prefix all of them share.
        migrations::decode_engine_state(state)?;
        let total_docs = u64::from_le_bytes(state[8..16].try_into().unwrap());
        let total_len = u64::from_le_bytes(state[16..24].try_into().unwrap());
        if total_docs != doc_len.len() as u64 {
            return Err(Error::Corrupt("bm25 document total disagrees with doc_len"));
        }
        Ok(Self::from_parts(postings, doc_len, total_docs, total_len))
    }
}

impl<'a> Memory<'a> {
    /// A [`Sections`] view over this engine's own structures — the source for
    /// an ordinary snapshot.
    pub(super) fn sections(&self) -> Sections<'_, 'a> {
        Sections {
            facts: &self.facts,
            fact_aux: &self.fact_aux,
            entities: &self.entities,
            temporal: &self.temporal,
            texts: &self.texts,
            metas: &self.metas,
            tag_lists: &self.tag_lists,
            bm25: &self.bm25,
            tags_idx: &self.tags_idx,
            entity_facts: &self.entity_facts,
            vecs: &self.vecs,
            hnsw: &self.hnsw,
        }
    }

    /// Emits every snapshot section in canonical order, handing each to `f`
    /// as its `kind` and one-or-more byte pieces (concatenated = the section
    /// body). The rebuildable sections come from `s`; the ride-through ones
    /// (by-name, edges, interner, id counters — untouched by `maintain`) come
    /// from `self`. Most sections are a single owned buffer produced on the fly
    /// and dropped after `f` returns; the dominant vector pool is handed as its
    /// two borrowed pieces (`base`, `tail`) with no owned copy. Called twice by
    /// the writer (size/hash pass, then write pass), so it must be deterministic
    /// and side-effect free.
    fn emit_sections_from(&self, s: &Sections<'_, '_>, f: &mut SectionFn<'_>) -> Result<(), Error> {
        for (mk, pk, arena) in [
            (kind::FACTS_META, kind::FACTS_POOL, arena_sections(s.facts)),
            (kind::AUX_META, kind::AUX_POOL, arena_sections(s.fact_aux)),
            (
                kind::ENTITIES_META,
                kind::ENTITIES_POOL,
                arena_sections(s.entities),
            ),
            (
                kind::BY_NAME_META,
                kind::BY_NAME_POOL,
                arena_sections(&self.by_name),
            ),
            (
                kind::EDGES_OUT_META,
                kind::EDGES_OUT_POOL,
                arena_sections(&self.edges_out),
            ),
            (
                kind::EDGES_IN_META,
                kind::EDGES_IN_POOL,
                arena_sections(&self.edges_in),
            ),
            (
                kind::EDGE_HIST_OUT_META,
                kind::EDGE_HIST_OUT_POOL,
                arena_sections(&self.edges_hist_out),
            ),
            (
                kind::EDGE_HIST_IN_META,
                kind::EDGE_HIST_IN_POOL,
                arena_sections(&self.edges_hist_in),
            ),
            (
                kind::TEMPORAL_META,
                kind::TEMPORAL_POOL,
                arena_sections(s.temporal),
            ),
        ] {
            let (m, p) = arena;
            f(mk, &[&m])?;
            f(pk, &[&p])?;
        }
        let (mut i, mut p) = (Vec::new(), Vec::new());
        s.texts.dump_index(&mut i);
        s.texts.dump_pool(&mut p);
        f(kind::TEXTS_INDEX, &[&i])?;
        f(kind::TEXTS_POOL, &[&p])?;
        let (mut i, mut p) = (Vec::new(), Vec::new());
        s.metas.dump_index(&mut i);
        s.metas.dump_pool(&mut p);
        f(kind::METAS_INDEX, &[&i])?;
        f(kind::METAS_POOL, &[&p])?;
        let (mut i, mut p, mut t) = (Vec::new(), Vec::new(), Vec::new());
        self.terms.dump_index(&mut i);
        self.terms.dump_pool(&mut p);
        self.terms.dump_table(&mut t);
        f(kind::TERMS_INDEX, &[&i])?;
        f(kind::TERMS_POOL, &[&p])?;
        f(kind::TERMS_TABLE, &[&t])?;
        let (mut m, mut p) = (Vec::new(), Vec::new());
        s.tag_lists.dump_meta(&mut m);
        s.tag_lists.dump_pool(&mut p);
        f(kind::TAG_LISTS_META, &[&m])?;
        f(kind::TAG_LISTS_POOL, &[&p])?;
        for (k, bytes) in s.bm25.dump_pairs() {
            f(k, &[&bytes])?;
        }
        let [hm, hp, cm, cp] = s.tags_idx.dump_sections();
        f(kind::TAGS_HANDLES_META, &[&hm])?;
        f(kind::TAGS_HANDLES_POOL, &[&hp])?;
        f(kind::TAGS_CHUNKS_META, &[&cm])?;
        f(kind::TAGS_CHUNKS_POOL, &[&cp])?;
        let [hm, hp, cm, cp] = s.entity_facts.dump_sections();
        f(kind::ENTFACTS_HANDLES_META, &[&hm])?;
        f(kind::ENTFACTS_HANDLES_POOL, &[&hp])?;
        f(kind::ENTFACTS_CHUNKS_META, &[&cm])?;
        f(kind::ENTFACTS_CHUNKS_POOL, &[&cp])?;
        let mut state = Vec::with_capacity(STATE_LEN);
        state.extend_from_slice(&self.next_fact.to_le_bytes());
        state.extend_from_slice(&self.next_entity.to_le_bytes());
        state.extend_from_slice(&s.bm25.docs().to_le_bytes());
        state.extend_from_slice(&s.bm25.total_len().to_le_bytes());
        state.extend_from_slice(&self.bm25_tokenizer_version.to_le_bytes());
        state.extend_from_slice(&0u32.to_le_bytes());
        state.extend_from_slice(&self.next_edge.to_le_bytes());
        state.extend_from_slice(&0u32.to_le_bytes());
        f(kind::ENGINE_STATE, &[&state])?;
        // The vector pool is one flat section (empty when dim is 0), streamed
        // as its two borrowed pieces so the dominant pool needs no owned copy.
        f(kind::VEC_POOL, &s.vecs.pieces())?;
        // The HNSW graph: header, flat level-0 blocks, and the upper-level
        // arena + list pool (all empty in the flat regime).
        f(kind::HNSW_META, &[&s.hnsw.dump_meta()])?;
        f(kind::HNSW_LEVEL0, &[&s.hnsw.dump_level0()])?;
        let [um, up, lm, lp] = s.hnsw.dump_upper();
        f(kind::HNSW_UPPER_META, &[&um])?;
        f(kind::HNSW_UPPER_POOL, &[&up])?;
        f(kind::HNSW_LISTS_META, &[&lm])?;
        f(kind::HNSW_LISTS_POOL, &[&lp])?;
        Ok(())
    }

    /// Streams the whole engine into snapshot-container bytes through `sink`,
    /// never materializing the full image: a first pass computes
    /// each section's length and checksum, the header+table prefix is written,
    /// then a second pass streams the section bodies (the dominant vector pool
    /// straight from its borrowed pieces) while a running hash accumulates the
    /// file checksum, patched into the header at the end. Deterministic and
    /// canonical — byte-identical to [`Memory::snapshot_bytes`].
    ///
    /// # Errors
    ///
    /// Propagates whatever `sink` reports (e.g. an I/O error from a file sink).
    pub fn write_snapshot_to(&self, created_at: u64, sink: impl SnapshotSink) -> Result<(), Error> {
        self.write_snapshot_with(&self.sections(), created_at, sink)
    }

    /// The snapshot writer over an explicit [`Sections`] source — the shared
    /// core of [`Memory::write_snapshot_to`] (which passes `self`'s own
    /// sections) and the disk-first rebuild (which passes freshly rebuilt
    /// metadata with the big pools borrowing a `Scratch`). Since
    /// both drive the *same* emit, the disk-first output is byte-identical to a
    /// snapshot taken after an in-RAM `maintain`.
    pub(crate) fn write_snapshot_with(
        &self,
        s: &Sections<'_, '_>,
        created_at: u64,
        mut sink: impl SnapshotSink,
    ) -> Result<(), Error> {
        let mut cfg_bytes = Vec::new();
        self.cfg.encode(&mut cfg_bytes);
        let flags = if self.cfg.dim > 0 {
            crate::snapshot::FLAG_VECTORS
        } else {
            0
        };

        // Pass 1: (kind, len, hash) for every section — small and bounded.
        let mut metas: Vec<SectionMeta> = Vec::new();
        self.emit_sections_from(s, &mut |kind, pieces| {
            let mut h = Xxh3::new();
            let mut len = 0u64;
            for p in pieces {
                h.update(p);
                len += p.len() as u64;
            }
            metas.push(SectionMeta {
                kind,
                len,
                hash: h.digest(),
            });
            Ok(())
        })?;

        let Prefix {
            bytes: prefix,
            offsets,
            file_len: _,
        } = build_prefix(
            &cfg_bytes,
            flags,
            created_at,
            env!("CARGO_PKG_VERSION"),
            &metas,
        );
        sink.write(&prefix)?;
        let mut file_hash = Xxh3::new();
        file_hash.update(&prefix);

        // Pass 2: section bodies + alignment padding, into sink and hash.
        let zero = [0u8; 64]; // ALIGN — padding is always shorter than this.
        let mut idx = 0usize;
        self.emit_sections_from(s, &mut |_, pieces| {
            for p in pieces {
                sink.write(p)?;
                file_hash.update(p);
            }
            let n = pad_len(offsets[idx], metas[idx].len);
            sink.write(&zero[..n])?;
            file_hash.update(&zero[..n]);
            idx += 1;
            Ok(())
        })?;

        sink.patch(
            crate::snapshot::FILE_HASH_OFFSET,
            &file_hash.digest().to_le_bytes(),
        )
    }

    /// Serializes the whole engine into snapshot-container bytes.
    /// Deterministic and canonical: save → load → save is byte-identical.
    /// A thin wrapper over [`Memory::write_snapshot_to`] into a `Vec`; large
    /// databases should prefer streaming into a file sink.
    pub fn snapshot_bytes(&self, created_at: u64) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_snapshot_to(created_at, &mut out)
            .expect("writing a snapshot into a Vec is infallible");
        out
    }

    /// Writes a full snapshot and clears the journal.
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
    /// module docs for the validation inventory). Owned path: every
    /// section is copied into the arenas, so the returned engine borrows
    /// nothing from `bytes` and is a `Memory<'static>`.
    pub(super) fn load_snapshot(bytes: &[u8], cfg: Config) -> Result<Self, Error> {
        cfg.validate()?;
        let snap = Snapshot::parse(bytes)?;
        let cfg = Self::reconcile_config(&snap, cfg)?;
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
        // Absent edge sections mean an image older than the time-ordered
        // layout (or an empty database); `finish_load` migrates it.
        if let Some(edges) = edge_sections(&snap)? {
            mem.edges_out = Arena::load(ord(cfg.shards_edges), edges.out_meta, edges.out_pool)?;
            mem.edges_in = Arena::load(ord(cfg.shards_edges), edges.in_meta, edges.in_pool)?;
            mem.edges_hist_out = Arena::load(
                ord(cfg.shards_edges),
                edges.hist_out_meta,
                edges.hist_out_pool,
            )?;
            mem.edges_hist_in = Arena::load(
                ord(cfg.shards_edges),
                edges.hist_in_meta,
                edges.hist_in_pool,
            )?;
        }
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
        mem.metas = BlobHeap::load(
            blob,
            section(&snap, kind::METAS_INDEX)?,
            section(&snap, kind::METAS_POOL)?,
        )?;
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
        mem.vecs = VecPool::from_parts(cfg.dim, cfg.max_bytes, section(&snap, kind::VEC_POOL)?)?;
        mem.hnsw = crate::index::hnsw::HnswGraph::from_parts(
            cfg.hnsw_m,
            cfg.hnsw_m0,
            cfg.max_bytes,
            section(&snap, kind::HNSW_META)?,
            section(&snap, kind::HNSW_LEVEL0)?,
            section(&snap, kind::HNSW_UPPER_META)?,
            section(&snap, kind::HNSW_UPPER_POOL)?,
            section(&snap, kind::HNSW_LISTS_META)?,
            section(&snap, kind::HNSW_LISTS_POOL)?,
        )?;
        Self::finish_load(mem, &snap)
    }

    /// Zero-copy sibling of [`Memory::load_snapshot`]: the large byte
    /// pools (arenas, blob heaps, chunk pools, term dictionary, vectors,
    /// upper HNSW lists) *borrow* their sections straight out of `bytes`
    /// (an mmap'd snapshot), so opening an 8 GiB database residents only
    /// the pages actually touched. Small metadata is still
    /// rebuilt owned. The lifetime ties the engine to `bytes`; the handle
    /// is read-only, so copy-on-write never fires.
    pub(super) fn load_snapshot_borrowed(bytes: &'a [u8], cfg: Config) -> Result<Self, Error> {
        cfg.validate()?;
        let snap = Snapshot::parse(bytes)?;
        let cfg = Self::reconcile_config(&snap, cfg)?;
        let mut mem = Self::new(cfg)?;
        let cfg = &mem.cfg;
        let uni =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Uniform).with_max_bytes(cfg.max_bytes);
        let ord =
            |shards: usize| ArenaCfg::new(shards, ShardMode::Ordered).with_max_bytes(cfg.max_bytes);
        let blob = BlobHeapCfg::new()
            .with_max_bytes(cfg.max_bytes)
            .with_max_blob(cfg.max_blob);
        mem.facts = Arena::load_borrowed(
            uni(cfg.shards_facts),
            section(&snap, kind::FACTS_META)?,
            section(&snap, kind::FACTS_POOL)?,
        )?;
        mem.fact_aux = Arena::load_borrowed(
            uni(cfg.shards_facts),
            section(&snap, kind::AUX_META)?,
            section(&snap, kind::AUX_POOL)?,
        )?;
        mem.entities = Arena::load_borrowed(
            uni(cfg.shards_entities),
            section(&snap, kind::ENTITIES_META)?,
            section(&snap, kind::ENTITIES_POOL)?,
        )?;
        mem.by_name = Arena::load_borrowed(
            ord(cfg.shards_entities),
            section(&snap, kind::BY_NAME_META)?,
            section(&snap, kind::BY_NAME_POOL)?,
        )?;
        // A legacy image's edges are rebuilt owned by `finish_load` rather
        // than borrowed: their bytes have to be re-keyed, so there is nothing
        // to map. Everything else still borrows.
        if let Some(edges) = edge_sections(&snap)? {
            mem.edges_out =
                Arena::load_borrowed(ord(cfg.shards_edges), edges.out_meta, edges.out_pool)?;
            mem.edges_in =
                Arena::load_borrowed(ord(cfg.shards_edges), edges.in_meta, edges.in_pool)?;
            mem.edges_hist_out = Arena::load_borrowed(
                ord(cfg.shards_edges),
                edges.hist_out_meta,
                edges.hist_out_pool,
            )?;
            mem.edges_hist_in = Arena::load_borrowed(
                ord(cfg.shards_edges),
                edges.hist_in_meta,
                edges.hist_in_pool,
            )?;
        }
        mem.temporal = Arena::load_borrowed(
            ord(cfg.shards_temporal),
            section(&snap, kind::TEMPORAL_META)?,
            section(&snap, kind::TEMPORAL_POOL)?,
        )?;
        mem.texts = BlobHeap::load_borrowed(
            blob,
            section(&snap, kind::TEXTS_INDEX)?,
            section(&snap, kind::TEXTS_POOL)?,
        )?;
        mem.metas = BlobHeap::load_borrowed(
            blob,
            section(&snap, kind::METAS_INDEX)?,
            section(&snap, kind::METAS_POOL)?,
        )?;
        mem.terms = Interner::load_borrowed(
            blob,
            section(&snap, kind::TERMS_INDEX)?,
            section(&snap, kind::TERMS_POOL)?,
            section(&snap, kind::TERMS_TABLE)?,
        )?;
        mem.tag_lists = ChunkPool::load_borrowed(
            ChunkPoolCfg::new().with_max_bytes(cfg.max_bytes),
            section(&snap, kind::TAG_LISTS_META)?,
            section(&snap, kind::TAG_LISTS_POOL)?,
        )?;
        mem.bm25 = Bm25Index::load_from_borrowed(&snap, cfg)?;
        mem.tags_idx = IdListIndex::load_sections_borrowed(
            cfg.shards_postings,
            cfg.max_bytes,
            section(&snap, kind::TAGS_HANDLES_META)?,
            section(&snap, kind::TAGS_HANDLES_POOL)?,
            section(&snap, kind::TAGS_CHUNKS_META)?,
            section(&snap, kind::TAGS_CHUNKS_POOL)?,
        )?;
        mem.entity_facts = IdListIndex::load_sections_borrowed(
            cfg.shards_entities,
            cfg.max_bytes,
            section(&snap, kind::ENTFACTS_HANDLES_META)?,
            section(&snap, kind::ENTFACTS_HANDLES_POOL)?,
            section(&snap, kind::ENTFACTS_CHUNKS_META)?,
            section(&snap, kind::ENTFACTS_CHUNKS_POOL)?,
        )?;
        mem.vecs =
            VecPool::from_parts_borrowed(cfg.dim, cfg.max_bytes, section(&snap, kind::VEC_POOL)?)?;
        mem.hnsw = crate::index::hnsw::HnswGraph::from_parts_borrowed(
            cfg.hnsw_m,
            cfg.hnsw_m0,
            cfg.max_bytes,
            section(&snap, kind::HNSW_META)?,
            section(&snap, kind::HNSW_LEVEL0)?,
            section(&snap, kind::HNSW_UPPER_META)?,
            section(&snap, kind::HNSW_UPPER_POOL)?,
            section(&snap, kind::HNSW_LISTS_META)?,
            section(&snap, kind::HNSW_LISTS_POOL)?,
        )?;
        Self::finish_load(mem, &snap)
    }

    /// Checks the stored config against the caller's (structural fields
    /// must match; tuning fields follow the caller) and adopts the
    /// snapshot's lineage identity. Shared by both load paths.
    fn reconcile_config(snap: &Snapshot<'_>, mut cfg: Config) -> Result<Config, Error> {
        let stored = Config::decode(snap.config())?;
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
        // The lineage identity is the snapshot's, not the caller's: a
        // caller passing 0 adopts the stored uuid; a nonzero caller value
        // is an assertion "this must be that database" and must match.
        if cfg.db_uuid != 0 && stored.db_uuid != cfg.db_uuid {
            return Err(Error::ConfigMismatch("stored db_uuid differs"));
        }
        cfg.db_uuid = stored.db_uuid;
        Ok(cfg)
    }

    /// Finishes a load once every section is in place: reads the id
    /// counters, checks they cover the record counts, and range-validates
    /// references. Deliberately does **not** scan the large byte pools —
    /// stored-text UTF-8 and the vector fact↔slot bijection are deferred to
    /// [`Memory::verify`], so an overlay/read-only open faults
    /// in only the metadata, not the text or vector pools. The accessors stay
    /// panic-free on any bytes regardless (checked `from_utf8`, bounds-checked
    /// vector reads). Shared by both load paths.
    fn finish_load(mut mem: Self, snap: &Snapshot<'_>) -> Result<Self, Error> {
        let state = migrations::decode_engine_state(section(snap, kind::ENGINE_STATE)?)?;
        mem.next_fact = state.next_fact;
        mem.next_entity = state.next_entity;
        mem.bm25_tokenizer_version = state.bm25_tokenizer_version;
        mem.next_edge = state.next_edge;
        // Current-format edge sections were absent: either the image predates
        // the time-ordered layout, or it has no edges at all.
        let cfg = mem.cfg.clone();
        if mem.edges_hist_out.is_empty() && mem.edges_out.is_empty() {
            mem.migrate_edges(snap, &cfg)?;
        }
        let derived_next_edge = mem
            .edges_hist_out
            .iter()
            .map(|edge| edge.edge.0)
            .max()
            .map(|edge| edge.saturating_add(1))
            .unwrap_or(0);
        if mem.next_edge < derived_next_edge {
            if !state.predates_edge_versions {
                return Err(Error::Corrupt("engine edge id counter below record count"));
            }
            mem.next_edge = derived_next_edge;
        }
        if (mem.next_fact as usize) < mem.facts.len()
            || (mem.next_entity as usize) < mem.entities.len()
        {
            return Err(Error::Corrupt("engine id counters below record counts"));
        }
        mem.tombstones = mem.facts.iter().filter(|fact| fact.is_tombstone()).count();
        mem.validate_references()?;
        Ok(mem)
    }

    /// Range-checks every stored id so the engine's panicking accessors
    /// are sound on loaded data (module docs). O(records) — the price of
    /// panic-freedom on hostile input, linear and cache-friendly. Does **not**
    /// touch the large text or vector byte pools: stored-text UTF-8 and the
    /// vector fact↔slot bijection are deferred to [`Memory::verify`]
    /// so an overlay/read-only open faults in only the
    /// metadata. The accessors that read those pools are panic-free on any
    /// bytes on their own (checked `from_utf8`, bounds-checked slot reads).
    fn validate_references(&self) -> Result<(), Error> {
        let texts = self.texts.len() as u32;
        let terms = self.terms.len() as u32;
        // The HNSW graph is validated against the pool length it indexes
        // (owned level0 + small upper lists; it does not read vector slots),
        // so this stays cheap and eager.
        self.hnsw.validate(&self.vecs)?;
        for fact in self.facts.iter() {
            if fact.id.0 >= self.next_fact
                || fact.text.0 >= texts
                || (fact.entity.0 != NONE_U32 && fact.entity.0 >= self.next_entity)
                || (fact.revises.0 != NONE_U32 && fact.revises.0 >= self.next_fact)
                || fact.kind != 0
            {
                return Err(Error::Corrupt("fact record references out of range"));
            }
            // The has-vector bijection touches the vector pool and is deferred
            // to `verify()`; the cheap direction stays — a fact without the
            // flag must carry no slot.
            if !fact.has_vector() && fact.vector != NONE_U32 {
                return Err(Error::Corrupt("fact without a vector flag carries a slot"));
            }
        }
        let metas = self.metas.len() as u32;
        let mut visited = alloc::vec![false; self.tag_lists.chunks()];
        for aux in self.fact_aux.iter() {
            if aux.id.0 >= self.next_fact || (aux.meta.0 != NONE_U32 && aux.meta.0 >= metas) {
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
                    || edge.edge.0 >= self.next_edge
                    || (edge.fact.0 != NONE_U32 && edge.fact.0 >= self.next_fact)
                    || !self.entities.contains(&edge.a.0.to_be_bytes())
                    || !self.entities.contains(&edge.b.0.to_be_bytes())
                {
                    return Err(Error::Corrupt("edge record references out of range"));
                }
            }
        }
        if self.edges_out.len() != self.edges_in.len()
            || self.edges_hist_out.len() != self.edges_hist_in.len()
        {
            return Err(Error::Corrupt("edge mirrors disagree"));
        }
        for edge in self.edges_out.iter() {
            if !self.edges_in.contains(&edge_key(edge.b, edge.rel, edge.a)) {
                return Err(Error::Corrupt("edge mirrors disagree"));
            }
            // The current slot names its open version directly, so this is a
            // point lookup rather than a search through the triple's history.
            let version = self
                .edges_hist_out
                .get(&edge_history_key(edge.a, edge.valid_from, edge.edge))
                .ok_or(Error::Corrupt("current edge has no history record"))?;
            if version.valid_to != VALID_TO_OPEN
                || version.rel != edge.rel
                || version.b != edge.b
                || version.fact != edge.fact
            {
                return Err(Error::Corrupt("current edge disagrees with its history"));
            }
        }
        for edge in self.edges_hist_out.iter() {
            if edge.a.0 >= self.next_entity
                || edge.b.0 >= self.next_entity
                || edge.edge.0 >= self.next_edge
                || edge.rel.0 >= terms
                || edge.kind != 0
                || edge.valid_from > edge.valid_to
                || (edge.fact.0 != NONE_U32 && edge.fact.0 >= self.next_fact)
                || !self.entities.contains(&edge.a.0.to_be_bytes())
                || !self.entities.contains(&edge.b.0.to_be_bytes())
            {
                return Err(Error::Corrupt("edge history references out of range"));
            }
            if !self
                .edges_hist_in
                .contains(&edge_history_key(edge.b, edge.valid_from, edge.edge))
            {
                return Err(Error::Corrupt("edge history mirrors disagree"));
            }
            // An open version must be reachable as a current edge: the two
            // structures are one fact stored twice, and recall trusts the
            // current graph to be exactly the open versions.
            if edge.valid_to == VALID_TO_OPEN
                && !self.edges_out.contains(&edge_key(edge.a, edge.rel, edge.b))
            {
                return Err(Error::Corrupt("open edge version is not a current edge"));
            }
        }
        for slot in self.temporal.iter() {
            if slot.fact.0 >= self.next_fact {
                return Err(Error::Corrupt("temporal record references out of range"));
            }
        }
        Ok(())
    }

    /// Runs the integrity checks that `open` **defers** for speed and memory
    /// — the on-demand equivalent of SQLite's `integrity_check`.
    ///
    /// A load (owned, overlay or read-only) validates only the metadata, so
    /// the large byte pools stay non-resident on an mmap'd base — an overlay
    /// open of a multi-gigabyte database faults in only what it must. This
    /// method sweeps the deferred pools and confirms the whole image is
    /// well-formed: every stored text is valid UTF-8, the vector pool is
    /// self-consistent, and facts flagged with a vector map one-to-one onto
    /// pool slots that name them back. It reads the text and vector pools in
    /// full, so it costs one linear pass over them (and residents them).
    ///
    /// Skipping it is safe: the accessors that read these pools tolerate bad
    /// bytes on their own (invalid text hides the fact, vector reads are
    /// bounds-checked), so a corrupt image never panics — `verify` only turns
    /// that latent corruption into an explicit [`Error::Corrupt`].
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for the first inconsistency found.
    pub fn verify(&self) -> Result<(), Error> {
        // Text: every stored blob is valid UTF-8. Accessors already tolerate
        // invalid text gracefully; this is the eager confirmation.
        for (_, text) in self.texts.iter() {
            if core::str::from_utf8(text).is_err() {
                return Err(Error::Corrupt("stored text is not valid UTF-8"));
            }
        }
        // Metadata: every referenced blob decodes to a well-formed key→value
        // map (bounds, UTF-8, strictly ascending unique keys). Ranges were
        // validated at load; this is the content confirmation, like the text
        // pass above.
        let mut pairs = Vec::new();
        for aux in self.fact_aux.iter() {
            if aux.meta.0 != NONE_U32 {
                crate::metadata::decode(self.metas.get(aux.meta), &mut pairs)?;
            }
        }
        // Vectors: structural self-check, then the fact↔slot bijection (each
        // HAS_VECTOR fact points at a slot that names it back; no slot is
        // orphaned).
        self.vecs.validate()?;
        let vslots = self.vecs.len() as u32;
        let mut with_vec = 0u32;
        for fact in self.facts.iter() {
            if fact.has_vector() {
                if fact.vector >= vslots || self.vecs.slot_fact(fact.vector as usize) != fact.id.0 {
                    return Err(Error::Corrupt(
                        "fact vector slot is out of range or mismatched",
                    ));
                }
                with_vec += 1;
            }
        }
        if with_vec != vslots {
            return Err(Error::Corrupt("vector pool has orphan slots"));
        }
        Ok(())
    }

    /// Attributes [`Memory::verify`]'s content checks to individual facts — the
    /// salvage predicate for `recover`. Walks every live
    /// (non-tombstone) fact and returns those whose stored text is not valid
    /// UTF-8, that are flagged with a vector whose slot is out of range or does
    /// not name the fact back, or whose metadata blob does not decode to a
    /// well-formed key→value map. It reads the text, vector and metadata pools
    /// (like `verify`), so it residents them; the accessors it uses are
    /// panic-free on any bytes. Unlike `verify`, it does not fail on the first
    /// problem — it
    /// reports each faulty fact so the caller can `forget` it and rebuild a
    /// clean image from the survivors.
    pub fn faulty_facts(&self) -> Vec<(FactId, FactFault)> {
        let vslots = self.vecs.len() as u32;
        let metas = self.metas.len() as u32;
        let mut pairs = Vec::new();
        let mut out = Vec::new();
        for i in 0..self.next_fact {
            let id = FactId(i);
            let Some(record) = self.fact(id) else {
                continue; // unknown or tombstoned
            };
            if record.is_tombstone() {
                continue;
            }
            if core::str::from_utf8(self.texts.get(record.text)).is_err() {
                out.push((id, FactFault::Text));
                continue;
            }
            if record.has_vector()
                && (record.vector >= vslots || self.vecs.slot_fact(record.vector as usize) != id.0)
            {
                out.push((id, FactFault::Vector));
                continue;
            }
            // Metadata: a referenced blob that is out of range or does not decode
            // to a well-formed key→value map. `metadata_of` hides such a fact's
            // metadata gracefully; here it becomes an explicit salvage fault.
            if let Some(aux) = self.fact_aux.get(&id.0.to_be_bytes())
                && aux.meta.0 != NONE_U32
                && (aux.meta.0 >= metas
                    || crate::metadata::decode(self.metas.get(aux.meta), &mut pairs).is_err())
            {
                out.push((id, FactFault::Metadata));
            }
        }
        out
    }
}
