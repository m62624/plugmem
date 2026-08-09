//! Reading images written by older versions of this crate.
//!
//! Every format the engine has ever written must still open. New images are
//! not expected to open in an older binary — the compatibility promise runs
//! **forwards only** — so a migration reads the old bytes once, rebuilds the
//! current structures from them, and the next checkpoint persists the result.
//! Nothing on the steady-state read path knows an old format exists.
//!
//! Keeping all of that here, rather than sprinkled through
//! [`persist`](super::persist), makes the set of supported legacy shapes
//! something you can read in one sitting:
//!
//! | image | what it lacks | how it is upgraded |
//! |---|---|---|
//! | engine state, 24 bytes | tokenizer version, edge-version counter | defaults; counters re-derived |
//! | engine state, 32 bytes | edge-version counter | re-derived from history |
//! | edges without history | the whole edge-history arena | one open version synthesized per current edge |
//! | edges keyed by triple | the time-ordered history key | every version re-keyed by `valid_from`; current edges re-derived from the open ones |
//! | 8-byte per-document BM25 records | the term-set summary | widened, marked unknown; `maintain` fills them from the postings |
//! | no tag-catalog section | bounded tag discovery index | rebuilt from authoritative tag postings and current facts |
//! | no vector-space section | identity of the model that produced vectors | left untracked; explicit reembed establishes it safely |
//!
//! The legacy record layouts live here as their own [`Slot`] types. They are
//! deliberately duplicated rather than shared with [`crate::model`]: the
//! current layouts are free to change, and a migration must keep reading the
//! bytes that were actually written.

use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, TermId, key};

use crate::config::Config;
use crate::error::Error;
use crate::id::{EdgeId, EntityId, FactId};
use crate::index::bm25::DocLenSlot;
use crate::model::{EdgeHistorySlot, VALID_TO_OPEN};
use crate::snapshot::Snapshot;

use super::Memory;

/// Section kinds this crate reads but no longer writes.
pub(super) mod legacy_kind {
    /// Current edges, `[a | rel | b]` keyed, 16-byte slots.
    pub const EDGES_OUT_META: u16 = 9;
    pub const EDGES_OUT_POOL: u16 = 10;
    pub const EDGES_IN_META: u16 = 11;
    pub const EDGES_IN_POOL: u16 = 12;
    /// Per-document BM25 records without the term-set summary, 8-byte slots.
    pub const BM25_DOCLEN_META: u16 = 26;
    pub const BM25_DOCLEN_POOL: u16 = 27;
    /// Edge history, `[a | rel | b | edge]` keyed.
    pub const EDGE_HIST_OUT_META: u16 = 46;
    pub const EDGE_HIST_OUT_POOL: u16 = 47;
    pub const EDGE_HIST_IN_META: u16 = 48;
    pub const EDGE_HIST_IN_POOL: u16 = 49;
}

/// Byte length of the original engine-state section.
const STATE_V1_LEN: usize = 24;
/// Byte length of the tokenizer-version engine-state section.
const STATE_V2_LEN: usize = 32;
/// Byte length of the current engine-state section.
pub(super) const STATE_LEN: usize = 40;

/// Offsets inside the engine-state section, each derived from the previous
/// field's width.
mod state_at {
    use core::mem::size_of;

    pub(super) const NEXT_FACT: usize = 0;
    pub(super) const NEXT_ENTITY: usize = NEXT_FACT + size_of::<u32>();
    pub(super) const BM25_DOCS: usize = NEXT_ENTITY + size_of::<u32>();
    pub(super) const BM25_TOTAL_LEN: usize = BM25_DOCS + size_of::<u64>();
    pub(super) const TOKENIZER_VERSION: usize = BM25_TOTAL_LEN + size_of::<u64>();
    pub(super) const RESERVED: usize = TOKENIZER_VERSION + size_of::<u32>();
    pub(super) const NEXT_EDGE: usize = RESERVED + size_of::<u32>();
}

/// The engine-state section, decoded from any width this crate has written.
pub(super) struct EngineState {
    pub(super) next_fact: u32,
    pub(super) next_entity: u32,
    pub(super) bm25_tokenizer_version: u32,
    pub(super) next_edge: u32,
    /// The image predates the edge-version counter, so `next_edge` is a
    /// default rather than a stored value and missing edge history is
    /// expected rather than corruption.
    pub(super) predates_edge_versions: bool,
}

/// Decodes the engine-state section, accepting every width this crate has
/// written and filling in defaults for the fields an older one omitted.
pub(super) fn decode_engine_state(bytes: &[u8]) -> Result<EngineState, Error> {
    if bytes.len() != STATE_V1_LEN && bytes.len() != STATE_V2_LEN && bytes.len() != STATE_LEN {
        return Err(Error::Corrupt("engine state section has a wrong length"));
    }
    let at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    Ok(EngineState {
        next_fact: at(state_at::NEXT_FACT),
        next_entity: at(state_at::NEXT_ENTITY),
        bm25_tokenizer_version: if bytes.len() >= STATE_V2_LEN {
            at(state_at::TOKENIZER_VERSION)
        } else {
            super::maintain::TOKENIZER_INDEX_VERSION
        },
        next_edge: if bytes.len() >= STATE_LEN {
            at(state_at::NEXT_EDGE)
        } else {
            0
        },
        predates_edge_versions: bytes.len() < STATE_LEN,
    })
}

/// A current edge as written before the slot carried its open version's
/// identity: key `[a | rel | b]`, payload `fact`.
#[derive(Clone, Copy)]
struct LegacyEdgeSlot {
    a: EntityId,
    rel: TermId,
    b: EntityId,
    fact: FactId,
}

impl Slot for LegacyEdgeSlot {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 12;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.a.0);
        key::write_u32(&mut out[4..], self.rel.0);
        key::write_u32(&mut out[8..], self.b.0);
        key::write_u32(&mut out[12..], self.fact.0);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            a: EntityId(key::read_u32(bytes)),
            rel: TermId(key::read_u32(&bytes[4..])),
            b: EntityId(key::read_u32(&bytes[8..])),
            fact: FactId(key::read_u32(&bytes[12..])),
        }
    }
}

/// An edge version as written before history was time-ordered: key
/// `[a | rel | b | edge]`, so an entity's versions were grouped by triple.
#[derive(Clone, Copy)]
struct LegacyEdgeHistorySlot {
    a: EntityId,
    rel: TermId,
    b: EntityId,
    edge: EdgeId,
    fact: FactId,
    flags: u16,
    kind: u16,
    recorded_at: u64,
    valid_from: u64,
    valid_to: u64,
}

impl Slot for LegacyEdgeHistorySlot {
    const SIZE: usize = 48;
    const KEY_LEN: usize = 16;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.a.0);
        key::write_u32(&mut out[4..], self.rel.0);
        key::write_u32(&mut out[8..], self.b.0);
        key::write_u32(&mut out[12..], self.edge.0);
        key::write_u32(&mut out[16..], self.fact.0);
        out[20..22].copy_from_slice(&self.flags.to_be_bytes());
        out[22..24].copy_from_slice(&self.kind.to_be_bytes());
        key::write_u64(&mut out[24..], self.recorded_at);
        key::write_u64(&mut out[32..], self.valid_from);
        key::write_u64(&mut out[40..], self.valid_to);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            a: EntityId(key::read_u32(bytes)),
            rel: TermId(key::read_u32(&bytes[4..])),
            b: EntityId(key::read_u32(&bytes[8..])),
            edge: EdgeId(key::read_u32(&bytes[12..])),
            fact: FactId(key::read_u32(&bytes[16..])),
            flags: u16::from_be_bytes(bytes[20..22].try_into().unwrap()),
            kind: u16::from_be_bytes(bytes[22..24].try_into().unwrap()),
            recorded_at: key::read_u64(&bytes[24..]),
            valid_from: key::read_u64(&bytes[32..]),
            valid_to: key::read_u64(&bytes[40..]),
        }
    }
}

impl Memory<'_> {
    /// Rebuilds the optional derived tag catalogue for an image written before
    /// section kind 60 existed. The next checkpoint persists the compact
    /// sorted section; interned tag strings and fact tag lists never move.
    pub(super) fn migrate_tag_catalog(&mut self, has_catalog: bool) -> bool {
        if has_catalog {
            return false;
        }
        let mut counts = alloc::vec::Vec::with_capacity(self.tags_idx.keys());
        for slot in self.tags_idx.slots() {
            let count = self
                .tags_idx
                .entries(slot.key)
                .filter(|(id, _)| {
                    self.fact(*id)
                        .is_some_and(|fact| !fact.is_tombstone() && !fact.is_closed())
                })
                .count();
            if let Ok(count) = u32::try_from(count)
                && count != 0
            {
                counts.push((TermId(slot.key), count));
            }
        }
        self.tag_catalog = super::tags::TagCatalog::from_counts(counts, &self.terms);
        true
    }

    /// Rebuilds the edge arenas from a pre-time-ordered image, if this
    /// snapshot is one. Returns `true` when a migration ran.
    ///
    /// Called only when the current-format edge sections were absent, so the
    /// engine's own edge arenas are still empty and can be filled directly.
    pub(super) fn migrate_edges(
        &mut self,
        snap: &Snapshot<'_>,
        cfg: &Config,
    ) -> Result<bool, Error> {
        let Some(current) = legacy_current_edges(snap, cfg)? else {
            return Ok(false);
        };
        match legacy_history(snap, cfg)? {
            // Versions exist: re-key them by `valid_from` and take the current
            // graph from the ones still open. That derivation is exact — the
            // old writer opened a current edge and its version together, and
            // closed them together — and it is what lets the current slot
            // carry its version's identity, which it never stored before.
            Some(history) => {
                for old in history.iter() {
                    self.adopt_history_version(EdgeHistorySlot {
                        a: old.a,
                        rel: old.rel,
                        b: old.b,
                        edge: old.edge,
                        fact: old.fact,
                        flags: old.flags,
                        kind: old.kind,
                        recorded_at: old.recorded_at,
                        valid_from: old.valid_from,
                        valid_to: old.valid_to,
                    })?;
                }
                if self.edges_out.len() != current.len() {
                    return Err(Error::Corrupt(
                        "legacy edge history does not cover every current edge",
                    ));
                }
            }
            // Older still: no history at all. Every current edge becomes one
            // open version. Their validity is unknown, so it starts at the
            // epoch — the image never recorded when the edge was created, and
            // inventing a later instant would hide the edge from `as_of`
            // queries that legitimately saw it.
            None => {
                for old in current.iter() {
                    let edge = EdgeId(self.next_edge);
                    self.next_edge = self.next_edge.saturating_add(1);
                    self.adopt_history_version(EdgeHistorySlot {
                        a: old.a,
                        rel: old.rel,
                        b: old.b,
                        edge,
                        fact: old.fact,
                        flags: 0,
                        kind: 0,
                        recorded_at: 0,
                        valid_from: 0,
                        valid_to: VALID_TO_OPEN,
                    })?;
                }
            }
        }
        Ok(true)
    }

    /// Inserts one migrated version into both history mirrors, and into the
    /// current graph as well when it is still open.
    fn adopt_history_version(&mut self, version: EdgeHistorySlot) -> Result<(), Error> {
        self.insert_history_edge(version)?;
        if version.valid_to == VALID_TO_OPEN {
            self.insert_current_edge(
                version.a,
                version.rel,
                version.b,
                version.fact,
                version.edge,
                version.valid_from,
            )?;
        }
        Ok(())
    }
}

/// Loads the legacy current-edge arena, or `None` when this image has no
/// legacy edge sections at all (it is either current-format or empty).
fn legacy_current_edges(
    snap: &Snapshot<'_>,
    cfg: &Config,
) -> Result<Option<Arena<'static, LegacyEdgeSlot>>, Error> {
    let Some(pair) = section_pair(
        snap,
        legacy_kind::EDGES_OUT_META,
        legacy_kind::EDGES_OUT_POOL,
    )?
    else {
        return Ok(None);
    };
    // The mirror is validated by loading it; its contents are re-derived, so
    // nothing else reads it.
    if section_pair(snap, legacy_kind::EDGES_IN_META, legacy_kind::EDGES_IN_POOL)?.is_none() {
        return Err(Error::Corrupt("snapshot has incomplete edge sections"));
    }
    Ok(Some(Arena::load(ordered(cfg), pair.0, pair.1)?))
}

/// Loads the legacy edge-history arena, or `None` when the image predates it.
fn legacy_history(
    snap: &Snapshot<'_>,
    cfg: &Config,
) -> Result<Option<Arena<'static, LegacyEdgeHistorySlot>>, Error> {
    let out = section_pair(
        snap,
        legacy_kind::EDGE_HIST_OUT_META,
        legacy_kind::EDGE_HIST_OUT_POOL,
    )?;
    let has_in = section_pair(
        snap,
        legacy_kind::EDGE_HIST_IN_META,
        legacy_kind::EDGE_HIST_IN_POOL,
    )?
    .is_some();
    match (out, has_in) {
        (Some(pair), true) => Ok(Some(Arena::load(ordered(cfg), pair.0, pair.1)?)),
        (None, false) => Ok(None),
        _ => Err(Error::Corrupt(
            "snapshot has incomplete edge history sections",
        )),
    }
}

/// A per-document BM25 record as written before it carried the term-set
/// summary: key `[fact]`, payload `len` and two reserved bytes.
#[derive(Clone, Copy)]
struct LegacyDocLenSlot {
    fact: FactId,
    len: u16,
}

impl Slot for LegacyDocLenSlot {
    const SIZE: usize = 8;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.fact.0);
        out[4..6].copy_from_slice(&self.len.to_be_bytes());
        out[6..8].copy_from_slice(&[0, 0]);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            fact: FactId(key::read_u32(bytes)),
            len: u16::from_be_bytes(bytes[4..6].try_into().unwrap()),
        }
    }
}

/// Reads the per-document BM25 records of a pre-signature image, widened into
/// the current slot, or `None` when this image is already current-format.
///
/// The upgraded records are marked "unknown" rather than guessed: recovering a
/// term set here would mean tokenizing every stored text at open time. The
/// write path falls back to reading the text for such a document — exactly what
/// it did before signatures existed — and the first `maintain` fills the
/// summaries in from the postings, which is the same information without the
/// tokenizer.
///
/// The result is always owned, even for a borrowed open: the slot widened, so
/// the old bytes cannot be aliased as new records.
pub(super) fn legacy_doc_len(
    snap: &Snapshot<'_>,
    cfg: &Config,
) -> Result<Option<Arena<'static, DocLenSlot>>, Error> {
    let Some((meta, pool)) = section_pair(
        snap,
        legacy_kind::BM25_DOCLEN_META,
        legacy_kind::BM25_DOCLEN_POOL,
    )?
    else {
        return Ok(None);
    };
    let old = Arena::<LegacyDocLenSlot>::load(doc_len_cfg(cfg), meta, pool)?;
    let mut out = Arena::new(doc_len_cfg(cfg))?;
    for doc in old.iter() {
        out.insert(&DocLenSlot {
            fact: doc.fact,
            len: doc.len,
            distinct: 0,
            sig: 0,
        })?;
    }
    Ok(Some(out))
}

/// Arena configuration of the per-document BM25 records, shared by the legacy
/// reader and the current loader so a migration cannot land in a differently
/// shaped arena.
pub(super) fn doc_len_cfg(cfg: &Config) -> ArenaCfg {
    ArenaCfg::new(cfg.shards_postings, ShardMode::Uniform).with_max_bytes(cfg.max_bytes)
}

/// The `(meta, pool)` byte pair one arena is stored as.
type ArenaImage<'s> = (&'s [u8], &'s [u8]);

/// Both halves of a section pair, or `None` when neither is present.
fn section_pair<'s>(
    snap: &Snapshot<'s>,
    meta: u16,
    pool: u16,
) -> Result<Option<ArenaImage<'s>>, Error> {
    match (snap.section(meta), snap.section(pool)) {
        (Some(m), Some(p)) => Ok(Some((m, p))),
        (None, None) => Ok(None),
        _ => Err(Error::Corrupt("snapshot section pair is incomplete")),
    }
}

fn ordered(cfg: &Config) -> ArenaCfg {
    ArenaCfg::new(cfg.shards_edges, ShardMode::Ordered).with_max_bytes(cfg.max_bytes)
}
