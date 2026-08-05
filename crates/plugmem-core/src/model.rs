//! Record layouts of the data model.
//!
//! Every record is a fixed-size [`Slot`] living in an [`Arena`]; the byte
//! layouts below are **format contracts** — the snapshot is a memcpy of
//! the arenas, so changing an offset here changes the file format. The
//! layout tests compare against hand-written reference buffers: breaking a
//! layout breaks a test.
//!
//! All integer fields are big-endian — mandatory for key prefixes (the
//! arena sorts by raw bytes) and kept for payloads too, so a slot has one
//! endianness throughout.
//!
//! [`Arena`]: plugmem_arena::Arena

use core::mem::size_of;

use plugmem_arena::{BlobId, ListHandle, Slot, TermId, key};

use crate::id::{EdgeId, EntityId, FactId, NONE_U32};

/// `valid_to` value of an open fact ("true now").
pub const VALID_TO_OPEN: u64 = u64::MAX;

/// Bit flags of [`FactRecord::flags`].
pub mod fact_flags {
    /// The fact is deleted; recall never returns it, `maintain` purges it.
    pub const TOMBSTONE: u16 = 1;
    /// The validity interval is closed (`valid_to < u64::MAX`) — the fact
    /// was revised.
    pub const CLOSED: u16 = 1 << 1;
    /// A vector slot is attached ([`crate::model::FactRecord::vector`]
    /// is meaningful).
    pub const HAS_VECTOR: u16 = 1 << 2;
}

/// The unit of memory: one fact (48-byte slot, Uniform arena).
///
/// | off | size | field |
/// |---|---|---|
/// | 0 | 4 | `id` (key) |
/// | 4 | 4 | `entity` |
/// | 8 | 2 | `flags` |
/// | 10 | 2 | `kind` (reserved, 0 in v1) |
/// | 12 | 4 | `text` |
/// | 16 | 4 | `vector` |
/// | 20 | 4 | `revises` |
/// | 24 | 8 | `recorded_at` |
/// | 32 | 8 | `valid_from` |
/// | 40 | 8 | `valid_to` |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FactRecord {
    /// Fact id — the key.
    pub id: FactId,
    /// Subject entity, or [`EntityId::NONE`].
    pub entity: EntityId,
    /// Bit set from [`fact_flags`].
    pub flags: u16,
    /// Reserved in v1; must be `0` (fact typing is a tag convention).
    pub kind: u16,
    /// Fact text (UTF-8) in the blob heap.
    pub text: BlobId,
    /// Slot index in the vector arena, or [`NONE_U32`]; meaningful only
    /// with [`fact_flags::HAS_VECTOR`].
    pub vector: u32,
    /// Predecessor in the revision chain, or [`FactId::NONE`]. May name a
    /// *burned* id — a predecessor that was forgotten and physically
    /// purged by `maintain`; resolving it then yields `None`, the same
    /// answer a tombstoned record gives.
    pub revises: FactId,
    /// Knowledge axis: when the memory learned this. Immutable.
    pub recorded_at: u64,
    /// Truth axis: start of the validity interval.
    pub valid_from: u64,
    /// Truth axis: end of the validity interval; [`VALID_TO_OPEN`] = open.
    pub valid_to: u64,
}

impl FactRecord {
    /// `true` when the tombstone flag is set.
    pub fn is_tombstone(&self) -> bool {
        self.flags & fact_flags::TOMBSTONE != 0
    }

    /// `true` when the validity interval is closed.
    pub fn is_closed(&self) -> bool {
        self.flags & fact_flags::CLOSED != 0
    }

    /// `true` when a vector slot is attached.
    pub fn has_vector(&self) -> bool {
        self.flags & fact_flags::HAS_VECTOR != 0
    }

    /// The `as_of(t)` liveness rule: not a tombstone, already
    /// recorded at `t`, and `t` inside `[valid_from, valid_to)`.
    pub fn is_live_at(&self, t: u64) -> bool {
        !self.is_tombstone() && self.recorded_at <= t && self.valid_from <= t && t < self.valid_to
    }
}

impl Slot for FactRecord {
    const SIZE: usize = 48;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id.0);
        key::write_u32(&mut out[4..], self.entity.0);
        out[8..10].copy_from_slice(&self.flags.to_be_bytes());
        out[10..12].copy_from_slice(&self.kind.to_be_bytes());
        key::write_u32(&mut out[12..], self.text.0);
        key::write_u32(&mut out[16..], self.vector);
        key::write_u32(&mut out[20..], self.revises.0);
        key::write_u64(&mut out[24..], self.recorded_at);
        key::write_u64(&mut out[32..], self.valid_from);
        key::write_u64(&mut out[40..], self.valid_to);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            id: FactId(key::read_u32(bytes)),
            entity: EntityId(key::read_u32(&bytes[4..])),
            flags: u16::from_be_bytes(bytes[8..10].try_into().unwrap()),
            kind: u16::from_be_bytes(bytes[10..12].try_into().unwrap()),
            text: BlobId(key::read_u32(&bytes[12..])),
            vector: key::read_u32(&bytes[16..]),
            revises: FactId(key::read_u32(&bytes[20..])),
            recorded_at: key::read_u64(&bytes[24..]),
            valid_from: key::read_u64(&bytes[32..]),
            valid_to: key::read_u64(&bytes[40..]),
        }
    }
}

/// Per-fact auxiliary record: the tag-list handle and the optional metadata
/// blob (20-byte slot, Uniform arena; layout
/// `[id 4 | ListHandle 12 | meta 4]`).
///
/// Split from [`FactRecord`] so the hot 48-byte record stays hot: tags and
/// metadata are touched only by tag-filtered queries, `show`/`export` and
/// `maintain`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FactAux {
    /// Fact id — the key.
    pub id: FactId,
    /// The fact's tag list (`TermId` values) in the tag `ChunkPool`.
    pub tags: ListHandle,
    /// The fact's metadata blob in the `metas` heap (a canonical key→value
    /// encoding, see `crate::metadata`), or [`BlobId`]`(`[`NONE_U32`]`)` when
    /// the fact carries no metadata. The engine never interprets the bytes.
    pub meta: BlobId,
}

impl Slot for FactAux {
    const SIZE: usize = 20;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id.0);
        out[4..16].copy_from_slice(&self.tags.to_bytes());
        key::write_u32(&mut out[16..], self.meta.0);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            id: FactId(key::read_u32(bytes)),
            tags: ListHandle::from_bytes(bytes[4..16].try_into().unwrap()),
            meta: BlobId(key::read_u32(&bytes[16..])),
        }
    }
}

/// A graph node (24-byte slot, Uniform arena).
///
/// | off | size | field |
/// |---|---|---|
/// | 0 | 4 | `id` (key) |
/// | 4 | 4 | `name` |
/// | 8 | 4 | `name_term` |
/// | 12 | 8 | `created_at` |
/// | 20 | 4 | `flags` (reserved, 0 in v1) |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntityRecord {
    /// Entity id — the key.
    pub id: EntityId,
    /// Canonical name as first entered (blob heap, UTF-8).
    pub name: BlobId,
    /// Interned *normalized* name — the lookup key for name resolution.
    pub name_term: TermId,
    /// When the entity was first mentioned.
    pub created_at: u64,
    /// Reserved in v1; must be `0`.
    pub flags: u32,
}

impl Slot for EntityRecord {
    const SIZE: usize = 24;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id.0);
        key::write_u32(&mut out[4..], self.name.0);
        key::write_u32(&mut out[8..], self.name_term.0);
        key::write_u64(&mut out[12..], self.created_at);
        key::write_u32(&mut out[20..], self.flags);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            id: EntityId(key::read_u32(bytes)),
            name: BlobId(key::read_u32(&bytes[4..])),
            name_term: TermId(key::read_u32(&bytes[8..])),
            created_at: key::read_u64(&bytes[12..]),
            flags: key::read_u32(&bytes[20..]),
        }
    }
}

/// Name → entity resolution record (8-byte slot, Ordered arena,
/// the whole slot is the key: `[name_term BE | id BE]`).
///
/// The normalized name is unique (lookup-or-create), so a prefix scan on
/// `name_term` yields at most one record; the full pair keeps the slot
/// unique and self-describing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntityByName {
    /// Interned normalized name.
    pub name_term: TermId,
    /// The entity carrying that name.
    pub id: EntityId,
}

impl Slot for EntityByName {
    const SIZE: usize = 8;
    const KEY_LEN: usize = 8;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.name_term.0);
        key::write_u32(&mut out[4..], self.id.0);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            name_term: TermId(key::read_u32(bytes)),
            id: EntityId(key::read_u32(&bytes[4..])),
        }
    }
}

/// Byte layout of [`EdgeSlot`]. Every offset is the previous field's offset
/// plus its width, so a field cannot be moved by editing one number: the
/// layout is a chain, and `SIZE`/`KEY_LEN` fall out of it.
mod edge_at {
    use core::mem::size_of;

    pub(super) const A: usize = 0;
    pub(super) const REL: usize = A + size_of::<u32>();
    pub(super) const B: usize = REL + size_of::<u32>();
    /// End of the key: `(a, rel, b)` identifies a current edge.
    pub(super) const KEY_LEN: usize = B + size_of::<u32>();
    pub(super) const FACT: usize = KEY_LEN;
    pub(super) const EDGE: usize = FACT + size_of::<u32>();
    pub(super) const VALID_FROM: usize = EDGE + size_of::<u32>();
    pub(super) const SIZE: usize = VALID_FROM + size_of::<u64>();
}

/// A typed graph edge, currently open (28-byte slot, Ordered arena, key
/// `[a BE | rel BE | b BE]`, payload `fact | edge | valid_from`).
///
/// Stored twice, in two mirrored arenas: the out-arena keys by
/// `(src, rel, dst)`, the in-arena by `(dst, rel, src)` — `a`/`b` are
/// whichever end comes first in that arena's key. Neighbor traversal is a
/// prefix range scan. An edge is unique per `(src, rel, dst)`; re-linking
/// closes this version and opens a new one.
///
/// The slot carries the identity of its open [`EdgeHistorySlot`] version —
/// `edge` and `valid_from` are exactly that record's key tail — so closing an
/// edge addresses its history record directly instead of searching for the
/// open version among the triple's other versions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeSlot {
    /// First key component (out-arena: source; in-arena: destination).
    pub a: EntityId,
    /// Interned relation term (`"works_at"`, `"owns"`, …).
    pub rel: TermId,
    /// Second key component (out-arena: destination; in-arena: source).
    pub b: EntityId,
    /// Provenance fact, or [`FactId::NONE`]. Like
    /// [`FactRecord::revises`], it may name a burned id once the
    /// provenance fact has been forgotten and purged.
    pub fact: FactId,
    /// The open edge version this slot mirrors.
    pub edge: EdgeId,
    /// Start of the open version's validity — with `edge`, the key tail of
    /// its [`EdgeHistorySlot`].
    pub valid_from: u64,
}

impl Slot for EdgeSlot {
    const SIZE: usize = edge_at::SIZE;
    const KEY_LEN: usize = edge_at::KEY_LEN;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(&mut out[edge_at::A..], self.a.0);
        key::write_u32(&mut out[edge_at::REL..], self.rel.0);
        key::write_u32(&mut out[edge_at::B..], self.b.0);
        key::write_u32(&mut out[edge_at::FACT..], self.fact.0);
        key::write_u32(&mut out[edge_at::EDGE..], self.edge.0);
        key::write_u64(&mut out[edge_at::VALID_FROM..], self.valid_from);
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            a: EntityId(key::read_u32(&bytes[edge_at::A..])),
            rel: TermId(key::read_u32(&bytes[edge_at::REL..])),
            b: EntityId(key::read_u32(&bytes[edge_at::B..])),
            fact: FactId(key::read_u32(&bytes[edge_at::FACT..])),
            edge: EdgeId(key::read_u32(&bytes[edge_at::EDGE..])),
            valid_from: key::read_u64(&bytes[edge_at::VALID_FROM..]),
        }
    }
}

/// The key of a current edge: `[a | rel | b]`.
pub(crate) fn edge_key(a: EntityId, rel: TermId, b: EntityId) -> [u8; edge_at::KEY_LEN] {
    let mut out = [0u8; edge_at::KEY_LEN];
    key::write_u32(&mut out[edge_at::A..], a.0);
    key::write_u32(&mut out[edge_at::REL..], rel.0);
    key::write_u32(&mut out[edge_at::B..], b.0);
    out
}

/// Byte layout of [`EdgeHistorySlot`], derived field by field like
/// [`edge_at`].
mod edge_hist_at {
    use core::mem::size_of;

    pub(super) const A: usize = 0;
    pub(super) const VALID_FROM: usize = A + size_of::<u32>();
    pub(super) const EDGE: usize = VALID_FROM + size_of::<u64>();
    /// End of the key: `(a, valid_from, edge)` orders an entity's versions by
    /// the instant they became true, `edge` breaking ties.
    pub(super) const KEY_LEN: usize = EDGE + size_of::<u32>();
    pub(super) const REL: usize = KEY_LEN;
    pub(super) const B: usize = REL + size_of::<u32>();
    pub(super) const FACT: usize = B + size_of::<u32>();
    pub(super) const FLAGS: usize = FACT + size_of::<u32>();
    pub(super) const KIND: usize = FLAGS + size_of::<u16>();
    pub(super) const RECORDED_AT: usize = KIND + size_of::<u16>();
    pub(super) const VALID_TO: usize = RECORDED_AT + size_of::<u64>();
    pub(super) const SIZE: usize = VALID_TO + size_of::<u64>();
}

/// A temporal typed graph edge version (48-byte slot, Ordered arena, key
/// `[a BE | valid_from BE | edge BE]`).
///
/// Stored twice, in two mirrored history arenas with the same orientation as
/// [`EdgeSlot`]. The hot current graph still uses [`EdgeSlot`]; this record is
/// the source of truth for historical `as_of` traversal.
///
/// The key is **time-ordered per entity**, not grouped by relation. An
/// `as_of(t)` traversal wants the versions valid at one instant, and at most
/// one version of a `(a, rel, b)` triple is valid at any instant, so grouping
/// by triple forces a walk through every version of every triple to find the
/// few that answer. Ordering by `valid_from` instead lets the traversal start
/// at `t` and walk backwards through the versions that most recently became
/// true — the candidates — and stop when it has enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeHistorySlot {
    /// First key component (out-arena: source; in-arena: destination).
    pub a: EntityId,
    /// Interned relation term.
    pub rel: TermId,
    /// Second key component (out-arena: destination; in-arena: source).
    pub b: EntityId,
    /// Edge-version id.
    pub edge: EdgeId,
    /// Provenance fact, or [`FactId::NONE`].
    pub fact: FactId,
    /// Edge flags; see [`edge_flags`].
    pub flags: u16,
    /// Reserved for future lifecycle states.
    pub kind: u16,
    /// Knowledge axis: when the edge version was recorded.
    pub recorded_at: u64,
    /// Truth axis: start of the edge validity interval.
    pub valid_from: u64,
    /// Truth axis: end of the edge validity interval; [`VALID_TO_OPEN`] =
    /// current.
    pub valid_to: u64,
}

impl EdgeHistorySlot {
    /// `true` when the edge version is active at `t`.
    pub fn active_at(&self, t: u64) -> bool {
        self.valid_from <= t && t < self.valid_to
    }

    /// `true` when this version is the current open edge.
    pub fn is_open(&self) -> bool {
        self.valid_to == VALID_TO_OPEN
    }
}

/// Bit flags of [`EdgeHistorySlot::flags`].
pub mod edge_flags {
    /// The edge validity interval is closed (`valid_to < u64::MAX`).
    pub const CLOSED: u16 = 1;
}

impl Slot for EdgeHistorySlot {
    const SIZE: usize = edge_hist_at::SIZE;
    const KEY_LEN: usize = edge_hist_at::KEY_LEN;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(&mut out[edge_hist_at::A..], self.a.0);
        key::write_u64(&mut out[edge_hist_at::VALID_FROM..], self.valid_from);
        key::write_u32(&mut out[edge_hist_at::EDGE..], self.edge.0);
        key::write_u32(&mut out[edge_hist_at::REL..], self.rel.0);
        key::write_u32(&mut out[edge_hist_at::B..], self.b.0);
        key::write_u32(&mut out[edge_hist_at::FACT..], self.fact.0);
        let flags = edge_hist_at::FLAGS;
        out[flags..flags + size_of::<u16>()].copy_from_slice(&self.flags.to_be_bytes());
        let kind = edge_hist_at::KIND;
        out[kind..kind + size_of::<u16>()].copy_from_slice(&self.kind.to_be_bytes());
        key::write_u64(&mut out[edge_hist_at::RECORDED_AT..], self.recorded_at);
        key::write_u64(&mut out[edge_hist_at::VALID_TO..], self.valid_to);
    }

    fn read(bytes: &[u8]) -> Self {
        let flags = edge_hist_at::FLAGS;
        let kind = edge_hist_at::KIND;
        Self {
            a: EntityId(key::read_u32(&bytes[edge_hist_at::A..])),
            rel: TermId(key::read_u32(&bytes[edge_hist_at::REL..])),
            b: EntityId(key::read_u32(&bytes[edge_hist_at::B..])),
            edge: EdgeId(key::read_u32(&bytes[edge_hist_at::EDGE..])),
            fact: FactId(key::read_u32(&bytes[edge_hist_at::FACT..])),
            flags: u16::from_be_bytes(bytes[flags..flags + size_of::<u16>()].try_into().unwrap()),
            kind: u16::from_be_bytes(bytes[kind..kind + size_of::<u16>()].try_into().unwrap()),
            recorded_at: key::read_u64(&bytes[edge_hist_at::RECORDED_AT..]),
            valid_from: key::read_u64(&bytes[edge_hist_at::VALID_FROM..]),
            valid_to: key::read_u64(&bytes[edge_hist_at::VALID_TO..]),
        }
    }
}

/// The lowest key of `a`'s current-edge run — the inclusive lower bound of a
/// neighbor scan.
pub(crate) fn edge_floor(a: EntityId) -> [u8; edge_at::KEY_LEN] {
    edge_key(a, TermId(0), EntityId(0))
}

/// The exclusive upper bound of `a`'s current-edge run. Saturating leaves the
/// range empty for [`EntityId::NONE`], which is a sentinel and never names a
/// stored entity.
pub(crate) fn edge_end(a: EntityId) -> [u8; edge_at::KEY_LEN] {
    edge_key(EntityId(a.0.saturating_add(1)), TermId(0), EntityId(0))
}

/// The key of one edge version: `[a | valid_from | edge]`.
pub(crate) fn edge_history_key(
    a: EntityId,
    valid_from: u64,
    edge: EdgeId,
) -> [u8; edge_hist_at::KEY_LEN] {
    let mut out = [0u8; edge_hist_at::KEY_LEN];
    key::write_u32(&mut out[edge_hist_at::A..], a.0);
    key::write_u64(&mut out[edge_hist_at::VALID_FROM..], valid_from);
    key::write_u32(&mut out[edge_hist_at::EDGE..], edge.0);
    out
}

/// The lowest key of `a`'s version run — the inclusive lower bound of a
/// per-entity history scan.
pub(crate) fn edge_history_floor(a: EntityId) -> [u8; edge_hist_at::KEY_LEN] {
    edge_history_key(a, 0, EdgeId(0))
}

/// The exclusive upper bound of `a`'s versions that had already become true at
/// `as_of`, i.e. everything with `valid_from <= as_of`.
///
/// Saturating past `u64::MAX` excludes only a version with
/// `valid_from == u64::MAX`, which can never be valid at any instant: validity
/// needs `t < valid_to <= u64::MAX` and `valid_from <= t`.
pub(crate) fn edge_history_ceiling(a: EntityId, as_of: u64) -> [u8; edge_hist_at::KEY_LEN] {
    edge_history_key(a, as_of.saturating_add(1), EdgeId(0))
}

/// Closes an edge version in place: sets [`edge_flags::CLOSED`] and
/// `valid_to`. `payload` is the slot bytes *after* the key, exactly as
/// [`Arena::payload_mut`](plugmem_arena::Arena::payload_mut) hands them over,
/// so every offset is shifted by the key length here rather than at the call
/// site.
pub(crate) fn close_edge_history_payload(payload: &mut [u8], valid_to: u64) {
    const KEY: usize = edge_hist_at::KEY_LEN;
    const FLAGS: usize = edge_hist_at::FLAGS - KEY;
    const VALID_TO: usize = edge_hist_at::VALID_TO - KEY;
    let flags = u16::from_be_bytes(payload[FLAGS..FLAGS + size_of::<u16>()].try_into().unwrap())
        | edge_flags::CLOSED;
    payload[FLAGS..FLAGS + size_of::<u16>()].copy_from_slice(&flags.to_be_bytes());
    key::write_u64(&mut payload[VALID_TO..], valid_to);
}

/// Temporal index record (12-byte slot, Ordered arena, the whole
/// slot is the key: `[recorded_at BE | fact BE]`, no payload).
///
/// Range scans answer "what was recorded in this window"; validity
/// filtering happens per candidate on its [`FactRecord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TemporalSlot {
    /// When the fact was recorded (knowledge axis).
    pub recorded_at: u64,
    /// The fact recorded at that moment.
    pub fact: FactId,
}

impl Slot for TemporalSlot {
    const SIZE: usize = 12;
    const KEY_LEN: usize = 12;

    fn write(&self, out: &mut [u8]) {
        key::write_pair(out, self.recorded_at, self.fact.0);
    }

    fn read(bytes: &[u8]) -> Self {
        let (recorded_at, fact) = key::read_pair(bytes);
        Self {
            recorded_at,
            fact: FactId(fact),
        }
    }
}

/// Compile-time layout self-checks: a slot size that drifts is a format
/// break, catch it before any test runs.
const _: () = {
    assert!(FactRecord::SIZE == 48 && FactRecord::KEY_LEN == 4);
    assert!(FactAux::SIZE == 20 && FactAux::KEY_LEN == 4);
    assert!(EntityRecord::SIZE == 24 && EntityRecord::KEY_LEN == 4);
    assert!(EntityByName::SIZE == 8 && EntityByName::KEY_LEN == 8);
    assert!(EdgeSlot::SIZE == 28 && EdgeSlot::KEY_LEN == 12);
    assert!(EdgeHistorySlot::SIZE == 48 && EdgeHistorySlot::KEY_LEN == 16);
    assert!(TemporalSlot::SIZE == 12 && TemporalSlot::KEY_LEN == 12);
    // NONE sentinels must agree across the id kinds and the raw fields.
    assert!(NONE_U32 == u32::MAX);
};
