//! Record layouts of the data model (specs/02).
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

use plugmem_arena::{BlobId, ListHandle, Slot, TermId, key};

use crate::id::{EntityId, FactId, NONE_U32};

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

/// The unit of memory: one fact (specs/02, 48-byte slot, Uniform arena).
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

    /// The `as_of(t)` liveness rule (specs/02): not a tombstone, already
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

/// Per-fact auxiliary record: the tag-list handle (specs/02, 16-byte slot,
/// Uniform arena; layout `[id 4 | ListHandle 12]`).
///
/// Split from [`FactRecord`] so the hot 48-byte record stays hot: tags are
/// touched only by tag-filtered queries and `maintain`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactAux {
    /// Fact id — the key.
    pub id: FactId,
    /// The fact's tag list (`TermId` values) in the tag `ChunkPool`.
    pub tags: ListHandle,
}

impl Slot for FactAux {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id.0);
        out[4..16].copy_from_slice(&self.tags.to_bytes());
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            id: FactId(key::read_u32(bytes)),
            tags: ListHandle::from_bytes(bytes[4..16].try_into().unwrap()),
        }
    }
}

/// A graph node (specs/02, 24-byte slot, Uniform arena).
///
/// | off | size | field |
/// |---|---|---|
/// | 0 | 4 | `id` (key) |
/// | 4 | 4 | `name` |
/// | 8 | 4 | `name_term` |
/// | 12 | 8 | `created_at` |
/// | 20 | 4 | `flags` (reserved, 0 in v1) |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Name → entity resolution record (specs/02, 8-byte slot, Ordered arena,
/// the whole slot is the key: `[name_term BE | id BE]`).
///
/// The normalized name is unique (lookup-or-create), so a prefix scan on
/// `name_term` yields at most one record; the full pair keeps the slot
/// unique and self-describing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A typed graph edge (specs/02, 16-byte slot, Ordered arena, key
/// `[a BE | rel BE | b BE]`, payload `fact`).
///
/// Stored twice, in two mirrored arenas: the out-arena keys by
/// `(src, rel, dst)`, the in-arena by `(dst, rel, src)` — `a`/`b` are
/// whichever end comes first in that arena's key. Neighbor traversal is a
/// prefix range scan. An edge is unique per `(src, rel, dst)`; re-linking
/// updates the provenance payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl Slot for EdgeSlot {
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

/// Temporal index record (specs/02, 12-byte slot, Ordered arena, the whole
/// slot is the key: `[recorded_at BE | fact BE]`, no payload).
///
/// Range scans answer "what was recorded in this window"; validity
/// filtering happens per candidate on its [`FactRecord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    assert!(FactAux::SIZE == 16 && FactAux::KEY_LEN == 4);
    assert!(EntityRecord::SIZE == 24 && EntityRecord::KEY_LEN == 4);
    assert!(EntityByName::SIZE == 8 && EntityByName::KEY_LEN == 8);
    assert!(EdgeSlot::SIZE == 16 && EdgeSlot::KEY_LEN == 12);
    assert!(TemporalSlot::SIZE == 12 && TemporalSlot::KEY_LEN == 12);
    // NONE sentinels must agree across the id kinds and the raw fields.
    assert!(NONE_U32 == u32::MAX);
};
