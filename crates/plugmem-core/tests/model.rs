//! Layout tests (specs/02 test plan): every record's `write` is compared
//! byte-for-byte against a hand-written reference buffer — the layouts are
//! the file format, so these tests *are* the format contract. Plus
//! roundtrip properties and the `as_of` liveness rule.

use plugmem_arena::{Arena, ArenaCfg, ChunkPool, ChunkPoolCfg, ListHandle, ShardMode, Slot};
use plugmem_core::{
    BlobId, EdgeSlot, EntityByName, EntityId, EntityRecord, FactAux, FactId, FactRecord,
    TemporalSlot, TermId, VALID_TO_OPEN, fact_flags,
};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::*;

/// Serializes one record into its exact-size buffer.
fn bytes_of<T: Slot>(rec: &T) -> Vec<u8> {
    let mut out = vec![0u8; T::SIZE];
    rec.write(&mut out);
    out
}

#[test]
fn fact_record_reference_layout() {
    let rec = FactRecord {
        id: FactId(0x0102_0304),
        entity: EntityId(0x1112_1314),
        flags: fact_flags::CLOSED | fact_flags::HAS_VECTOR,
        kind: 0,
        text: BlobId(0x2122_2324),
        vector: 0x3132_3334,
        revises: FactId(0x4142_4344),
        recorded_at: 0x5152_5354_5556_5758,
        valid_from: 0x6162_6364_6566_6768,
        valid_to: 0x7172_7374_7576_7778,
    };
    #[rustfmt::skip]
    let want = [
        0x01, 0x02, 0x03, 0x04,                         // id (key, BE)
        0x11, 0x12, 0x13, 0x14,                         // entity
        0x00, 0x06,                                     // flags: CLOSED|HAS_VECTOR
        0x00, 0x00,                                     // kind (reserved)
        0x21, 0x22, 0x23, 0x24,                         // text
        0x31, 0x32, 0x33, 0x34,                         // vector
        0x41, 0x42, 0x43, 0x44,                         // revises
        0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, // recorded_at
        0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, // valid_from
        0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, // valid_to
    ];
    assert_eq!(bytes_of(&rec), want);
    assert_eq!(FactRecord::read(&want), rec);
}

#[test]
fn fact_aux_reference_layout() {
    // A handle with real pool state: two pushes land in chunk 0.
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut tags = ListHandle::EMPTY;
    pool.push(&mut tags, &7u32.to_be_bytes()).unwrap();
    pool.push(&mut tags, &9u32.to_be_bytes()).unwrap();
    let rec = FactAux {
        id: FactId(0x0A0B_0C0D),
        tags,
        meta: BlobId(0x1A1B_1C1D),
    };
    #[rustfmt::skip]
    let want = [
        0x0A, 0x0B, 0x0C, 0x0D,   // id (key, BE)
        0x00, 0x00, 0x00, 0x00,   // handle.head = chunk 0
        0x00, 0x00, 0x00, 0x00,   // handle.tail = chunk 0
        0x00, 0x00, 0x00, 0x02,   // handle.len = 2
        0x1A, 0x1B, 0x1C, 0x1D,   // meta blob id (BE)
    ];
    assert_eq!(bytes_of(&rec), want);
    let back = FactAux::read(&want);
    assert_eq!(back, rec);
    // The restored handle still reads its list from the pool.
    let restored: Vec<u8> = pool.iter(&back.tags).flatten().copied().collect();
    assert_eq!(restored.len(), 8);
}

#[test]
fn entity_record_reference_layout() {
    let rec = EntityRecord {
        id: EntityId(0x0102_0304),
        name: BlobId(0x1112_1314),
        name_term: TermId(0x2122_2324),
        created_at: 0x3132_3334_3536_3738,
        flags: 0,
    };
    #[rustfmt::skip]
    let want = [
        0x01, 0x02, 0x03, 0x04,                         // id (key, BE)
        0x11, 0x12, 0x13, 0x14,                         // name
        0x21, 0x22, 0x23, 0x24,                         // name_term
        0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, // created_at
        0x00, 0x00, 0x00, 0x00,                         // flags (reserved)
    ];
    assert_eq!(bytes_of(&rec), want);
    assert_eq!(EntityRecord::read(&want), rec);
}

#[test]
fn entity_by_name_reference_layout() {
    let rec = EntityByName {
        name_term: TermId(0x0102_0304),
        id: EntityId(0x1112_1314),
    };
    let want = [0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14];
    assert_eq!(bytes_of(&rec), want);
    assert_eq!(EntityByName::read(&want), rec);
}

#[test]
fn edge_slot_reference_layout() {
    let rec = EdgeSlot {
        a: EntityId(0x0102_0304),
        rel: TermId(0x1112_1314),
        b: EntityId(0x2122_2324),
        fact: FactId::NONE,
    };
    #[rustfmt::skip]
    let want = [
        0x01, 0x02, 0x03, 0x04,   // a (key, BE)
        0x11, 0x12, 0x13, 0x14,   // rel (key, BE)
        0x21, 0x22, 0x23, 0x24,   // b (key, BE)
        0xFF, 0xFF, 0xFF, 0xFF,   // fact = NONE
    ];
    assert_eq!(bytes_of(&rec), want);
    assert_eq!(EdgeSlot::read(&want), rec);
}

#[test]
fn temporal_slot_reference_layout() {
    let rec = TemporalSlot {
        recorded_at: 0x0102_0304_0506_0708,
        fact: FactId(0x1112_1314),
    };
    let want = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x11, 0x12, 0x13, 0x14,
    ];
    assert_eq!(bytes_of(&rec), want);
    assert_eq!(TemporalSlot::read(&want), rec);
}

#[test]
fn fact_flag_helpers() {
    let mut rec = FactRecord {
        id: FactId(1),
        entity: EntityId::NONE,
        flags: 0,
        kind: 0,
        text: BlobId(0),
        vector: plugmem_core::NONE_U32,
        revises: FactId::NONE,
        recorded_at: 100,
        valid_from: 100,
        valid_to: VALID_TO_OPEN,
    };
    assert!(!rec.is_tombstone() && !rec.is_closed() && !rec.has_vector());
    rec.flags = fact_flags::TOMBSTONE | fact_flags::CLOSED | fact_flags::HAS_VECTOR;
    assert!(rec.is_tombstone() && rec.is_closed() && rec.has_vector());
}

#[test]
fn as_of_liveness_rule() {
    // An open fact recorded at t=100, valid from t=50 (backdated).
    let mut rec = FactRecord {
        id: FactId(1),
        entity: EntityId::NONE,
        flags: 0,
        kind: 0,
        text: BlobId(0),
        vector: plugmem_core::NONE_U32,
        revises: FactId::NONE,
        recorded_at: 100,
        valid_from: 50,
        valid_to: VALID_TO_OPEN,
    };
    assert!(!rec.is_live_at(49), "before validity");
    assert!(
        !rec.is_live_at(99),
        "valid but not yet recorded (knowledge axis)"
    );
    assert!(rec.is_live_at(100));
    assert!(rec.is_live_at(u64::MAX - 1), "open interval has no end");

    // Closing the interval at t=200 (a revision took over).
    rec.valid_to = 200;
    rec.flags = fact_flags::CLOSED;
    assert!(rec.is_live_at(199));
    assert!(!rec.is_live_at(200), "valid_to is exclusive");

    // A tombstone is dead at every t.
    rec.flags |= fact_flags::TOMBSTONE;
    assert!(!rec.is_live_at(150));
}

/// Every record type must actually live in an arena: keys order correctly
/// and survive storage. Smoke-level here; the engine tests will exercise
/// the real access patterns.
#[test]
fn records_store_and_sort_in_arenas() {
    let mut facts = Arena::<FactRecord>::new(ArenaCfg::new(4, ShardMode::Uniform)).unwrap();
    let mut by_name = Arena::<EntityByName>::new(ArenaCfg::new(1, ShardMode::Ordered)).unwrap();
    let mut edges = Arena::<EdgeSlot>::new(ArenaCfg::new(1, ShardMode::Ordered)).unwrap();
    for i in (0..64u32).rev() {
        facts
            .insert(&FactRecord {
                id: FactId(i),
                entity: EntityId::NONE,
                flags: 0,
                kind: 0,
                text: BlobId(i),
                vector: plugmem_core::NONE_U32,
                revises: FactId::NONE,
                recorded_at: u64::from(i),
                valid_from: u64::from(i),
                valid_to: VALID_TO_OPEN,
            })
            .unwrap();
        by_name
            .insert(&EntityByName {
                name_term: TermId(i / 2),
                id: EntityId(i),
            })
            .unwrap();
        edges
            .insert(&EdgeSlot {
                a: EntityId(i / 8),
                rel: TermId(i / 4),
                b: EntityId(i),
                fact: FactId(i),
            })
            .unwrap();
    }
    assert_eq!(facts.len(), 64);
    // Prefix scan on `name_term = 3` finds exactly its two entities.
    let from = bytes_of(&EntityByName {
        name_term: TermId(3),
        id: EntityId(0),
    });
    let to = bytes_of(&EntityByName {
        name_term: TermId(4),
        id: EntityId(0),
    });
    let hits: Vec<u32> = by_name.range(&from, &to).map(|r| r.id.0).collect();
    assert_eq!(hits, [6, 7]);
    // Prefix scan on `(a=2, rel=5)` walks that neighbor run in order
    // (edges with a = i/8, rel = i/4 put i = 20..24 under that prefix).
    let from = bytes_of(&EdgeSlot {
        a: EntityId(2),
        rel: TermId(5),
        b: EntityId(0),
        fact: FactId(0),
    })[..12]
        .to_vec();
    let to = bytes_of(&EdgeSlot {
        a: EntityId(2),
        rel: TermId(6),
        b: EntityId(0),
        fact: FactId(0),
    })[..12]
        .to_vec();
    let hits: Vec<u32> = edges.range(&from, &to).map(|e| e.b.0).collect();
    assert_eq!(hits, [20, 21, 22, 23]);
}

#[cfg(not(target_family = "wasm"))]
fn arb_fact() -> impl Strategy<Value = FactRecord> {
    (
        any::<u32>(),
        any::<u32>(),
        any::<u16>(),
        any::<u32>(),
        any::<u32>(),
        any::<u32>(),
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(
            |(id, entity, flags, text, vector, revises, recorded_at, valid_from, valid_to)| {
                FactRecord {
                    id: FactId(id),
                    entity: EntityId(entity),
                    flags,
                    kind: 0,
                    text: BlobId(text),
                    vector,
                    revises: FactId(revises),
                    recorded_at,
                    valid_from,
                    valid_to,
                }
            },
        )
}

#[cfg(not(target_family = "wasm"))]
proptest! {
    // Roundtrip: read is the exact inverse of write for every field value.
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn fact_roundtrip(rec in arb_fact()) {
        prop_assert_eq!(FactRecord::read(&bytes_of(&rec)), rec);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn entity_roundtrip(id in any::<u32>(), name in any::<u32>(), term in any::<u32>(),
                        at in any::<u64>()) {
        let rec = EntityRecord {
            id: EntityId(id),
            name: BlobId(name),
            name_term: TermId(term),
            created_at: at,
            flags: 0,
        };
        prop_assert_eq!(EntityRecord::read(&bytes_of(&rec)), rec);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn edge_and_temporal_roundtrip(a in any::<u32>(), rel in any::<u32>(), b in any::<u32>(),
                                   fact in any::<u32>(), at in any::<u64>()) {
        let edge = EdgeSlot {
            a: EntityId(a),
            rel: TermId(rel),
            b: EntityId(b),
            fact: FactId(fact),
        };
        prop_assert_eq!(EdgeSlot::read(&bytes_of(&edge)), edge);
        let t = TemporalSlot { recorded_at: at, fact: FactId(fact) };
        prop_assert_eq!(TemporalSlot::read(&bytes_of(&t)), t);
    }

    // Key ordering contract: byte comparison of encoded keys must equal
    // the intended (recorded_at, fact) ordering.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn temporal_key_order_matches_semantic_order(
        a in (any::<u64>(), any::<u32>()), b in (any::<u64>(), any::<u32>())
    ) {
        let ka = bytes_of(&TemporalSlot { recorded_at: a.0, fact: FactId(a.1) });
        let kb = bytes_of(&TemporalSlot { recorded_at: b.0, fact: FactId(b.1) });
        prop_assert_eq!(ka.cmp(&kb), a.cmp(&b));
    }
}
