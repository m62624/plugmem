//! Per-fact metadata (specs/02, specs/05): a small key→value map stored as one
//! canonical blob. Covers the single-order guarantee, journal/snapshot
//! determinism, compaction, and deferred content validation.

use plugmem_core::{Config, Error, FactFault, FactId, MemStorage, Memory, RememberInput, Storage};

fn cfg() -> Config {
    let mut cfg = Config::default();
    cfg.shards_facts = 8;
    cfg.shards_entities = 4;
    cfg.shards_edges = 4;
    cfg.shards_temporal = 4;
    cfg.shards_postings = 16;
    cfg
}

fn engine() -> (Memory<'static>, MemStorage) {
    (Memory::new(cfg()).unwrap(), MemStorage::new())
}

/// `remember` with deliberately unsorted keys; `metadata_of` returns them in
/// one canonical (ascending) order, with the right count.
#[test]
fn metadata_is_stored_and_read_back_in_canonical_order() {
    let (mut mem, mut store) = engine();
    // Fact 0: metadata with deliberately unsorted keys.
    let pairs = [
        ("uri", "s3://b/x"),
        ("mime", "application/pdf"),
        ("page", "3"),
    ];
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&pairs),
            ..RememberInput::text(100, "a scanned contract")
        },
    )
    .unwrap();
    // Fact 1: no metadata. Fact 2: an explicitly empty map (stored as none).
    mem.remember(&mut store, RememberInput::text(200, "no metadata"))
        .unwrap();
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&[]),
            ..RememberInput::text(300, "empty map")
        },
    )
    .unwrap();

    let mut out = Vec::new();
    assert!(mem.metadata_of(FactId(0), &mut out));
    assert_eq!(
        out,
        vec![
            ("mime", "application/pdf"),
            ("page", "3"),
            ("uri", "s3://b/x"),
        ],
        "keys come back sorted regardless of input order"
    );
    assert!(!mem.metadata_of(FactId(1), &mut out));
    assert!(out.is_empty(), "no metadata clears the buffer");
    assert!(!mem.metadata_of(FactId(2), &mut out), "empty map = none");
}

/// Cross-layer parity: the raw pairs `core::metadata_of` yields and the
/// `BTreeMap` a host builds from them agree on BOTH order and count. Order is
/// fixed once (at encode); collecting into a map neither reorders nor drops a
/// key. Uses deliberately shuffled input to prove the single canonical order.
#[test]
fn core_pairs_and_host_btreemap_agree_on_order_and_count() {
    use std::collections::BTreeMap;
    let (mut mem, mut store) = engine();
    let shuffled = [
        ("m", "1"),
        ("a", "2"),
        ("zzz", "3"),
        ("b", "4"),
        ("aa", "5"),
    ];
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&shuffled),
            ..RememberInput::text(100, "parity")
        },
    )
    .unwrap();

    // The core view: raw pairs in stored order.
    let mut pairs = Vec::new();
    assert!(mem.metadata_of(FactId(0), &mut pairs));

    // The host view: the same pairs collected into a BTreeMap (what
    // `metadata_map` does), then iterated back out.
    let map: BTreeMap<&str, &str> = pairs.iter().copied().collect();
    let map_seq: Vec<(&str, &str)> = map.into_iter().collect();

    assert_eq!(pairs.len(), shuffled.len(), "no key lost");
    assert_eq!(pairs.len(), map_seq.len(), "core and map agree on count");
    assert_eq!(pairs, map_seq, "core order == map iteration order");
    // And it is the one canonical (ascending) order.
    assert_eq!(
        pairs,
        vec![
            ("a", "2"),
            ("aa", "5"),
            ("b", "4"),
            ("m", "1"),
            ("zzz", "3"),
        ]
    );
}

/// A duplicate key in the input is rejected before anything is stored.
#[test]
fn duplicate_metadata_key_is_rejected() {
    let (mut mem, mut store) = engine();
    let err = mem
        .remember(
            &mut store,
            RememberInput {
                metadata: Some(&[("k", "a"), ("k", "b")]),
                ..RememberInput::text(100, "dup")
            },
        )
        .unwrap_err();
    assert!(matches!(err, Error::Invalid(_)));
    assert_eq!(mem.facts_len(), 0, "the whole op aborted");
}

/// Replaying the journal reproduces the metadata exactly (order and content).
#[test]
fn journal_replay_reproduces_metadata() {
    let (mut mem, mut store) = engine();
    let pairs = [("z", "last"), ("a", "first"), ("m", "mid")];
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&pairs),
            ..RememberInput::text(100, "journaled")
        },
    )
    .unwrap();

    let journal = store.read_journal().unwrap();
    let (replayed, report) = Memory::from_bytes(None, &journal, cfg()).unwrap();
    assert_eq!(report.replayed, 1);
    let mut a = Vec::new();
    let mut b = Vec::new();
    mem.metadata_of(FactId(0), &mut a);
    replayed.metadata_of(FactId(0), &mut b);
    assert_eq!(a, b);
    assert_eq!(a, vec![("a", "first"), ("m", "mid"), ("z", "last")]);
}

/// Snapshot is deterministic with metadata: save → load → save is identical.
#[test]
fn snapshot_roundtrips_byte_identical_with_metadata() {
    let (mut mem, mut store) = engine();
    for (i, kv) in [("uri", "a"), ("uri", "b")].iter().enumerate() {
        mem.remember(
            &mut store,
            RememberInput {
                metadata: Some(&[*kv, ("n", "1")]),
                ..RememberInput::text(100 + i as u64, "fact")
            },
        )
        .unwrap();
    }
    let first = mem.snapshot_bytes(0);
    let (loaded, _) = Memory::from_bytes(Some(&first), &[], cfg()).unwrap();
    assert!(loaded.verify().is_ok());
    assert_eq!(first, loaded.snapshot_bytes(0), "save→load→save is stable");
}

/// `maintain` carries a live fact's metadata across compaction and drops a
/// purged fact's; the compacted image still verifies.
#[test]
fn maintain_preserves_live_metadata_and_purges_the_rest() {
    let (mut mem, mut store) = engine();
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&[("keep", "yes")]),
            ..RememberInput::text(100, "survivor")
        },
    )
    .unwrap();
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&[("drop", "me")]),
            ..RememberInput::text(200, "doomed")
        },
    )
    .unwrap();
    mem.forget(&mut store, 250, FactId(1)).unwrap();
    let report = mem.maintain(&mut store, 300).unwrap();
    assert_eq!(report.purged, 1);

    let mut out = Vec::new();
    assert!(mem.metadata_of(FactId(0), &mut out));
    assert_eq!(out, vec![("keep", "yes")]);
    assert!(!mem.metadata_of(FactId(1), &mut out), "purged fact is gone");
    assert!(mem.verify().is_ok());
}

/// A metadata value corrupted past the (skipped) checksums opens fine, hides
/// the fact's metadata through the tolerant accessor, and is caught by
/// `verify()` and attributed by `faulty_facts()`.
#[test]
fn verify_and_faulty_facts_catch_corrupt_metadata() {
    let (mut mem, mut store) = engine();
    mem.remember(
        &mut store,
        RememberInput {
            metadata: Some(&[("k", "UNIQUEMETAMARKER")]),
            ..RememberInput::text(100, "has metadata")
        },
    )
    .unwrap();
    let clean = mem.snapshot_bytes(0);
    let (loaded, _) = Memory::from_bytes(Some(&clean), &[], cfg()).unwrap();
    assert!(loaded.verify().is_ok(), "a clean image verifies");

    // Flip a byte inside the stored metadata value to break its UTF-8.
    let at = clean
        .windows(b"UNIQUEMETAMARKER".len())
        .position(|w| w == b"UNIQUEMETAMARKER")
        .expect("the metadata value is stored verbatim");
    let mut bad = clean.clone();
    bad[at] = 0xFF;

    let (loaded, _) = Memory::from_bytes(Some(&bad), &[], cfg())
        .expect("the trust/sparse default open does not scan metadata");
    // Tolerant read: the fact is fine, only its metadata is hidden.
    assert!(loaded.get(FactId(0)).is_some());
    let mut out = Vec::new();
    assert!(
        !loaded.metadata_of(FactId(0), &mut out),
        "bad metadata hidden"
    );
    // Explicit integrity check catches it, and salvage attributes it.
    assert_eq!(
        loaded.verify(),
        Err(Error::Corrupt("metadata string is not UTF-8"))
    );
    assert_eq!(
        loaded.faulty_facts(),
        vec![(FactId(0), FactFault::Metadata)]
    );
}
