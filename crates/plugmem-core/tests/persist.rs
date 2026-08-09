//! Engine snapshot tests (test plan, engine level): canonical
//! roundtrip, snapshot + journal-tail replay, config compatibility gates,
//! and corruption rejection.

use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot};
use plugmem_core::model::EdgeHistorySlot;
use plugmem_core::{
    Config, EdgeSlot, Error, FactId, LinkInput, MAX_SHARDS, MaintenanceOptions, MemScratch,
    MemStorage, Memory, RecallQuery, RememberInput, Scratch, Storage, TagQuery, UnlinkInput,
    snapshot::{FORMAT_VERSION, Snapshot, SnapshotWriter},
};

fn cfg() -> Config {
    Config::default()
}

const DAY: u64 = 86_400_000;
const SNAP_HEADER: usize = 64;
const SNAP_ENTRY: usize = 32;
const SNAP_ALIGN: usize = 64;
const SNAP_FLAGS_AT: usize = 6;
const SNAP_SECTION_COUNT_AT: usize = 8;
const SNAP_CONFIG_LEN_AT: usize = 16;
/// Width of one size field inside the config block (all are `u64`).
const CFG_U64: usize = 8;
/// Offset of `shards_facts` in the config block: it follows `dim`,
/// `max_bytes`, `max_text` and `max_blob`, and the five shard counts are
/// consecutive from there.
const CFG_SHARDS_AT: usize = 4 * CFG_U64;
const SNAP_CREATED_AT: usize = 28;
const KIND_ENGINE_STATE: u16 = 36;
const KIND_EDGES_OUT_META: u16 = 50;
const KIND_EDGES_OUT_POOL: u16 = 51;
const KIND_EDGES_IN_META: u16 = 52;
const KIND_EDGES_IN_POOL: u16 = 53;
const KIND_EDGE_HIST_OUT_META: u16 = 54;
const KIND_EDGE_HIST_OUT_POOL: u16 = 55;
const KIND_EDGE_HIST_IN_META: u16 = 56;
const KIND_EDGE_HIST_IN_POOL: u16 = 57;
/// The eight sections a current image writes for the graph.
const KIND_EDGE_SECTIONS: [u16; 8] = [
    KIND_EDGES_OUT_META,
    KIND_EDGES_OUT_POOL,
    KIND_EDGES_IN_META,
    KIND_EDGES_IN_POOL,
    KIND_EDGE_HIST_OUT_META,
    KIND_EDGE_HIST_OUT_POOL,
    KIND_EDGE_HIST_IN_META,
    KIND_EDGE_HIST_IN_POOL,
];
/// The section kinds older images used for the graph: current edges keyed by
/// triple, and (from the version that introduced history) versions keyed by
/// triple plus edge id.
const LEGACY_KIND_EDGES_OUT_META: u16 = 9;
const LEGACY_KIND_EDGES_OUT_POOL: u16 = 10;
const LEGACY_KIND_EDGES_IN_META: u16 = 11;
const LEGACY_KIND_EDGES_IN_POOL: u16 = 12;
const LEGACY_KIND_EDGE_HIST_OUT_META: u16 = 46;
const LEGACY_KIND_EDGE_HIST_OUT_POOL: u16 = 47;
const LEGACY_KIND_EDGE_HIST_IN_META: u16 = 48;
const LEGACY_KIND_EDGE_HIST_IN_POOL: u16 = 49;
const KIND_BM25_DOCLEN_META: u16 = 58;
const KIND_BM25_DOCLEN_POOL: u16 = 59;
const KIND_TAG_CATALOG: u16 = 60;
const KIND_VECTOR_SPACE: u16 = 61;
const LEGACY_KIND_BM25_DOCLEN_META: u16 = 26;
const LEGACY_KIND_BM25_DOCLEN_POOL: u16 = 27;
const STATE_V2_LEN: usize = 32;

/// A workload touching every structure: entities, tags, links, revisions,
/// tombstones.
fn workload(mem: &mut Memory<'_>, store: &mut MemStorage) {
    for i in 0..50u64 {
        mem.remember(
            store,
            RememberInput {
                entity: Some(["user", "plugmem", "кот Барсик"][(i % 3) as usize]),
                tags: if i % 2 == 0 { &["pref"] } else { &[] },
                links: if i % 10 == 0 {
                    &[("works_on", "plugmem")]
                } else {
                    &[]
                },
                ..RememberInput::text((i + 1) * DAY, "some fact text about работа and tokio")
            },
        )
        .unwrap();
    }
    mem.revise(
        store,
        FactId(3),
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(60 * DAY, "revised statement")
        },
    )
    .unwrap();
    mem.forget(store, 61 * DAY, FactId(7)).unwrap();
    mem.link(
        store,
        LinkInput {
            now: 62 * DAY,
            src: "package",
            rel: "depends_on",
            dst: "runtime",
            provenance: None,
        },
    )
    .unwrap();
    mem.unlink(
        store,
        UnlinkInput {
            now: 63 * DAY,
            src: "package",
            rel: "depends_on",
            dst: "runtime",
        },
    )
    .unwrap();
}

fn assert_equal(a: &mut Memory<'_>, b: &mut Memory<'_>) {
    assert_eq!(a.stats(), b.stats());
    assert_eq!(a.facts_len(), b.facts_len());
    assert_eq!(a.entities_len(), b.entities_len());
    for id in 0..a.facts_len() as u32 {
        let id = FactId(id);
        match (a.get(id), b.get(id)) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(x.text, y.text);
                assert_eq!(x.record, y.record);
            }
            (x, y) => panic!("fact {id:?}: {x:?} vs {y:?}"),
        }
    }
    // Query equivalence across every source.
    let q = RecallQuery {
        entities: &["plugmem"],
        range: Some((0, 100 * DAY)),
        ..RecallQuery::text(100 * DAY, "работа tokio")
    };
    assert_eq!(a.recall(q).unwrap().rendered, b.recall(q).unwrap().rendered);
    let current_graph = RecallQuery {
        entities: &["package"],
        ..RecallQuery::text(100 * DAY, "")
    };
    assert_eq!(
        a.recall(current_graph).unwrap().edges,
        b.recall(current_graph).unwrap().edges
    );
    let historical_graph = RecallQuery {
        entities: &["package"],
        as_of: Some(62 * DAY),
        ..RecallQuery::text(100 * DAY, "")
    };
    assert_eq!(
        a.recall(historical_graph).unwrap().edges,
        b.recall(historical_graph).unwrap().edges
    );
}

#[test]
fn snapshot_roundtrip_is_canonical_and_complete() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(999);

    let (mut loaded, report) = Memory::from_bytes(Some(&bytes), &[], cfg()).unwrap();
    assert_eq!(report.replayed, 0);
    assert_equal(&mut mem, &mut loaded);
    // Canonical: save → load → save is byte-identical.
    assert_eq!(loaded.snapshot_bytes(999), bytes);
    // The loaded engine keeps working.
    let out = loaded
        .remember(&mut store, RememberInput::text(200 * DAY, "fresh"))
        .unwrap();
    assert_eq!(out.id.0 as usize, mem.facts_len());
}

#[test]
fn legacy_snapshot_without_tag_catalog_is_migrated_and_next_snapshot_persists_it() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    mem.remember(
        &mut store,
        RememberInput {
            tags: &["project", "rust"],
            ..RememberInput::text(1, "tagged")
        },
    )
    .unwrap();
    mem.remember(
        &mut store,
        RememberInput {
            tags: &["project"],
            ..RememberInput::text(2, "also tagged")
        },
    )
    .unwrap();
    let current = mem.snapshot_bytes(3);
    let legacy = omit_sections(&current, &[KIND_TAG_CATALOG]);
    assert_eq!(u16::from_le_bytes(legacy[4..6].try_into().unwrap()), 1);
    assert_eq!(FORMAT_VERSION, 1);
    assert!(
        Snapshot::parse(&legacy)
            .unwrap()
            .section(KIND_TAG_CATALOG)
            .is_none()
    );

    let (migrated, _) = Memory::from_bytes(Some(&legacy), &[], cfg()).unwrap();
    let page = migrated.list_tags(TagQuery::default()).unwrap();
    assert_eq!(
        page.items
            .iter()
            .map(|item| (item.name.as_str(), item.count))
            .collect::<Vec<_>>(),
        [("project", 2), ("rust", 1)]
    );
    let upgraded = migrated.snapshot_bytes(4);
    assert_eq!(u16::from_le_bytes(upgraded[4..6].try_into().unwrap()), 1);
    assert!(
        Snapshot::parse(&upgraded)
            .unwrap()
            .section(KIND_TAG_CATALOG)
            .is_some(),
        "the next checkpoint writes the current derived section"
    );
}

#[test]
fn legacy_vectors_open_untracked_and_the_next_snapshot_uses_the_current_shape() {
    let mut config = cfg();
    config.dim = 4;
    let (mut mem, mut store) = (Memory::new(config.clone()).unwrap(), MemStorage::new());
    mem.claim_vector_space(&mut store, "old-model").unwrap();
    mem.remember(
        &mut store,
        RememberInput {
            vector: Some(&[1.0, 0.0, 0.0, 0.0]),
            ..RememberInput::text(1, "legacy vector")
        },
    )
    .unwrap();
    let current = mem.snapshot_bytes(2);
    assert_eq!(
        Snapshot::parse(&current)
            .unwrap()
            .section(KIND_VECTOR_SPACE),
        Some(&b"old-model"[..])
    );

    let legacy = omit_sections(&current, &[KIND_VECTOR_SPACE]);
    assert_eq!(u16::from_le_bytes(legacy[4..6].try_into().unwrap()), 1);
    let (mut migrated, _) = Memory::from_bytes(Some(&legacy), &[], config).unwrap();
    assert_eq!(migrated.vector_space(), None);
    assert_eq!(migrated.stats().vectors, 1);
    assert_eq!(
        migrated
            .claim_vector_space(&mut MemStorage::new(), "old-model")
            .unwrap_err(),
        Error::UntrackedVectorSpace,
        "a missing identity cannot be guessed even when the dimension matches"
    );

    let upgraded = migrated.snapshot_bytes(3);
    assert_eq!(u16::from_le_bytes(upgraded[4..6].try_into().unwrap()), 1);
    assert_eq!(
        Snapshot::parse(&upgraded)
            .unwrap()
            .section(KIND_VECTOR_SPACE),
        Some(&b""[..]),
        "the next checkpoint writes the current optional section without inventing a model"
    );
}

#[test]
fn snapshot_plus_journal_tail_replays_and_skips() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    // Snapshot mid-life, then more operations accumulate in the journal.
    mem.snapshot(&mut store, 70 * DAY).unwrap();
    assert!(store.read_journal().unwrap().is_empty());
    mem.remember(
        &mut store,
        RememberInput::text(80 * DAY, "after the snapshot"),
    )
    .unwrap();
    mem.forget(&mut store, 81 * DAY, FactId(1)).unwrap();

    let (mut reopened, report) = Memory::open(&mut store, cfg()).unwrap();
    assert_eq!(report.replayed, 2);
    assert_eq!(report.skipped, 0);
    assert_equal(&mut mem, &mut reopened);

    // A journal whose head overlaps the snapshot (double-applied tail) is
    // skipped idempotently: append the snapshot-era journal again.
    let mut overlap = MemStorage::new();
    overlap.write_snapshot(&mem.snapshot_bytes(0)).unwrap();
    let mut probe = MemStorage::new();
    let mut fresh = Memory::new(cfg()).unwrap();
    fresh
        .remember(&mut probe, RememberInput::text(1, "will be skipped"))
        .unwrap();
    overlap
        .append_journal(&probe.read_journal().unwrap())
        .unwrap();
    let (reopened, report) = Memory::open(&mut overlap, cfg()).unwrap();
    assert_eq!(report.skipped, 1);
    assert_eq!(reopened.facts_len(), mem.facts_len());
}

/// A read-only borrowed open over the same snapshot bytes is
/// observably identical to an owned open: same facts, same recall across
/// every source. The borrowed engine copies nothing — it reads straight
/// out of `bytes`.
#[test]
fn readonly_borrowed_open_matches_owned() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    // A checkpointed database has an empty journal after a snapshot; the
    // read-only path requires exactly that.
    mem.snapshot(&mut store, 70 * DAY).unwrap();
    let bytes = mem.snapshot_bytes(0);

    let (mut owned, _) = Memory::from_bytes(Some(&bytes), &[], cfg()).unwrap();
    let mut borrowed = Memory::from_bytes_borrowed(&bytes, &[], cfg()).unwrap();
    assert_equal(&mut owned, &mut borrowed);

    // A non-empty journal is refused by the read-only path: the read-only
    // handle has no write verbs, so it must open a checkpointed snapshot.
    let mut probe_store = MemStorage::new();
    let mut probe = Memory::new(cfg()).unwrap();
    probe
        .remember(&mut probe_store, RememberInput::text(1, "journal record"))
        .unwrap();
    let journal = probe_store.read_journal().unwrap();
    assert_eq!(
        Memory::from_bytes_borrowed(&bytes, &journal, cfg()).unwrap_err(),
        Error::Invalid("read-only open requires a checkpointed (empty) journal")
    );
}

#[test]
fn overlay_open_replays_journal_and_matches_owned() {
    // Checkpoint, then a journal tail of writes that the overlay open must
    // replay into the borrowed base without cloning it.
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    mem.snapshot(&mut store, 70 * DAY).unwrap();
    for i in 0..20u64 {
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some("plugmem"),
                ..RememberInput::text((80 + i) * DAY, "post-checkpoint fact tokio работа")
            },
        )
        .unwrap();
    }
    mem.revise(
        &mut store,
        FactId(2),
        RememberInput::text(101 * DAY, "post-checkpoint revision"),
    )
    .unwrap();
    mem.forget(&mut store, 102 * DAY, FactId(4)).unwrap();
    mem.link(
        &mut store,
        LinkInput {
            now: 103 * DAY,
            src: "plugmem",
            rel: "uses",
            dst: "overlay",
            provenance: None,
        },
    )
    .unwrap();
    mem.unlink(
        &mut store,
        UnlinkInput {
            now: 104 * DAY,
            src: "plugmem",
            rel: "uses",
            dst: "overlay",
        },
    )
    .unwrap();

    let snap = store.read_snapshot().unwrap().unwrap();
    let journal = store.read_journal().unwrap();

    // Owned and overlay both open snapshot + journal; overlay borrows the base.
    let (mut owned, _) = Memory::from_bytes(Some(&snap), &journal, cfg()).unwrap();
    let (mut overlay, _) = Memory::from_bytes_overlay(&snap, &journal, cfg()).unwrap();

    assert_equal(&mut owned, &mut overlay);
    assert_eq!(overlay.stats().edges, mem.stats().edges);
    assert_eq!(overlay.stats().edge_versions, mem.stats().edge_versions);
    let plugmem = overlay.entity("plugmem").expect("source entity exists");
    let overlay_entity = overlay
        .entity("overlay")
        .expect("destination entity exists");
    let historical_tail_edge = RecallQuery {
        entities: &["plugmem"],
        as_of: Some(103 * DAY),
        ..RecallQuery::text(110 * DAY, "")
    };
    let edges = overlay.recall(historical_tail_edge).unwrap().edges;
    assert!(
        edges
            .iter()
            .any(|edge| edge.src == plugmem && edge.dst == overlay_entity),
        "overlay replay keeps the closed post-snapshot edge history"
    );
    // Canonical: overlay dumps byte-identically to owned, and to the live
    // engine that produced the snapshot + journal.
    assert_eq!(overlay.snapshot_bytes(0), owned.snapshot_bytes(0));
    assert_eq!(overlay.snapshot_bytes(0), mem.snapshot_bytes(0));
}

#[test]
fn overlay_open_replays_maintain_from_the_journal() {
    // A journal that contains an Op::Maintain: replay must re-run the compaction
    // over the overlay (rebuilding the in-place arenas and chunk pools) and land
    // byte-for-byte where the owned path does.
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    mem.snapshot(&mut store, 70 * DAY).unwrap();
    mem.forget(&mut store, 80 * DAY, FactId(5)).unwrap();
    mem.forget(&mut store, 81 * DAY, FactId(9)).unwrap();
    mem.maintain(&mut store, 82 * DAY).unwrap();
    mem.remember(
        &mut store,
        RememberInput::text(90 * DAY, "after maintain работа"),
    )
    .unwrap();

    let snap = store.read_snapshot().unwrap().unwrap();
    let journal = store.read_journal().unwrap();

    let (mut owned, _) = Memory::from_bytes(Some(&snap), &journal, cfg()).unwrap();
    let (mut overlay, _) = Memory::from_bytes_overlay(&snap, &journal, cfg()).unwrap();

    assert_equal(&mut owned, &mut overlay);
    assert_eq!(overlay.snapshot_bytes(0), owned.snapshot_bytes(0));
    assert_eq!(overlay.snapshot_bytes(0), mem.snapshot_bytes(0));
}

#[test]
fn config_gates_reject_structural_drift() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(0);

    let mut other = cfg();
    other.max_text = 2048;
    assert_eq!(
        Memory::from_bytes(Some(&bytes), &[], other).unwrap_err(),
        Error::ConfigMismatch("stored size limits differ")
    );
    // The HNSW degrees shape the stored graph, so they are a gate too — and
    // they are named as one. Left to be noticed further down, in
    // `HnswGraph::from_parts`, the only symptom is a section whose length is
    // wrong, reported as `Corrupt`: a healthy file accused of damage, which
    // sends the reader looking for the wrong bug entirely.
    for (label, mutate) in [
        ("m", (|c: &mut Config| c.hnsw_m = 32) as fn(&mut Config)),
        ("m0", |c: &mut Config| c.hnsw_m0 = 64),
    ] {
        let mut other = cfg();
        mutate(&mut other);
        assert_eq!(
            Memory::from_bytes(Some(&bytes), &[], other).unwrap_err(),
            Error::ConfigMismatch("stored hnsw degrees differ"),
            "hnsw_{label} must be refused as a config mismatch, not as corruption"
        );
    }

    // Tuning fields may differ freely — this is what makes them tunable at all:
    // reopening with new weights is how a caller changes the ranking.
    let mut other = cfg();
    other.w_bm25 = 2.0;
    other.rrf_k = 30;
    other.hnsw_ef_search = 128;
    other.half_life_days = 30;
    assert!(Memory::from_bytes(Some(&bytes), &[], other).is_ok());
}

/// The shard counts are not a gate. They describe how the file on disk is laid
/// out, and the loader needs the stored ones to read it at all — what the
/// caller happens to carry is irrelevant to that, and refusing over it would
/// mean every caller had to learn the new numbers each time a database
/// re-sharded itself.
#[test]
fn the_stored_shard_layout_is_adopted_rather_than_matched() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    // Put the database on a layout no caller would guess.
    mem.maintain_with_options(&mut store, 100 * DAY, MaintenanceOptions::full())
        .unwrap();
    let stored = mem.stats().shards;
    let bytes = mem.snapshot_bytes(0);

    for other in [cfg(), {
        let mut c = cfg();
        c.shards_facts = 1024;
        c.shards_entities = 1024;
        c.shards_edges = 1024;
        c.shards_temporal = 1024;
        c.shards_postings = 1024;
        c
    }] {
        let (loaded, _) = Memory::from_bytes(Some(&bytes), &[], other).unwrap();
        assert_eq!(loaded.stats().shards, stored);
        // The adopted layout is what the file says, so re-saving describes the
        // same database rather than a differently-shaped one.
        assert_eq!(loaded.snapshot_bytes(0), bytes);
    }
}

#[test]
fn a_snapshot_claiming_absurdly_many_shards_is_refused_before_allocating() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(0);

    // A shard count is an allocation size taken from the file: the arena turns
    // it straight into per-shard vectors plus a page pool. Left unbounded, the
    // value below asks the loader for gigabytes before anything else is read.
    let hostile = (MAX_SHARDS * 2) as u64;
    for (field, message) in [
        (0usize, "shards_facts exceeds MAX_SHARDS"),
        (1, "shards_entities exceeds MAX_SHARDS"),
        (2, "shards_edges exceeds MAX_SHARDS"),
        (3, "shards_temporal exceeds MAX_SHARDS"),
        (4, "shards_postings exceeds MAX_SHARDS"),
    ] {
        let mut damaged = bytes.clone();
        let at = SNAP_HEADER + CFG_SHARDS_AT + field * CFG_U64;
        damaged[at..at + CFG_U64].copy_from_slice(&hostile.to_le_bytes());
        assert_eq!(
            Memory::from_bytes(Some(&damaged), &[], cfg()).unwrap_err(),
            Error::ConfigMismatch(message)
        );
    }
}

#[test]
fn structural_corruption_is_a_typed_error_at_load() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(0);

    // The default open is trust/sparse: only *structural* damage
    // is rejected at load — truncations, bad magic, unknown version. Container
    // checksums are checked on demand by scrub, content by verify(); those and
    // the never-panic contract are covered by the sweep test below.
    for cut in (0..bytes.len()).step_by(513) {
        assert!(Memory::from_bytes(Some(&bytes[..cut]), &[], cfg()).is_err());
    }
    let mut b = bytes.clone();
    b[0] ^= 0xFF; // magic
    assert!(Memory::from_bytes(Some(&b), &[], cfg()).is_err());
    let mut b = bytes.clone();
    b[10] = 1; // reserved header byte must be zero
    assert!(Memory::from_bytes(Some(&b), &[], cfg()).is_err());
}

#[test]
fn incomplete_edge_history_sections_are_rejected() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    mem.link(
        &mut store,
        LinkInput {
            now: DAY,
            src: "package",
            rel: "depends_on",
            dst: "runtime",
            provenance: None,
        },
    )
    .unwrap();
    let mut bytes = mem.snapshot_bytes(0);
    retag_section(&mut bytes, KIND_EDGE_HIST_IN_POOL, 60_000);

    let err = Memory::from_bytes(Some(&bytes), &[], cfg()).unwrap_err();
    assert_eq!(err, Error::Corrupt("snapshot has incomplete edge sections"));
    let err = Memory::from_bytes_borrowed(&bytes, &[], cfg()).unwrap_err();
    assert_eq!(err, Error::Corrupt("snapshot has incomplete edge sections"));
}

#[test]
fn current_snapshot_with_edges_requires_edge_history_sections() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    mem.link(
        &mut store,
        LinkInput {
            now: DAY,
            src: "package",
            rel: "depends_on",
            dst: "runtime",
            provenance: None,
        },
    )
    .unwrap();
    let mut bytes = mem.snapshot_bytes(0);
    for kind in [
        KIND_EDGE_HIST_OUT_META,
        KIND_EDGE_HIST_OUT_POOL,
        KIND_EDGE_HIST_IN_META,
        KIND_EDGE_HIST_IN_POOL,
    ] {
        retag_section(&mut bytes, kind, kind + 60_000);
    }

    // The current-edge halves survive, so this is not an older image — it is
    // a current one missing half its graph.
    let err = Memory::from_bytes(Some(&bytes), &[], cfg()).unwrap_err();
    assert_eq!(err, Error::Corrupt("snapshot has incomplete edge sections"));
    let err = Memory::from_bytes_borrowed(&bytes, &[], cfg()).unwrap_err();
    assert_eq!(err, Error::Corrupt("snapshot has incomplete edge sections"));
}

#[test]
fn snapshot_rejects_edge_counter_below_history_records() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    mem.link(
        &mut store,
        LinkInput {
            now: DAY,
            src: "package",
            rel: "depends_on",
            dst: "runtime",
            provenance: None,
        },
    )
    .unwrap();
    let mut bytes = mem.snapshot_bytes(0);
    let state = section_body(&bytes, KIND_ENGINE_STATE);
    bytes[state.start + 32..state.start + 36].copy_from_slice(&0u32.to_le_bytes());

    let err = Memory::from_bytes(Some(&bytes), &[], cfg()).unwrap_err();
    assert_eq!(
        err,
        Error::Corrupt("engine edge id counter below record count")
    );
    let err = Memory::from_bytes_borrowed(&bytes, &[], cfg()).unwrap_err();
    assert_eq!(
        err,
        Error::Corrupt("engine edge id counter below record count")
    );
}

#[test]
fn legacy_snapshot_without_edge_history_migrates_current_edges() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    mem.link(
        &mut store,
        LinkInput {
            now: DAY,
            src: "package",
            rel: "depends_on",
            dst: "runtime",
            provenance: None,
        },
    )
    .unwrap();
    let legacy = legacy_snapshot(&mem.snapshot_bytes(0), LegacyShape::EdgesOnly);
    let (loaded, _) = Memory::from_bytes(Some(&legacy), &[], cfg()).unwrap();
    loaded.verify().unwrap();

    let stats = loaded.stats();
    assert_eq!(stats.edges, 1);
    assert_eq!(stats.edge_versions, 1);
    assert_eq!(stats.next_edge, 1);
    let current = loaded
        .recall(RecallQuery {
            entities: &["package"],
            ..RecallQuery::text(2 * DAY, "")
        })
        .unwrap();
    assert_eq!(current.edges.len(), 1);
    let historical = loaded
        .recall(RecallQuery {
            entities: &["package"],
            as_of: Some(DAY),
            ..RecallQuery::text(2 * DAY, "")
        })
        .unwrap();
    assert_eq!(historical.edges, current.edges);

    let borrowed = Memory::from_bytes_borrowed(&legacy, &[], cfg()).unwrap();
    assert_eq!(borrowed.stats().edges, 1);
    assert_eq!(borrowed.stats().edge_versions, 1);
}

/// The 0.3.0 shape: history exists but is keyed by triple. Re-keying it by
/// `valid_from` has to preserve every version, every interval, and the
/// current graph — which is re-derived from the versions still open.
#[test]
fn legacy_triple_keyed_history_migrates_to_the_time_ordered_layout() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    // Two triples, one relinked twice and left closed, one left open, plus a
    // third that is opened, closed and reopened — so the fixture carries
    // several versions per triple and both interval shapes.
    for (round, dst) in [(1u64, "runtime"), (2, "compiler"), (3, "runtime")] {
        mem.link(
            &mut store,
            LinkInput {
                now: round * DAY,
                src: "package",
                rel: "depends_on",
                dst,
                provenance: None,
            },
        )
        .unwrap();
    }
    mem.unlink(
        &mut store,
        UnlinkInput {
            now: 4 * DAY,
            src: "package",
            rel: "depends_on",
            dst: "compiler",
        },
    )
    .unwrap();
    let expected = mem.stats();
    let current_before = mem
        .recall(RecallQuery {
            entities: &["package"],
            ..RecallQuery::text(5 * DAY, "")
        })
        .unwrap();
    let historical_before = mem
        .recall(RecallQuery {
            entities: &["package"],
            as_of: Some(3 * DAY),
            ..RecallQuery::text(5 * DAY, "")
        })
        .unwrap();

    let legacy = legacy_snapshot(&mem.snapshot_bytes(0), LegacyShape::TripleKeyedHistory);
    let (loaded, _) = Memory::from_bytes(Some(&legacy), &[], cfg()).unwrap();
    loaded.verify().unwrap();

    let stats = loaded.stats();
    assert_eq!(stats.edges, expected.edges);
    assert_eq!(stats.edge_versions, expected.edge_versions);
    assert_eq!(stats.next_edge, expected.next_edge);
    assert_eq!(
        loaded
            .recall(RecallQuery {
                entities: &["package"],
                ..RecallQuery::text(5 * DAY, "")
            })
            .unwrap()
            .edges,
        current_before.edges,
    );
    assert_eq!(
        loaded
            .recall(RecallQuery {
                entities: &["package"],
                as_of: Some(3 * DAY),
                ..RecallQuery::text(5 * DAY, "")
            })
            .unwrap()
            .edges,
        historical_before.edges,
    );

    // Re-snapshotting the migrated engine writes the current layout, and that
    // image loads without migrating again.
    let upgraded = loaded.snapshot_bytes(0);
    let snap = Snapshot::parse(&upgraded).unwrap();
    for kind in KIND_EDGE_SECTIONS {
        assert!(snap.section(kind).is_some(), "section {kind} was written");
    }
    for kind in [LEGACY_KIND_EDGES_OUT_META, LEGACY_KIND_EDGE_HIST_OUT_META] {
        assert!(snap.section(kind).is_none(), "legacy {kind} was rewritten");
    }
    let (again, _) = Memory::from_bytes(Some(&upgraded), &[], cfg()).unwrap();
    again.verify().unwrap();
    assert_eq!(again.stats().edge_versions, expected.edge_versions);

    let borrowed = Memory::from_bytes_borrowed(&legacy, &[], cfg()).unwrap();
    assert_eq!(borrowed.stats().edge_versions, expected.edge_versions);
}

/// A pre-signature image opens, answers identically, and is upgraded by
/// `maintain` without the tokenizer.
///
/// The term-set summary cannot be recovered at open time — that would mean
/// tokenizing every stored text — so migration marks the documents "unknown"
/// and the write path falls back to reading their text, which is what it did
/// before the summary existed. Compaction then fills the summaries in from the
/// postings, which hold the same term sets transposed.
#[test]
fn legacy_documents_migrate_unsummarized_and_maintain_fills_them_in() {
    let mut store = MemStorage::new();
    let mut mem = Memory::new(cfg()).unwrap();
    let texts = [
        "lives in Berlin and works remotely",
        "lives in Berlin and works remotely most days",
        "prefers oat milk in coffee",
    ];
    for (i, text) in texts.iter().enumerate() {
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some("subject"),
                ..RememberInput::text(DAY + i as u64, text)
            },
        )
        .unwrap();
    }
    // One edge, so the fixture's edge downgrade has something to carry.
    mem.link(
        &mut store,
        LinkInput {
            now: DAY,
            src: "subject",
            rel: "knows",
            dst: "other",
            provenance: None,
        },
    )
    .unwrap();

    // The same corpus in both formats, so the two engines differ in exactly
    // one thing: whether their documents carry a term-set summary.
    let current = mem.snapshot_bytes(0);
    let legacy = legacy_snapshot(&current, LegacyShape::TripleKeyedHistory);
    let legacy_snap = Snapshot::parse(&legacy).unwrap();
    assert!(
        legacy_snap.section(LEGACY_KIND_BM25_DOCLEN_META).is_some()
            && legacy_snap.section(KIND_BM25_DOCLEN_META).is_none(),
        "the fixture must carry only the pre-signature records"
    );

    let (mut summarized, _) = Memory::from_bytes(Some(&current), &[], cfg()).unwrap();
    let (mut loaded, _) = Memory::from_bytes(Some(&legacy), &[], cfg()).unwrap();
    loaded.verify().unwrap();
    assert_eq!(loaded.stats(), summarized.stats());

    // Migrated documents carry no summary at all.
    assert!(
        document_summaries(&loaded.snapshot_bytes(0))
            .iter()
            .all(|&(_, distinct, sig)| distinct == 0 && sig == 0),
        "migration must not invent summaries it cannot know"
    );

    // The same write against both engines must produce the same hints: the
    // unsummarized one reaches them by reading text, the summarized one by
    // ruling candidates out first.
    let probe = |mem: &mut Memory<'_>| {
        let mut store = MemStorage::new();
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some("subject"),
                ..RememberInput::text(2 * DAY, "lives in Berlin and works remotely still")
            },
        )
        .unwrap()
        .similar
    };
    let hints_before = probe(&mut summarized);
    assert!(!hints_before.is_empty(), "the corpus must produce a hint");
    assert_eq!(
        probe(&mut loaded),
        hints_before,
        "an unsummarized image must answer exactly as the summarized one does"
    );

    // Compaction rebuilds the summaries from the postings.
    let mut replay_store = MemStorage::new();
    loaded.maintain(&mut replay_store, 4 * DAY).unwrap();
    let compacted = loaded.snapshot_bytes(0);
    let summaries = document_summaries(&compacted);
    assert!(!summaries.is_empty());
    assert!(
        summaries
            .iter()
            .all(|&(_, distinct, sig)| distinct > 0 && sig != 0),
        "maintain must summarize every document: {summaries:?}"
    );

    // And a document indexed after the migration was summarized on the way in,
    // so the two sources agree on the same corpus.
    let (again, _) = Memory::from_bytes(Some(&compacted), &[], cfg()).unwrap();
    again.verify().unwrap();
    let snap = Snapshot::parse(&compacted).unwrap();
    assert!(snap.section(KIND_BM25_DOCLEN_META).is_some());
    assert!(snap.section(LEGACY_KIND_BM25_DOCLEN_META).is_none());
}

/// The graph's cross-references are checked by `verify`, not by `open`.
///
/// Whether the two edge mirrors hold the same edges, whether an open version
/// is reachable as a current edge, and whether an edge's endpoints exist are
/// consistency properties: an accessor indexes with none of them, so a
/// disagreement makes recall wrong rather than unsafe, and checking them costs
/// a random lookup per edge on every open. Each therefore has to behave the
/// same way — the image opens, reading it does not panic, and `verify` names
/// the problem.
#[test]
fn broken_graph_cross_references_open_and_are_reported_by_verify() {
    let mut store = MemStorage::new();
    let mut mem = Memory::new(cfg()).unwrap();
    for (i, (src, dst)) in [("a", "b"), ("b", "c"), ("c", "d")].iter().enumerate() {
        mem.link(
            &mut store,
            LinkInput {
                now: (i as u64 + 1) * DAY,
                src,
                rel: "knows",
                dst,
                provenance: None,
            },
        )
        .unwrap();
    }
    let healthy = mem.snapshot_bytes(0);
    Memory::from_bytes(Some(&healthy), &[], cfg())
        .unwrap()
        .0
        .verify()
        .unwrap();

    // One mirror loses an edge and gains an unrelated one, so the counts still
    // match but the correspondence does not.
    let mut mirrored = load_edges(&healthy, KIND_EDGES_IN_META, KIND_EDGES_IN_POOL);
    let victim = mirrored.iter().next().expect("edges exist");
    let mut swapped = Arena::<EdgeSlot>::new(ordered_cfg()).unwrap();
    for edge in mirrored.iter() {
        if edge.a == victim.a && edge.b == victim.b {
            swapped
                .insert(&EdgeSlot {
                    a: plugmem_core::EntityId(victim.a.0),
                    b: plugmem_core::EntityId(victim.b.0 + 1),
                    ..victim
                })
                .unwrap();
        } else {
            swapped.insert(&edge).unwrap();
        }
    }
    mirrored = swapped;
    let (meta, pool) = arena_pair(&mirrored);
    let broken = rewrite_sections(
        &healthy,
        &[(KIND_EDGES_IN_META, meta), (KIND_EDGES_IN_POOL, pool)],
    );
    assert_reported(&broken, "edge mirrors disagree");

    // Both current mirrors lose the same edge, so their counts still match and
    // its history version is left open with nothing to reach it from.
    let mut out = Arena::<EdgeSlot>::new(ordered_cfg()).unwrap();
    let mut inn = Arena::<EdgeSlot>::new(ordered_cfg()).unwrap();
    let full_out = load_edges(&healthy, KIND_EDGES_OUT_META, KIND_EDGES_OUT_POOL);
    let dropped = full_out.iter().next().expect("edges exist");
    for edge in full_out.iter() {
        if edge.a == dropped.a && edge.b == dropped.b {
            continue;
        }
        out.insert(&edge).unwrap();
        inn.insert(&EdgeSlot {
            a: edge.b,
            b: edge.a,
            ..edge
        })
        .unwrap();
    }
    let (out_meta, out_pool) = arena_pair(&out);
    let (in_meta, in_pool) = arena_pair(&inn);
    let orphaned = rewrite_sections(
        &healthy,
        &[
            (KIND_EDGES_OUT_META, out_meta),
            (KIND_EDGES_OUT_POOL, out_pool),
            (KIND_EDGES_IN_META, in_meta),
            (KIND_EDGES_IN_POOL, in_pool),
        ],
    );
    assert_reported(&orphaned, "open edge version is not a current edge");
}

/// Opens `bytes`, confirms reading it is panic-free, and that `verify` fails
/// with `message`.
fn assert_reported(bytes: &[u8], message: &'static str) {
    let (mem, _) =
        Memory::from_bytes(Some(bytes), &[], cfg()).expect("a corrupt graph still opens");
    // Rendering walks the edges the graph source returned; an endpoint that
    // does not resolve is skipped, never unwrapped.
    let recalled = mem
        .recall(RecallQuery {
            entities: &["a", "b", "c", "d"],
            k: 64,
            ..RecallQuery::text(10 * DAY, "")
        })
        .unwrap();
    let _ = recalled.rendered;
    for id in 0..mem.facts_len() as u32 + 1 {
        let _ = mem.get(FactId(id));
    }
    assert_eq!(mem.verify(), Err(Error::Corrupt(message)));
}

fn load_edges(bytes: &[u8], meta: u16, pool: u16) -> Arena<'static, EdgeSlot> {
    let snap = Snapshot::parse(bytes).expect("snapshot parses");
    Arena::load(
        ordered_cfg(),
        snap.section(meta).expect("edge section"),
        snap.section(pool).expect("edge section"),
    )
    .expect("edges load")
}

/// Copies `bytes` with the named sections replaced.
fn rewrite_sections(bytes: &[u8], replacements: &[(u16, Vec<u8>)]) -> Vec<u8> {
    let flags = u16::from_le_bytes(bytes[SNAP_FLAGS_AT..SNAP_FLAGS_AT + 2].try_into().unwrap());
    let config_len = u32::from_le_bytes(
        bytes[SNAP_CONFIG_LEN_AT..SNAP_CONFIG_LEN_AT + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    let created_at = u64::from_le_bytes(
        bytes[SNAP_CREATED_AT..SNAP_CREATED_AT + 8]
            .try_into()
            .unwrap(),
    );
    let mut writer = SnapshotWriter::new();
    for (_, kind, offset, len) in section_entries(bytes) {
        let body = match replacements.iter().find(|(k, _)| *k == kind) {
            Some((_, replacement)) => replacement.clone(),
            None => bytes[offset..offset + len].to_vec(),
        };
        writer.section(kind, body).unwrap();
    }
    writer.finish(
        &bytes[SNAP_HEADER..SNAP_HEADER + config_len],
        flags,
        created_at,
        "0.2.0",
    )
}

/// Rebuilds a valid v1 image without selected optional/legacy sections.
fn omit_sections(bytes: &[u8], omitted: &[u16]) -> Vec<u8> {
    let flags = u16::from_le_bytes(bytes[SNAP_FLAGS_AT..SNAP_FLAGS_AT + 2].try_into().unwrap());
    let config_len = u32::from_le_bytes(
        bytes[SNAP_CONFIG_LEN_AT..SNAP_CONFIG_LEN_AT + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    let created_at = u64::from_le_bytes(
        bytes[SNAP_CREATED_AT..SNAP_CREATED_AT + 8]
            .try_into()
            .unwrap(),
    );
    let mut writer = SnapshotWriter::new();
    for (_, kind, offset, len) in section_entries(bytes) {
        if !omitted.contains(&kind) {
            writer
                .section(kind, bytes[offset..offset + len].to_vec())
                .unwrap();
        }
    }
    writer.finish(
        &bytes[SNAP_HEADER..SNAP_HEADER + config_len],
        flags,
        created_at,
        "legacy-test",
    )
}

/// `(fact, distinct, sig)` of every per-document record in a snapshot.
fn document_summaries(bytes: &[u8]) -> Vec<(u32, u16, u64)> {
    let snap = Snapshot::parse(bytes).expect("snapshot parses");
    let doc_len = Arena::<plugmem_core::index::bm25::DocLenSlot>::load(
        doc_len_cfg(),
        snap.section(KIND_BM25_DOCLEN_META)
            .expect("document records present"),
        snap.section(KIND_BM25_DOCLEN_POOL)
            .expect("document records present"),
    )
    .expect("document records load");
    doc_len
        .iter()
        .map(|doc| (doc.fact.0, doc.distinct, doc.sig))
        .collect()
}

// The bitflip sweep relies on `catch_unwind` (unwinding) and is heavy — native
// only, like the proptest sections.
#[cfg(not(target_family = "wasm"))]
#[test]
fn an_open_never_panics_through_access_and_verify_catches_corruption() {
    // The default open is trust/sparse: it does not verify the
    // container checksums, so a corrupt image can reach the engine — content
    // validation is deferred. The contract stays panic-free: a load either
    // errors typed (metadata is still range-checked) or opens, and then every
    // accessor tolerates the bad bytes. `verify()` turns latent corruption into
    // an explicit error.
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(0);

    for at in (0..bytes.len()).step_by(29) {
        let mut b = bytes.clone();
        b[at] ^= 0x40;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Ok((m, _)) = Memory::from_bytes(Some(&b), &[], cfg()) else {
                return; // a typed load error is a fine outcome
            };
            // Access sweep: nothing may panic on the corrupt image.
            let stats = m.stats();
            for i in 0..stats.next_fact {
                let _ = m.get(FactId(i));
            }
            let _ = m.recall(RecallQuery::text(DAY, "работа tokio"));
            let _ = m.snapshot_bytes(0);
            let _ = m.verify(); // Ok or Err, never a panic
        }));
        assert!(outcome.is_ok(), "an access panicked after a flip at {at}");
    }
}

#[test]
fn verify_accepts_a_clean_image_and_reports_deferred_text_corruption() {
    // `verify()` is the on-demand integrity check (SQLite's `integrity_check`):
    // a clean image passes; a stored text corrupted past the (skipped) checksums
    // opens fine and is caught by `verify()`, while `get` hides the fact and
    // nothing panics.
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(DAY, "UNIQUETEXTMARKER here")
        },
    )
    .unwrap();
    let clean = mem.snapshot_bytes(0);
    let (loaded, _) = Memory::from_bytes(Some(&clean), &[], cfg()).unwrap();
    assert!(loaded.verify().is_ok(), "a clean image verifies");

    // Corrupt the stored text: find the marker and make a byte invalid UTF-8.
    let at = clean
        .windows(b"UNIQUETEXTMARKER".len())
        .position(|w| w == b"UNIQUETEXTMARKER")
        .expect("the marker text is stored verbatim");
    let mut bad = clean.clone();
    bad[at] = 0xFF; // not a valid UTF-8 start byte

    let (mut loaded, _) = Memory::from_bytes(Some(&bad), &[], cfg())
        .expect("the trust/sparse default open does not scan the text");
    assert!(
        loaded.get(FactId(0)).is_none(),
        "an unreadable text hides the fact, no panic"
    );
    // Every accessor tolerates the bad bytes: recall renders the fact with an
    // empty body, and a same-subject remember runs lexical similar-detection
    // over the unreadable text — neither panics.
    let mut store = MemStorage::new();
    let out = loaded
        .recall(RecallQuery::text(2 * DAY, "UNIQUETEXTMARKER"))
        .unwrap();
    assert!(
        !out.rendered.contains("UNIQUETEXTMARKER"),
        "corrupt body is empty"
    );
    loaded
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(3 * DAY, "another user fact")
            },
        )
        .unwrap();
    assert_eq!(
        loaded.verify(),
        Err(Error::Corrupt("stored text is not valid UTF-8")),
        "verify() reports the deferred text corruption"
    );
}

#[test]
fn empty_engine_snapshot_roundtrips() {
    let mem = Memory::new(cfg()).unwrap();
    let bytes = mem.snapshot_bytes(0);
    let (mut loaded, _) = Memory::from_bytes(Some(&bytes), &[], cfg()).unwrap();
    assert_eq!(loaded.facts_len(), 0);
    let mut store = MemStorage::new();
    loaded
        .remember(&mut store, RememberInput::text(1, "first"))
        .unwrap();
    assert_eq!(loaded.facts_len(), 1);
}

#[test]
fn db_uuid_is_minted_once_and_gates_opens() {
    let uuid = 0xDEAD_BEEF_0123_4567_89AB_CDEF_0000_0001u128;
    let mut named = cfg();
    named.db_uuid = uuid;
    let (mut mem, mut store) = (Memory::new(named.clone()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    assert_eq!(mem.stats().db_uuid, uuid);
    mem.snapshot(&mut store, 200 * DAY).unwrap();
    let snap = mem.snapshot_bytes(0);

    // A caller passing 0 adopts the stored identity — and stays canonical.
    let (adopted, _) = Memory::open(&mut store, cfg()).unwrap();
    assert_eq!(adopted.stats().db_uuid, uuid);
    assert_eq!(adopted.snapshot_bytes(0), snap);

    // A matching nonzero assertion opens fine.
    let (matched, _) = Memory::open(&mut store, named).unwrap();
    assert_eq!(matched.stats().db_uuid, uuid);

    // A different nonzero assertion is a typed refusal: wrong database.
    let mut other = cfg();
    other.db_uuid = uuid + 1;
    assert_eq!(
        Memory::open(&mut store, other).unwrap_err(),
        Error::ConfigMismatch("stored db_uuid differs")
    );

    // The identity survives maintain and re-saves (lineage, not state).
    let (mut kept, _) = Memory::open(&mut store, cfg()).unwrap();
    kept.maintain(&mut store, 300 * DAY).unwrap();
    assert_eq!(kept.stats().db_uuid, uuid);
}

/// A sink that records every write so the test can prove the writer streams
/// section by section instead of assembling one full-image
/// buffer. Storage lives outside the sink, so the impl is on `&mut`.
#[derive(Default)]
struct RecordingSink {
    out: Vec<u8>,
    writes: Vec<usize>,
}

impl plugmem_core::snapshot::SnapshotSink for &mut RecordingSink {
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.writes.push(bytes.len());
        self.out.extend_from_slice(bytes);
        Ok(())
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        let at = at as usize;
        self.out[at..at + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

#[test]
fn write_snapshot_to_streams_and_matches_snapshot_bytes() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);

    let canonical = mem.snapshot_bytes(999);
    let mut sink = RecordingSink::default();
    mem.write_snapshot_to(999, &mut sink).unwrap();

    // Streaming is byte-identical to the materialized path.
    assert_eq!(
        sink.out, canonical,
        "streamed bytes must equal snapshot_bytes"
    );

    // It streams: many bounded writes, and no single write is the whole image
    // (that would mean a full-image buffer was assembled).
    assert!(
        sink.writes.len() > 40,
        "expected one write per section body + padding, got {}",
        sink.writes.len()
    );
    let largest = sink.writes.iter().copied().max().unwrap();
    assert!(
        largest < sink.out.len(),
        "no write should span the whole image (largest {largest}, file {})",
        sink.out.len()
    );
}

/// Milestone H: the disk-first rebuild must produce a snapshot
/// **byte-identical** to one taken after an in-RAM `maintain` — the load-bearing
/// invariant that lets a big database be maintained/recovered on either path.
/// `MemScratch` drives the streaming path deterministically, with no files.
fn vector_corpus(mem: &mut Memory<'_>, store: &mut MemStorage, n: u64) {
    for i in 0..n {
        let v: Vec<f32> = (0..8).map(|k| ((i * 7 + k) % 13) as f32 / 13.0).collect();
        mem.remember(
            store,
            RememberInput {
                entity: Some(["user", "plugmem", "кот"][(i % 3) as usize]),
                tags: if i % 2 == 0 { &["pref"] } else { &[] },
                vector: Some(&v),
                ..RememberInput::text((i + 1) * DAY, "факт about работа and tokio vectors")
            },
        )
        .unwrap();
    }
}

/// Runs a disk-first snapshot and an in-RAM `maintain`+snapshot from the same
/// state and asserts the bytes match. Returns the purge count.
fn assert_disk_first_matches(mem: &mut Memory<'_>, store: &mut MemStorage, now: u64) -> usize {
    // Disk-first is read-only over `mem`: stream the big pools through scratch.
    let mut disk = Vec::new();
    let (mut ts, mut vs) = (MemScratch::new(), MemScratch::new());
    let purged = mem
        .snapshot_disk_first(now, &mut ts, &mut vs, &mut disk)
        .unwrap();
    // In-RAM: maintain (mutates `mem`) then snapshot.
    mem.maintain(store, now).unwrap();
    let in_ram = mem.snapshot_bytes(now);
    assert_eq!(
        disk, in_ram,
        "disk-first output must be byte-identical to in-RAM"
    );
    purged
}

#[test]
fn disk_first_maintain_is_byte_identical_to_in_memory() {
    let mut c = cfg();
    c.dim = 8;
    c.flat_to_hnsw = 16; // low, so 56 survivors actually build the HNSW graph

    let mut mem = Memory::new(c.clone()).unwrap();
    let mut store = MemStorage::new();
    vector_corpus(&mut mem, &mut store, 60);
    // Tombstones (to purge) and a revision (a closed record, kept).
    for id in [5u32, 13, 27, 41] {
        mem.forget(&mut store, 100 * DAY, FactId(id)).unwrap();
    }
    mem.revise(
        &mut store,
        FactId(9),
        RememberInput::text(101 * DAY, "revised"),
    )
    .unwrap();

    // Round 1: the HNSW graph is built from scratch on both paths.
    let purged = assert_disk_first_matches(&mut mem, &mut store, 200 * DAY);
    assert_eq!(purged, 4, "the four tombstones were purged");

    // The disk-first image is a valid, loadable snapshot.
    let mut disk = Vec::new();
    let (mut ts, mut vs) = (MemScratch::new(), MemScratch::new());
    mem.snapshot_disk_first(201 * DAY, &mut ts, &mut vs, &mut disk)
        .unwrap();
    let (reloaded, _) = Memory::from_bytes(Some(&disk), &[], c).unwrap();
    assert_eq!(reloaded.stats().facts, mem.stats().facts);

    // Round 2: the graph now exists, so the rebuild takes the carry-over
    // (remapped) path — still byte-identical.
    mem.forget(&mut store, 300 * DAY, FactId(20)).unwrap();
    mem.forget(&mut store, 300 * DAY, FactId(33)).unwrap();
    let purged = assert_disk_first_matches(&mut mem, &mut store, 400 * DAY);
    assert_eq!(purged, 2, "the two new tombstones were purged");
}

/// A `Scratch` that fails once a byte budget is exhausted — proves the
/// disk-first rebuild surfaces a staging failure as a typed error instead of
/// panicking or writing a truncated image.
struct FailingScratch {
    budget: usize,
}

impl Scratch for FailingScratch {
    type Error = &'static str;

    fn write(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        self.budget = self.budget.checked_sub(bytes.len()).ok_or("scratch full")?;
        Ok(())
    }

    fn len(&self) -> u64 {
        0
    }

    fn freeze(&mut self) -> Result<&[u8], &'static str> {
        Ok(&[])
    }
}

#[test]
fn disk_first_propagates_a_scratch_error() {
    let mut c = cfg();
    c.dim = 8;
    let mut mem = Memory::new(c).unwrap();
    let mut store = MemStorage::new();
    vector_corpus(&mut mem, &mut store, 5);
    mem.forget(&mut store, DAY, FactId(1)).unwrap();

    // The text scratch runs out mid-rebuild: the error propagates as Storage.
    let mut text = FailingScratch { budget: 10 };
    let mut vec = MemScratch::new();
    let mut out = Vec::new();
    assert!(matches!(
        mem.snapshot_disk_first(1, &mut text, &mut vec, &mut out),
        Err(Error::Storage(_))
    ));
}

fn align_up(v: usize) -> usize {
    v.div_ceil(SNAP_ALIGN) * SNAP_ALIGN
}

fn table_start(bytes: &[u8]) -> usize {
    let config_len = u32::from_le_bytes(
        bytes[SNAP_CONFIG_LEN_AT..SNAP_CONFIG_LEN_AT + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    align_up(SNAP_HEADER + config_len)
}

fn section_count(bytes: &[u8]) -> usize {
    u16::from_le_bytes(
        bytes[SNAP_SECTION_COUNT_AT..SNAP_SECTION_COUNT_AT + 2]
            .try_into()
            .unwrap(),
    ) as usize
}

fn section_entries(bytes: &[u8]) -> impl Iterator<Item = (usize, u16, usize, usize)> + '_ {
    let start = table_start(bytes);
    (0..section_count(bytes)).map(move |i| {
        let at = start + i * SNAP_ENTRY;
        let kind = u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
        let offset = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap()) as usize;
        let len = u64::from_le_bytes(bytes[at + 16..at + 24].try_into().unwrap()) as usize;
        (at, kind, offset, len)
    })
}

fn retag_section(bytes: &mut [u8], old: u16, new: u16) {
    let (at, _, _, _) = section_entries(bytes)
        .find(|(_, kind, _, _)| *kind == old)
        .expect("section exists");
    bytes[at..at + 2].copy_from_slice(&new.to_le_bytes());
}

fn section_body(bytes: &[u8], want: u16) -> std::ops::Range<usize> {
    let (_, _, offset, len) = section_entries(bytes)
        .find(|(_, kind, _, _)| *kind == want)
        .expect("section exists");
    offset..offset + len
}

/// A current edge as older versions wrote it: key `[a | rel | b]`, payload
/// `fact`, no link to its history version.
#[derive(Clone, Copy)]
struct LegacyEdgeSlot {
    a: u32,
    rel: u32,
    b: u32,
    fact: u32,
}

impl Slot for LegacyEdgeSlot {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 12;

    fn write(&self, out: &mut [u8]) {
        out[0..4].copy_from_slice(&self.a.to_be_bytes());
        out[4..8].copy_from_slice(&self.rel.to_be_bytes());
        out[8..12].copy_from_slice(&self.b.to_be_bytes());
        out[12..16].copy_from_slice(&self.fact.to_be_bytes());
    }

    fn read(b: &[u8]) -> Self {
        let at = |i: usize| u32::from_be_bytes(b[i..i + 4].try_into().unwrap());
        Self {
            a: at(0),
            rel: at(4),
            b: at(8),
            fact: at(12),
        }
    }
}

/// A per-document BM25 record as older versions wrote it: key `[fact]`,
/// payload `len` and two reserved bytes, with no term-set summary.
#[derive(Clone, Copy)]
struct LegacyDocLenSlot {
    fact: u32,
    len: u16,
}

impl Slot for LegacyDocLenSlot {
    const SIZE: usize = 8;
    const KEY_LEN: usize = 4;

    fn write(&self, out: &mut [u8]) {
        out[0..4].copy_from_slice(&self.fact.to_be_bytes());
        out[4..6].copy_from_slice(&self.len.to_be_bytes());
        out[6..8].copy_from_slice(&[0, 0]);
    }

    fn read(b: &[u8]) -> Self {
        Self {
            fact: u32::from_be_bytes(b[0..4].try_into().unwrap()),
            len: u16::from_be_bytes(b[4..6].try_into().unwrap()),
        }
    }
}

/// An edge version as the first history-carrying version wrote it: key
/// `[a | rel | b | edge]`, so versions were grouped by triple.
#[derive(Clone, Copy)]
struct LegacyEdgeHistorySlot {
    a: u32,
    rel: u32,
    b: u32,
    edge: u32,
    fact: u32,
    flags: u16,
    recorded_at: u64,
    valid_from: u64,
    valid_to: u64,
}

impl Slot for LegacyEdgeHistorySlot {
    const SIZE: usize = 48;
    const KEY_LEN: usize = 16;

    fn write(&self, out: &mut [u8]) {
        out[0..4].copy_from_slice(&self.a.to_be_bytes());
        out[4..8].copy_from_slice(&self.rel.to_be_bytes());
        out[8..12].copy_from_slice(&self.b.to_be_bytes());
        out[12..16].copy_from_slice(&self.edge.to_be_bytes());
        out[16..20].copy_from_slice(&self.fact.to_be_bytes());
        out[20..22].copy_from_slice(&self.flags.to_be_bytes());
        out[22..24].copy_from_slice(&0u16.to_be_bytes());
        out[24..32].copy_from_slice(&self.recorded_at.to_be_bytes());
        out[32..40].copy_from_slice(&self.valid_from.to_be_bytes());
        out[40..48].copy_from_slice(&self.valid_to.to_be_bytes());
    }

    fn read(b: &[u8]) -> Self {
        let u32_at = |i: usize| u32::from_be_bytes(b[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_be_bytes(b[i..i + 8].try_into().unwrap());
        Self {
            a: u32_at(0),
            rel: u32_at(4),
            b: u32_at(8),
            edge: u32_at(12),
            fact: u32_at(16),
            flags: u16::from_be_bytes(b[20..22].try_into().unwrap()),
            recorded_at: u64_at(24),
            valid_from: u64_at(32),
            valid_to: u64_at(40),
        }
    }
}

/// How much of the graph an older image carried.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LegacyShape {
    /// Current edges only — before edge history existed.
    EdgesOnly,
    /// Current edges plus versions keyed by triple.
    TripleKeyedHistory,
}

fn ordered_cfg() -> ArenaCfg {
    ArenaCfg::new(cfg().shards_edges, ShardMode::Ordered)
}

fn doc_len_cfg() -> ArenaCfg {
    ArenaCfg::new(cfg().shards_postings, ShardMode::Uniform)
}

fn arena_pair<T: Slot>(a: &Arena<'_, T>) -> (Vec<u8>, Vec<u8>) {
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    a.dump_meta(&mut meta);
    a.dump_pool(&mut pool);
    (meta, pool)
}

/// Rewrites a current snapshot into one an older version of this crate would
/// have produced: the graph goes back into the legacy sections, the
/// per-document BM25 records lose their term-set summary, and the engine state
/// loses the fields that version did not have. The rest of the image is copied
/// verbatim.
///
/// Both downgrades happen together because both layouts changed after the same
/// release: an image old enough to hold triple-keyed edges also holds 8-byte
/// document records, so a fixture with one and not the other would be a shape
/// that never existed.
///
/// Deriving the fixture from a real snapshot — rather than hand-assembling
/// one — keeps it honest: every other section is exactly what the engine
/// writes, so the test exercises the migration and nothing else.
fn legacy_snapshot(bytes: &[u8], shape: LegacyShape) -> Vec<u8> {
    let snap = Snapshot::parse(bytes).expect("current snapshot parses");
    let section = |kind: u16| snap.section(kind).expect("edge section is present");
    let hist_out = Arena::<EdgeHistorySlot>::load(
        ordered_cfg(),
        section(KIND_EDGE_HIST_OUT_META),
        section(KIND_EDGE_HIST_OUT_POOL),
    )
    .expect("history loads");
    let current = Arena::<EdgeSlot>::load(
        ordered_cfg(),
        section(KIND_EDGES_OUT_META),
        section(KIND_EDGES_OUT_POOL),
    )
    .expect("current edges load");

    // Legacy current edges: the same triples, without the version link.
    let mut legacy_out = Arena::<LegacyEdgeSlot>::new(ordered_cfg()).unwrap();
    let mut legacy_in = Arena::<LegacyEdgeSlot>::new(ordered_cfg()).unwrap();
    for edge in current.iter() {
        let slot = LegacyEdgeSlot {
            a: edge.a.0,
            rel: edge.rel.0,
            b: edge.b.0,
            fact: edge.fact.0,
        };
        legacy_out.insert(&slot).unwrap();
        legacy_in
            .insert(&LegacyEdgeSlot {
                a: slot.b,
                b: slot.a,
                ..slot
            })
            .unwrap();
    }
    let mut legacy_hist_out = Arena::<LegacyEdgeHistorySlot>::new(ordered_cfg()).unwrap();
    let mut legacy_hist_in = Arena::<LegacyEdgeHistorySlot>::new(ordered_cfg()).unwrap();
    for version in hist_out.iter() {
        let slot = LegacyEdgeHistorySlot {
            a: version.a.0,
            rel: version.rel.0,
            b: version.b.0,
            edge: version.edge.0,
            fact: version.fact.0,
            flags: version.flags,
            recorded_at: version.recorded_at,
            valid_from: version.valid_from,
            valid_to: version.valid_to,
        };
        legacy_hist_out.insert(&slot).unwrap();
        legacy_hist_in
            .insert(&LegacyEdgeHistorySlot {
                a: slot.b,
                b: slot.a,
                ..slot
            })
            .unwrap();
    }

    // Per-document records, narrowed back to `[fact | len | pad]`.
    let doc_len = Arena::<plugmem_core::index::bm25::DocLenSlot>::load(
        doc_len_cfg(),
        section(KIND_BM25_DOCLEN_META),
        section(KIND_BM25_DOCLEN_POOL),
    )
    .expect("document records load");
    let mut legacy_doc_len = Arena::<LegacyDocLenSlot>::new(doc_len_cfg()).unwrap();
    for doc in doc_len.iter() {
        legacy_doc_len
            .insert(&LegacyDocLenSlot {
                fact: doc.fact.0,
                len: doc.len,
            })
            .unwrap();
    }

    let flags = u16::from_le_bytes(bytes[SNAP_FLAGS_AT..SNAP_FLAGS_AT + 2].try_into().unwrap());
    let config_len = u32::from_le_bytes(
        bytes[SNAP_CONFIG_LEN_AT..SNAP_CONFIG_LEN_AT + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    let created_at = u64::from_le_bytes(
        bytes[SNAP_CREATED_AT..SNAP_CREATED_AT + 8]
            .try_into()
            .unwrap(),
    );
    let config = &bytes[SNAP_HEADER..SNAP_HEADER + config_len];
    let mut writer = SnapshotWriter::new();
    for (_, kind, offset, len) in section_entries(bytes) {
        if KIND_EDGE_SECTIONS.contains(&kind)
            || kind == KIND_BM25_DOCLEN_META
            || kind == KIND_BM25_DOCLEN_POOL
        {
            continue;
        }
        let mut section = bytes[offset..offset + len].to_vec();
        if kind == KIND_ENGINE_STATE && shape == LegacyShape::EdgesOnly {
            // That version had no edge-version counter.
            section.truncate(STATE_V2_LEN);
        }
        writer.section(kind, section).unwrap();
    }
    let (meta, pool) = arena_pair(&legacy_doc_len);
    writer.section(LEGACY_KIND_BM25_DOCLEN_META, meta).unwrap();
    writer.section(LEGACY_KIND_BM25_DOCLEN_POOL, pool).unwrap();
    let (meta, pool) = arena_pair(&legacy_out);
    writer.section(LEGACY_KIND_EDGES_OUT_META, meta).unwrap();
    writer.section(LEGACY_KIND_EDGES_OUT_POOL, pool).unwrap();
    let (meta, pool) = arena_pair(&legacy_in);
    writer.section(LEGACY_KIND_EDGES_IN_META, meta).unwrap();
    writer.section(LEGACY_KIND_EDGES_IN_POOL, pool).unwrap();
    if shape == LegacyShape::TripleKeyedHistory {
        let (meta, pool) = arena_pair(&legacy_hist_out);
        writer
            .section(LEGACY_KIND_EDGE_HIST_OUT_META, meta)
            .unwrap();
        writer
            .section(LEGACY_KIND_EDGE_HIST_OUT_POOL, pool)
            .unwrap();
        let (meta, pool) = arena_pair(&legacy_hist_in);
        writer.section(LEGACY_KIND_EDGE_HIST_IN_META, meta).unwrap();
        writer.section(LEGACY_KIND_EDGE_HIST_IN_POOL, pool).unwrap();
    }
    writer.finish(config, flags, created_at, "0.2.0")
}

/// `next_fact` counts ids ever issued, so the loader can only check it from
/// below — a long-lived database legitimately outruns its record count. That
/// makes it the wrong thing to size work with: a file is free to claim four
/// billion over a handful of records, and every loop or array that trusted it
/// became a four-billion-step spin or a multi-gigabyte allocation.
#[test]
fn an_inflated_id_counter_does_not_size_the_work() {
    /// Offset of `next_fact` in the engine-state section: it is first.
    const STATE_NEXT_FACT_AT: usize = 0;

    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(0);

    let mut state = bytes[section_body(&bytes, KIND_ENGINE_STATE)].to_vec();
    let honest = u32::from_le_bytes(state[..4].try_into().unwrap());
    // Far above the record count, and far above what any array indexed by it
    // could be allowed to reach.
    let inflated = 4_000_000_000u32;
    state[STATE_NEXT_FACT_AT..STATE_NEXT_FACT_AT + 4].copy_from_slice(&inflated.to_le_bytes());
    let damaged = rewrite_sections(&bytes, &[(KIND_ENGINE_STATE, state)]);

    // Opening is fine: the counter is above the records, which is legal.
    let (mut loaded, _) = Memory::from_bytes(Some(&damaged), &[], cfg()).unwrap();
    assert_eq!(loaded.stats().next_fact, inflated);
    assert_eq!(loaded.stats().facts, mem.stats().facts);

    // The passes that used to walk the id space now walk the records, so each
    // of these returns instead of spinning through four billion ids or asking
    // the allocator for an array that size.
    assert_eq!(loaded.faulty_facts(), Vec::new());
    loaded.verify().unwrap();
    let mut fresh = MemStorage::new();
    fresh.write_snapshot(&damaged).unwrap();
    loaded
        .maintain_with_options(&mut fresh, 200 * DAY, MaintenanceOptions::full())
        .unwrap();
    assert_eq!(loaded.stats().facts, mem.stats().facts - 1);
    // The burned ids survive the pass: the counter is state, not a guess.
    assert_eq!(loaded.stats().next_fact, inflated);
    assert!(honest < inflated);
}
