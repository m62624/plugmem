//! Engine snapshot tests (specs/03 test plan, engine level): canonical
//! roundtrip, snapshot + journal-tail replay, config compatibility gates,
//! and corruption rejection.

use plugmem_core::{
    Config, Error, FactId, LinkInput, MemStorage, Memory, RecallQuery, RememberInput, Storage,
};

fn cfg() -> Config {
    let mut cfg = Config::default();
    cfg.shards_facts = 8;
    cfg.shards_entities = 4;
    cfg.shards_edges = 4;
    cfg.shards_temporal = 4;
    cfg.shards_postings = 16;
    cfg
}

const DAY: u64 = 86_400_000;

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
            src: "plugmem",
            rel: "depends_on",
            dst: "tokio",
            provenance: None,
        },
    )
    .unwrap();
}

fn assert_equal(a: &mut Memory<'_>, b: &mut Memory<'_>) {
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

/// A read-only borrowed open (specs/16 §8) over the same snapshot bytes is
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

    let snap = store.read_snapshot().unwrap().unwrap();
    let journal = store.read_journal().unwrap();

    // Owned and overlay both open snapshot + journal; overlay borrows the base.
    let (mut owned, _) = Memory::from_bytes(Some(&snap), &journal, cfg()).unwrap();
    let (mut overlay, _) = Memory::from_bytes_overlay(&snap, &journal, cfg()).unwrap();

    assert_equal(&mut owned, &mut overlay);
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
    other.shards_facts = 16;
    assert_eq!(
        Memory::from_bytes(Some(&bytes), &[], other).unwrap_err(),
        Error::ConfigMismatch("stored shard counts differ")
    );
    let mut other = cfg();
    other.max_text = 2048;
    assert_eq!(
        Memory::from_bytes(Some(&bytes), &[], other).unwrap_err(),
        Error::ConfigMismatch("stored size limits differ")
    );
    // Tuning fields may differ freely.
    let mut other = cfg();
    other.w_bm25 = 2.0;
    other.rrf_k = 30;
    assert!(Memory::from_bytes(Some(&bytes), &[], other).is_ok());
}

#[test]
fn corrupt_snapshots_are_typed_errors() {
    let (mut mem, mut store) = (Memory::new(cfg()).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    let bytes = mem.snapshot_bytes(0);

    // Any single-byte flip fails typed (container checksums), never
    // panics — sample the sweep to keep the test fast.
    for at in (0..bytes.len()).step_by(97) {
        let mut b = bytes.clone();
        b[at] ^= 0x40;
        assert!(
            Memory::from_bytes(Some(&b), &[], cfg()).is_err(),
            "flip at {at} accepted"
        );
    }
    // Truncations too.
    for cut in (0..bytes.len()).step_by(513) {
        assert!(Memory::from_bytes(Some(&bytes[..cut]), &[], cfg()).is_err());
    }
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
