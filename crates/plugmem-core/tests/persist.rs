//! Engine snapshot tests (test plan, engine level): canonical
//! roundtrip, snapshot + journal-tail replay, config compatibility gates,
//! and corruption rejection.

use plugmem_core::{
    Config, Error, FactId, LinkInput, MemScratch, MemStorage, Memory, RecallQuery, RememberInput,
    Scratch, Storage,
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
