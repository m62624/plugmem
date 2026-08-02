//! `maintain` — tombstone purge and satellite compaction (
//! B). Coverage: observable state is preserved, space is
//! reclaimed, ids/chains/edges stay stable, the pass is canonical and
//! replayable byte-for-byte, the compacted image has no orphan chunks or
//! dangling references, vectors survive the bijection, and a random
//! workload is observation-equivalent before and after (proptest).

use plugmem_core::{
    Config, EntityId, FactId, LinkInput, MaintenanceMode, MaintenanceOptions, MemStorage, Memory,
    RecallQuery, RememberInput, Storage,
};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::*;

const DAY: u64 = 86_400_000;

fn cfg(dim: usize) -> Config {
    let mut c = Config::default();
    c.dim = dim;
    c.shards_facts = 16;
    c.shards_entities = 8;
    c.shards_edges = 8;
    c.shards_temporal = 8;
    c.shards_postings = 32;
    c
}

/// A deterministic LCG for vector payloads.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next()).collect()
    }
}

/// Content of one live fact, excluding the storage-internal `text` blob id
/// and `vector` slot index (both legitimately change under compaction).
type Content = (String, EntityId, u16, u16, FactId, u64, u64, u64, Vec<u32>);

fn content(mem: &Memory<'_>, id: FactId) -> Option<Content> {
    let v = mem.get(id)?;
    let r = v.record;
    let mut tags = Vec::new();
    mem.tags_of(id, &mut tags);
    Some((
        v.text.to_string(),
        r.entity,
        r.flags,
        r.kind,
        r.revises,
        r.recorded_at,
        r.valid_from,
        r.valid_to,
        tags.iter().map(|t| t.0).collect(),
    ))
}

/// Every fact id's content, in id order over the full id space (burned
/// and tombstoned ids read as `None` alike — that equality is the point).
fn all_content(mem: &Memory<'_>) -> Vec<Option<Content>> {
    (0..mem.stats().next_fact)
        .map(|id| content(mem, FactId(id)))
        .collect()
}

/// A deterministic pseudo-embedding for step `seed` — same in every path,
/// so matching facts get matching vectors.
fn embed(seed: usize, dim: usize) -> Vec<f32> {
    let mut lcg = Lcg(0x51ED ^ seed as u64);
    lcg.vector(dim)
}

/// A fixed battery of recall queries whose rendered blocks capture the
/// observable behavior of every source (the vector source is queried only
/// when the engine has a vector layer).
fn battery(mem: &mut Memory<'_>) -> Vec<String> {
    let dim = mem.cfg().dim;
    let mut out: Vec<String> = [
        RecallQuery::text(100 * DAY, "memory tokio work fact"),
        RecallQuery {
            tags: &["pref"],
            ..RecallQuery::text(100 * DAY, "engine")
        },
        RecallQuery {
            entities: &["user", "plugmem"],
            ..RecallQuery::text(100 * DAY, "")
        },
        RecallQuery {
            range: Some((0, 100 * DAY)),
            include_closed: true,
            ..RecallQuery::text(100 * DAY, "text")
        },
    ]
    .into_iter()
    .map(|q| mem.recall(q).unwrap().rendered)
    .collect();
    if dim > 0 {
        let qv = embed(2, dim);
        let q = RecallQuery {
            vector: Some(&qv),
            k: 12,
            ..RecallQuery::text(100 * DAY, "")
        };
        out.push(mem.recall(q).unwrap().rendered);
    }
    out
}

/// A workload touching every structure: entities, tags, links, a revision
/// and a tombstone.
fn workload(mem: &mut Memory<'_>, store: &mut MemStorage) {
    let entities = ["user", "plugmem", "кот Барсик", "tokio"];
    for i in 0..60u64 {
        mem.remember(
            store,
            RememberInput {
                entity: Some(entities[(i % 4) as usize]),
                tags: if i % 2 == 0 { &["pref"] } else { &[] },
                links: if i % 10 == 0 {
                    &[("works_on", "plugmem")]
                } else {
                    &[]
                },
                // Repeated "tokio" exercises tf accumulation on the
                // maintain re-tokenization path.
                ..RememberInput::text((i + 1) * DAY, "tokio memory fact about tokio and работа")
            },
        )
        .unwrap();
    }
    mem.revise(
        store,
        FactId(3),
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(70 * DAY, "a revised statement here")
        },
    )
    .unwrap();
    mem.forget(store, 71 * DAY, FactId(7)).unwrap();
    mem.forget(store, 71 * DAY, FactId(12)).unwrap();
    mem.link(
        store,
        LinkInput {
            now: 72 * DAY,
            src: "plugmem",
            rel: "depends_on",
            dst: "tokio",
            provenance: None,
        },
    )
    .unwrap();
}

#[test]
fn maintain_preserves_observable_state() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);

    let before_content = all_content(&mem);
    let before_battery = battery(&mut mem);

    let report = mem.maintain(&mut store, 80 * DAY).unwrap();
    assert_eq!(report.purged, 2, "two facts were forgotten");

    assert_eq!(all_content(&mem), before_content, "fact content changed");
    assert_eq!(battery(&mut mem), before_battery, "recall behavior changed");
    // Ids stayed put: allocation continues from the persisted counter,
    // never reusing the purged ids.
    let next = mem.stats().next_fact;
    let out = mem
        .remember(&mut store, RememberInput::text(90 * DAY, "fresh"))
        .unwrap();
    assert_eq!(out.id.0, next);
}

#[test]
fn auto_maintain_noop_does_not_append_a_marker() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    let report = mem.maintain(&mut store, DAY).unwrap();
    assert!(report.no_op);
    assert_eq!(report.purged, 0);
    assert!(!report.structural_compacted);
    assert_eq!(store.read_journal().unwrap().len(), 0);
}

#[test]
fn ordinary_compaction_reuses_bm25_postings_without_reindexing() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    let keep = mem
        .remember(
            &mut store,
            RememberInput::text(DAY, "alpha beta searchable survivor"),
        )
        .unwrap()
        .id;
    let drop = mem
        .remember(&mut store, RememberInput::text(DAY, "alpha beta removed"))
        .unwrap()
        .id;
    mem.forget(&mut store, 2 * DAY, drop).unwrap();

    let report = mem.maintain(&mut store, 3 * DAY).unwrap();
    assert!(report.structural_compacted);
    assert!(report.bm25_compacted);
    assert!(!report.bm25_reindexed);
    assert_eq!(report.purged, 1);
    assert_eq!(report.tombstones_before, 1);
    assert_eq!(mem.stats().tombstones, 0);

    let out = mem
        .recall(RecallQuery::text(4 * DAY, "searchable survivor"))
        .unwrap();
    assert_eq!(out.facts.first().map(|f| f.id), Some(keep));
    assert!(out.facts.iter().all(|f| f.id != drop));
}

#[test]
fn explicit_reindex_text_rebuilds_bm25_without_compacting_when_no_tombstones() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    mem.remember(&mut store, RememberInput::text(DAY, "alpha beta gamma"))
        .unwrap();
    let report = mem
        .maintain_with_options(
            &mut store,
            2 * DAY,
            MaintenanceOptions {
                mode: MaintenanceMode::ReindexText,
                max_hnsw_inserts: Some(0),
            },
        )
        .unwrap();
    assert!(!report.no_op);
    assert!(!report.structural_compacted);
    assert!(report.bm25_reindexed);
    assert!(!report.bm25_compacted);
    assert_eq!(report.purged, 0);
}

#[test]
fn vector_optimization_can_be_bounded() {
    let dim = 16;
    let mut c = cfg(dim);
    c.flat_to_hnsw = 8;
    let (mut mem, mut store) = (Memory::new(c).unwrap(), MemStorage::new());
    let mut rng = Lcg(0xCAFE);
    for i in 0..32u64 {
        let v = rng.vector(dim);
        mem.remember(
            &mut store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(i * DAY, "vector fact")
            },
        )
        .unwrap();
    }

    let report = mem
        .maintain_with_options(
            &mut store,
            40 * DAY,
            MaintenanceOptions {
                mode: MaintenanceMode::OptimizeVectors,
                max_hnsw_inserts: Some(5),
            },
        )
        .unwrap();
    assert!(report.hnsw_rebuilt);
    assert_eq!(report.hnsw_inserted, 5);
    assert_eq!(mem.stats().hnsw_indexed, 5);
    assert!(mem.maintenance_needed(MaintenanceOptions {
        mode: MaintenanceMode::OptimizeVectors,
        max_hnsw_inserts: Some(5),
    }));
}

#[test]
fn maintain_reclaims_space() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    let big = "x".repeat(2000);
    for i in 0..40u64 {
        mem.remember(&mut store, RememberInput::text(i * DAY, &big))
            .unwrap();
    }
    let before = mem.snapshot_bytes(0).len();
    // Forget half of them.
    for i in (0..40u32).step_by(2) {
        mem.forget(&mut store, 100 * DAY, FactId(i)).unwrap();
    }
    let report = mem.maintain(&mut store, 101 * DAY).unwrap();
    assert_eq!(report.purged, 20);
    assert_eq!(mem.facts_len(), 20, "purged records are physically gone");
    assert!(
        report.bytes_after < report.bytes_before,
        "no reclaim: {} -> {}",
        report.bytes_before,
        report.bytes_after
    );
    // The snapshot shrank too, and survivors are intact.
    assert!(mem.snapshot_bytes(0).len() < before);
    for i in 0..40u32 {
        let present = mem.get(FactId(i)).is_some();
        assert_eq!(
            present,
            i % 2 == 1,
            "fact {i} liveness wrong after maintain"
        );
    }
}

#[test]
fn maintain_is_canonical_and_replayable() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    mem.maintain(&mut store, 80 * DAY).unwrap();
    // A little more work after the maintain marker.
    mem.remember(&mut store, RememberInput::text(90 * DAY, "post-maintain"))
        .unwrap();
    let snap = mem.snapshot_bytes(0);

    // Replaying the journal (which contains the Maintain marker) must
    // reproduce the exact same image, byte for byte.
    let (mut reopened, report) = Memory::open(&mut store, cfg(0)).unwrap();
    assert!(report.replayed > 0 && report.skipped == 0);
    assert_eq!(reopened.snapshot_bytes(0), snap);

    // And save -> load -> save stays canonical.
    let (loaded, _) = Memory::from_bytes(Some(&snap), &[], cfg(0)).unwrap();
    assert_eq!(loaded.snapshot_bytes(0), snap);
    let _ = &mut reopened;
}

#[test]
fn maintain_output_roundtrips_without_orphans() {
    // The loader validates chunk chains (no cycles/orphans) and every
    // stored reference; a clean roundtrip after maintain proves the
    // compacted image is well-formed.
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    mem.maintain(&mut store, 80 * DAY).unwrap();
    let bytes = mem.snapshot_bytes(0);
    let (loaded, _) = Memory::from_bytes(Some(&bytes), &[], cfg(0)).unwrap();
    assert_eq!(loaded.facts_len(), mem.facts_len());
}

#[test]
fn maintain_preserves_ids_chains_and_edges() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    workload(&mut mem, &mut store);
    // Fact 3 was revised: it is closed and fact 60 (the revision) points
    // back at it. Capture the chain before.
    let revised = mem.get(FactId(3)).unwrap().record;
    assert!(revised.is_closed());
    let revision = (0..mem.facts_len() as u32)
        .map(FactId)
        .find(|&f| mem.get(f).map(|v| v.record.revises) == Some(FactId(3)))
        .unwrap();

    mem.maintain(&mut store, 80 * DAY).unwrap();

    // The closed original survives with its interval intact; the revision
    // still points at it (chain preserved).
    let after = mem.get(FactId(3)).unwrap().record;
    assert!(after.is_closed());
    assert_eq!(after.valid_to, revised.valid_to);
    assert_eq!(mem.get(revision).unwrap().record.revises, FactId(3));

    // The edge and its graph recall survive.
    let out = mem
        .recall(RecallQuery {
            entities: &["plugmem"],
            ..RecallQuery::text(80 * DAY, "")
        })
        .unwrap();
    assert!(out.rendered.contains("depends_on"));
}

#[test]
fn maintain_compacts_vectors() {
    let dim = 48;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(0xbeef);
    let mut vecs = Vec::new();
    for i in 0..40u64 {
        let v = rng.vector(dim);
        mem.remember(
            &mut store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(i * DAY, "vectorized fact")
            },
        )
        .unwrap();
        vecs.push(v);
    }
    // Forget the even ids, then compact.
    for i in (0..40u32).step_by(2) {
        mem.forget(&mut store, 100 * DAY, FactId(i)).unwrap();
    }
    mem.maintain(&mut store, 101 * DAY).unwrap();

    // A surviving vector still recalls its own fact first.
    let survivor = 11usize;
    let out = mem
        .recall(RecallQuery {
            vector: Some(&vecs[survivor]),
            k: 5,
            ..RecallQuery::text(101 * DAY, "")
        })
        .unwrap();
    assert_eq!(out.facts[0].id, FactId(survivor as u32));

    // The compacted image roundtrips: the fact<->slot bijection holds
    // (the loader rejects orphan slots and mismatched references).
    let bytes = mem.snapshot_bytes(0);
    let (loaded, _) = Memory::from_bytes(Some(&bytes), &[], cfg(dim)).unwrap();
    assert_eq!(loaded.facts_len(), mem.facts_len());
}

#[test]
fn maintain_on_empty_engine_and_idempotent() {
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    // Empty engine: a no-op that still journals a marker.
    assert_eq!(mem.maintain(&mut store, 1).unwrap().purged, 0);

    workload(&mut mem, &mut store);
    mem.maintain(&mut store, 80 * DAY).unwrap();
    let once = mem.snapshot_bytes(0);
    // A second maintain with nothing new to purge is a canonical no-op.
    let report = mem.maintain(&mut store, 81 * DAY).unwrap();
    assert_eq!(
        report.purged, 0,
        "the first pass removed the records; burned ids purge nothing"
    );
    assert_eq!(
        report.bytes_before, report.bytes_after,
        "nothing left to reclaim"
    );
    assert_eq!(
        mem.snapshot_bytes(0),
        once,
        "second maintain changed the image"
    );
}

/// Drives one workload (with vectors). `maintain_at` marks which step
/// indices trigger a compaction; `now` advances identically whether or not
/// a maintain fires there, so two paths over the same steps stay aligned.
#[cfg(not(target_family = "wasm"))]
fn drive(steps: &[u8], dim: usize, maintain: bool) -> (Memory<'static>, MemStorage) {
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let names = ["user", "plugmem", "кот Барсик", "tokio", "работа"];
    let tags = ["pref", "health", "a", "b"];
    let mut now = 0u64;
    for (i, step) in steps.iter().enumerate() {
        now += DAY;
        let e = names[i % names.len()];
        match step {
            0..=2 => {
                let v = embed(i, dim);
                let _ = mem.remember(
                    &mut store,
                    RememberInput {
                        entity: Some(e),
                        tags: &[tags[i % tags.len()]],
                        links: if step == &2 {
                            &[("rel", "plugmem")]
                        } else {
                            &[]
                        },
                        vector: (dim > 0).then_some(&v[..]),
                        ..RememberInput::text(now, "some memory fact text tokio работа tokio")
                    },
                );
            }
            3 => {
                let _ = mem.revise(
                    &mut store,
                    FactId((i % 8) as u32),
                    RememberInput::text(now, "revised fact text"),
                );
            }
            4 => {
                let _ = mem.forget(&mut store, now, FactId((i % 8) as u32));
            }
            5 => {
                let _ = mem.link(
                    &mut store,
                    LinkInput {
                        now,
                        src: e,
                        rel: "rel",
                        dst: "tokio",
                        provenance: None,
                    },
                );
            }
            // A maintain step: fires only on the maintaining path, but the
            // clock advances on both so ids and timestamps stay aligned.
            _ => {
                if maintain {
                    mem.maintain(&mut store, now).unwrap();
                }
            }
        }
    }
    (mem, store)
}

#[cfg(not(target_family = "wasm"))]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]
    // The strong property: for any workload, running it with maintains
    // interleaved at arbitrary points is observation-equivalent to running
    // it with none — the content of every live fact and the rendered
    // output of every source (vectors included) match — and the maintained
    // path replays to a byte-identical image.
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd, forbidden under miri
    fn maintain_is_observation_preserving(steps in proptest::collection::vec(0u8..7, 0..60)) {
        let dim = 48;
        let (mut with, mut store) = drive(&steps, dim, true);
        let (mut without, _) = drive(&steps, dim, false);

        // Interleaved maintains never change what is observable.
        prop_assert_eq!(all_content(&with), all_content(&without));
        prop_assert_eq!(battery(&mut with), battery(&mut without));

        // Replaying the maintained path's journal (markers included)
        // reproduces its image byte for byte.
        let snap = with.snapshot_bytes(0);
        let (reopened, _) = Memory::open(&mut store, cfg(dim)).unwrap();
        prop_assert_eq!(reopened.snapshot_bytes(0), snap);
    }
}

#[test]
fn purge_is_physical_and_ids_stay_burned() {
    use plugmem_core::Error;
    let (mut mem, mut store) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    // A revision chain (a -> b), a provenance-carrying edge, and a plain
    // fact to forget.
    let a = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(DAY, "old statement")
            },
        )
        .unwrap()
        .id;
    let b = mem
        .revise(
            &mut store,
            a,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(2 * DAY, "new statement")
            },
        )
        .unwrap()
        .id;
    let c = mem
        .remember(&mut store, RememberInput::text(3 * DAY, "edge basis"))
        .unwrap()
        .id;
    mem.link(
        &mut store,
        LinkInput {
            now: 4 * DAY,
            src: "user",
            rel: "works_on",
            dst: "plugmem",
            provenance: Some(c),
        },
    )
    .unwrap();

    // Forget the closed predecessor and the provenance fact, then purge.
    mem.forget(&mut store, 5 * DAY, a).unwrap();
    mem.forget(&mut store, 5 * DAY, c).unwrap();
    let before = mem.stats();
    let report = mem.maintain(&mut store, 6 * DAY).unwrap();
    assert_eq!(report.purged, 2);

    // Physically gone; the id space is untouched.
    let after = mem.stats();
    assert_eq!(after.facts, before.facts - 2);
    assert_eq!(after.next_fact, before.next_fact);
    assert!(mem.get(a).is_none() && mem.get(c).is_none());

    // Burned ids answer like tombstoned ones: NotFound, never reuse.
    assert_eq!(
        mem.forget(&mut store, 7 * DAY, a).unwrap_err(),
        Error::NotFound(a)
    );
    assert_eq!(
        mem.revise(&mut store, a, RememberInput::text(7 * DAY, "x"))
            .unwrap_err(),
        Error::NotFound(a)
    );
    let fresh = mem
        .remember(&mut store, RememberInput::text(8 * DAY, "fresh"))
        .unwrap()
        .id;
    assert_eq!(fresh, FactId(before.next_fact));

    // Dangling references keep the burned id and read as None.
    assert_eq!(mem.get(b).unwrap().record.revises, a);
    let out = mem
        .recall(RecallQuery {
            entities: &["user"],
            ..RecallQuery::text(8 * DAY, "")
        })
        .unwrap();
    let edge = out
        .edges
        .iter()
        .find(|e| e.provenance == c)
        .expect("edge survives with its burned provenance id");
    assert!(mem.get(edge.provenance).is_none());

    // The image with holes and dangling ids roundtrips through the
    // untrusted loader.
    let bytes = mem.snapshot_bytes(0);
    let (loaded, _) = Memory::from_bytes(Some(&bytes), &[], cfg(0)).unwrap();
    assert_eq!(loaded.snapshot_bytes(0), bytes);
    assert_eq!(loaded.facts_len(), mem.facts_len());
}
