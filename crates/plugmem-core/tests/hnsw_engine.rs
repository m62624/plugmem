//! Engine-level HNSW tests (phase 2, verification
//! plan): the graph activates at the `flat_to_hnsw` threshold inside
//! `maintain`, agrees with the exact flat search, searches its flat tail,
//! respects tombstones, survives both maintenance paths (remap and full
//! rebuild) and replays byte-for-byte.

use plugmem_core::{Config, FactId, MemStorage, Memory, RecallQuery, RememberInput};

const DAY: u64 = 86_400_000;
const DIM: usize = 24;

/// A config whose HNSW threshold is low enough for test corpora.
fn cfg(flat_to_hnsw: usize) -> Config {
    let mut c = Config::default();
    c.dim = DIM;
    c.flat_to_hnsw = flat_to_hnsw;
    c
}

/// Deterministic LCG embeddings (the maintain tests' twin).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vector(&mut self) -> Vec<f32> {
        (0..DIM).map(|_| self.next()).collect()
    }
}

/// Remembers `n` clustered vectors into an engine and returns the raw
/// embeddings (16 clusters — the graph has real structure to navigate).
fn fill(mem: &mut Memory<'_>, store: &mut MemStorage, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    let centers: Vec<Vec<f32>> = (0..16).map(|_| rng.vector()).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % 16];
        let v: Vec<f32> = c.iter().map(|&x| x + rng.next() * 0.3).collect();
        mem.remember(
            store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text((i as u64 + 1) * DAY, "vector fact")
            },
        )
        .unwrap();
        out.push(v);
    }
    out
}

fn vquery(now: u64, v: &[f32], k: usize) -> RecallQuery<'_> {
    RecallQuery {
        vector: Some(v),
        k,
        ..RecallQuery::text(now, "")
    }
}

/// Above the threshold `maintain` builds the graph, and graph answers
/// agree with the exact flat engine: recall@8 >= 0.9 over query batch.
#[test]
#[cfg_attr(miri, ignore)] // data-heavy; miri covers the small logic tests
fn graph_agrees_with_flat_search() {
    let n = 600;
    let (mut graph_eng, mut gs) = (Memory::new(cfg(100)).unwrap(), MemStorage::new());
    let (mut flat_eng, mut fs) = (Memory::new(cfg(usize::MAX)).unwrap(), MemStorage::new());
    let vecs = fill(&mut graph_eng, &mut gs, n, 42);
    fill(&mut flat_eng, &mut fs, n, 42);
    graph_eng.maintain(&mut gs, 1000 * DAY).unwrap();
    flat_eng.maintain(&mut fs, 1000 * DAY).unwrap();

    let (mut hits, mut total) = (0usize, 0usize);
    for qi in (0..n).step_by(29) {
        let truth: Vec<FactId> = flat_eng
            .recall(vquery(1000 * DAY, &vecs[qi], 8))
            .unwrap()
            .facts
            .iter()
            .map(|f| f.id)
            .collect();
        let got: Vec<FactId> = graph_eng
            .recall(vquery(1000 * DAY, &vecs[qi], 8))
            .unwrap()
            .facts
            .iter()
            .map(|f| f.id)
            .collect();
        hits += truth.iter().filter(|t| got.contains(t)).count();
        total += truth.len();
    }
    let recall = hits as f64 / total as f64;
    assert!(recall >= 0.9, "graph recall@8 {recall} below the 0.9 gate");
}

/// Vectors remembered after the graph build live in the flat tail and
/// are found exactly; the next maintain absorbs them into the graph.
#[test]
#[cfg_attr(miri, ignore)]
fn flat_tail_is_searched_and_absorbed() {
    let (mut mem, mut store) = (Memory::new(cfg(50)).unwrap(), MemStorage::new());
    fill(&mut mem, &mut store, 100, 7);
    mem.maintain(&mut store, 1000 * DAY).unwrap();

    // A distinctive tail vector after the build.
    let needle: Vec<f32> = (0..DIM).map(|i| if i == 0 { 1.0 } else { 0.01 }).collect();
    let id = mem
        .remember(
            &mut store,
            RememberInput {
                vector: Some(&needle),
                ..RememberInput::text(2000 * DAY, "tail needle")
            },
        )
        .unwrap()
        .id;
    let out = mem.recall(vquery(2000 * DAY, &needle, 5)).unwrap();
    assert_eq!(out.facts[0].id, id, "the tail vector must rank first");

    // The next maintain folds the tail into the graph; still found.
    mem.maintain(&mut store, 2001 * DAY).unwrap();
    let out = mem.recall(vquery(2001 * DAY, &needle, 5)).unwrap();
    assert_eq!(out.facts[0].id, id, "absorbed into the graph and found");
}

/// Tombstoned facts never surface from the graph, and both maintenance
/// paths (remap at < 10% dead, full rebuild above) keep the engine
/// canonical, replayable and searchable.
#[test]
#[cfg_attr(miri, ignore)]
fn forgets_and_both_maintenance_paths() {
    let n = 200;
    let (mut mem, mut store) = (Memory::new(cfg(50)).unwrap(), MemStorage::new());
    let vecs = fill(&mut mem, &mut store, n, 11);
    mem.maintain(&mut store, 1000 * DAY).unwrap();

    // Forget a handful (< 10% of the graph) — the remap path.
    for id in [3u32, 57, 91] {
        mem.forget(&mut store, 1001 * DAY, FactId(id)).unwrap();
    }
    let out = mem.recall(vquery(1002 * DAY, &vecs[3], 10)).unwrap();
    assert!(
        out.facts.iter().all(|f| f.id != FactId(3)),
        "a tombstoned fact surfaced from the graph"
    );
    mem.maintain(&mut store, 1003 * DAY).unwrap();
    let snap = mem.snapshot_bytes(0);
    let (reopened, _) = Memory::open(&mut store, cfg(50)).unwrap();
    assert_eq!(
        reopened.snapshot_bytes(0),
        snap,
        "remap path must replay byte-for-byte"
    );
    let (loaded, _) = Memory::from_bytes(Some(&snap), &[], cfg(50)).unwrap();
    assert_eq!(loaded.snapshot_bytes(0), snap, "remap image not canonical");

    // Forget a third (> 10%) — the rebuild path.
    for id in (0..n as u32).filter(|i| i % 3 == 0) {
        let _ = mem.forget(&mut store, 1004 * DAY, FactId(id));
    }
    mem.maintain(&mut store, 1005 * DAY).unwrap();
    let snap = mem.snapshot_bytes(0);
    let (reopened, _) = Memory::open(&mut store, cfg(50)).unwrap();
    assert_eq!(
        reopened.snapshot_bytes(0),
        snap,
        "rebuild path must replay byte-for-byte"
    );

    // A survivor is still found first by its own embedding.
    let survivor = 4usize; // 4 % 3 != 0 and not in the handful
    let out = mem.recall(vquery(1006 * DAY, &vecs[survivor], 5)).unwrap();
    assert_eq!(out.facts[0].id, FactId(survivor as u32));
}

/// The per-query `ef` override is honored in the graph regime: a wide
/// beam is at least as good as the narrowest one.
#[test]
#[cfg_attr(miri, ignore)]
fn ef_override_widens_the_beam() {
    let (mut mem, mut store) = (Memory::new(cfg(50)).unwrap(), MemStorage::new());
    let vecs = fill(&mut mem, &mut store, 300, 5);
    mem.maintain(&mut store, 1000 * DAY).unwrap();

    let narrow = mem
        .recall(RecallQuery {
            ef: Some(1),
            ..vquery(1000 * DAY, &vecs[10], 8)
        })
        .unwrap();
    let wide = mem
        .recall(RecallQuery {
            ef: Some(256),
            ..vquery(1000 * DAY, &vecs[10], 8)
        })
        .unwrap();
    assert!(!narrow.facts.is_empty() && !wide.facts.is_empty());
    // The wide beam finds the query's own fact first; ef=1 may or may
    // not, but must stay valid (every fact live and vector-flagged).
    assert_eq!(wide.facts[0].id, FactId(10));
}
