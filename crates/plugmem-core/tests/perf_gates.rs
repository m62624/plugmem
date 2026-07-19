//! Deterministic performance gates for the engine layers (specs/07 §3).
//!
//! Same discipline as the arena's gates: fixed-seed workloads, ceilings
//! on work counters set from measured values ×1.2 — machine-independent,
//! lowered-only. Runs under `--features counters`.

#![cfg(feature = "counters")]

use plugmem_core::FactId;
use plugmem_core::index::bm25::{Bm25Index, Bm25Scratch};
use plugmem_core::index::vecpool::{VecPool, VecScratch};

/// The bench corpus shape: 10k docs × 8 terms, 3000-term vocabulary,
/// power-law skew (same generator as `benches/engine.rs`).
fn corpus() -> Vec<Vec<(u32, u8)>> {
    let mut s = 0xC0FF_EE00_0000_0001u64;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    (0..10_000)
        .map(|_| {
            let mut tfs: Vec<(u32, u8)> = Vec::with_capacity(8);
            for _ in 0..8 {
                let r = (rng() % 10_000) as f32 / 10_000.0;
                let term = ((r * r * r) * 3000.0) as u32;
                match tfs.iter_mut().find(|(t, _)| *t == term) {
                    Some((_, tf)) => *tf = tf.saturating_add(1),
                    None => tfs.push((term, 1)),
                }
            }
            tfs
        })
        .collect()
}

#[test]
fn bm25_decode_work_is_bounded() {
    let mut idx = Bm25Index::new(2048, usize::MAX).unwrap();
    for (fact, terms) in corpus().iter().enumerate() {
        idx.index_doc(FactId(fact as u32), terms).unwrap();
    }
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    idx.reset_decoded();
    // One common + one mid + one rare term, the bench query shape.
    idx.search(
        (1.2, 0.75),
        &[1, 400, 2500],
        8,
        &mut |_| true,
        &mut scratch,
        &mut out,
    );
    assert!(!out.is_empty());
    // Query cost is exactly Σ df of the query terms — the O(Σ df)
    // contract. Measured: df(1) + df(400) + df(2500) = 1_399 decodes on
    // the fixed corpus; the equality (not a ceiling) *is* the gate: any
    // extra decode means the scan is no longer bounded by the postings.
    let df_sum = u64::from(idx.df(1) + idx.df(400) + idx.df(2500));
    assert_eq!(idx.decoded(), df_sum, "decodes must equal Σ df");
    assert_eq!(df_sum, 1_399, "corpus drifted: Σ df changed");
}

#[test]
fn vector_rescore_work_is_bounded() {
    // The signature prefilter must cap exact dot products at
    // `max(4·k, 64)` regardless of corpus size — the whole point of the
    // two-phase search (a full linear rescore would be O(N)).
    let dim = 128;
    let n = 5000u32;
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut pool = VecPool::new(dim, usize::MAX);
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng()).collect();
        pool.push(FactId(i), &v).unwrap();
    }
    let query: Vec<f32> = (0..dim).map(|_| rng()).collect();
    let mut scratch = VecScratch::new();
    let mut out = Vec::new();
    let k = 8;
    pool.reset_dots();
    pool.search(&query, k, &mut |_| true, &mut scratch, &mut out)
        .unwrap();
    assert!(!out.is_empty());
    // Every admitted candidate is rescored once; with all admitted the
    // count is exactly the candidate budget.
    let budget = (4 * k).max(64) as u64;
    assert_eq!(
        pool.dots(),
        budget,
        "rescored dots must equal the candidate budget"
    );
}

#[test]
fn hnsw_search_work_is_bounded() {
    // Graph search must stay sublinear in the corpus: the deterministic
    // corpus fixes the exact distance-evaluation count, and the ceiling
    // (measured x1.2) guards complexity — a linear scan of the 2000
    // nodes would already cost 2000 evaluations before any neighbor
    // work.
    use plugmem_core::index::hnsw::{HnswGraph, HnswScratch};

    let dim = 64;
    let n = 2000u32;
    let mut s = 0xC0DE_0000_0000_0001u64;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut pool = VecPool::new(dim, usize::MAX);
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng()).collect();
        pool.push(FactId(i), &v).unwrap();
    }
    let mut scratch = HnswScratch::default();
    let mut graph = HnswGraph::new(16, 32, usize::MAX).unwrap();
    graph.insert_bulk(&pool, n, 200, &mut scratch).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng()).collect();
    let mut vec_scratch = VecScratch::new();
    let mut out = Vec::new();
    graph.reset_dist_evals();
    graph
        .search(&pool, &query, 64, &mut vec_scratch, &mut scratch, &mut out)
        .unwrap();
    assert!(!out.is_empty());
    // Measured 1197 on the fixed corpus (ef 64, m 16/32); ceiling x1.2.
    let evals = graph.dist_evals();
    assert!(
        evals <= 1_450,
        "graph search cost regressed: {evals} distance evaluations"
    );
}
