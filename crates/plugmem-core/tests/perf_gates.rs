//! Deterministic performance gates for the engine layers.
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

// Timing gate over a large corpus: meaningless and minutes-slow under the miri
// interpreter; UB coverage is per-operation, so small tests hit the same code.
#[test]
#[cfg_attr(miri, ignore)]
fn bm25_decode_work_is_bounded() {
    let mut idx = Bm25Index::new(2048, usize::MAX).unwrap();
    for (fact, terms) in corpus().iter().enumerate() {
        idx.index_doc(FactId(fact as u32), terms).unwrap();
    }
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    idx.reset_query_counters();
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

/// The counters that separate cheap work from expensive work.
///
/// `decoded` is varint arithmetic over a contiguous chunk chain — nanoseconds
/// per entry, and it is legitimately O(Σ df). `admitted` is not: every call is
/// a fact-record lookup in the engine's arena, a random probe that deepens with
/// the corpus. Only `k` documents can be returned, so a query that asks about
/// more than `k` is doing work it can never use — and that is what made the
/// degenerate query cost milliseconds.
///
/// `scored` stays at Σ df by design: a document length is still needed per
/// posting. What changed is where it comes from — a flat array rather than an
/// arena probe — so the count is the gate against *extra* scoring work, not
/// against the cost of scoring itself.
#[test]
#[cfg_attr(miri, ignore)]
fn bm25_probe_work_is_bounded() {
    let mut idx = Bm25Index::new(2048, usize::MAX).unwrap();
    for (fact, terms) in corpus().iter().enumerate() {
        idx.index_doc(FactId(fact as u32), terms).unwrap();
    }
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    let k = 8;

    // The ordinary query shape: one common + one mid + one rare term.
    idx.reset_query_counters();
    idx.search(
        (1.2, 0.75),
        &[1, 400, 2500],
        k,
        &mut |_| true,
        &mut scratch,
        &mut out,
    );
    assert_eq!(out.len(), k);
    assert_eq!(idx.scored(), 1_399, "a length per posting, no more");
    assert_eq!(
        idx.admitted(),
        k as u64,
        "with everything live, the filter is asked exactly k times"
    );

    // The degenerate query: the single most frequent term of the corpus, whose
    // posting list covers nearly half of it. The filter cost must not follow
    // `df` — this is the ceiling the millisecond worst case showed up in.
    idx.reset_query_counters();
    idx.search((1.2, 0.75), &[0], k, &mut |_| true, &mut scratch, &mut out);
    assert_eq!(out.len(), k);
    assert_eq!(
        u64::from(idx.df(0)),
        4_388,
        "corpus drifted: df of the most common term"
    );
    assert_eq!(idx.scored(), 4_388, "a length per posting, no more");
    assert_eq!(
        idx.admitted(),
        k as u64,
        "a common term must not cost one filter call per posting"
    );

    // A filter that rejects most of the ranking still terminates on the
    // survivors, and still never exceeds the candidate count.
    idx.reset_query_counters();
    idx.search(
        (1.2, 0.75),
        &[0],
        k,
        &mut |id| id.0.is_multiple_of(500),
        &mut scratch,
        &mut out,
    );
    assert_eq!(out.len(), k);
    assert!(
        idx.admitted() <= 4_388,
        "filtered fallback must not exceed the candidates: {}",
        idx.admitted()
    );
}

// 5000-vector pool: minutes under the miri interpreter, no extra UB coverage
// over the small vector tests.
#[test]
#[cfg_attr(miri, ignore)]
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

// 2000-node HNSW build: minutes under the miri interpreter, no extra UB
// coverage over the small graph tests.
#[test]
#[cfg_attr(miri, ignore)]
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
