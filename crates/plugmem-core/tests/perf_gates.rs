//! Deterministic performance gates for the engine layers (specs/07 §3).
//!
//! Same discipline as the arena's gates: fixed-seed workloads, ceilings
//! on work counters set from measured values ×1.2 — machine-independent,
//! lowered-only. Runs under `--features counters`.

#![cfg(feature = "counters")]

use plugmem_core::FactId;
use plugmem_core::index::bm25::{Bm25Index, Bm25Scratch};

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
