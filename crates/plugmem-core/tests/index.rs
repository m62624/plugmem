//! Index-layer tests (test plan): varint properties, the posting
//! store against a reference model, sorted-list intersection, and BM25
//! against golden scores computed by an independent script (only the
//! numbers live in the repo —).

use plugmem_core::FactId;
use plugmem_core::index::bm25::{Bm25Index, Bm25Scratch};
use plugmem_core::index::varint::{MAX_VARINT, decode_u32, encode_u32};
use plugmem_core::index::{IdListIndex, IntersectScratch, intersect};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::*;
#[cfg(not(target_family = "wasm"))]
use std::collections::BTreeMap;

#[test]
fn varint_boundaries() {
    // Canonical sizes at the 7-bit thresholds.
    for (v, want_len) in [
        (0u32, 1),
        (127, 1),
        (128, 2),
        (16_383, 2),
        (16_384, 3),
        (u32::MAX, 5),
    ] {
        let mut buf = [0u8; MAX_VARINT];
        let n = encode_u32(v, &mut buf);
        assert_eq!(n, want_len, "value {v}");
        assert_eq!(decode_u32(&buf[..n]), Some((v, n)), "value {v}");
    }
    // Truncation and a 5-byte value overflowing 32 bits are rejected.
    assert_eq!(decode_u32(&[]), None);
    assert_eq!(decode_u32(&[0x80]), None);
    assert_eq!(decode_u32(&[0x80, 0x80, 0x80, 0x80, 0x80]), None);
    assert_eq!(decode_u32(&[0x80, 0x80, 0x80, 0x80, 0x10]), None);
}

#[test]
fn posting_store_basics() {
    let mut idx = IdListIndex::new(16, usize::MAX).unwrap();
    assert_eq!(idx.count(7), 0);
    assert_eq!(idx.entries(7).count(), 0);
    // Id 0 is a valid first entry (delta 0 from the empty base).
    idx.push(7, FactId(0), 0).unwrap();
    idx.push(7, FactId(1), 0).unwrap();
    idx.push(7, FactId(300), 0).unwrap();
    idx.push(9, FactId(5), 0).unwrap();
    assert_eq!(idx.count(7), 3);
    assert_eq!(idx.count(9), 1);
    let ids: Vec<u32> = idx.entries(7).map(|(id, _)| id.0).collect();
    assert_eq!(ids, [0, 1, 300]);
    // tf defaults to 1 on the no-TF flavor.
    assert!(idx.entries(9).all(|(_, tf)| tf == 1));
    assert_eq!(idx.keys(), 2);
    assert!(idx.pool_bytes() > 0);
}

#[test]
fn intersection_table() {
    let mut idx = IdListIndex::new(16, usize::MAX).unwrap();
    for id in [1u32, 3, 5, 7, 9, 11] {
        idx.push(1, FactId(id), 0).unwrap();
    }
    for id in [2u32, 3, 5, 8, 11] {
        idx.push(2, FactId(id), 0).unwrap();
    }
    for id in [5u32, 11, 40] {
        idx.push(3, FactId(id), 0).unwrap();
    }
    let mut scratch = IntersectScratch::new();
    let mut out = Vec::new();
    let ids = |out: &Vec<FactId>| out.iter().map(|f| f.0).collect::<Vec<_>>();

    intersect(&idx, &[1], &mut scratch, &mut out);
    assert_eq!(ids(&out), [1, 3, 5, 7, 9, 11]);
    intersect(&idx, &[1, 2], &mut scratch, &mut out);
    assert_eq!(ids(&out), [3, 5, 11]);
    intersect(&idx, &[1, 2, 3], &mut scratch, &mut out);
    assert_eq!(ids(&out), [5, 11]);
    // An unknown key empties the intersection; empty keys yield nothing.
    intersect(&idx, &[1, 99], &mut scratch, &mut out);
    assert!(out.is_empty());
    intersect(&idx, &[], &mut scratch, &mut out);
    assert!(out.is_empty());
    // Duplicate keys are idempotent.
    intersect(&idx, &[3, 3], &mut scratch, &mut out);
    assert_eq!(ids(&out), [5, 11, 40]);
}

/// The golden corpus: term ids 1..=5, fact ids 0..=5. Scores
/// below were computed by an independent Python implementation of the
/// same formula (k1 = 1.2, b = 0.75); only the numbers are checked in.
fn golden_index() -> Bm25Index<'static> {
    let mut idx = Bm25Index::new(16, usize::MAX).unwrap();
    let docs: [&[(u32, u8)]; 6] = [
        &[(1, 2), (2, 1)],
        &[(1, 1), (5, 2)],
        &[(2, 3), (3, 1)],
        &[(3, 2), (4, 2)],
        &[(1, 1), (2, 1), (3, 1), (4, 1)],
        &[(4, 5)],
    ];
    for (fact, terms) in docs.iter().enumerate() {
        idx.index_doc(FactId(fact as u32), terms).unwrap();
    }
    idx
}

fn search(idx: &Bm25Index<'_>, terms: &[u32], k: usize) -> Vec<(u32, f32)> {
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    idx.search((1.2, 0.75), terms, k, &mut |_| true, &mut scratch, &mut out);
    out.into_iter().map(|(id, s)| (id.0, s)).collect()
}

fn assert_scores(got: &[(u32, f32)], want: &[(u32, f32)]) {
    assert_eq!(
        got.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
        want.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
        "ranking differs: got {got:?}, want {want:?}"
    );
    for (&(id, g), &(_, w)) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-4, "score of doc {id}: got {g}, want {w}");
    }
}

#[test]
fn bm25_matches_the_independent_reference() {
    let idx = golden_index();
    assert_eq!(idx.docs(), 6);
    assert!(idx.pool_bytes() > 0);
    assert_eq!(idx.df(1), 3);
    assert_eq!(idx.df(5), 1);
    // Reference idf values.
    assert!((idx.idf(3) - core::f32::consts::LN_2).abs() < 1e-4);
    assert!((idx.idf(1) - 1.540445).abs() < 1e-4);

    assert_scores(
        &search(&idx, &[1], 10),
        &[(0, 1.015145), (1, 0.760808), (4, 0.681034)],
    );
    assert_scores(
        &search(&idx, &[1, 2], 10),
        &[(0, 1.775953), (4, 1.362068), (2, 1.079177), (1, 0.760808)],
    );
    // The rare term dominates through its idf.
    assert_scores(&search(&idx, &[5], 10), &[(1, 2.25605)]);
    assert_scores(
        &search(&idx, &[1, 5], 10),
        &[(1, 3.016858), (0, 1.015145), (4, 0.681034)],
    );
    assert_scores(
        &search(&idx, &[3, 4], 10),
        &[(3, 1.883127), (4, 1.362068), (5, 1.177745), (2, 0.681034)],
    );
    // k truncates the ranking, not the scoring.
    assert_scores(&search(&idx, &[1, 2], 2), &[(0, 1.775953), (4, 1.362068)]);
    // Unknown terms contribute nothing; empty query finds nothing.
    assert_scores(&search(&idx, &[42], 10), &[]);
    assert_scores(&search(&idx, &[], 10), &[]);
}

#[test]
fn bm25_idf_is_monotone_in_df() {
    let idx = golden_index();
    // df 1 > df 2 > ... — rarer terms always weigh more.
    let idfs: Vec<f32> = (1..=6).map(|df| idx.idf(df)).collect();
    assert!(
        idfs.windows(2).all(|w| w[0] > w[1]),
        "idf not monotone: {idfs:?}"
    );
    assert!(idfs.iter().all(|&v| v > 0.0));
}

#[test]
fn bm25_live_filter_and_empty_index() {
    let idx = golden_index();
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    // Filter doc 0 out: the rest keep their scores and order.
    idx.search(
        (1.2, 0.75),
        &[1, 2],
        10,
        &mut |f| f.0 != 0,
        &mut scratch,
        &mut out,
    );
    let got: Vec<(u32, f32)> = out.iter().map(|&(id, s)| (id.0, s)).collect();
    assert_scores(&got, &[(4, 1.362068), (2, 1.079177), (1, 0.760808)]);

    // Empty index and k = 0 return cleanly.
    let empty = Bm25Index::new(16, usize::MAX).unwrap();
    empty.search((1.2, 0.75), &[1], 10, &mut |_| true, &mut scratch, &mut out);
    assert!(out.is_empty());
    idx.search((1.2, 0.75), &[1], 0, &mut |_| true, &mut scratch, &mut out);
    assert!(out.is_empty());
}

#[cfg(not(target_family = "wasm"))]
proptest! {
    // Varint roundtrip over the whole u32 range.
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn varint_roundtrip(v in any::<u32>()) {
        let mut buf = [0u8; MAX_VARINT];
        let n = encode_u32(v, &mut buf);
        prop_assert_eq!(decode_u32(&buf[..n]), Some((v, n)));
    }

    // The posting store is equivalent to a BTreeMap<key, Vec<(id, tf)>>
    // filled with ascending ids per key.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn posting_store_matches_model(
        ops in proptest::collection::vec(
            (0u32..8, 1u32..2000, 1u8..=255),
            0..300,
        )
    ) {
        let mut idx = IdListIndex::new(4, usize::MAX).unwrap();
        let mut tf_idx = Bm25Index::new(4, usize::MAX).unwrap();
        let mut model: BTreeMap<u32, Vec<(u32, u8)>> = BTreeMap::new();
        let mut next_id = 0u32;
        for (key, gap, tf) in ops {
            // Ascending ids across all pushes keeps every per-key list
            // ascending, as the engine guarantees.
            let id = next_id;
            next_id += gap;
            idx.push(key, FactId(id), 0).unwrap();
            tf_idx.index_doc(FactId(id), &[(key, tf)]).unwrap();
            model.entry(key).or_default().push((id, tf));
        }
        for (&key, want) in &model {
            let got: Vec<u32> = idx.entries(key).map(|(id, _)| id.0).collect();
            let want_ids: Vec<u32> = want.iter().map(|&(id, _)| id).collect();
            prop_assert_eq!(&got, &want_ids);
            prop_assert_eq!(idx.count(key) as usize, want.len());
            prop_assert_eq!(tf_idx.df(key) as usize, want.len());
        }
    }
}
