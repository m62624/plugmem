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

/// A sparse fact-id space must still score correctly.
///
/// Document lengths are answered from a flat array indexed by fact id, which
/// is only allocated while the ids stay dense — a snapshot is untrusted input
/// and nothing range-checks the ids inside its stored records, so an id far
/// past the document count must not be able to demand a gigabyte of memory
/// (and `usize` is 32 bits on wasm32, where that edge is nearer). The array
/// stops growing; the arena answers the rest. This checks that the fallback is
/// wired, not merely that the allocation is capped.
#[test]
fn bm25_scores_documents_beyond_the_flat_length_index() {
    let mut idx = Bm25Index::new(16, usize::MAX).unwrap();
    idx.index_doc(FactId(0), &[(1, 1)]).unwrap();
    // Far outside any dense window a two-document corpus would justify.
    idx.index_doc(FactId(u32::MAX - 1), &[(1, 1)]).unwrap();
    assert_eq!(idx.docs(), 2);

    let found = search(&idx, &[1], 10);
    assert_eq!(
        found.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
        vec![0, u32::MAX - 1],
        "both documents must score, whichever side of the flat index they are on"
    );
    // Same term, same length, so the sparse one must score identically.
    assert!((found[0].1 - found[1].1).abs() < 1e-6);
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

    /// The ranking must be exactly what a definition-following scorer
    /// produces — same documents, same order, same `f32` bits.
    ///
    /// The index does not evaluate the definition directly: it merges sorted
    /// posting runs instead of accumulating into a map, and it asks the `live`
    /// predicate only about documents in contention rather than about every
    /// candidate. Both are supposed to be invisible from outside, and the
    /// exact score equality is deliberate — a merge that summed terms in a
    /// different order would still be "correct" and would still show up here.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn bm25_search_matches_a_naive_scorer(
        docs in proptest::collection::vec(
            proptest::collection::vec((0u32..12, 1u8..=6), 1..7),
            1..40,
        ),
        query in proptest::collection::vec(0u32..16, 0..6),
        k in 1usize..12,
        live_mod in 1u32..5,
    ) {
        // Unique terms per document is the indexer's contract.
        let docs: Vec<Vec<(u32, u8)>> = docs
            .into_iter()
            .map(|terms| {
                let mut seen: BTreeMap<u32, u8> = BTreeMap::new();
                for (term, tf) in terms {
                    seen.entry(term).or_insert(tf);
                }
                seen.into_iter().collect()
            })
            .collect();

        let mut idx = Bm25Index::new(4, usize::MAX).unwrap();
        for (fact, terms) in docs.iter().enumerate() {
            idx.index_doc(FactId(fact as u32), terms).unwrap();
        }

        // The definition, evaluated straight: for each query term, add its
        // contribution to every document that holds it, in query order.
        let (k1, b) = (1.2f32, 0.75f32);
        let total_docs = docs.len() as u64;
        let total_len: u64 = docs
            .iter()
            .map(|d| d.iter().map(|&(_, tf)| u64::from(tf)).sum::<u64>())
            .sum();
        let avg_len = total_len as f32 / total_docs as f32;
        let mut want: Vec<(u32, f32)> = (0..docs.len() as u32).map(|id| (id, 0.0)).collect();
        for &term in &query {
            let df = docs.iter().filter(|d| d.iter().any(|&(t, _)| t == term)).count() as u32;
            if df == 0 {
                continue;
            }
            let idf = idx.idf(df);
            for (id, doc) in docs.iter().enumerate() {
                let Some(&(_, tf)) = doc.iter().find(|&&(t, _)| t == term) else {
                    continue;
                };
                let len: u32 = doc.iter().map(|&(_, tf)| u32::from(tf)).sum();
                let tf = f32::from(tf);
                let norm = tf * (k1 + 1.0)
                    / (tf + k1 * (1.0 - b + b * (len.min(u32::from(u16::MAX)) as f32) / avg_len));
                want[id].1 += idf * norm;
            }
        }
        let live = |id: u32| id.is_multiple_of(live_mod);
        want.retain(|&(id, score)| score != 0.0 && live(id));
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        want.truncate(k);

        let mut scratch = Bm25Scratch::new();
        let mut out = Vec::new();
        idx.search((k1, b), &query, k, &mut |f| live(f.0), &mut scratch, &mut out);
        let got: Vec<(u32, f32)> = out.iter().map(|&(id, s)| (id.0, s)).collect();
        prop_assert_eq!(&got, &want);
    }
}

/// `k` is the caller's, and both index searches are public entry points that
/// size a selection buffer from it. A `k` past `usize::MAX / 2` used to double
/// into a threshold *below* `k`, so the buffer compacted and then partitioned
/// at `k - 1` — off the end of what it had just filled. Answer the query
/// instead: every live candidate, since the corpus cannot supply more.
#[test]
fn a_degenerate_k_is_answered_rather_than_partitioned_out_of_range() {
    let mut idx = Bm25Index::new(64, usize::MAX).unwrap();
    for fact in 0..8u32 {
        idx.index_doc(FactId(fact), &[(1, (fact as u8 % 5) + 1)])
            .unwrap();
    }
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    for k in [usize::MAX, usize::MAX / 2 + 1, usize::MAX / 2] {
        idx.search((1.2, 0.75), &[1], k, &mut |_| true, &mut scratch, &mut out);
        assert_eq!(out.len(), 8, "k = {k} lost candidates");
        assert!(
            out.windows(2).all(|w| w[0].1 >= w[1].1),
            "k = {k} broke the ordering"
        );
    }
}
