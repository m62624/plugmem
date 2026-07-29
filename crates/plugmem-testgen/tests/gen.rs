//! testgen contract tests: determinism, validity of the stream against a
//! real engine (including canonical replay), and the corpus shape
//! promises of.

use plugmem_core::{Config, MemStorage, Memory, RecallQuery};
use plugmem_testgen::{Gen, GenOp, Profile, Vocabulary, apply, word_for};

fn profile(dim: usize) -> Profile {
    Profile {
        dim,
        w_maintain: 2,
        ..Profile::default()
    }
}

fn engine_cfg(dim: usize) -> Config {
    let mut cfg = Config::default();
    cfg.dim = dim;
    cfg.shards_facts = 16;
    cfg.shards_entities = 8;
    cfg.shards_edges = 8;
    cfg.shards_temporal = 8;
    cfg.shards_postings = 32;
    cfg
}

#[test]
fn same_seed_same_stream() {
    let a = Gen::new(42, profile(8)).ops(400);
    let b = Gen::new(42, profile(8)).ops(400);
    assert_eq!(
        a, b,
        "the stream must be a pure function of (seed, profile)"
    );
    // And chunked generation equals one-shot generation.
    let mut g = Gen::new(42, profile(8));
    let mut chunked = g.ops(150);
    chunked.extend(g.ops(250));
    assert_eq!(a, chunked);
}

#[test]
fn different_seeds_differ() {
    let a = Gen::new(1, profile(0)).ops(50);
    let b = Gen::new(2, profile(0)).ops(50);
    assert_ne!(a, b);
}

#[test]
fn stream_applies_cleanly_and_replays_canonically() {
    let dim = 16;
    let ops = Gen::new(0xC0FFEE, profile(dim)).ops(1200);

    // Every operation kind occurs in a stream this long — this test is
    // also what exercises every `apply` arm.
    for wanted in ["Remember", "Revise", "Forget", "Link", "Maintain"] {
        assert!(
            ops.iter().any(|op| match op {
                GenOp::Remember { .. } => wanted == "Remember",
                GenOp::Revise { .. } => wanted == "Revise",
                GenOp::Forget { .. } => wanted == "Forget",
                GenOp::Link { .. } => wanted == "Link",
                GenOp::Maintain { .. } => wanted == "Maintain",
            }),
            "no {wanted} in 1200 ops"
        );
    }

    // Validity by construction: every operation applies without error.
    let (mut mem, mut store) = (Memory::new(engine_cfg(dim)).unwrap(), MemStorage::new());
    for (i, op) in ops.iter().enumerate() {
        apply(&mut mem, &mut store, op).unwrap_or_else(|e| panic!("op {i} failed: {e:?}"));
    }

    // The corpus is searchable: the most frequent dictionary word hits.
    let now = mem.stats().next_fact as u64 * 1_000_000_000; // far future
    let head = word_for(0);
    let out = mem.recall(RecallQuery::text(now, &head)).unwrap();
    assert!(!out.rendered.is_empty(), "head-word recall found nothing");

    // Replaying the accumulated journal reproduces the image byte for
    // byte, maintains included.
    let snap = mem.snapshot_bytes(0);
    let (reopened, _) = Memory::open(&mut store, engine_cfg(dim)).unwrap();
    assert_eq!(reopened.snapshot_bytes(0), snap);
}

#[test]
fn vectors_are_unit_and_clustered() {
    let dim = 32;
    let ops = Gen::new(7, profile(dim)).ops(300);
    let mut seen = 0;
    for op in &ops {
        let (GenOp::Remember {
            vector: Some(v), ..
        }
        | GenOp::Revise {
            vector: Some(v), ..
        }) = op
        else {
            continue;
        };
        seen += 1;
        assert_eq!(v.len(), dim);
        let norm: f64 = v
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "vector norm {norm} is not unit");
    }
    assert!(seen > 100, "vectors must accompany remembers when dim > 0");
}

#[test]
fn zipf_head_dominates_the_tail() {
    let ops = Gen::new(11, profile(0)).ops(600);
    let head = word_for(0);
    let mid = word_for(15_000);
    let (mut head_n, mut mid_n) = (0usize, 0usize);
    for op in &ops {
        let (GenOp::Remember { text, .. } | GenOp::Revise { text, .. }) = op else {
            continue;
        };
        for w in text.split_whitespace() {
            head_n += usize::from(w == head);
            mid_n += usize::from(w == mid);
        }
    }
    assert!(
        head_n > 20 && head_n > 5 * mid_n.max(1),
        "Zipf shape is off: head {head_n}, mid {mid_n}"
    );
}

#[test]
fn time_axis_is_monotone_and_thickens() {
    let ops = Gen::new(3, profile(0)).ops(2000);
    let nows: Vec<u64> = ops
        .iter()
        .map(|op| match op {
            GenOp::Remember { now, .. }
            | GenOp::Revise { now, .. }
            | GenOp::Forget { now, .. }
            | GenOp::Link { now, .. }
            | GenOp::Maintain { now } => *now,
        })
        .collect();
    assert!(
        nows.windows(2).all(|w| w[0] < w[1]),
        "timestamps must strictly increase"
    );
    // Density rises toward the end: the first 500 operations span far
    // more time than the last 500.
    let early = nows[499] - nows[0];
    let late = nows[1999] - nows[1500];
    assert!(
        early > late * 4,
        "no thickening: early span {early}, late span {late}"
    );
}

#[test]
fn vocabulary_accessors_and_word_uniqueness() {
    let vocab = Vocabulary::new(0, 100, 1.07);
    assert_eq!(vocab.len(), 100);
    assert!(!vocab.is_empty());
    // Rank 0 is the shortest padded word.
    assert_eq!(vocab.word(0), word_for(0));
    // Unique and tokenizer-friendly (pure lowercase ASCII letters).
    let mut seen = std::collections::HashSet::new();
    for i in 0..5000 {
        let w = word_for(i);
        assert!(w.len() >= 4, "at least two syllables: {w}");
        assert!(w.bytes().all(|b| b.is_ascii_lowercase()));
        assert!(seen.insert(w), "duplicate word at index {i}");
    }
}

#[test]
fn zero_seed_is_remapped() {
    use plugmem_testgen::Rng;
    let mut zero = Rng::new(0);
    let mut one = Rng::new(1);
    assert_ne!(zero.next_u64(), 0, "the xorshift state must never be zero");
    assert_ne!(zero.next_u64(), one.next_u64());
}

#[test]
#[should_panic(expected = "at least one cluster")]
fn vectors_without_clusters_panic() {
    let _ = Gen::new(
        1,
        Profile {
            dim: 8,
            vector_clusters: 0,
            ..Profile::default()
        },
    );
}

#[test]
#[should_panic(expected = "positive total weight")]
fn zero_operation_mix_panics() {
    let _ = Gen::new(
        1,
        Profile {
            w_remember: 0,
            w_revise: 0,
            w_forget: 0,
            w_link: 0,
            w_maintain: 0,
            ..Profile::default()
        },
    );
}

#[test]
#[should_panic(expected = "dict_words must be positive")]
fn empty_dictionary_panics() {
    let _ = Gen::new(
        1,
        Profile {
            dict_words: 0,
            ..Profile::default()
        },
    );
}
