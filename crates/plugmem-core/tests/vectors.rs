//! The vector layer end-to-end (specs/04 §5, specs/11 A.5): remember with
//! an embedding, the flat vector recall source, quantized-cosine
//! similar-detection, persistence and journal replay, and the error
//! surface. Quantization accuracy and the two-phase search internals are
//! unit-tested in `src/index/vecpool.rs`.

use plugmem_core::{
    Config, Error, FactId, MemStorage, Memory, RecallQuery, RememberInput, Storage,
};

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

/// A tiny deterministic LCG — the repo forbids unseeded randomness.
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// A vector-only recall query (no text/tags/entities/range).
fn vquery<'a>(now: u64, v: &'a [f32], k: usize) -> RecallQuery<'a> {
    RecallQuery {
        now,
        text: None,
        vector: Some(v),
        tags: &[],
        entities: &[],
        as_of: None,
        range: None,
        k,
        token_budget: None,
        include_closed: false,
        ef: None,
    }
}

#[test]
fn recall_by_vector_surfaces_the_match() {
    let dim = 64;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(1);
    let a = rng.vector(dim);
    let b = rng.vector(dim);
    let c = rng.vector(dim);
    for (i, v) in [&a, &b, &c].into_iter().enumerate() {
        mem.remember(
            &mut store,
            RememberInput {
                vector: Some(v),
                ..RememberInput::text(1000, ["alpha", "beta", "gamma"][i])
            },
        )
        .unwrap();
    }
    // Query with b's exact vector: fact 1 must rank first.
    let out = mem.recall(vquery(1000, &b, 3)).unwrap();
    assert_eq!(out.facts[0].id, FactId(1));
    assert!(!out.rendered.is_empty());
}

#[test]
fn flat_recall_matches_bruteforce() {
    let dim = 128;
    let n = 300u64;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(0xabcd);
    let mut stored: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n {
        let v = rng.vector(dim);
        // Same recorded_at for all: recency boost is uniform, so the fused
        // order is the vector order — a clean comparison to brute force.
        mem.remember(
            &mut store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(1000, "v")
            },
        )
        .unwrap();
        stored.push(v);
    }
    let query = rng.vector(dim);
    let k = 10;
    let out = mem.recall(vquery(1000, &query, k)).unwrap();
    let got: Vec<u32> = out.facts.iter().map(|f| f.id.0).collect();

    // Brute-force top-k by true cosine.
    let mut truth: Vec<(u32, f32)> = stored
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine(&query, v)))
        .collect();
    truth.sort_by(|a, b| b.1.total_cmp(&a.1));
    let want: Vec<u32> = truth.iter().take(k).map(|&(i, _)| i).collect();

    let hits = got.iter().filter(|id| want.contains(id)).count();
    let recall = hits as f32 / k as f32;
    assert!(recall >= 0.8, "flat recall@{k} = {recall} (got {got:?})");
}

#[test]
fn vector_source_respects_liveness() {
    let dim = 32;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(7);
    let v = rng.vector(dim);
    let out = mem
        .remember(
            &mut store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(1000, "secret")
            },
        )
        .unwrap();
    // Present before forgetting.
    assert_eq!(mem.recall(vquery(1000, &v, 5)).unwrap().facts[0].id, out.id);
    // Tombstoned: the vector source must not surface it.
    mem.forget(&mut store, 2000, out.id).unwrap();
    assert!(mem.recall(vquery(2000, &v, 5)).unwrap().facts.is_empty());
    // as_of before it was recorded: also absent.
    let mut mem2 = Memory::new(cfg(dim)).unwrap();
    let mut store2 = MemStorage::new();
    mem2.remember(
        &mut store2,
        RememberInput {
            vector: Some(&v),
            ..RememberInput::text(5000, "later")
        },
    )
    .unwrap();
    let mut q = vquery(3000, &v, 5);
    q.as_of = Some(3000);
    assert!(mem2.recall(q).unwrap().facts.is_empty());
}

#[test]
fn vector_similar_detection_fires() {
    let dim = 64;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(99);
    let base = rng.vector(dim);
    // Two near-identical vectors on the same entity, disjoint text (so the
    // lexical detector stays silent and the cosine signal is what fires).
    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("user"),
            vector: Some(&base),
            ..RememberInput::text(1000, "one two three")
        },
    )
    .unwrap();
    let mut near = base.clone();
    near[0] += 0.001;
    let out = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                vector: Some(&near),
                ..RememberInput::text(2000, "four five six")
            },
        )
        .unwrap();
    assert_eq!(out.similar.len(), 1);
    assert_eq!(out.similar[0].id, FactId(0));
    assert_eq!(
        out.similar[0].reason,
        plugmem_core::SimilarReason::VectorCosine
    );
}

#[test]
fn snapshot_roundtrip_with_vectors_is_canonical() {
    let dim = 48;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(0xfeed);
    let mut vecs = Vec::new();
    for i in 0..40u64 {
        // Every other fact carries a vector — the load must handle the
        // mixed HAS_VECTOR / no-vector population.
        let v = rng.vector(dim);
        let has = i % 2 == 0;
        mem.remember(
            &mut store,
            RememberInput {
                vector: has.then_some(&v[..]),
                ..RememberInput::text(i * 1000, "fact text here")
            },
        )
        .unwrap();
        vecs.push((has, v));
    }
    let bytes = mem.snapshot_bytes(0);
    let (loaded, report) = Memory::from_bytes(Some(&bytes), &[], cfg(dim)).unwrap();
    assert_eq!(report.replayed, 0);
    // Canonical: save → load → save is byte-identical.
    assert_eq!(loaded.snapshot_bytes(0), bytes);
    // Vector recall agrees across the reload.
    let (has, ref v) = vecs[8];
    assert!(has);
    let a = mem.recall(vquery(40_000, v, 5)).unwrap();
    let b = loaded.recall(vquery(40_000, v, 5)).unwrap();
    assert_eq!(a.rendered, b.rendered);
    assert_eq!(a.facts[0].id, FactId(8));
}

#[test]
fn truncated_vector_snapshot_errors_at_load() {
    let dim = 32;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(5);
    for i in 0..20u64 {
        let v = rng.vector(dim);
        mem.remember(
            &mut store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(i * 1000, "text")
            },
        )
        .unwrap();
    }
    let bytes = mem.snapshot_bytes(0);
    // Truncations break the structure and fail typed at load, never panic.
    // Content flips (the vector section included) are trust/sparse — caught by
    // scrub/verify, exercised by `a_vector_open_never_panics_...`.
    for cut in (0..bytes.len()).step_by(53) {
        assert!(
            Memory::from_bytes(Some(&bytes[..cut]), &[], cfg(dim)).is_err(),
            "truncation to {cut} accepted"
        );
    }
}

// Bitflip sweep — `catch_unwind` (unwinding) and heavy: native only, like the
// proptest sections (specs/14 §3).
#[cfg(not(target_family = "wasm"))]
#[test]
fn a_vector_open_never_panics_and_verify_catches_corruption() {
    // The default trust/sparse open skips the checksums *and* the vector scan
    // (specs/16 §9), so a corrupt vector slot can reach the engine. Vector
    // search reads slots with bounds-checked arithmetic, so it never panics;
    // `verify()` catches the deferred corruption.
    let dim = 32;
    let c = cfg(dim);
    let (mut mem, mut store) = (Memory::new(c.clone()).unwrap(), MemStorage::new());
    let mut rng = Lcg(5);
    let mut last = Vec::new();
    for i in 0..20u64 {
        let v = rng.vector(dim);
        mem.remember(
            &mut store,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(i * 1000, "text")
            },
        )
        .unwrap();
        last = v;
    }
    let bytes = mem.snapshot_bytes(0);

    let mut verify_caught = false;
    for at in (0..bytes.len()).step_by(17) {
        let mut b = bytes.clone();
        b[at] ^= 0x20;
        let cc = c.clone();
        let q = last.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Ok((m, _)) = Memory::from_bytes(Some(&b), &[], cc) else {
                return false; // a typed load error is a fine outcome
            };
            let stats = m.stats();
            for i in 0..stats.next_fact {
                let _ = m.get(FactId(i));
            }
            let _ = m.recall(vquery(1000, &q, 5)); // search over maybe-corrupt slots
            let _ = m.snapshot_bytes(0);
            m.verify().is_err()
        }));
        match outcome {
            Ok(caught) => verify_caught |= caught,
            Err(_) => panic!("a vector access panicked after a flip at {at}"),
        }
    }
    assert!(
        verify_caught,
        "verify() must catch at least one deferred vector/content corruption"
    );
}

#[test]
fn journal_replay_reproduces_vectors() {
    let dim = 40;
    let (mut mem, mut store) = (Memory::new(cfg(dim)).unwrap(), MemStorage::new());
    let mut rng = Lcg(0x2468);
    for i in 0..30u64 {
        let v = rng.vector(dim);
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some(["user", "plugmem"][(i % 2) as usize]),
                vector: (i % 3 != 0).then_some(&v[..]),
                ..RememberInput::text(i * 1000, "some remembered text")
            },
        )
        .unwrap();
    }
    // Reopen from the journal alone (no snapshot): re-quantization must
    // reproduce the exact same image.
    let (reopened, report) = Memory::open(&mut store, cfg(dim)).unwrap();
    assert!(report.replayed > 0 && report.skipped == 0);
    assert_eq!(reopened.snapshot_bytes(0), mem.snapshot_bytes(0));
}

#[test]
fn vector_error_surface() {
    // dim 0: any vector is rejected.
    let (mut m0, mut s0) = (Memory::new(cfg(0)).unwrap(), MemStorage::new());
    let v = [1.0f32, 2.0];
    assert_eq!(
        m0.remember(
            &mut s0,
            RememberInput {
                vector: Some(&v),
                ..RememberInput::text(1, "x")
            }
        )
        .unwrap_err(),
        Error::Invalid("vector given but dim is 0")
    );

    // dim 64: wrong length is a DimMismatch; a zero vector is Invalid.
    let (mut m, mut s) = (Memory::new(cfg(64)).unwrap(), MemStorage::new());
    let short = [0.5f32; 10];
    assert!(matches!(
        m.remember(
            &mut s,
            RememberInput {
                vector: Some(&short),
                ..RememberInput::text(1, "x")
            }
        )
        .unwrap_err(),
        Error::DimMismatch { got: 10, want: 64 }
    ));
    let zero = [0.0f32; 64];
    assert_eq!(
        m.remember(
            &mut s,
            RememberInput {
                vector: Some(&zero),
                ..RememberInput::text(1, "x")
            }
        )
        .unwrap_err(),
        Error::Invalid("vector must be nonzero")
    );
}

#[test]
fn journal_vector_dim_mismatch_is_corrupt() {
    // A journal written at dim 4, reopened claiming dim 8: the replay
    // guard rejects the vector rather than mis-quantizing it.
    let (mut mem, mut store) = (Memory::new(cfg(4)).unwrap(), MemStorage::new());
    mem.remember(
        &mut store,
        RememberInput {
            vector: Some(&[1.0, 2.0, 3.0, 4.0]),
            ..RememberInput::text(1, "x")
        },
    )
    .unwrap();
    let journal = store.read_journal().unwrap();
    assert_eq!(
        Memory::from_bytes(None, &journal, cfg(8)).unwrap_err(),
        Error::Corrupt("journal vector dimension disagrees with dim")
    );
}
