//! Emits `#TSV` latency rows for the engine's recall sources, in the same
//! shape the arena stand prints, so [`plugmem-bench-charts`] can render a
//! core chart next to the arena ones.
//!
//! Native-only: `plugmem-testgen` (the corpus generator the vector paths
//! need) is a dev-dependency gated off wasm, so on wasm this compiles to an
//! empty stub. Run it in release, and feed the output to the chart tool:
//!
//! ```text
//! cargo run --release -p plugmem-core --example bench_ops > core.tsv
//! cargo run -p plugmem-bench-charts -- core.tsv
//! ```
//!
//! Each op is timed best-of-N after a warm-up (recall/search allocate
//! nothing after warm-up, so the best sample is a clean lower bound). The
//! two lexical/tag sources build a synthetic index directly; the two
//! vector sources need the whole engine (quantization, the flat/HNSW
//! regimes), so they run over a testgen corpus at the specs gate sizes.
//!
//! [`plugmem-bench-charts`]: ../../../tools/bench-charts

#[cfg(not(target_family = "wasm"))]
fn main() {
    use std::time::Instant;

    use plugmem_core::index::bm25::{Bm25Index, Bm25Scratch};
    use plugmem_core::index::{IdListIndex, IntersectScratch, intersect};
    use plugmem_core::{Config, FactId, MemStorage, Memory, RecallQuery, RecallResult};
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    /// Best-of-`reps` wall time of `f`, in microseconds, after `warm`
    /// warm-up calls. `f` returns a value so the optimizer cannot elide it.
    fn best_us(warm: u32, reps: u32, mut f: impl FnMut() -> usize) -> f64 {
        for _ in 0..warm {
            std::hint::black_box(f());
        }
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(f());
            best = best.min(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        best
    }

    /// Prints one machine-readable row for the chart tool.
    fn row(op: &str, us: f64) {
        // n \t runtime \t structure \t metric \t value
        println!("core\tnative\t{op}\tlatency_us\t{us:.1}");
    }

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    // --- lexical: BM25 over a skewed 10k-doc corpus, a 3-term query ------
    {
        const VOCAB: u64 = 3000;
        let mut rng = xorshift(0xC0FF_EE00_0000_0001);
        let docs: Vec<Vec<(u32, u8)>> = (0..10_000)
            .map(|_| {
                let mut tfs: Vec<(u32, u8)> = Vec::with_capacity(8);
                for _ in 0..8 {
                    let r = (rng() % 10_000) as f32 / 10_000.0;
                    let term = ((r * r * r) * VOCAB as f32) as u32;
                    match tfs.iter_mut().find(|(t, _)| *t == term) {
                        Some((_, tf)) => *tf = tf.saturating_add(1),
                        None => tfs.push((term, 1)),
                    }
                }
                tfs
            })
            .collect();
        let mut idx = Bm25Index::new(2048, usize::MAX).unwrap();
        for (fact, terms) in docs.iter().enumerate() {
            idx.index_doc(FactId(fact as u32), terms).unwrap();
        }
        let query = [1u32, 400, 2500];
        let mut scratch = Bm25Scratch::new();
        let mut out = Vec::new();
        row(
            "BM25 (3 terms, 10k)",
            best_us(50, 500, || {
                idx.search(
                    (1.2, 0.75),
                    &query,
                    8,
                    &mut |_| true,
                    &mut scratch,
                    &mut out,
                );
                out.len()
            }),
        );
    }

    // --- tags: intersect three id-lists over 100k facts ------------------
    {
        let mut idx = IdListIndex::new(512, usize::MAX).unwrap();
        let mut rng = xorshift(0x7A67_0000_0000_0001);
        for id in 0..100_000u32 {
            let r = rng();
            if r.is_multiple_of(10) {
                idx.push(1, FactId(id), 0).unwrap();
            }
            if r.is_multiple_of(20) {
                idx.push(2, FactId(id), 0).unwrap();
            }
            if r.is_multiple_of(50) {
                idx.push(3, FactId(id), 0).unwrap();
            }
        }
        let mut scratch = IntersectScratch::new();
        let mut out = Vec::new();
        row(
            "tags (3 lists, 100k)",
            best_us(50, 500, || {
                intersect(&idx, &[1, 2, 3], &mut scratch, &mut out);
                out.len()
            }),
        );
    }

    // --- vectors: a corpus member queried at k = 8 in both regimes -------
    // Below the flat→HNSW threshold (24k): a flat two-phase scan.
    row(
        "flat vector (24k, d384)",
        vector_recall_us(0x5EC0_0000_0000_0384, 24_000, false),
    );
    // Above it (30k) with the graph built by `maintain`: HNSW search.
    row(
        "HNSW (30k, d384)",
        vector_recall_us(0x4A5A_0000_0000_0001, 30_000, true),
    );

    /// Builds a dim-384 vector engine of `vectors` corpus members and
    /// returns the best k=8 vector-recall time (µs). With `build_graph`,
    /// `maintain` builds the HNSW graph first (search then goes through
    /// it); without, the flat regime is measured.
    fn vector_recall_us(seed: u64, vectors: usize, build_graph: bool) -> f64 {
        const DIM: usize = 384;
        let profile = Profile {
            dim: DIM,
            w_revise: 0,
            w_forget: 0,
            w_link: 0,
            ..Profile::default()
        };
        let ops = Gen::new(seed, profile).ops(vectors);
        let mut cfg = Config::default();
        cfg.dim = DIM;
        let mut mem = Memory::new(cfg).unwrap();
        let mut store = MemStorage::new();
        for op in &ops {
            apply(&mut mem, &mut store, op).unwrap();
        }
        let now = ops
            .iter()
            .map(|op| match op {
                GenOp::Remember { now, .. }
                | GenOp::Revise { now, .. }
                | GenOp::Forget { now, .. }
                | GenOp::Link { now, .. }
                | GenOp::Maintain { now } => *now,
            })
            .max()
            .unwrap_or(0)
            + 1;
        if build_graph {
            mem.maintain(&mut store, now).unwrap();
        }
        let query = ops
            .iter()
            .find_map(|op| match op {
                GenOp::Remember {
                    vector: Some(v), ..
                } => Some(v.clone()),
                _ => None,
            })
            .expect("every remember carries a vector at dim > 0");
        let mut out = RecallResult::default();
        let q = RecallQuery {
            vector: Some(&query),
            k: 8,
            text: None,
            ..RecallQuery::text(now, "")
        };
        best_us(30, 200, || {
            mem.recall_into(q, &mut out).unwrap();
            out.facts.len()
        })
    }
}

#[cfg(target_family = "wasm")]
fn main() {}
