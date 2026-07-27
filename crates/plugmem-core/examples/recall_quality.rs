//! Recall quality baseline for vector search — the companion to the
//! `search_matrix` **speed** bench (`benches/engine.rs`). Criterion answers
//! "how fast?"; this answers "how correct?".
//!
//! For each (fill, dim) point it builds the engine over a testgen corpus,
//! then for a sample of query vectors compares the engine's top-k against the
//! **ground truth**: an exact brute-force cosine ranking over the *original
//! f32* vectors (which testgen still holds — the engine only keeps the int8
//! quantization). `recall@k` is the fraction of the true nearest neighbours
//! the engine actually returned, averaged over the query sample; the minimum
//! is the worst single query.
//!
//! This is the number a later vector optimization (a signature-tier layout, a
//! binary HNSW traversal) trades against: rerun it after the change and see
//! whether quality dropped a little or a lot. It measures quality only, not
//! time — the two together are the full "before" picture.
//!
//! Native-only: `plugmem-testgen` is a dev-dependency gated off wasm, so on
//! wasm this compiles to an empty stub.
//!
//! ```text
//! cargo run --release -p plugmem-core --example recall_quality
//! ```

#[cfg(not(target_family = "wasm"))]
fn main() {
    use plugmem_core::{
        Config, FactId, MemStorage, Memory, RecallQuery, RecallResult, RecallScratch,
    };
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    /// Cosine similarity between two equal-length f32 vectors. Zero-norm
    /// vectors score 0 (never NaN), matching the engine's own guard.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (&x, &y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom > 0.0 { dot / denom } else { 0.0 }
    }

    /// Exact top-`k` nearest neighbours of `query` over the corpus by cosine
    /// on the original f32 vectors, excluding the query's own fact.
    fn ground_truth(
        corpus: &[(FactId, Vec<f32>)],
        query: &[f32],
        self_id: FactId,
        k: usize,
    ) -> Vec<FactId> {
        let mut scored: Vec<(f32, FactId)> = corpus
            .iter()
            .filter(|(id, _)| *id != self_id)
            .map(|(id, v)| (cosine(query, v), *id))
            .collect();
        // Descending by score; the highest cosines are the nearest.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    const K: usize = 8;
    const FILLS: [usize; 3] = [2_000, 20_000, 60_000];
    const DIMS: [usize; 2] = [384, 768];
    // Query sample per point — enough for a stable mean without making the
    // O(fill) brute force dominate.
    const QUERIES: usize = 100;

    println!("# recall@{K} vs exact f32 cosine (ground truth), self-excluded");
    println!("dim\tfill\tmode\trecall_mean\trecall_min\tqueries");

    for &dim in &DIMS {
        for &fill in &FILLS {
            let profile = Profile {
                dim,
                w_revise: 0,
                w_forget: 0,
                w_link: 0,
                ..Profile::default()
            };
            let seed = 0x5EC0_0000_0000_0000 ^ (fill as u64) ^ ((dim as u64) << 40);
            let ops = Gen::new(seed, profile).ops(fill);

            let mut cfg = Config::default();
            cfg.dim = dim;
            // Isolate vector quality: with a single (vector) source active and
            // the recency boost off, the fused ranking collapses to the pure
            // vector top-k, so recall reflects quantization + HNSW alone, not
            // the recency re-ranking a real recall would apply.
            cfg.w_recency = 0.0;
            let mut mem = Memory::new(cfg).unwrap();
            let mut store = MemStorage::new();

            // Apply the stream, capturing each remembered vector paired with the
            // fact id the engine minted for it — this is the ground-truth corpus.
            let mut corpus: Vec<(FactId, Vec<f32>)> = Vec::with_capacity(fill);
            for op in &ops {
                let outcome = apply(&mut mem, &mut store, op).unwrap();
                if let (
                    GenOp::Remember {
                        vector: Some(v), ..
                    },
                    Some(o),
                ) = (op, outcome)
                {
                    corpus.push((o.id, v.clone()));
                }
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
            mem.maintain(&mut store, now).unwrap();
            let mode = if fill >= 24_000 { "hnsw" } else { "flat" };

            // Sample queries evenly across the corpus.
            let step = (corpus.len() / QUERIES).max(1);
            let mut out = RecallResult::default();
            let mut scratch = RecallScratch::new();
            let mut sum = 0.0f64;
            let mut min = 1.0f32;
            let mut n = 0usize;
            for (self_id, qv) in corpus.iter().step_by(step) {
                let truth = ground_truth(&corpus, qv, *self_id, K);
                if truth.is_empty() {
                    continue;
                }
                let q = RecallQuery {
                    vector: Some(qv),
                    k: K,
                    text: None,
                    ..RecallQuery::text(now, "")
                };
                mem.recall_into(q, &mut scratch, &mut out).unwrap();
                let hits = out
                    .facts
                    .iter()
                    .filter(|f| f.id != *self_id && truth.contains(&f.id))
                    .count();
                let recall = hits as f32 / truth.len() as f32;
                sum += recall as f64;
                min = min.min(recall);
                n += 1;
            }
            let mean = if n > 0 { sum / n as f64 } else { 0.0 };
            println!("{dim}\t{fill}\t{mode}\t{mean:.3}\t{min:.3}\t{n}");
        }
    }
}

#[cfg(target_family = "wasm")]
fn main() {}
