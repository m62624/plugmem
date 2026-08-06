//! Recall quality on **real** embeddings (Ollama `nomic-embed-text`, dim 768),
//! the reality check on the synthetic `recall_quality` baseline: testgen
//! vectors are random, so their "8 nearest" are almost arbitrary and any
//! quantization noise reorders them. Real embeddings have genuine semantic
//! structure (similar sentences cluster), so this is the honest number.
//!
//! It also sweeps the **query k**: the flat search rescores the best
//! `max(4k, 64)` signature candidates, so asking for a larger k widens that
//! sieve. recall@8 is then measured among the top-k the engine returns — this
//! shows how much recall a wider sieve buys and what it costs in time, the
//! exact "raise recall without losing much speed" trade-off, with no code
//! change.
//!
//! Needs a running Ollama with the model pulled:
//! ```text
//! ollama pull nomic-embed-text
//! cargo run --release -p plugmem-host --example recall_ollama
//! ```

use std::time::Instant;

use plugmem_core::{
    Config, FactId, MemStorage, Memory, RecallQuery, RecallResult, RecallScratch, RememberInput,
};
use plugmem_host::{Embedder, OpenAiCompatEmbedder};

const DIM: usize = 768;
const K_EVAL: usize = 8; // recall@8, the metric
const K_SWEEP: [usize; 4] = [8, 16, 32, 64]; // query k = sieve width

/// A meaningful English corpus: topic templates × fillers, so the real
/// embedder places genuine semantic clusters (variations of one template land
/// near each other, different topics land apart). ~2000 unique sentences.
fn corpus() -> Vec<String> {
    let templates: &[(&str, &[&str])] = &[
        (
            "{} prefers strong typing and pins dependency versions",
            &[
                "The backend team",
                "Our new hire",
                "The reviewer",
                "Alex",
                "The staff engineer",
            ],
        ),
        (
            "the deployment to {} failed and was rolled back",
            &[
                "staging",
                "production",
                "the EU region",
                "the canary fleet",
                "us-east-1",
            ],
        ),
        (
            "{} is allergic to peanuts and avoids shellfish",
            &[
                "the guest",
                "my colleague",
                "her brother",
                "the client",
                "our intern",
            ],
        ),
        (
            "the flight to {} was delayed by three hours",
            &["Tokyo", "Berlin", "São Paulo", "Reykjavík", "Singapore"],
        ),
        (
            "{} scored the winning goal in the final minute",
            &[
                "the striker",
                "the substitute",
                "our captain",
                "the rookie",
                "the veteran",
            ],
        ),
        (
            "{} recommends the new espresso blend from the roastery",
            &[
                "the barista",
                "my neighbor",
                "the food critic",
                "her mentor",
                "the owner",
            ],
        ),
        (
            "the quarterly revenue for {} grew fifteen percent",
            &[
                "the cloud division",
                "the retail arm",
                "the mobile unit",
                "the ads business",
                "the hardware line",
            ],
        ),
        (
            "{} practices the violin every morning before work",
            &[
                "the student",
                "the conductor",
                "my aunt",
                "the teacher",
                "the prodigy",
            ],
        ),
        (
            "the hiking trail near {} is closed for maintenance",
            &[
                "the lake",
                "the ridge",
                "the waterfall",
                "the old mine",
                "the summit",
            ],
        ),
        (
            "{} switched from Postgres to a columnar warehouse",
            &[
                "the analytics team",
                "the startup",
                "the data platform",
                "the finance group",
                "the ETL pipeline",
            ],
        ),
        (
            "{} adopted a rescue dog from the local shelter",
            &[
                "the family",
                "my roommate",
                "the retiree",
                "the couple",
                "the vet",
            ],
        ),
        (
            "the museum exhibit about {} opens next spring",
            &[
                "ancient Rome",
                "deep-sea life",
                "modern jazz",
                "space travel",
                "textile art",
            ],
        ),
        (
            "{} tuned the model and cut latency in half",
            &[
                "the ML engineer",
                "the research group",
                "the intern",
                "the platform team",
                "the contractor",
            ],
        ),
        (
            "{} planted tomatoes and basil in the community garden",
            &[
                "the volunteers",
                "my grandmother",
                "the schoolkids",
                "the chef",
                "the botanist",
            ],
        ),
        (
            "the concert by {} sold out within minutes",
            &[
                "the orchestra",
                "the indie band",
                "the pianist",
                "the choir",
                "the DJ",
            ],
        ),
        (
            "{} reviewed the pull request and requested changes",
            &[
                "the maintainer",
                "the lead",
                "a bot",
                "the security team",
                "the architect",
            ],
        ),
        (
            "the recipe for {} calls for saffron and fresh mint",
            &[
                "the paella",
                "the lamb tagine",
                "the rice pudding",
                "the seafood stew",
                "the pilaf",
            ],
        ),
        (
            "{} negotiated a lower rate on the cloud contract",
            &[
                "the CFO",
                "procurement",
                "the founder",
                "the ops lead",
                "the consultant",
            ],
        ),
        (
            "{} sprained an ankle during the marathon",
            &[
                "the runner",
                "my trainer",
                "the pacer",
                "the champion",
                "the amateur",
            ],
        ),
        (
            "the library extended its hours during {}",
            &[
                "exam week",
                "the summer",
                "the festival",
                "the renovation",
                "the holidays",
            ],
        ),
    ];
    let mut out = Vec::new();
    // Multiply each template by a second filler axis (adverbial tails) to reach
    // ~2000 sentences with fine-grained near-duplicates inside each cluster.
    let tails: &[&str] = &[
        "",
        ", according to the notes",
        " last week",
        " again this quarter",
        ", the team confirmed",
        " despite the weather",
        " as planned",
        " to everyone's surprise",
        " after a long discussion",
        " for the record",
        " on Tuesday",
        " without any warning",
        " per the new policy",
        " once more",
        " with great enthusiasm",
        " ahead of schedule",
        " under budget",
        " over the objections",
        " citing fresh data",
        " and it was noted",
    ];
    for (tpl, fills) in templates {
        for f in *fills {
            for t in tails {
                out.push(format!("{}{}", tpl.replace("{}", f), t));
            }
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

fn ground_truth(vecs: &[Vec<f32>], q: usize, k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = vecs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != q)
        .map(|(i, v)| (cosine(&vecs[q], v), i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn main() {
    let texts = corpus();
    println!("corpus: {} sentences", texts.len());

    // Embed the whole corpus in batches through the real model.
    let emb = OpenAiCompatEmbedder::new(
        "http://localhost:11434/v1/embeddings",
        "nomic-embed-text",
        DIM,
    );
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let t0 = Instant::now();
    for chunk in refs.chunks(64) {
        vecs.extend(emb.embed(chunk).expect("ollama embed"));
    }
    println!("embedded in {:?}", t0.elapsed());

    // Build the engine over the real vectors (flat regime — under the 24k
    // HNSW threshold; recency off so the fused rank is the pure vector top-k).
    let mut cfg = Config::default();
    cfg.dim = DIM;
    cfg.w_recency = 0.0;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();
    let mut fid_of: Vec<FactId> = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        let o = mem
            .remember(
                &mut store,
                RememberInput {
                    vector: Some(&vecs[i]),
                    ..RememberInput::text((i + 1) as u64, text)
                },
            )
            .unwrap();
        fid_of.push(o.id);
    }
    let now = texts.len() as u64 + 1;

    // Sample ~200 queries evenly; ground truth is exact f32 cosine.
    let step = (texts.len() / 200).max(1);
    let queries: Vec<usize> = (0..texts.len()).step_by(step).collect();
    let truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|&q| ground_truth(&vecs, q, K_EVAL))
        .collect();

    println!("\n# real embeddings — recall@{K_EVAL} vs exact f32 cosine (self-excluded)");
    println!("# (synthetic testgen baseline for comparison: flat d768 2k = 0.854)");
    println!("query_k\tsieve\trecall_mean\trecall_min\tsearch_us");
    let mut out = RecallResult::default();
    let mut scratch = RecallScratch::new();
    for &kq in &K_SWEEP {
        let sieve = (4 * kq).max(64);
        let (mut sum, mut min, mut n) = (0.0f64, 1.0f32, 0usize);
        let t = Instant::now();
        for (&q, truth) in queries.iter().zip(&truths) {
            let query = RecallQuery {
                vector: Some(&vecs[q]),
                k: kq,
                text: None,
                ..RecallQuery::text(now, "")
            };
            mem.recall_into(query, &mut scratch, &mut out).unwrap();
            let self_fid = fid_of[q];
            // recall@K_EVAL: how many of the true 8 nearest appear anywhere in
            // the top-k the engine returned.
            let hits = truth
                .iter()
                .filter(|&&ti| {
                    let tfid = fid_of[ti];
                    out.facts.iter().any(|f| f.id == tfid && f.id != self_fid)
                })
                .count();
            sum += hits as f64 / truth.len() as f64;
            min = min.min(hits as f32 / truth.len() as f32);
            n += 1;
        }
        let per_query_us = t.elapsed().as_micros() as f64 / n as f64;
        println!(
            "{kq}\t{sieve}\t{:.3}\t{min:.3}\t{per_query_us:.1}",
            sum / n as f64
        );
    }
}
