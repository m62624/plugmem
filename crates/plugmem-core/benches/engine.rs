//! Criterion benchmarks for the engine layers. Groups are
//! added as each layer lands; every index gets a group — the mandate from
//! Informational, not CI gates (gates are counter tests).

use core::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use plugmem_arena::Slot;
use plugmem_core::tokenizer::Tokenizer;
use plugmem_core::{BlobId, EntityId, FactId, FactRecord, VALID_TO_OPEN};

/// A realistic mixed-language fact corpus line, ~150 bytes (the testgen
/// text-length center).
const SAMPLE: &str = "User предпочитает tokio 1.47 строгим версиям — записано \
2026-07-18; см. проект plugmem, тэги #pref #rust-42 и немного 東京 текста.";

/// One tokenizer benchmark case over a reused instance.
fn tok_case(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    text: &str,
) {
    g.throughput(Throughput::Bytes(text.len() as u64));
    g.bench_function(name, |b| {
        let mut tk = Tokenizer::new();
        b.iter(|| {
            let mut n = 0usize;
            tk.tokenize(black_box(text), &mut |t| n += t.len());
            n
        });
    });
}

fn bench_tokenizer(c: &mut Criterion) {
    let mut g = c.benchmark_group("tokenizer");
    tok_case(&mut g, "mixed_190b", SAMPLE);
    // Pure-ASCII path (the common English case, no folding work).
    tok_case(
        &mut g,
        "ascii_72b",
        "the quick brown fox jumps over the lazy dog 0123456789 again and again",
    );
    // CJK path: bigram machine + per-char segments.
    tok_case(
        &mut g,
        "cjk_60b",
        "彼は寿司が好きだ。東京都に住んでいる。トーキョー",
    );
    g.finish();
}

fn bench_record_codec(c: &mut Criterion) {
    let mut g = c.benchmark_group("record_codec");
    let rec = FactRecord {
        id: FactId(123_456),
        entity: EntityId(42),
        flags: 0,
        kind: 0,
        text: BlobId(9_000),
        vector: 7,
        revises: FactId::NONE,
        recorded_at: 1_784_000_000_000,
        valid_from: 1_784_000_000_000,
        valid_to: VALID_TO_OPEN,
    };
    let mut buf = [0u8; FactRecord::SIZE];
    g.throughput(Throughput::Elements(1));
    g.bench_function("fact_write", |b| {
        b.iter(|| {
            rec.write(black_box(&mut buf));
            buf
        });
    });
    rec.write(&mut buf);
    g.bench_function("fact_read", |b| {
        b.iter(|| FactRecord::read(black_box(&buf)));
    });
    g.finish();
}

/// Deterministic pseudo-random u64 stream.
fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// A skewed synthetic corpus: `docs` documents of `terms_per_doc` terms
/// drawn from a 3000-term vocabulary with a power-law bias (common terms
/// dominate, as in real text).
fn corpus(docs: usize, terms_per_doc: usize) -> Vec<Vec<(u32, u8)>> {
    const VOCAB: u64 = 3000;
    let mut rng = xorshift(0xC0FF_EE00_0000_0001);
    (0..docs)
        .map(|_| {
            let mut tfs: Vec<(u32, u8)> = Vec::with_capacity(terms_per_doc);
            for _ in 0..terms_per_doc {
                let r = (rng() % 10_000) as f32 / 10_000.0;
                let term = ((r * r * r) * VOCAB as f32) as u32;
                match tfs.iter_mut().find(|(t, _)| *t == term) {
                    Some((_, tf)) => *tf = tf.saturating_add(1),
                    None => tfs.push((term, 1)),
                }
            }
            tfs
        })
        .collect()
}

fn bench_bm25(c: &mut Criterion) {
    use plugmem_core::index::bm25::{Bm25Index, Bm25Scratch};

    let mut g = c.benchmark_group("bm25");
    const DOCS: usize = 10_000;
    let docs = corpus(DOCS, 8);
    g.throughput(Throughput::Elements(DOCS as u64));
    g.bench_function("index_10k_docs", |b| {
        b.iter(|| {
            let mut idx = Bm25Index::new(2048, usize::MAX).unwrap();
            for (fact, terms) in docs.iter().enumerate() {
                idx.index_doc(plugmem_core::FactId(fact as u32), terms)
                    .unwrap();
            }
            idx
        });
    });
    let mut idx = Bm25Index::new(2048, usize::MAX).unwrap();
    for (fact, terms) in docs.iter().enumerate() {
        idx.index_doc(plugmem_core::FactId(fact as u32), terms)
            .unwrap();
    }
    // A mixed query: one very common term, one mid, one rare.
    let query = [1u32, 400, 2500];
    let mut scratch = Bm25Scratch::new();
    let mut out = Vec::new();
    g.throughput(Throughput::Elements(1));
    g.bench_function("search_3_terms_of_10k", |b| {
        b.iter(|| {
            idx.search(
                (1.2, 0.75),
                black_box(&query),
                8,
                &mut |_| true,
                &mut scratch,
                &mut out,
            );
            out.len()
        });
    });
    g.finish();
}

fn bench_idlist(c: &mut Criterion) {
    use plugmem_core::index::{IdListIndex, IntersectScratch, intersect};

    let mut g = c.benchmark_group("idlist");
    // Three tag lists over 100k facts: 10%, 5% and 2% selectivity.
    let mut idx = IdListIndex::new(512, usize::MAX).unwrap();
    let mut rng = xorshift(0x7A67_0000_0000_0001);
    for id in 0..100_000u32 {
        let r = rng();
        if r.is_multiple_of(10) {
            idx.push(1, plugmem_core::FactId(id), 0).unwrap();
        }
        if r.is_multiple_of(20) {
            idx.push(2, plugmem_core::FactId(id), 0).unwrap();
        }
        if r.is_multiple_of(50) {
            idx.push(3, plugmem_core::FactId(id), 0).unwrap();
        }
    }
    let mut scratch = IntersectScratch::new();
    let mut out = Vec::new();
    g.throughput(Throughput::Elements(1));
    g.bench_function("intersect_3_tags_of_100k", |b| {
        b.iter(|| {
            intersect(&idx, black_box(&[1, 2, 3]), &mut scratch, &mut out);
            out.len()
        });
    });
    g.finish();
}

/// The last timestamp of a generated stream (queries run "after" it).
fn last_now(ops: &[plugmem_testgen::GenOp]) -> u64 {
    use plugmem_testgen::GenOp;
    ops.iter()
        .map(|op| match op {
            GenOp::Remember { now, .. }
            | GenOp::Revise { now, .. }
            | GenOp::Forget { now, .. }
            | GenOp::Link { now, .. }
            | GenOp::Maintain { now } => *now,
        })
        .max()
        .unwrap_or(0)
}

fn bench_recall(c: &mut Criterion) {
    use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult, RecallScratch};
    use plugmem_testgen::{Gen, GenOp, Profile, apply, word_for};

    let mut g = c.benchmark_group("recall");
    g.sample_size(50);
    let mut cfg = Config::default();
    cfg.shards_postings = 2048;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();
    // The corpus passport via testgen: Zipf vocabulary,
    // normal text lengths, hub entities, tags, links, revisions and
    // forgets in the default mix.
    let ops = Gen::new(0x5EED_0000_0000_0007, Profile::default()).ops(100_000);
    for op in &ops {
        apply(&mut mem, &mut store, op).unwrap();
    }
    // Query pieces come from the stream itself: the head entity and tag
    // of the corpus, and dictionary words of known frequency classes
    // (rank 1 = common, 400 = mid, 2500 = rare).
    let entity = ops
        .iter()
        .find_map(|op| match op {
            GenOp::Remember {
                entity: Some(e), ..
            } => Some(e.clone()),
            _ => None,
        })
        .expect("the stream contains entity-bearing remembers");
    let tag = ops
        .iter()
        .find_map(|op| match op {
            GenOp::Remember { tags, .. } if !tags.is_empty() => Some(tags[0].clone()),
            _ => None,
        })
        .expect("the stream contains tagged remembers");
    let now = last_now(&ops) + 1;
    let text = format!("{} {} {}", word_for(1), word_for(400), word_for(2500));

    let mut out = RecallResult::default();
    let mut scratch = RecallScratch::new();
    let anchors = [entity.as_str()];
    let q = RecallQuery {
        entities: &anchors,
        ..RecallQuery::text(now, &text)
    };
    g.throughput(Throughput::Elements(1));
    g.bench_function("structural_100k", |b| {
        b.iter(|| {
            mem.recall_into(black_box(q), &mut scratch, &mut out)
                .unwrap();
            out.facts.len()
        });
    });
    // Tag filter plus a recorded_at window spanning the last ~5k
    // operations (range cost is proportional to the window by design;
    // the time axis thickens toward the end, so a fixed *time* share
    // would cover most of the corpus).
    let mut nows: Vec<u64> = ops
        .iter()
        .map(|op| match op {
            GenOp::Remember { now, .. }
            | GenOp::Revise { now, .. }
            | GenOp::Forget { now, .. }
            | GenOp::Link { now, .. }
            | GenOp::Maintain { now } => *now,
        })
        .collect();
    nows.sort_unstable();
    let from = nows[nows.len().saturating_sub(5_000)];
    let tags = [tag.as_str()];
    let tagged = RecallQuery {
        tags: &tags,
        range: Some((from, now)),
        ..RecallQuery::text(now, &text)
    };
    g.bench_function("tags_and_range_100k", |b| {
        b.iter(|| {
            mem.recall_into(black_box(tagged), &mut scratch, &mut out)
                .unwrap();
            out.facts.len()
        });
    });
    g.finish();
}

fn bench_vec(c: &mut Criterion) {
    use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult, RecallScratch};
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    // The A.5(9) gate corpus: 24k vectors of dim 384 — the
    // flat-to-HNSW threshold — searched at k = 8.
    let mut g = c.benchmark_group("vec");
    g.sample_size(50);
    const DIM: usize = 384;
    const VECTORS: usize = 24_000;
    let profile = Profile {
        dim: DIM,
        w_revise: 0,
        w_forget: 0,
        w_link: 0,
        ..Profile::default()
    };
    let ops = Gen::new(0x5EC0_0000_0000_0384, profile).ops(VECTORS);
    let mut cfg = Config::default();
    cfg.dim = DIM;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();
    for op in &ops {
        apply(&mut mem, &mut store, op).unwrap();
    }
    // The query is a real corpus member: a cluster inhabitant, so the
    // search has genuine near neighbors to rank.
    let query = ops
        .iter()
        .find_map(|op| match op {
            GenOp::Remember {
                vector: Some(v), ..
            } => Some(v.clone()),
            _ => None,
        })
        .expect("every remember carries a vector at dim > 0");
    let now = last_now(&ops) + 1;
    let mut out = RecallResult::default();
    let mut scratch = RecallScratch::new();
    let q = RecallQuery {
        vector: Some(&query),
        k: 8,
        text: None,
        ..RecallQuery::text(now, "")
    };
    g.throughput(Throughput::Elements(1));
    g.bench_function("flat_24k_d384_k8", |b| {
        b.iter(|| {
            mem.recall_into(black_box(q), &mut scratch, &mut out)
                .unwrap();
            out.facts.len()
        });
    });
    g.finish();
}

fn bench_hnsw(c: &mut Criterion) {
    use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult, RecallScratch};
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    // Above the default flat_to_hnsw threshold (24k): `maintain` builds
    // the graph, search goes through it. The one-time build cost is
    // printed to stderr (it is a maintain cost, not a query cost).
    let mut g = c.benchmark_group("hnsw");
    g.sample_size(50);
    const DIM: usize = 384;
    const VECTORS: usize = 30_000;
    let profile = Profile {
        dim: DIM,
        w_revise: 0,
        w_forget: 0,
        w_link: 0,
        ..Profile::default()
    };
    let ops = Gen::new(0x4A5A_0000_0000_0001, profile).ops(VECTORS);
    let mut cfg = Config::default();
    cfg.dim = DIM;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();
    for op in &ops {
        apply(&mut mem, &mut store, op).unwrap();
    }
    let now = last_now(&ops) + 1;
    let t0 = std::time::Instant::now();
    mem.maintain(&mut store, now).unwrap();
    eprintln!(
        "hnsw: maintain graph build over {VECTORS} vectors (dim {DIM}) took {:?}",
        t0.elapsed()
    );
    let query = ops
        .iter()
        .find_map(|op| match op {
            GenOp::Remember {
                vector: Some(v), ..
            } => Some(v.clone()),
            _ => None,
        })
        .expect("every remember carries a vector");
    let mut out = RecallResult::default();
    let mut scratch = RecallScratch::new();
    let q = RecallQuery {
        vector: Some(&query),
        k: 8,
        text: None,
        ..RecallQuery::text(now, "")
    };
    g.throughput(Throughput::Elements(1));
    g.bench_function("search_30k_d384_k8_ef64", |b| {
        b.iter(|| {
            mem.recall_into(black_box(q), &mut scratch, &mut out)
                .unwrap();
            out.facts.len()
        });
    });
    g.finish();
}

fn bench_search_matrix(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult, RecallScratch};
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    // Baseline sweep for vector search across **fill** (how many memories) and
    // **difficulty** (embedding dimension), at the default k = 8. This fixes
    // the current numbers so a later vector optimization — a signature-tier
    // layout (signatures resident, int8 in mmap) or a binary HNSW traversal —
    // can be measured against a committed reference instead of memory.
    //
    // Fill straddles the flat/HNSW split (specs default flat_to_hnsw = 24k):
    // 2k and 20k stay flat (exact int8 rescoring over all candidates), 60k is
    // searched through the graph that `maintain` builds. The one-time build
    // cost and the resolved mode are printed to stderr (a maintain cost, not a
    // query cost). Difficulty is the dimension: 384 and 768 are the common
    // embedding sizes; the per-vector stride roughly doubles between them.
    let mut g = c.benchmark_group("search_matrix");
    g.sample_size(20);
    const FILLS: [usize; 3] = [2_000, 20_000, 60_000];
    const DIMS: [usize; 2] = [384, 768];

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
            let mut mem = Memory::new(cfg).unwrap();
            let mut store = MemStorage::new();
            for op in &ops {
                apply(&mut mem, &mut store, op).unwrap();
            }
            let now = last_now(&ops) + 1;
            // Past the threshold this builds the graph; below it search stays
            // flat and maintain is a cheap no-op for the vector index.
            let t0 = std::time::Instant::now();
            mem.maintain(&mut store, now).unwrap();
            let mode = if fill >= 24_000 { "hnsw" } else { "flat" };
            eprintln!(
                "search_matrix: fill={fill} dim={dim} mode={mode} maintain={:?}",
                t0.elapsed()
            );
            // A real corpus member as the query: genuine near neighbors to rank.
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
            let mut scratch = RecallScratch::new();
            let q = RecallQuery {
                vector: Some(&query),
                k: 8,
                text: None,
                ..RecallQuery::text(now, "")
            };
            g.throughput(Throughput::Elements(1));
            g.bench_function(BenchmarkId::new(format!("d{dim}"), fill), |b| {
                b.iter(|| {
                    mem.recall_into(black_box(q), &mut scratch, &mut out)
                        .unwrap();
                    out.facts.len()
                });
            });
        }
    }
    g.finish();
}

fn bench_scrub(c: &mut Criterion) {
    use plugmem_core::snapshot::Snapshot;
    use plugmem_core::{Config, MemStorage, Memory, RememberInput};

    // The on-demand container scrub: sweep the slice budget over
    // one warm snapshot buffer to find the knee — below it, per-`next()`
    // overhead (parse of the cursor state, the section-boundary checks)
    // dominates; above it, the streaming xxh3 throughput plateaus. This is the
    // measurement behind `DEFAULT_SCRUB_BUDGET`; the host mmap bench
    // (`plugmem-host`) adds the cold-page cost, which shifts absolute time but
    // not the knee.
    let mut g = c.benchmark_group("scrub");
    g.sample_size(20);
    let mut mem = Memory::new(Config::default()).unwrap();
    let mut store = MemStorage::new();
    // Wide texts so the container is large enough (~8 MiB) to sweep MiB-scale
    // budgets.
    let big = "lorem ipsum dolor sit amet ".repeat(75); // ~2 KiB
    for i in 0..4_000u64 {
        mem.remember(&mut store, RememberInput::text(i + 1, &big))
            .unwrap();
    }
    let bytes = mem.snapshot_bytes(4_001);
    let total = bytes.len() as u64;
    eprintln!("scrub: snapshot is {total} bytes");
    g.throughput(Throughput::Bytes(total));
    for budget in [64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20] {
        g.bench_function(format!("budget_{}KiB", budget >> 10), |b| {
            b.iter(|| {
                let snap = Snapshot::parse(black_box(&bytes)).unwrap();
                let mut done = 0u64;
                for step in snap.scrub_with_budget(budget) {
                    done = step.unwrap().done_bytes;
                }
                done
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_tokenizer,
    bench_record_codec,
    bench_bm25,
    bench_idlist,
    bench_recall,
    bench_vec,
    bench_hnsw,
    bench_search_matrix,
    bench_scrub
);
criterion_main!(benches);
