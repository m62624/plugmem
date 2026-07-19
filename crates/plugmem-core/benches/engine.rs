//! Criterion benchmarks for the engine layers (specs/07 §4). Groups are
//! added as each layer lands; every index gets a group — the mandate from
//! specs/07. Informational, not CI gates (gates are counter tests).

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
    use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult};
    use plugmem_testgen::{Gen, GenOp, Profile, apply, word_for};

    let mut g = c.benchmark_group("recall");
    g.sample_size(50);
    let mut cfg = Config::default();
    cfg.shards_postings = 2048;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();
    // The specs/07 §6 corpus passport via testgen: Zipf vocabulary,
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
    let anchors = [entity.as_str()];
    let q = RecallQuery {
        entities: &anchors,
        ..RecallQuery::text(now, &text)
    };
    g.throughput(Throughput::Elements(1));
    g.bench_function("structural_100k", |b| {
        b.iter(|| {
            mem.recall_into(black_box(q), &mut out).unwrap();
            out.facts.len()
        });
    });
    // Tag filter plus a recorded_at window over the dense (late) part of
    // the time axis (range cost is proportional to the window by design).
    let tags = [tag.as_str()];
    let tagged = RecallQuery {
        tags: &tags,
        range: Some((now * 9 / 10, now)),
        ..RecallQuery::text(now, &text)
    };
    g.bench_function("tags_and_range_100k", |b| {
        b.iter(|| {
            mem.recall_into(black_box(tagged), &mut out).unwrap();
            out.facts.len()
        });
    });
    g.finish();
}

fn bench_vec(c: &mut Criterion) {
    use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult};
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    // The specs/11 A.5(9) gate corpus: 24k vectors of dim 384 — the
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
    let q = RecallQuery {
        vector: Some(&query),
        k: 8,
        text: None,
        ..RecallQuery::text(now, "")
    };
    g.throughput(Throughput::Elements(1));
    g.bench_function("flat_24k_d384_k8", |b| {
        b.iter(|| {
            mem.recall_into(black_box(q), &mut out).unwrap();
            out.facts.len()
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_tokenizer,
    bench_record_codec,
    bench_bm25,
    bench_idlist,
    bench_recall,
    bench_vec
);
criterion_main!(benches);
