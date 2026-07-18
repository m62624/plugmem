//! Criterion benchmarks for the engine layers (specs/07 §4). Groups are
//! added as each layer lands; every index gets a group — the mandate from
//! specs/07. Informational, not CI gates (gates are counter tests).

use core::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use plugmem_arena::Slot;
use plugmem_core::tokenizer::tokenize;
use plugmem_core::{BlobId, EntityId, FactId, FactRecord, VALID_TO_OPEN};

/// A realistic mixed-language fact corpus line, ~150 bytes (the testgen
/// text-length center).
const SAMPLE: &str = "User предпочитает tokio 1.47 строгим версиям — записано \
2026-07-18; см. проект plugmem, тэги #pref #rust-42 и немного 東京 текста.";

fn bench_tokenizer(c: &mut Criterion) {
    let mut g = c.benchmark_group("tokenizer");
    let bytes = SAMPLE.len() as u64;
    g.throughput(Throughput::Bytes(bytes));
    g.bench_function("mixed_150b", |b| {
        let mut buf = String::new();
        b.iter(|| {
            let mut n = 0usize;
            tokenize(black_box(SAMPLE), &mut buf, |t| n += t.len());
            n
        });
    });
    // Pure-ASCII path (the common English case, no lowercase expansion).
    let ascii = "the quick brown fox jumps over the lazy dog 0123456789 again and again";
    g.throughput(Throughput::Bytes(ascii.len() as u64));
    g.bench_function("ascii_72b", |b| {
        let mut buf = String::new();
        b.iter(|| {
            let mut n = 0usize;
            tokenize(black_box(ascii), &mut buf, |t| n += t.len());
            n
        });
    });
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

criterion_group!(benches, bench_tokenizer, bench_record_codec);
criterion_main!(benches);
