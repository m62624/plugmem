//! Criterion benchmarks for the four storage structures (specs/01 matrix).
//!
//! Run with `cargo bench -p plugmem-arena`. These are informational (trends,
//! spec targets), not CI gates — the gates are deterministic counter tests
//! (specs/07). Spec targets on a modern native x86-64 desktop @100k records:
//! insert <= 300 ns, get <= 150 ns.
//!
//! All inputs are deterministic (xorshift64 with fixed seeds): reruns
//! measure the code, not the workload.

use core::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, BlobId, ChunkPool, ChunkPoolCfg, Interner, ListHandle,
    ShardMode, Slot, key,
};

/// A realistic 16-byte record: 12-byte composite key `[u64 | u32]` (entity
/// id + revision, or timestamp + fact id) and a 4-byte payload.
#[derive(Clone, Copy)]
struct Rec16 {
    hi: u64,
    lo: u32,
    val: u32,
}

impl Rec16 {
    fn new(hi: u64, val: u32) -> Self {
        Self { hi, lo: 0, val }
    }

    fn key(hi: u64) -> [u8; 12] {
        let mut k = [0u8; 12];
        key::write_pair(&mut k, hi, 0);
        k
    }
}

impl Slot for Rec16 {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 12;
    fn write(&self, out: &mut [u8]) {
        key::write_pair(out, self.hi, self.lo);
        out[12..16].copy_from_slice(&self.val.to_be_bytes());
    }
    fn read(bytes: &[u8]) -> Self {
        let (hi, lo) = key::read_pair(bytes);
        Self {
            hi,
            lo,
            val: u32::from_be_bytes(bytes[12..16].try_into().unwrap()),
        }
    }
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

fn random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = xorshift(seed);
    (0..n).map(|_| rng()).collect()
}

fn build_arena(keys: &[u64], mode: ShardMode) -> Arena<'_, Rec16> {
    let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, mode)).unwrap();
    for &k in keys {
        a.insert(&Rec16::new(k, 1)).unwrap();
    }
    a
}

fn bench_arena_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("arena_insert");
    g.sample_size(10);
    for n in [1_000usize, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("seq", n), &n, |b, &n| {
            b.iter(|| {
                let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Uniform)).unwrap();
                for i in 0..n as u64 {
                    a.insert(&Rec16::new(i, 1)).unwrap();
                }
                a
            });
        });
        let keys = random_keys(n, 0xB0B5_1234_5678_9ABC);
        g.bench_with_input(BenchmarkId::new("random", n), &keys, |b, keys| {
            b.iter(|| build_arena(keys, ShardMode::Uniform));
        });
    }
    g.finish();
}

fn bench_arena_get(c: &mut Criterion) {
    let mut g = c.benchmark_group("arena_get");
    // 10k lookups per iteration keeps one iteration ~1 ms while the
    // reported number stays per-element.
    const LOOKUPS: usize = 10_000;
    for n in [100_000usize, 1_000_000] {
        let keys = random_keys(n, 0xB0B5_1234_5678_9ABC);
        let arena = build_arena(&keys, ShardMode::Uniform);
        // Hits: an evenly strided sample of present keys (out of insertion
        // order, so the walk is not accidentally cache-warm).
        let hits: Vec<u64> = keys.iter().step_by(n / LOOKUPS).copied().collect();
        // Misses: a disjoint random stream (collisions with 64-bit keys are
        // measure-zero).
        let misses = random_keys(LOOKUPS, 0xDEAD_BEEF_CAFE_F00D);
        g.throughput(Throughput::Elements(hits.len() as u64));
        g.bench_with_input(BenchmarkId::new("hit", n), &hits, |b, hits| {
            b.iter(|| {
                let mut found = 0u32;
                for &k in hits {
                    found += u32::from(arena.contains(&Rec16::key(k)));
                }
                found
            });
        });
        g.throughput(Throughput::Elements(misses.len() as u64));
        g.bench_with_input(BenchmarkId::new("miss", n), &misses, |b, misses| {
            b.iter(|| {
                let mut found = 0u32;
                for &k in misses {
                    found += u32::from(arena.contains(&Rec16::key(k)));
                }
                found
            });
        });
    }
    g.finish();
}

fn bench_arena_range(c: &mut Criterion) {
    let mut g = c.benchmark_group("arena_range");
    const N: usize = 100_000;
    const SCAN: usize = 1_000;
    let keys = random_keys(N, 0xB0B5_1234_5678_9ABC);
    let arena = build_arena(&keys, ShardMode::Ordered);
    // A window of ~SCAN consecutive (by value) keys somewhere in the middle.
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    let from = Rec16::key(sorted[N / 2]);
    let to = Rec16::key(sorted[N / 2 + SCAN]);
    g.throughput(Throughput::Elements(SCAN as u64));
    g.bench_function("scan_1k_of_100k", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for rec in arena.range(&from, &to) {
                acc += u64::from(rec.val);
            }
            acc
        });
    });
    g.finish();
}

fn bench_interner(c: &mut Criterion) {
    let mut g = c.benchmark_group("interner");
    const N: usize = 10_000;
    let words: Vec<String> = (0..N).map(|i| format!("term-{i}")).collect();
    let fresh: Vec<String> = (0..N).map(|i| format!("fresh-{i}")).collect();
    let mut warm = Interner::new(BlobHeapCfg::new());
    for w in &words {
        warm.intern(w).unwrap();
    }
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("intern_hit_10k", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for w in &words {
                acc += u64::from(warm.intern(w).unwrap().0);
            }
            acc
        });
    });
    g.bench_function("intern_miss_10k", |b| {
        b.iter_batched(
            || warm.clone(),
            |mut it| {
                for w in &fresh {
                    it.intern(w).unwrap();
                }
                it
            },
            BatchSize::LargeInput,
        );
    });
    let ids: Vec<_> = words.iter().map(|w| warm.intern(w).unwrap()).collect();
    g.bench_function("resolve_10k", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for &id in &ids {
                acc += warm.resolve(id).len();
            }
            acc
        });
    });
    g.finish();
}

fn bench_blob_heap(c: &mut Criterion) {
    let mut g = c.benchmark_group("blob_heap");
    const N: usize = 100_000;
    let mut rng = xorshift(0x0B10_B0B5_0000_0001);
    let blobs: Vec<Vec<u8>> = (0..N)
        .map(|_| {
            let len = 16 + (rng() % 185) as usize;
            (0..len).map(|i| i as u8).collect()
        })
        .collect();
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("push_100k_var_len", |b| {
        b.iter(|| {
            let mut h = BlobHeap::new(BlobHeapCfg::new());
            for blob in &blobs {
                h.push(blob).unwrap();
            }
            h
        });
    });
    let mut heap = BlobHeap::new(BlobHeapCfg::new());
    for blob in &blobs {
        heap.push(blob).unwrap();
    }
    let ids: Vec<u32> = (0..10_000).map(|_| (rng() % N as u64) as u32).collect();
    g.throughput(Throughput::Elements(ids.len() as u64));
    g.bench_function("get_10k", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for &id in &ids {
                acc += heap.get(BlobId(id)).len();
            }
            acc
        });
    });
    g.finish();
}

fn bench_chunk_pool(c: &mut Criterion) {
    let mut g = c.benchmark_group("chunk_pool");
    const N: usize = 100_000;
    const LISTS: usize = 1_024;
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("push_8b_round_robin", |b| {
        b.iter(|| {
            let mut pool = ChunkPool::new(ChunkPoolCfg::new());
            let mut lists = vec![ListHandle::EMPTY; LISTS];
            for i in 0..N {
                pool.push(&mut lists[i % LISTS], &(i as u64).to_be_bytes())
                    .unwrap();
            }
            (pool, lists)
        });
    });
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut lists = vec![ListHandle::EMPTY; LISTS];
    for i in 0..N {
        pool.push(&mut lists[i % LISTS], &(i as u64).to_be_bytes())
            .unwrap();
    }
    g.bench_function("iter_all_lists", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for list in &lists {
                for chunk in pool.iter(list) {
                    acc += black_box(chunk).len();
                }
            }
            acc
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_arena_insert,
    bench_arena_get,
    bench_arena_range,
    bench_interner,
    bench_blob_heap,
    bench_chunk_pool
);
criterion_main!(benches);
