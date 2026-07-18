//! Reproducible cross-runtime micro-benchmark — **zero dependencies**.
//!
//! Compares the arena against the standard library's collections of the
//! same *workload class* (ordered map / lookup map / flat sorted array) on:
//!
//! - insert throughput and **per-insert tail latency** (p50 / p99 / max —
//!   this is where page splits, tree rebalancing, hash-table rehashes and
//!   O(n) array shifts become visible);
//! - point lookup and ordered scan throughput;
//! - retained and **peak** memory, and the number of **allocator calls**
//!   during a build (measured with a counting global allocator).
//!
//! Runs identically on native and inside WebAssembly runtimes; the corpus
//! size is the first CLI argument (default 100000), so the same binary
//! measures both the design center and the scale ceiling:
//!
//! ```text
//! # native
//! cargo run --release --example bench_repro            # 100k
//! cargo run --release --example bench_repro -- 1000000 # 1M
//!
//! # wasm (either runtime)
//! cargo build --release --example bench_repro --target wasm32-wasip1
//! wasmtime run target/wasm32-wasip1/release/examples/bench_repro.wasm 1000000
//! wasmer run  target/wasm32-wasip1/release/examples/bench_repro.wasm -- 1000000
//! ```
//!
//! The whole matrix (every structure x every runtime x both sizes) is
//! scripted by `cargo run --release -p plugmem-bench-matrix`.
//!
//! Methodology: fixed workload (seeded xorshift keys — identical streams
//! everywhere), throughput = best of 3, latency percentiles from a separate
//! single instrumented pass (each insert bracketed by `Instant::now()`; the
//! clock-call overhead inflates p50 by a constant, spikes are unaffected —
//! compare percentiles *within* a runtime, not across). No criterion here
//! on purpose: this example must not pull a single crate, so the wasm
//! builds stay trivial and the numbers are easy to audit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, BlobId, ChunkPool, ChunkPoolCfg, Interner, ListHandle,
    ShardMode, Slot, TermId, key,
};

// --- counting allocator ---------------------------------------------------

/// Global allocator wrapper counting live bytes, peak bytes and calls.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates verbatim to `System`; the counters are metadata only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        let live = LIVE.fetch_add(new_size, Ordering::Relaxed) + new_size - layout.size();
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        PEAK.fetch_max(live, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Allocator activity of one measured build.
struct AllocStat {
    calls: usize,
    peak: usize,
    retained: usize,
}

impl AllocStat {
    /// Rescales the byte counters so the per-N report reads per stored
    /// item instead (used when stored items != stream length, e.g. the
    /// interner keeps only distinct terms). Call counts stay as measured.
    fn scaled(self, items: usize, n: usize) -> Self {
        // u128: `bytes * n` overflows a 32-bit usize on wasm32.
        let per = |v: usize| (v as u128 * n as u128 / items.max(1) as u128) as usize;
        Self {
            calls: self.calls,
            peak: per(self.peak),
            retained: per(self.retained),
        }
    }
}

/// Runs `f` with the allocator counters scoped to it.
fn with_alloc_stats<T>(f: impl FnOnce() -> T) -> (T, AllocStat) {
    let live0 = LIVE.load(Ordering::Relaxed);
    let calls0 = CALLS.load(Ordering::Relaxed);
    PEAK.store(live0, Ordering::Relaxed);
    let out = f();
    (
        out,
        AllocStat {
            calls: CALLS.load(Ordering::Relaxed) - calls0,
            peak: PEAK.load(Ordering::Relaxed) - live0,
            retained: LIVE.load(Ordering::Relaxed) - live0,
        },
    )
}

// --- workload -------------------------------------------------------------

/// The same 16-byte record class as `benches/storage.rs`: 12-byte composite
/// key, 4-byte payload.
#[derive(Clone, Copy)]
struct Rec16 {
    hi: u64,
    lo: u32,
    val: u32,
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

fn rec(hi: u64) -> Rec16 {
    Rec16 { hi, lo: 0, val: 1 }
}

fn rec_key(hi: u64) -> [u8; 12] {
    let mut k = [0u8; 12];
    key::write_pair(&mut k, hi, 0);
    k
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

const LOOKUPS: usize = 10_000;
const SCAN: usize = 1_000;
const REPS: usize = 3;

/// Runs `f` REPS times, returns the best wall time in nanoseconds.
fn best_ns<T>(mut f: impl FnMut() -> T) -> u64 {
    let mut best = u64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        let out = f();
        let ns = t.elapsed().as_nanos() as u64;
        std::hint::black_box(out);
        best = best.min(ns);
    }
    best
}

/// Per-insert latency percentiles of one instrumented build pass.
struct Tail {
    p50: u64,
    p99: u64,
    max: u64,
}

/// Times `insert(key)` for every key individually and extracts percentiles.
fn tail_of(keys: &[u64], mut insert: impl FnMut(u64)) -> Tail {
    let mut ns: Vec<u64> = Vec::with_capacity(keys.len());
    for &k in keys {
        let t = Instant::now();
        insert(k);
        ns.push(t.elapsed().as_nanos() as u64);
    }
    ns.sort_unstable();
    Tail {
        p50: ns[ns.len() / 2],
        p99: ns[ns.len() * 99 / 100],
        max: *ns.last().unwrap(),
    }
}

// --- reporting ------------------------------------------------------------

/// One structure's measured metrics; `None` = not applicable to its class.
struct Row {
    name: &'static str,
    insert_ns: Option<u64>,
    get_ns: Option<u64>,
    scan_ns: Option<u64>,
    tail: Option<Tail>,
    alloc: Option<AllocStat>,
}

fn emit(n: usize, rows: &[Row]) {
    println!(
        "bench_repro: N={n}, lookups={LOOKUPS}, scan={SCAN}, reps={REPS} (best kept), \
         latency pass: single, clock overhead included in p50"
    );
    // Machine-readable: one `#M <structure> <metric> <value>` line each.
    for r in rows {
        let mut m: Vec<(&str, f64)> = Vec::new();
        if let Some(v) = r.insert_ns {
            m.push(("insert_ns", v as f64 / n as f64));
        }
        if let Some(v) = r.get_ns {
            m.push(("get_ns", v as f64 / LOOKUPS as f64));
        }
        if let Some(v) = r.scan_ns {
            m.push(("scan_ns", v as f64 / SCAN as f64));
        }
        if let Some(t) = &r.tail {
            m.push(("ins_p50", t.p50 as f64));
            m.push(("ins_p99", t.p99 as f64));
            m.push(("ins_max", t.max as f64));
        }
        if let Some(a) = &r.alloc {
            m.push(("mem_b", a.retained as f64 / n as f64));
            m.push(("mem_peak_b", a.peak as f64 / n as f64));
            m.push(("allocs", a.calls as f64));
        }
        for (metric, value) in m {
            println!("#M\t{}\t{metric}\t{value:.1}", r.name);
        }
    }
}

// --- main -----------------------------------------------------------------

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100_000);
    let keys: Vec<u64> = {
        let mut rng = xorshift(0xB0B5_1234_5678_9ABC);
        (0..n).map(|_| rng()).collect()
    };
    let hit_keys: Vec<u64> = keys.iter().step_by(n / LOOKUPS).copied().collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    let (scan_from, scan_to) = (sorted[n / 2], sorted[n / 2 + SCAN]);

    let mut rows = Vec::new();

    // --- plugmem arena, Ordered (the ordered-map class) -------------------
    {
        let build = || {
            let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Ordered)).unwrap();
            for &k in &keys {
                a.insert(&rec(k)).unwrap();
            }
            a
        };
        let insert_ns = best_ns(build);
        let (arena, alloc) = with_alloc_stats(build);
        let mut tailed = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Ordered)).unwrap();
        let tail = tail_of(&keys, |k| {
            tailed.insert(&rec(k)).unwrap();
        });
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(arena.contains(&rec_key(k)));
            }
            acc
        });
        let scan_ns = best_ns(|| {
            let mut acc = 0u64;
            for r in arena.range(&rec_key(scan_from), &rec_key(scan_to)) {
                acc += u64::from(r.val);
            }
            acc
        });
        rows.push(Row {
            name: "plugmem Arena (Ordered)",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: Some(scan_ns),
            tail: Some(tail),
            alloc: Some(alloc),
        });
    }

    // --- plugmem arena, Uniform (the lookup-map class) --------------------
    {
        let build = || {
            let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Uniform)).unwrap();
            for &k in &keys {
                a.insert(&rec(k)).unwrap();
            }
            a
        };
        let insert_ns = best_ns(build);
        let (arena, alloc) = with_alloc_stats(build);
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(arena.contains(&rec_key(k)));
            }
            acc
        });
        rows.push(Row {
            name: "plugmem Arena (Uniform)",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: None,
            tail: None,
            alloc: Some(alloc),
        });
    }

    // --- BTreeMap (std ordered map — the same class) ----------------------
    {
        let build = || {
            let mut m: BTreeMap<u64, u32> = BTreeMap::new();
            for &k in &keys {
                m.insert(k, 1);
            }
            m
        };
        let insert_ns = best_ns(build);
        let (map, alloc) = with_alloc_stats(build);
        let mut tailed: BTreeMap<u64, u32> = BTreeMap::new();
        let tail = tail_of(&keys, |k| {
            tailed.insert(k, 1);
        });
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(map.contains_key(&k));
            }
            acc
        });
        let scan_ns = best_ns(|| {
            let mut acc = 0u64;
            for (_, v) in map.range(scan_from..scan_to) {
                acc += u64::from(*v);
            }
            acc
        });
        rows.push(Row {
            name: "std BTreeMap",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: Some(scan_ns),
            tail: Some(tail),
            alloc: Some(alloc),
        });
    }

    // --- HashMap (std lookup map — no ordering) ---------------------------
    {
        let build = || {
            let mut m: HashMap<u64, u32> = HashMap::new();
            for &k in &keys {
                m.insert(k, 1);
            }
            m
        };
        let insert_ns = best_ns(build);
        let (map, alloc) = with_alloc_stats(build);
        let mut tailed: HashMap<u64, u32> = HashMap::new();
        let tail = tail_of(&keys, |k| {
            tailed.insert(k, 1);
        });
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(map.contains_key(&k));
            }
            acc
        });
        rows.push(Row {
            name: "std HashMap",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: None,
            tail: Some(tail),
            alloc: Some(alloc),
        });
    }

    // --- sorted Vec, bulk build (flat baseline's best case) ---------------
    {
        let build = || {
            let mut v: Vec<(u64, u32)> = keys.iter().map(|&k| (k, 1)).collect();
            v.sort_unstable_by_key(|&(k, _)| k);
            v
        };
        let insert_ns = best_ns(build);
        let (vec, alloc) = with_alloc_stats(build);
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(vec.binary_search_by_key(&k, |&(k, _)| k).is_ok());
            }
            acc
        });
        let scan_ns = best_ns(|| {
            let start = vec.partition_point(|&(k, _)| k < scan_from);
            let mut acc = 0u64;
            for &(k, v) in &vec[start..] {
                if k >= scan_to {
                    break;
                }
                acc += u64::from(v);
            }
            acc
        });
        rows.push(Row {
            name: "sorted Vec (bulk)",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: Some(scan_ns),
            tail: None,
            alloc: Some(alloc),
        });
    }

    // --- sorted Vec, incremental (why the arena exists) -------------------
    // Keeping a flat array sorted while inserting is O(n) per insert: the
    // whole tail shifts. One instrumented pass doubles as the throughput
    // measurement (a full best-of-3 would take minutes). Skipped at 1M —
    // the quadratic build would run for hours.
    if n <= 100_000 {
        let mut v: Vec<(u64, u32)> = Vec::new();
        let t = Instant::now();
        let tail = tail_of(&keys, |k| {
            let at = v.partition_point(|&(existing, _)| existing < k);
            v.insert(at, (k, 1));
        });
        let insert_ns = t.elapsed().as_nanos() as u64;
        std::hint::black_box(&v);
        rows.push(Row {
            name: "sorted Vec (incremental)",
            insert_ns: Some(insert_ns),
            get_ns: None,
            scan_ns: None,
            tail: Some(tail),
            alloc: None,
        });
    }

    // === companion structures vs their std-class baselines =================
    // Metrics reuse the same keys: insert_ns = push/intern (per stream
    // element), get_ns = get/resolve (per lookup), scan_ns = full iteration
    // (per element), mem/allocs = build stats. Memory is per *stored* item
    // (blobs / values / distinct terms).

    // --- BlobHeap vs Vec<Vec<u8>>: append-only blob storage ---------------
    {
        // Variable-length blobs, 16..=200 bytes (avg ~108), deterministic.
        let mut rng = xorshift(0x0B10_B0B5_0000_0001);
        let blobs: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let len = 16 + (rng() % 185) as usize;
                (0..len).map(|i| i as u8).collect()
            })
            .collect();
        let ids: Vec<u32> = (0..LOOKUPS).map(|_| (rng() % n as u64) as u32).collect();

        let build = || {
            let mut h = BlobHeap::new(BlobHeapCfg::new());
            for b in &blobs {
                h.push(b).unwrap();
            }
            h
        };
        let insert_ns = best_ns(build);
        let (heap, alloc) = with_alloc_stats(build);
        let get_ns = best_ns(|| {
            let mut acc = 0usize;
            for &id in &ids {
                acc += heap.get(BlobId(id)).len();
            }
            acc
        });
        rows.push(Row {
            name: "plugmem BlobHeap",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: None,
            tail: None,
            alloc: Some(alloc),
        });

        let build_std = || {
            let mut v: Vec<Vec<u8>> = Vec::new();
            for b in &blobs {
                v.push(b.clone()); // one heap allocation per blob
            }
            v
        };
        let insert_ns = best_ns(build_std);
        let (vecs, alloc) = with_alloc_stats(build_std);
        let get_ns = best_ns(|| {
            let mut acc = 0usize;
            for &id in &ids {
                acc += vecs[id as usize].len();
            }
            acc
        });
        rows.push(Row {
            name: "Vec<Vec<u8>> (blob baseline)",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: None,
            tail: None,
            alloc: Some(alloc),
        });
    }

    // --- ChunkPool vs one Vec<u8> per list: many small lists --------------
    {
        const LISTS: usize = 1024;
        let values: Vec<[u8; 8]> = keys.iter().map(|k| k.to_be_bytes()).collect();

        let build = || {
            let mut pool = ChunkPool::new(ChunkPoolCfg::new());
            let mut lists = vec![ListHandle::EMPTY; LISTS];
            for (i, v) in values.iter().enumerate() {
                pool.push(&mut lists[i % LISTS], v).unwrap();
            }
            (pool, lists)
        };
        let insert_ns = best_ns(build);
        let ((pool, lists), alloc) = with_alloc_stats(build);
        // Iteration sums every byte on both sides, so the chain-hop cost
        // of the pool is compared against a contiguous slice doing the
        // same work; the total is normalized to read per *value* after
        // emit's division by SCAN.
        let scan_total = best_ns(|| {
            let mut acc = 0usize;
            for list in &lists {
                for chunk in pool.iter(list) {
                    for &b in chunk {
                        acc += b as usize;
                    }
                }
            }
            acc
        });
        rows.push(Row {
            name: "plugmem ChunkPool",
            insert_ns: Some(insert_ns),
            get_ns: None,
            scan_ns: Some(scan_total * SCAN as u64 / n as u64),
            tail: None,
            alloc: Some(alloc),
        });

        let build_std = || {
            let mut lists: Vec<Vec<u8>> = vec![Vec::new(); LISTS];
            for (i, v) in values.iter().enumerate() {
                lists[i % LISTS].extend_from_slice(v);
            }
            lists
        };
        let insert_ns = best_ns(build_std);
        let (std_lists, alloc) = with_alloc_stats(build_std);
        let scan_total = best_ns(|| {
            let mut acc = 0usize;
            for list in &std_lists {
                for &b in list {
                    acc += b as usize;
                }
            }
            acc
        });
        rows.push(Row {
            name: "Vec<u8> per list (chunk baseline)",
            insert_ns: Some(insert_ns),
            get_ns: None,
            scan_ns: Some(scan_total * SCAN as u64 / n as u64),
            tail: None,
            alloc: Some(alloc),
        });
    }

    // --- Interner vs HashMap<String,u32> + Vec<String> --------------------
    {
        // Vocabulary of n/10 distinct terms; the stream draws uniformly, so
        // ~90% of intern calls are hits after warm-up (the realistic mix).
        let vocab = (n / 10).max(1);
        let mut rng = xorshift(0x1234_5678_9ABC_DEF0);
        let words: Vec<String> = (0..n)
            .map(|_| format!("term-{}", rng() % vocab as u64))
            .collect();

        let build = || {
            let mut it = Interner::new(BlobHeapCfg::new());
            for w in &words {
                it.intern(w).unwrap();
            }
            it
        };
        let insert_ns = best_ns(build);
        let (interner, alloc) = with_alloc_stats(build);
        let distinct = interner.len();
        let resolve_ids: Vec<u32> = (0..LOOKUPS).map(|i| (i % distinct) as u32).collect();
        let get_ns = best_ns(|| {
            let mut acc = 0usize;
            for &id in &resolve_ids {
                acc += interner.resolve(TermId(id)).len();
            }
            acc
        });
        rows.push(Row {
            name: "plugmem Interner",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: None,
            tail: None,
            alloc: Some(alloc.scaled(distinct, n)),
        });

        let build_std = || {
            let mut map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
            let mut names: Vec<String> = Vec::new();
            for w in &words {
                if !map.contains_key(w.as_str()) {
                    map.insert(w.clone(), names.len() as u32);
                    names.push(w.clone());
                }
            }
            (map, names)
        };
        let insert_ns = best_ns(build_std);
        let ((_, names), alloc) = with_alloc_stats(build_std);
        let get_ns = best_ns(|| {
            let mut acc = 0usize;
            for &id in &resolve_ids {
                acc += names[id as usize].len();
            }
            acc
        });
        rows.push(Row {
            name: "HashMap+Vec (intern baseline)",
            insert_ns: Some(insert_ns),
            get_ns: Some(get_ns),
            scan_ns: None,
            tail: None,
            alloc: Some(alloc.scaled(distinct, n)),
        });
    }

    emit(n, &rows);
}
