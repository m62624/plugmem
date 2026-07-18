//! Reproducible cross-runtime micro-benchmark — **zero dependencies**.
//!
//! Compares the arena against the standard library's collections of the
//! same *workload class* (ordered map / lookup map / flat sorted array) on
//! insert, point lookup, ordered scan, and peak memory. Runs identically on
//! native and inside WebAssembly runtimes, so the numbers behind the README
//! chart can be re-created by anyone with:
//!
//! ```text
//! # native
//! cargo run --release --example bench_repro
//!
//! # wasm (either runtime)
//! cargo build --release --example bench_repro --target wasm32-wasip1
//! wasmtime run target/wasm32-wasip1/release/examples/bench_repro.wasm
//! wasmer run  target/wasm32-wasip1/release/examples/bench_repro.wasm
//! ```
//!
//! Methodology: fixed workload (100k records, 16-byte payload class, seeded
//! xorshift keys — identical streams everywhere), 3 repetitions per
//! measurement, best time kept (median-of-noise floor), memory measured with
//! a counting global allocator (bytes retained by the structure after
//! build). No criterion here on purpose: this example must not pull a
//! single crate, so the wasm builds stay trivial and the numbers are easy
//! to audit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, key};

/// Global allocator wrapper counting live bytes (for structure footprints).
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates verbatim to `System`; the counter is metadata only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size, Ordering::Relaxed);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

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

const N: usize = 100_000;
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

/// One structure's measured row.
struct Row {
    name: &'static str,
    insert_ns: u64,
    get_ns: u64,
    /// `None` = the structure has no ordered scan (unordered class).
    scan_ns: Option<u64>,
    mem_bytes: usize,
}

fn print_rows(rows: &[Row]) {
    println!("\nstructure\tinsert ns/elem\tget ns/op\tscan ns/elem\tmem B/elem");
    for r in rows {
        println!(
            "{}\t{:.1}\t{:.1}\t{}\t{:.1}",
            r.name,
            r.insert_ns as f64 / N as f64,
            r.get_ns as f64 / LOOKUPS as f64,
            r.scan_ns
                .map(|ns| format!("{:.1}", ns as f64 / SCAN as f64))
                .unwrap_or_else(|| "-".into()),
            r.mem_bytes as f64 / N as f64,
        );
    }
}

fn main() {
    let keys: Vec<u64> = {
        let mut rng = xorshift(0xB0B5_1234_5678_9ABC);
        (0..N).map(|_| rng()).collect()
    };
    let hit_keys: Vec<u64> = keys.iter().step_by(N / LOOKUPS).copied().collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    let (scan_from, scan_to) = (sorted[N / 2], sorted[N / 2 + SCAN]);

    let mut rows = Vec::new();

    // --- plugmem arena, Ordered (the ordered-map class) -------------------
    {
        let mem_before = LIVE.load(Ordering::Relaxed);
        let mut arena = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Ordered)).unwrap();
        let insert_ns = best_ns(|| {
            let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Ordered)).unwrap();
            for &k in &keys {
                a.insert(&Rec16 {
                    hi: k,
                    lo: 0,
                    val: 1,
                })
                .unwrap();
            }
            a.len()
        });
        for &k in &keys {
            arena
                .insert(&Rec16 {
                    hi: k,
                    lo: 0,
                    val: 1,
                })
                .unwrap();
        }
        let mem_bytes = LIVE.load(Ordering::Relaxed) - mem_before;
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(arena.contains(&rec_key(k)));
            }
            acc
        });
        let scan_ns = best_ns(|| {
            let mut acc = 0u64;
            for rec in arena.range(&rec_key(scan_from), &rec_key(scan_to)) {
                acc += u64::from(rec.val);
            }
            acc
        });
        rows.push(Row {
            name: "plugmem Arena (Ordered)",
            insert_ns,
            get_ns,
            scan_ns: Some(scan_ns),
            mem_bytes,
        });
    }

    // --- plugmem arena, Uniform (the lookup-map class) --------------------
    {
        let mem_before = LIVE.load(Ordering::Relaxed);
        let mut arena = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Uniform)).unwrap();
        let insert_ns = best_ns(|| {
            let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Uniform)).unwrap();
            for &k in &keys {
                a.insert(&Rec16 {
                    hi: k,
                    lo: 0,
                    val: 1,
                })
                .unwrap();
            }
            a.len()
        });
        for &k in &keys {
            arena
                .insert(&Rec16 {
                    hi: k,
                    lo: 0,
                    val: 1,
                })
                .unwrap();
        }
        let mem_bytes = LIVE.load(Ordering::Relaxed) - mem_before;
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(arena.contains(&rec_key(k)));
            }
            acc
        });
        rows.push(Row {
            name: "plugmem Arena (Uniform)",
            insert_ns,
            get_ns,
            scan_ns: None,
            mem_bytes,
        });
    }

    // --- BTreeMap (std ordered map) ---------------------------------------
    {
        let mem_before = LIVE.load(Ordering::Relaxed);
        let mut map: BTreeMap<u64, u32> = BTreeMap::new();
        let insert_ns = best_ns(|| {
            let mut m: BTreeMap<u64, u32> = BTreeMap::new();
            for &k in &keys {
                m.insert(k, 1);
            }
            m.len()
        });
        for &k in &keys {
            map.insert(k, 1);
        }
        let mem_bytes = LIVE.load(Ordering::Relaxed) - mem_before;
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
            insert_ns,
            get_ns,
            scan_ns: Some(scan_ns),
            mem_bytes,
        });
    }

    // --- HashMap (std lookup map) -----------------------------------------
    {
        let mem_before = LIVE.load(Ordering::Relaxed);
        let mut map: HashMap<u64, u32> = HashMap::new();
        let insert_ns = best_ns(|| {
            let mut m: HashMap<u64, u32> = HashMap::new();
            for &k in &keys {
                m.insert(k, 1);
            }
            m.len()
        });
        for &k in &keys {
            map.insert(k, 1);
        }
        let mem_bytes = LIVE.load(Ordering::Relaxed) - mem_before;
        let get_ns = best_ns(|| {
            let mut acc = 0u32;
            for &k in &hit_keys {
                acc += u32::from(map.contains_key(&k));
            }
            acc
        });
        rows.push(Row {
            name: "std HashMap",
            insert_ns,
            get_ns,
            scan_ns: None,
            mem_bytes,
        });
    }

    // --- sorted Vec (flat baseline: bulk build + binary search) -----------
    // Incremental sorted insertion into a Vec is O(n) per element (the
    // whole tail shifts) — the arena exists precisely to avoid that — so
    // the flat baseline gets its best case: bulk push + one sort.
    {
        let mem_before = LIVE.load(Ordering::Relaxed);
        let mut vec: Vec<(u64, u32)> = Vec::new();
        let insert_ns = best_ns(|| {
            let mut v: Vec<(u64, u32)> = keys.iter().map(|&k| (k, 1)).collect();
            v.sort_unstable_by_key(|&(k, _)| k);
            v.len()
        });
        vec.extend(keys.iter().map(|&k| (k, 1)));
        vec.sort_unstable_by_key(|&(k, _)| k);
        let mem_bytes = LIVE.load(Ordering::Relaxed) - mem_before;
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
            insert_ns,
            get_ns,
            scan_ns: Some(scan_ns),
            mem_bytes,
        });
    }

    println!(
        "plugmem-arena bench_repro: N={N}, lookups={LOOKUPS}, scan={SCAN}, reps={REPS} (best kept)"
    );
    print_rows(&rows);
}
