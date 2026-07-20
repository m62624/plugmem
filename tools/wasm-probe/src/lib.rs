//! Cross-target equivalence probe (specs/14 §3).
//!
//! Runs one deterministic scenario through the public engine API — facts
//! with vectors, entities, tags, links, `forget`, `revise`, a `maintain`
//! with physical purge, a `recall` — and returns the xxh3 hash of the
//! resulting snapshot bytes.
//!
//! The same source builds four ways: native (this crate's `main`),
//! `wasm32-unknown-unknown`, `wasm64-unknown-unknown` (nightly,
//! `-Zbuild-std=core,alloc`) and with `+simd128`. If every build prints
//! the same hash, the snapshot format is proven pointer-width- and
//! target-independent — CI asserts exactly that by executing the wasm
//! artifacts under wasmtime and wasmer and comparing against native.
#![cfg_attr(target_family = "wasm", no_std)]

extern crate alloc;

use alloc::vec::Vec;
use plugmem_core::{Config, LinkInput, MemStorage, Memory, RecallQuery, RememberInput};
use xxhash_rust::xxh3::xxh3_64;

/// Tiny deterministic PRNG (xorshift64*), identical on every target.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A float in (-1, 1) built from the top PRNG bits — bit-exact on
    /// every target (a single division, no libm).
    fn f32_unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

const DIM: usize = 64;
const FACTS: u64 = 200;

/// Runs the scenario and returns `xxh3(snapshot_bytes)`.
pub fn scenario_hash() -> u64 {
    let mut cfg = Config::default();
    cfg.dim = DIM;
    cfg.db_uuid = 0x5EED_0000_0000_0003;
    let mut store = MemStorage::default();
    let (mut mem, _) = Memory::open(&mut store, cfg).expect("open");
    let mut rng = Rng(0xDEAD_BEEF_CAFE_0001);

    let tags_pool = ["alpha", "beta", "gamma", "delta"];
    let ents = ["user", "plugmem", "wasm", "engine"];

    let mut vec_buf = [0.0f32; DIM];
    for i in 0..FACTS {
        for v in vec_buf.iter_mut() {
            *v = rng.f32_unit();
        }
        let now = 1_700_000_000_000 + i * 60_000;
        let text = alloc::format!("fact number {i} about topic {}", i % 17);
        let tag = tags_pool[(i % 4) as usize];
        let entity = ents[(i % 4) as usize];
        let input = RememberInput {
            now,
            text: &text,
            entity: Some(entity),
            tags: &[tag],
            links: &[("relates_to", ents[((i + 1) % 4) as usize])],
            vector: Some(&vec_buf),
            valid_from: None,
        };
        let out = mem.remember(&mut store, input).expect("remember");
        if i % 7 == 0 {
            mem.forget(&mut store, now + 1, out.id).expect("forget");
        } else if i % 11 == 0 {
            for v in vec_buf.iter_mut() {
                *v = rng.f32_unit();
            }
            let rtext = alloc::format!("revised fact {i}");
            let rin = RememberInput {
                now: now + 2,
                text: &rtext,
                entity: Some(entity),
                tags: &[tag],
                links: &[],
                vector: Some(&vec_buf),
                valid_from: None,
            };
            mem.revise(&mut store, out.id, rin).expect("revise");
        }
    }
    mem.link(
        &mut store,
        LinkInput {
            now: 1_700_999_000_000,
            src: "user",
            rel: "works_on",
            dst: "plugmem",
            provenance: None,
        },
    )
    .expect("link");
    // Physical purge + the vector-graph maintenance policy path.
    mem.maintain(&mut store, 1_701_000_000_000)
        .expect("maintain");
    // Exercise recall; a pure query must not disturb the state.
    let mut q = RecallQuery::text(1_701_000_000_001, "fact about topic");
    q.k = 8;
    let _ = mem.recall(q).expect("recall");

    let bytes: Vec<u8> = mem.snapshot_bytes(1_701_000_000_002);
    xxh3_64(&bytes)
}

/// Wasm entry point, invoked by the runtime CLI (`--invoke run`). Both
/// wasmtime and wasmer print the returned value as a decimal `i64`, so
/// the native runner prints the same representation for comparison.
#[cfg(target_family = "wasm")]
#[unsafe(no_mangle)]
pub extern "C" fn run() -> u64 {
    scenario_hash()
}

// ---- no_std glue for pure-wasm targets (no OS, no std) --------------------
#[cfg(target_family = "wasm")]
mod glue {
    use core::alloc::{GlobalAlloc, Layout};
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// Fixed heap in linear memory; the scenario needs a few MiB, 64 MiB
    /// leaves a wide margin. A bump allocator never frees — irrelevant
    /// for a run-once probe.
    const HEAP_BYTES: usize = 64 * 1024 * 1024;
    static mut HEAP: [u8; HEAP_BYTES] = [0; HEAP_BYTES];
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Bump;

    // SAFETY: the returned pointers lie inside HEAP, are aligned by the
    // rounding below, and are never handed out twice — NEXT only grows,
    // and the CAS loop makes each grab exclusive (single-threaded wasm
    // anyway). dealloc is a legal no-op for a bump allocator.
    unsafe impl GlobalAlloc for Bump {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align().max(16);
            let mut old = NEXT.load(Ordering::Relaxed);
            loop {
                let base = core::ptr::addr_of!(HEAP) as usize;
                let start = (base + old + align - 1) & !(align - 1);
                let end = start - base + layout.size();
                if end > HEAP_BYTES {
                    return core::ptr::null_mut();
                }
                match NEXT.compare_exchange(old, end, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => return start as *mut u8,
                    Err(v) => old = v,
                }
            }
        }

        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
    }

    #[global_allocator]
    static A: Bump = Bump;

    /// The engine is panic-free on valid inputs; a panic here is a probe
    /// bug. Divergence is enough — the runtime call simply never returns
    /// and CI times out with a visible failure.
    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }
}
