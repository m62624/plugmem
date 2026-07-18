//! The zero-allocation invariant (specs/07 §2): after warm-up, `recall`
//! and `get` perform **zero** allocator calls — scratches and result
//! buffers are reused, growth happens only on new size maximums.
//!
//! Own test binary: it swaps the global allocator for a counting wrapper.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RecallResult, RememberInput};

static ALLOCS: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocs<R>(f: impl FnOnce() -> R) -> (u64, R) {
    let before = ALLOCS.load(Ordering::Relaxed);
    let result = f();
    (ALLOCS.load(Ordering::Relaxed) - before, result)
}

#[test]
fn recall_and_get_allocate_nothing_after_warmup() {
    let mut cfg = Config::default();
    cfg.shards_facts = 64;
    cfg.shards_entities = 16;
    cfg.shards_edges = 16;
    cfg.shards_temporal = 16;
    cfg.shards_postings = 128;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();

    // A corpus wide enough to exercise every source: entities, links,
    // tags, and varied text.
    let entities = ["user", "plugmem", "tokio", "work", "home"];
    for i in 0..500u64 {
        let entity = entities[(i % 5) as usize];
        let text = match i % 4 {
            0 => "prefers tokio for async work and testing",
            1 => "building a memory engine with flat arenas",
            2 => "went to the office and discussed the roadmap",
            _ => "the weather was nice for a walk",
        };
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some(entity),
                tags: if i % 3 == 0 { &["pref"] } else { &[] },
                links: if i % 7 == 0 {
                    &[("works_on", "plugmem")]
                } else {
                    &[]
                },
                ..RememberInput::text(i * 1000, text)
            },
        )
        .unwrap();
    }

    let q = RecallQuery {
        entities: &["user", "plugmem"],
        tags: &["pref"],
        range: Some((0, 600_000)),
        k: 16,
        ..RecallQuery::text(600_000, "tokio memory arenas roadmap")
    };
    let mut out = RecallResult::default();
    // Warm-up: two passes let every scratch and result buffer reach its
    // high-water mark.
    mem.recall_into(q, &mut out).unwrap();
    mem.recall_into(q, &mut out).unwrap();

    let (n, _) = allocs(|| mem.recall_into(q, &mut out).unwrap());
    assert_eq!(n, 0, "recall allocated {n} times after warm-up");
    assert!(!out.facts.is_empty(), "the gate must measure a real query");

    let (n, view) = allocs(|| mem.get(plugmem_core::FactId(42)));
    assert_eq!(n, 0, "get allocated {n} times");
    assert!(view.is_some());
}
