//! Opening a snapshot from arbitrary bytes must never panic.
//!
//! The engine's read accessors are deliberately panicking on contract
//! violations (`resolve` on an unknown term, indexing a slot that must exist).
//! What makes that sound is the load path: it range-checks every persisted id
//! before returning an engine, so after a successful open no stored reference
//! can violate a contract. That argument is only as good as the checking, and
//! the checking reads untrusted bytes — a snapshot is a file on disk that
//! anything could have written to.
//!
//! So this target asserts the two halves of the claim together:
//!
//! 1. `from_bytes` on any input returns `Ok` or a typed `Err`, never a panic
//!    and never a wild read;
//! 2. when it returns `Ok`, the accessors that trust the load are then
//!    exercised — including `verify`, which is the deferred half of the
//!    validation, and a recall touching every source.

#![no_main]

use libfuzzer_sys::fuzz_target;
use plugmem_core::{Config, FactId, Memory, RecallQuery};

fuzz_target!(|data: &[u8]| {
    let cfg = Config::default();

    // Owned open: the bytes are copied into the engine's pools.
    if let Ok((mem, _report)) = Memory::from_bytes(Some(data), &[], cfg.clone()) {
        exercise(&mem);
    }

    // Borrowed open: the same validation, but the pools alias `data`, so a
    // missed bound here reads the fuzzer's buffer rather than an owned copy.
    if let Ok(mem) = Memory::from_bytes_borrowed(data, &[], cfg.clone()) {
        exercise(&mem);
    }

    // Overlay open: borrowed base, mutable engine. Writes go to an owned tail,
    // so this covers the copy-on-write path over untrusted page metadata.
    if let Ok((mem, _report)) = Memory::from_bytes_overlay(data, &[], cfg) {
        exercise(&mem);
    }
});

/// Drives the accessors whose panic-freedom the load path is what guarantees.
fn exercise(mem: &Memory<'_>) {
    let stats = mem.stats();

    // `verify` is the validation the load defers: text UTF-8, the fact/vector
    // bijection, metadata decoding. It reads the big pools the load skipped.
    let _ = mem.verify();

    // Every fact id in the counter's range, including burned and tombstoned
    // ones, plus one past the end.
    for id in 0..stats.next_fact.min(512) {
        let id = FactId(id);
        let _ = mem.get(id);
        let mut tags = Vec::new();
        mem.tags_of(id, &mut tags);
        for tag in &tags {
            // Resolving a term id read out of the image: the load promised
            // this cannot be out of range.
            let _ = mem.term(*tag);
        }
        let mut meta = Vec::new();
        let _ = mem.metadata_of(id, &mut meta);
    }
    let _ = mem.get(FactId(stats.next_fact));

    for id in 0..stats.next_entity.min(512) {
        let _ = mem.entity_name(plugmem_core::EntityId(id));
    }

    // A recall that lights up every source at once: lexical, graph anchors,
    // a temporal window, tags and closed facts. Rendering walks entity names
    // and relation terms of whatever the image claimed.
    let _ = mem.recall(RecallQuery {
        text: Some("fuzz query text"),
        tags: &["tag"],
        entities: &["entity"],
        range: Some((0, u64::MAX)),
        include_closed: true,
        k: 64,
        ..RecallQuery::text(u64::MAX / 2, "fuzz query text")
    });

    // The same query at an instant in the past drives the edge-history walk.
    let _ = mem.recall(RecallQuery {
        entities: &["entity"],
        as_of: Some(1),
        k: 64,
        ..RecallQuery::text(u64::MAX / 2, "")
    });

    // Re-emitting the image must not panic on anything the load accepted.
    let _ = mem.snapshot_bytes(0);
}
