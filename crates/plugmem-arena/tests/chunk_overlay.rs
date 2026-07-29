//! Overlay write-path tests for `ChunkPool` (per-chunk
//! copy-on-write).
//!
//! A pool opened with [`ChunkPool::load_overlay`] borrows its chunks from a
//! longer-lived buffer (a memory-mapped snapshot in production) yet stays fully
//! mutable: because a chunk carries its chain/free-list link in its own bytes,
//! the first write to a borrowed chunk (a value push or a link update) copies
//! just that chunk into an owned overlay, freshly grown chunks live in an owned
//! tail, and the borrowed base is never mutated or cloned. These tests pin that
//! contract against an owned pool driven through the same operations.

use plugmem_arena::{ChunkPool, ChunkPoolCfg, ListHandle};

/// Dumps a pool's two sections.
fn dump(p: &ChunkPool<'_>) -> (Vec<u8>, Vec<u8>) {
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    p.dump_meta(&mut meta);
    p.dump_pool(&mut pool);
    (meta, pool)
}

/// Collects one list's bytes across its chunk chain.
fn collect(p: &ChunkPool<'_>, h: &ListHandle) -> Vec<u8> {
    p.iter(h).flatten().copied().collect()
}

/// Builds a base pool of two interleaved lists spanning several chunks and
/// returns it with the two handles — the image an overlay opens over.
fn base_image() -> (ChunkPool<'static>, ListHandle, ListHandle) {
    let mut p = ChunkPool::new(ChunkPoolCfg::new());
    let (mut evens, mut odds) = (ListHandle::EMPTY, ListHandle::EMPTY);
    for n in 0..200u32 {
        let list = if n % 2 == 0 { &mut evens } else { &mut odds };
        p.push(list, &n.to_be_bytes()).unwrap();
    }
    (p, evens, odds)
}

#[test]
fn overlay_open_matches_owned_and_leaves_base_untouched() {
    let (base, evens, odds) = base_image();
    let (meta, pool) = dump(&base);
    let base_snapshot = pool.clone();

    let mut owned = ChunkPool::load(ChunkPoolCfg::new(), &meta, &pool).unwrap();
    let mut overlay = ChunkPool::load_overlay(ChunkPoolCfg::new(), &meta, &pool).unwrap();

    // Chunk indices survive load, so the base handles address both pools.
    let apply = |p: &mut ChunkPool<'_>| {
        let (mut evens, mut odds) = (evens, odds);
        // Extend existing lists: fills the borrowed tail chunks (CoW) and then
        // allocates fresh chunks.
        for n in 200..600u32 {
            let list = if n % 2 == 0 { &mut evens } else { &mut odds };
            p.push(list, &n.to_be_bytes()).unwrap();
        }
        // Free one whole chain — recycles its chunks (link writes CoW base
        // chunks) — then build a new list that reuses the freed chunks.
        p.free(&mut odds);
        let mut fresh = ListHandle::EMPTY;
        for n in 0..150u32 {
            p.push(&mut fresh, &(n + 10_000).to_be_bytes()).unwrap();
        }
        (evens, fresh)
    };
    let (owned_evens, owned_fresh) = apply(&mut owned);
    let (overlay_evens, overlay_fresh) = apply(&mut overlay);

    // Surviving lists read identically.
    assert_eq!(
        collect(&overlay, &overlay_evens),
        collect(&owned, &owned_evens)
    );
    assert_eq!(
        collect(&overlay, &overlay_fresh),
        collect(&owned, &owned_fresh)
    );
    assert_eq!(overlay.chunks(), owned.chunks());

    // Canonical: the overlay dumps byte-identically to the owned pool.
    assert_eq!(dump(&overlay), dump(&owned));

    // The borrowed base buffer was never mutated by all that writing.
    assert_eq!(
        pool, base_snapshot,
        "overlay must not mutate the borrowed base"
    );
}

#[test]
fn overlay_writes_span_base_and_grown_chunks() {
    let (base, evens, _odds) = base_image();
    let (meta, pool) = dump(&base);
    let base_snapshot = pool.clone();
    let base_chunks = base.chunks();

    let mut p = ChunkPool::load_overlay(ChunkPoolCfg::new(), &meta, &pool).unwrap();

    // Read a list straight from borrowed base chunks (no write yet).
    assert_eq!(collect(&p, &evens).len(), 100 * 4);

    // Push into the list: appends to the borrowed tail chunk (CoW that chunk)
    // and, once it fills, allocates grown chunks past the base.
    let mut evens = evens;
    for n in (200..1000u32).step_by(2) {
        p.push(&mut evens, &n.to_be_bytes()).unwrap();
    }
    assert!(p.chunks() > base_chunks, "the overlay grew fresh chunks");

    // The list now spans base (CoW), and grown chunks — read it back whole.
    let got: Vec<u32> = collect(&p, &evens)
        .chunks_exact(4)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
        .collect();
    let want: Vec<u32> = (0..1000u32).filter(|n| n % 2 == 0).collect();
    assert_eq!(got, want);

    // Base bytes intact through reads, CoW pushes, and chunk growth.
    assert_eq!(pool, base_snapshot);
}
