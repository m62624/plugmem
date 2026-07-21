//! Overlay write-path tests for `Arena` (specs/16 §9, per-page copy-on-write).
//!
//! An arena opened with [`Arena::load_overlay`] borrows its base pages from a
//! longer-lived buffer (a memory-mapped snapshot in production) yet stays
//! fully mutable: the first write to a base page copies just that page into a
//! flat owned overlay pool, freshly grown pages live in an owned tail, and the
//! borrowed base is never mutated or cloned as a whole. These tests pin that
//! contract: every backing branch is exercised, an overlay arena is observably
//! identical to an owned arena driven through the same operations, dumps are
//! byte-identical, and the borrowed base bytes are untouched.

use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, key};

/// 16-byte record with an 8-byte key (mirrors `tests/borrowed.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rec {
    k: u64,
    v: u64,
}

impl Slot for Rec {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 8;
    fn write(&self, out: &mut [u8]) {
        key::write_u64(out, self.k);
        key::write_u64(&mut out[8..], self.v);
    }
    fn read(bytes: &[u8]) -> Self {
        Self {
            k: key::read_u64(bytes),
            v: key::read_u64(&bytes[8..]),
        }
    }
}

fn kbuf(k: u64) -> [u8; 8] {
    let mut b = [0u8; 8];
    key::write_u64(&mut b, k);
    b
}

/// Dumps an arena's two sections.
fn dump(a: &Arena<'_, Rec>) -> (Vec<u8>, Vec<u8>) {
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    a.dump_meta(&mut meta);
    a.dump_pool(&mut pool);
    (meta, pool)
}

/// Builds a base arena of `n` records (keys `0, step, 2*step, ...`) and returns
/// its config plus dumped sections — the on-disk image an overlay opens over.
fn base_image(shards: usize, n: u64, step: u64) -> (ArenaCfg, Vec<u8>, Vec<u8>) {
    let cfg = ArenaCfg::new(shards, ShardMode::Ordered);
    let mut a = Arena::<Rec>::new(cfg).unwrap();
    for i in 0..n {
        a.insert(&Rec { k: i * step, v: i }).unwrap();
    }
    let (meta, pool) = dump(&a);
    (cfg, meta, pool)
}

#[test]
fn overlay_open_matches_owned_and_leaves_base_untouched() {
    // Base of 2000 records across few shards → multi-page chains, so mutations
    // hit existing base pages, split full ones, and grow new ones.
    let (cfg, meta, pool) = base_image(4, 2000, 3);
    let base_snapshot = pool.clone();

    // Two engines from the same image: one owns a copy, one overlays the
    // borrowed base. Every op is applied to both.
    let mut owned = Arena::<Rec>::load(cfg, &meta, &pool).unwrap();
    let mut overlay = Arena::<Rec>::load_overlay(cfg, &meta, &pool).unwrap();

    let apply = |a: &mut Arena<'_, Rec>| {
        // Inserts landing between existing keys → in-page shifts (CoW a base
        // page) and, on full pages, splits (grow + cross-page move).
        for i in 0..2000u64 {
            a.insert(&Rec {
                k: i * 3 + 1,
                v: i + 10_000,
            })
            .unwrap();
        }
        // Payload updates on existing records → CoW the covering base page.
        for i in 0..500u64 {
            if let Some(p) = a.payload_mut(&kbuf(i * 3)) {
                p.copy_from_slice(&(i + 777).to_be_bytes());
            }
        }
        // Removals → in-page shifts and, when a page empties, free-list reuse.
        for i in 0..400u64 {
            assert!(a.remove(&kbuf(i * 3 + 1)));
        }
        // A second insert wave reuses freed pages and grows fresh ones.
        for i in 0..300u64 {
            a.insert(&Rec {
                k: 1_000_000 + i,
                v: i,
            })
            .unwrap();
        }
    };
    apply(&mut owned);
    apply(&mut overlay);

    // Observationally identical: same records in the same order.
    assert_eq!(
        overlay.iter().collect::<Vec<_>>(),
        owned.iter().collect::<Vec<_>>()
    );
    assert_eq!(overlay.len(), owned.len());

    // Point reads agree across the whole key range, present and absent alike.
    for k in [0u64, 1, 2, 3, 4, 1198, 1_000_000, 1_000_299, 9_999_999] {
        assert_eq!(overlay.get(&kbuf(k)), owned.get(&kbuf(k)), "get {k}");
    }

    // Canonical: the overlay dumps byte-identically to the owned engine.
    assert_eq!(dump(&overlay), dump(&owned));

    // The borrowed base buffer was never mutated by all that writing.
    assert_eq!(
        pool, base_snapshot,
        "overlay must not mutate the borrowed base"
    );
}

#[test]
fn overlay_reads_span_base_over_and_grown_pages() {
    // Small base: a handful of pages, all borrowed.
    let (cfg, meta, pool) = base_image(2, 600, 2);
    let base_snapshot = pool.clone();
    let mut a = Arena::<Rec>::load_overlay(cfg, &meta, &pool).unwrap();

    // Untouched base page: a record present at open, on a page we never write.
    assert_eq!(a.get(&kbuf(0)), Some(Rec { k: 0, v: 0 }));

    // Force a copy-on-write of a base page via an in-place payload edit, then
    // read the record back from the OVERLAY copy.
    assert!(a.payload_mut(&kbuf(200)).is_some());
    a.payload_mut(&kbuf(200))
        .unwrap()
        .copy_from_slice(&42u64.to_be_bytes());
    assert_eq!(a.get(&kbuf(200)), Some(Rec { k: 200, v: 42 }));

    // Grow new pages past the base by appending high keys, then read one back
    // from the GROWN tail.
    for i in 0..500u64 {
        a.insert(&Rec {
            k: 10_000_000 + i,
            v: i,
        })
        .unwrap();
    }
    assert_eq!(
        a.get(&kbuf(10_000_499)),
        Some(Rec {
            k: 10_000_499,
            v: 499
        })
    );

    // A neighbouring untouched base record still reads from the base.
    assert_eq!(a.get(&kbuf(2)), Some(Rec { k: 2, v: 1 }));

    // Base bytes intact through overlay reads, one CoW, and page growth.
    assert_eq!(pool, base_snapshot);
}

#[test]
fn overlay_single_slot_pages_split_across_the_boundary() {
    // A slot larger than half a page makes every page hold exactly one slot,
    // so inserts drive the degenerate `spp == 1` split path — a cross-page
    // move under the overlay backing.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Big {
        k: u64,
        _pad: [u8; 2560],
    }
    impl Slot for Big {
        const SIZE: usize = 2568; // > PAGE_BYTES / 2 → one slot per page
        const KEY_LEN: usize = 8;
        fn write(&self, out: &mut [u8]) {
            key::write_u64(out, self.k);
            out[8..].copy_from_slice(&self._pad);
        }
        fn read(bytes: &[u8]) -> Self {
            let mut _pad = [0u8; 2560];
            _pad.copy_from_slice(&bytes[8..]);
            Self {
                k: key::read_u64(bytes),
                _pad,
            }
        }
    }
    let big = |k: u64| Big {
        k,
        _pad: [k as u8; 2560],
    };

    let cfg = ArenaCfg::new(1, ShardMode::Ordered);
    let mut src = Arena::<Big>::new(cfg).unwrap();
    for k in [10u64, 20, 30] {
        src.insert(&big(k)).unwrap();
    }
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    src.dump_meta(&mut meta);
    src.dump_pool(&mut pool);
    let base_snapshot = pool.clone();

    let mut owned = Arena::<Big>::load(cfg, &meta, &pool).unwrap();
    let mut overlay = Arena::<Big>::load_overlay(cfg, &meta, &pool).unwrap();

    // Insert a key before the first (pos == 0 split branch) and one between
    // existing keys — both split single-slot pages.
    for a in [&mut owned, &mut overlay] {
        a.insert(&big(5)).unwrap();
        a.insert(&big(25)).unwrap();
    }
    assert_eq!(
        overlay.iter().map(|r| r.k).collect::<Vec<_>>(),
        owned.iter().map(|r| r.k).collect::<Vec<_>>()
    );
    assert_eq!(overlay.get(&kbuf(5)), owned.get(&kbuf(5)));
    let (mut m, mut p) = (Vec::new(), Vec::new());
    overlay.dump_meta(&mut m);
    overlay.dump_pool(&mut p);
    let (mut mo, mut po) = (Vec::new(), Vec::new());
    owned.dump_meta(&mut mo);
    owned.dump_pool(&mut po);
    assert_eq!((m, p), (mo, po));
    assert_eq!(pool, base_snapshot);
}
