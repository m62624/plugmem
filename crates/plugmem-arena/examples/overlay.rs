//! Opening a serialized pool **over borrowed bytes** and mutating it without
//! cloning the base — run it with
//! `cargo run -p plugmem-arena --example overlay`.
//!
//! This is written to be lifted into your own project; it uses nothing but the
//! crate's public API. The scenario is the one that motivates the overlay: you
//! have a large serialized structure on disk, you memory-map it, and you want
//! to *keep writing* to it without first copying the whole thing into RAM.
//!
//! The deal the overlay makes:
//!
//! - the pages present at open are **borrowed** from the mapped image and read
//!   straight out of it — only the pages you actually touch become resident;
//! - the first write to a borrowed page copies just *that* page into an owned
//!   overlay (per-page copy-on-write); pages you grow live in an owned tail;
//! - the borrowed image is never mutated and never cloned as a whole.
//!
//! Staying true to the crate's flat philosophy, the overlay is plain flat
//! `Vec`s — no `Box`, no map, no tree. A structure opened this way is
//! byte-for-byte identical, on dump, to one that owned its bytes all along.

use plugmem_arena::{Arena, ArenaCfg, ChunkPool, ChunkPoolCfg, ListHandle, ShardMode, Slot, key};

/// A tiny fixed-size record: an 8-byte big-endian key plus an 8-byte payload.
#[derive(Debug, PartialEq)]
struct Row {
    id: u64,
    value: u64,
}

impl Slot for Row {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 8;
    fn write(&self, out: &mut [u8]) {
        key::write_u64(out, self.id);
        key::write_u64(&mut out[8..], self.value);
    }
    fn read(bytes: &[u8]) -> Self {
        Row {
            id: key::read_u64(bytes),
            value: key::read_u64(&bytes[8..]),
        }
    }
}

fn key8(id: u64) -> [u8; 8] {
    let mut k = [0u8; 8];
    key::write_u64(&mut k, id);
    k
}

fn main() {
    arena_overlay();
    chunk_pool_overlay();
    println!("\nboth overlays stayed byte-identical to owned — and never touched the base.");
}

/// An [`Arena`] opened over a borrowed image, then mutated in place.
fn arena_overlay() {
    // 1. Build a structure and serialize it — this stands in for the bytes you
    //    would read (or memory-map) from a file.
    let mut source = Arena::<Row>::new(ArenaCfg::new(4, ShardMode::Ordered)).unwrap();
    for id in 0..1000 {
        source.insert(&Row { id, value: id * 10 }).unwrap();
    }
    let (mut meta, mut image) = (Vec::new(), Vec::new());
    source.dump_meta(&mut meta);
    source.dump_pool(&mut image);

    // Keep a copy to prove the borrowed image is never written to.
    let image_at_open = image.clone();

    // 2. Open OVER the borrowed image. No page is copied yet: reads come
    //    straight from `image`.
    let mut overlay = Arena::<Row>::load_overlay(*source.cfg(), &meta, &image).unwrap();
    assert_eq!(
        overlay.get(&key8(500)),
        Some(Row {
            id: 500,
            value: 5000
        })
    );

    // 3. Mutate it. Inserts between existing keys copy just the touched pages
    //    up; appends past the end grow an owned tail. The base is untouched.
    for id in 0..1000 {
        overlay
            .insert(&Row {
                id: 1_000_000 + id,
                value: id,
            })
            .unwrap();
    }
    if let Some(payload) = overlay.payload_mut(&key8(500)) {
        payload.copy_from_slice(&42u64.to_be_bytes());
    }

    assert_eq!(overlay.len(), 2000);
    assert_eq!(overlay.get(&key8(500)), Some(Row { id: 500, value: 42 }));
    assert_eq!(
        image, image_at_open,
        "the borrowed image must not be mutated"
    );

    // 4. Dumping the overlay yields exactly the bytes an all-owned arena driven
    //    through the same operations would produce.
    let mut owned = Arena::<Row>::load(*source.cfg(), &meta, &image).unwrap();
    for id in 0..1000 {
        owned
            .insert(&Row {
                id: 1_000_000 + id,
                value: id,
            })
            .unwrap();
    }
    owned
        .payload_mut(&key8(500))
        .unwrap()
        .copy_from_slice(&42u64.to_be_bytes());

    let dump = |a: &Arena<'_, Row>| {
        let (mut m, mut p) = (Vec::new(), Vec::new());
        a.dump_meta(&mut m);
        a.dump_pool(&mut p);
        (m, p)
    };
    assert_eq!(dump(&overlay), dump(&owned));
    println!("arena:      1000 borrowed rows + 1000 grown + an in-place edit — base intact.");
}

/// A [`ChunkPool`] opened over a borrowed image, then appended to. A chunk
/// keeps its chain link in its own bytes, so writing copies just that chunk up.
fn chunk_pool_overlay() {
    let mut source = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    for n in 0..100u32 {
        source.push(&mut list, &n.to_be_bytes()).unwrap();
    }
    let (mut meta, mut image) = (Vec::new(), Vec::new());
    source.dump_meta(&mut meta);
    source.dump_pool(&mut image);
    let image_at_open = image.clone();

    // The list handle addresses the reloaded pool (chunk indices are stable).
    let mut overlay = ChunkPool::load_overlay(ChunkPoolCfg::new(), &meta, &image).unwrap();
    for n in 100..400u32 {
        overlay.push(&mut list, &n.to_be_bytes()).unwrap();
    }

    let values: Vec<u32> = overlay
        .iter(&list)
        .flatten()
        .copied()
        .collect::<Vec<u8>>()
        .chunks_exact(4)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(values, (0..400).collect::<Vec<_>>());
    assert_eq!(
        image, image_at_open,
        "the borrowed image must not be mutated"
    );
    println!("chunk pool: 100 borrowed values + 300 appended across chunks — base intact.");
}
