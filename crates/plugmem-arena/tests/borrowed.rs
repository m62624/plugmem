//! Zero-copy borrowed-load tests (specs/16): every structure can rebuild
//! itself borrowing its pool from a longer-lived buffer (a memory-mapped
//! snapshot in production) instead of copying it. Each test asserts the
//! borrowed structure is observably identical to the owned one and dumps
//! back to the very same bytes, plus that a mutation copies-up (`Cow`) and
//! leaves the source buffer untouched.

use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, ChunkPool, ChunkPoolCfg, Interner, ListHandle,
    ShardMode, Slot, key,
};

/// 16-byte record with an 8-byte key.
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

#[test]
fn arena_borrowed_load_equals_owned() {
    let mut a = Arena::<Rec>::new(ArenaCfg::new(4, ShardMode::Ordered)).unwrap();
    for i in 0..2000u64 {
        a.insert(&Rec { k: i * 3, v: i }).unwrap();
    }
    for i in 0..600u64 {
        let mut k = [0u8; 8];
        key::write_u64(&mut k, i * 3);
        assert!(a.remove(&k));
    }
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    a.dump_meta(&mut meta);
    a.dump_pool(&mut pool);

    let owned = Arena::<Rec>::load(*a.cfg(), &meta, &pool).unwrap();
    let borrowed = Arena::<Rec>::load_borrowed(*a.cfg(), &meta, &pool).unwrap();

    // Same records, same order.
    let want: Vec<_> = a.iter().collect();
    assert_eq!(owned.iter().collect::<Vec<_>>(), want);
    assert_eq!(borrowed.iter().collect::<Vec<_>>(), want);

    // Canonical: both dump back to the exact input bytes.
    let redump = |x: &Arena<'_, Rec>| {
        let (mut m, mut p) = (Vec::new(), Vec::new());
        x.dump_meta(&mut m);
        x.dump_pool(&mut p);
        (m, p)
    };
    assert_eq!(redump(&owned), (meta.clone(), pool.clone()));
    assert_eq!(redump(&borrowed), (meta, pool));
}

#[test]
fn arena_borrowed_mutation_copies_up_and_spares_the_source() {
    let mut a = Arena::<Rec>::new(ArenaCfg::new(2, ShardMode::Ordered)).unwrap();
    for i in 0..50u64 {
        a.insert(&Rec { k: i, v: i }).unwrap();
    }
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    a.dump_meta(&mut meta);
    a.dump_pool(&mut pool);
    let snapshot = pool.clone();

    let mut borrowed =
        Arena::<Rec>::load_borrowed(ArenaCfg::new(2, ShardMode::Ordered), &meta, &pool).unwrap();
    // A mutation triggers Cow::to_mut — the source buffer must stay intact.
    borrowed.insert(&Rec { k: 999, v: 999 }).unwrap();
    assert_eq!(pool, snapshot, "the borrowed source pool was not mutated");
    assert_eq!(
        borrowed.get(&{
            let mut k = [0u8; 8];
            key::write_u64(&mut k, 999);
            k
        }),
        Some(Rec { k: 999, v: 999 })
    );
}

#[test]
fn blob_heap_borrowed_load_equals_owned() {
    let mut h = BlobHeap::new(BlobHeapCfg::new());
    let ids: Vec<_> = ["alpha", "", "a longer blob of bytes", "z"]
        .iter()
        .map(|s| h.push(s.as_bytes()).unwrap())
        .collect();
    let (mut index, mut pool) = (Vec::new(), Vec::new());
    h.dump_index(&mut index);
    h.dump_pool(&mut pool);

    let owned = BlobHeap::load(BlobHeapCfg::new(), &index, &pool).unwrap();
    let borrowed = BlobHeap::load_borrowed(BlobHeapCfg::new(), &index, &pool).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        let want = h.get(id);
        assert_eq!(owned.get(id), want, "owned blob {i}");
        assert_eq!(borrowed.get(id), want, "borrowed blob {i}");
    }
    assert_eq!(borrowed.len(), h.len());
}

#[test]
fn chunk_pool_borrowed_load_equals_owned() {
    let mut c = ChunkPool::new(ChunkPoolCfg::new());
    // Two lists interleaved so chains thread through the pool.
    let mut evens = ListHandle::EMPTY;
    let mut odds = ListHandle::EMPTY;
    for n in 0..80u32 {
        let list = if n % 2 == 0 { &mut evens } else { &mut odds };
        c.push(list, &n.to_be_bytes()).unwrap();
    }
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    c.dump_meta(&mut meta);
    c.dump_pool(&mut pool);

    let owned = ChunkPool::load(ChunkPoolCfg::new(), &meta, &pool).unwrap();
    let borrowed = ChunkPool::load_borrowed(ChunkPoolCfg::new(), &meta, &pool).unwrap();

    // Chunk indices are preserved by load, so a handle built before the dump
    // iterates the reloaded pools identically.
    let collect =
        |p: &ChunkPool<'_>, h: &ListHandle| -> Vec<u8> { p.iter(h).flatten().copied().collect() };
    assert_eq!(collect(&owned, &evens), collect(&c, &evens));
    assert_eq!(collect(&borrowed, &evens), collect(&c, &evens));
    assert_eq!(collect(&borrowed, &odds), collect(&c, &odds));
    assert_eq!(borrowed.chunks(), c.chunks());
}

#[test]
fn interner_borrowed_load_equals_owned() {
    let mut terms = Interner::new(BlobHeapCfg::new());
    let words = ["tokio", "async", "runtime", "tokio", "wasm"];
    let ids: Vec<_> = words.iter().map(|w| terms.intern(w).unwrap()).collect();
    let (mut index, mut pool, mut table) = (Vec::new(), Vec::new(), Vec::new());
    terms.dump_index(&mut index);
    terms.dump_pool(&mut pool);
    terms.dump_table(&mut table);

    let owned = Interner::load(BlobHeapCfg::new(), &index, &pool, &table).unwrap();
    let borrowed = Interner::load_borrowed(BlobHeapCfg::new(), &index, &pool, &table).unwrap();
    for (w, &id) in words.iter().zip(&ids) {
        assert_eq!(owned.lookup(w), Some(id));
        assert_eq!(borrowed.lookup(w), Some(id));
        assert_eq!(borrowed.resolve(id), *w);
    }
    assert_eq!(borrowed.len(), terms.len());
}
