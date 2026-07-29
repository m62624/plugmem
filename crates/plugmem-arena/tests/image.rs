//! Serialized-image tests (test plan, structure level): canonical
//! roundtrips (dump → load → dump reproduces identical bytes), state
//! equivalence after load, and a corruption matrix per validation rule —
//! plus a whole-section bitflip sweep that must never panic.

use plugmem_arena::{
    Arena, ArenaCfg, BlobHeap, BlobHeapCfg, ChunkPool, ChunkPoolCfg, Error, Interner, ListHandle,
    PAGE_BYTES, ShardMode, Slot, key,
};

/// 16-byte record with an 8-byte key, as the engine's edge slots use.
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

fn dump_arena(a: &Arena<'_, Rec>) -> (Vec<u8>, Vec<u8>) {
    let (mut meta, mut pool) = (Vec::new(), Vec::new());
    a.dump_meta(&mut meta);
    a.dump_pool(&mut pool);
    (meta, pool)
}

/// A populated arena with removals, so free pages and short chains exist.
fn sample_arena(mode: ShardMode) -> Arena<'static, Rec> {
    let mut a = Arena::<Rec>::new(ArenaCfg::new(4, mode)).unwrap();
    for i in 0..3000u64 {
        a.insert(&Rec { k: i * 7, v: i }).unwrap();
    }
    // Remove enough of one key region to empty pages into the free-list.
    for i in 0..1200u64 {
        let mut k = [0u8; 8];
        key::write_u64(&mut k, i * 7);
        assert!(a.remove(&k));
    }
    a
}

/// Small-scale roundtrip that runs under miri: `dump_pool` walks right up
/// to the uninitialized page tails — the interpreter proves it never
/// reads past the initialized prefixes.
#[test]
fn arena_dump_reads_no_uninitialized_bytes() {
    let mut a = Arena::<Rec>::new(ArenaCfg::new(2, ShardMode::Ordered)).unwrap();
    for i in 0..40u64 {
        a.insert(&Rec { k: i, v: i }).unwrap();
    }
    for i in 0..15u64 {
        let mut k = [0u8; 8];
        key::write_u64(&mut k, i);
        assert!(a.remove(&k));
    }
    let (meta, pool) = dump_arena(&a);
    let b = Arena::<Rec>::load(*a.cfg(), &meta, &pool).unwrap();
    assert_eq!(b.iter().collect::<Vec<_>>(), a.iter().collect::<Vec<_>>());
    assert_eq!(dump_arena(&b), (meta, pool));
}

#[test]
#[cfg_attr(miri, ignore)] // 3000-insert sample: minutes under the interpreter,
// and the uninit-adjacent path is covered by the small test above
fn arena_roundtrip_is_canonical_and_equivalent() {
    for mode in [ShardMode::Uniform, ShardMode::Ordered] {
        let a = sample_arena(mode);
        let (meta, pool) = dump_arena(&a);
        let b = Arena::<Rec>::load(*a.cfg(), &meta, &pool).unwrap();
        // Same records...
        assert_eq!(b.len(), a.len());
        let want: Vec<Rec> = a.iter().collect();
        let got: Vec<Rec> = b.iter().collect();
        assert_eq!(got, want);
        // ...and byte-identical re-dump (canonical form).
        assert_eq!(dump_arena(&b), (meta, pool));
        // The loaded arena keeps working: lookups, inserts, removals.
        let mut b = b;
        assert!(b.contains(&{
            let mut k = [0u8; 8];
            key::write_u64(&mut k, 2999 * 7);
            k
        }));
        b.insert(&Rec { k: 1, v: 42 }).unwrap();
        assert_eq!(b.len(), a.len() + 1);
    }
}

#[test]
fn empty_arena_roundtrip() {
    let a = Arena::<Rec>::new(ArenaCfg::new(2, ShardMode::Ordered)).unwrap();
    let (meta, pool) = dump_arena(&a);
    assert!(pool.is_empty());
    let b = Arena::<Rec>::load(*a.cfg(), &meta, &pool).unwrap();
    assert!(b.is_empty());
    assert_eq!(dump_arena(&b), (meta, pool));
}

#[test]
#[cfg_attr(miri, ignore)] // uses the 3000-insert sample arena
fn arena_load_rejects_each_inconsistency() {
    let a = sample_arena(ShardMode::Ordered);
    let cfg = *a.cfg();
    let (meta, pool) = dump_arena(&a);
    let corrupt = |m: &[u8], p: &[u8]| match Arena::<Rec>::load(cfg, m, p) {
        Err(Error::Corrupt(what)) => what,
        other => panic!("expected Corrupt, got {other:?}"),
    };

    // Truncated meta.
    assert_eq!(
        corrupt(&meta[..10], &pool),
        "arena meta shorter than its header"
    );
    assert_eq!(
        corrupt(&meta[..meta.len() - 1], &pool),
        "arena meta length mismatch"
    );
    // Wrong pool size.
    assert_eq!(
        corrupt(&meta, &pool[..pool.len() - 1]),
        "arena pool length mismatch"
    );
    // Config disagreement.
    let mut m = meta.clone();
    m[0..4].copy_from_slice(&8u32.to_le_bytes());
    assert_eq!(
        Arena::<Rec>::load(ArenaCfg::new(4, ShardMode::Ordered), &m, &pool).unwrap_err(),
        Error::Corrupt("arena meta shard count disagrees with cfg")
    );
    let mut m = meta.clone();
    m[20] = 0; // Uniform, but cfg says Ordered
    assert_eq!(
        corrupt(&m, &pool),
        "arena meta shard mode disagrees with cfg"
    );
    let mut m = meta.clone();
    m[21] = 1;
    assert_eq!(corrupt(&m, &pool), "arena meta reserved bytes must be zero");
    // Total mismatch.
    let mut m = meta.clone();
    m[12..20].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(
        corrupt(&m, &pool),
        "arena record total disagrees with page counts"
    );
    // A chain head pointing out of bounds.
    let mut m = meta.clone();
    m[24..28].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
    assert_eq!(corrupt(&m, &pool), "arena chain page out of bounds");
    // A self-cycle: page 0's next pointing at itself (page 0 is in some
    // chain of this sample; the walk revisits it).
    let pages = (pool.len() / PAGE_BYTES) as u32;
    assert!(pages > 2);
    let next_base = 24 + 4 * 4;
    let mut m = meta.clone();
    m[next_base..next_base + 4].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(corrupt(&m, &pool), "arena page linked more than once");
    // Slot-count overflow.
    let counts_base = next_base + pages as usize * 4;
    let mut m = meta.clone();
    m[counts_base..counts_base + 2].copy_from_slice(&(PAGE_BYTES as u16).to_le_bytes());
    assert_eq!(
        corrupt(&m, &pool),
        "arena page count exceeds slots per page"
    );
    // An invalid cfg fails with the same typed errors as `new`.
    assert_eq!(
        Arena::<Rec>::load(ArenaCfg::new(3, ShardMode::Ordered), &meta, &pool).unwrap_err(),
        Error::BadShardCount { got: 3 }
    );
}

#[test]
#[cfg_attr(miri, ignore)] // uses the 3000-insert sample arena
fn arena_load_never_panics_on_flipped_meta() {
    let a = sample_arena(ShardMode::Uniform);
    let (meta, pool) = dump_arena(&a);
    for at in 0..meta.len() {
        for bit in [0x01u8, 0x80] {
            let mut m = meta.clone();
            m[at] ^= bit;
            // Any outcome but a panic is acceptable: most flips are caught
            // as Corrupt; flips in key-irrelevant count bytes may load.
            let _ = Arena::<Rec>::load(*a.cfg(), &m, &pool);
        }
    }
}

#[test]
fn blob_heap_roundtrip_and_corruption() {
    let cfg = BlobHeapCfg::new();
    let mut h = BlobHeap::new(cfg);
    let blobs: Vec<Vec<u8>> = (0..100u32).map(|i| (0..(i % 17) as u8).collect()).collect();
    for b in &blobs {
        h.push(b).unwrap();
    }
    let (mut index, mut pool) = (Vec::new(), Vec::new());
    h.dump_index(&mut index);
    h.dump_pool(&mut pool);
    let loaded = BlobHeap::load(cfg, &index, &pool).unwrap();
    assert_eq!(loaded, h);
    let (mut i2, mut p2) = (Vec::new(), Vec::new());
    loaded.dump_index(&mut i2);
    loaded.dump_pool(&mut p2);
    assert_eq!((i2, p2), (index.clone(), pool.clone()));

    let corrupt = |i: &[u8], p: &[u8]| match BlobHeap::load(cfg, i, p) {
        Err(Error::Corrupt(what)) => what,
        other => panic!("expected Corrupt, got {other:?}"),
    };
    assert_eq!(
        corrupt(&index[..3], &pool),
        "blob index shorter than its header"
    );
    assert_eq!(
        corrupt(&index[..index.len() - 1], &pool),
        "blob index length mismatch"
    );
    // Lengths overrunning and undercovering the pool.
    let mut i = index.clone();
    let last = i.len() - 4;
    i[last..].copy_from_slice(&u32::MAX.to_le_bytes());
    let what = corrupt(&i, &pool);
    assert!(
        what == "blob lengths overrun the pool"
            || what == "blob length exceeds the configured max_blob",
        "got {what}"
    );
    // A truncated pool trips the overrun check mid-walk; an extended pool
    // leaves bytes uncovered.
    assert_eq!(
        corrupt(&index, &pool[..pool.len() - 1]),
        "blob lengths overrun the pool"
    );
    let mut p = pool.clone();
    p.push(0);
    assert_eq!(corrupt(&index, &p), "blob lengths do not cover the pool");
    // The id-space sentinel needs a crafted 4-byte index (count check
    // precedes the length check).
    assert_eq!(
        corrupt(&u32::MAX.to_le_bytes(), &pool),
        "blob count overflows the id space"
    );
    // Oversized blob length vs cfg.
    let tight = BlobHeapCfg::new().with_max_blob(4);
    assert_eq!(
        BlobHeap::load(tight, &index, &pool).unwrap_err(),
        Error::Corrupt("blob length exceeds the configured max_blob")
    );
    // Pool over the byte ceiling.
    let tiny = BlobHeapCfg::new().with_max_bytes(8);
    assert_eq!(
        BlobHeap::load(tiny, &index, &pool).unwrap_err(),
        Error::Corrupt("blob pool exceeds the configured ceiling")
    );
}

#[test]
fn chunk_pool_roundtrip_canonicalizes_freed_chunks() {
    let cfg = ChunkPoolCfg::new();
    let mut pool = ChunkPool::new(cfg);
    let mut keep = ListHandle::EMPTY;
    let mut dead = ListHandle::EMPTY;
    for i in 0..200u32 {
        pool.push(&mut keep, &i.to_be_bytes()).unwrap();
        pool.push(&mut dead, &[0xEE; 8]).unwrap();
    }
    pool.free(&mut dead); // stale bytes + stale used values in free chunks

    let (mut meta, mut bytes) = (Vec::new(), Vec::new());
    pool.dump_meta(&mut meta);
    pool.dump_pool(&mut bytes);
    // Canonicalization: no 0xAB payload byte survives into the image
    // (links are the first 4 bytes of each chunk and can hold anything).
    assert!(
        bytes
            .chunks_exact(64)
            .all(|chunk| !chunk[4..].contains(&0xEE)),
        "freed chunk payload leaked into the dump"
    );

    let loaded = ChunkPool::load(cfg, &meta, &bytes).unwrap();
    // The kept list reads identically through the loaded pool.
    let want: Vec<u8> = pool.iter(&keep).flatten().copied().collect();
    let got: Vec<u8> = loaded.iter(&keep).flatten().copied().collect();
    assert_eq!(got, want);
    // Canonical re-dump.
    let (mut m2, mut b2) = (Vec::new(), Vec::new());
    loaded.dump_meta(&mut m2);
    loaded.dump_pool(&mut b2);
    assert_eq!((m2, b2), (meta.clone(), bytes.clone()));
    // The loaded pool keeps working and reuses the freed chain.
    let mut loaded = loaded;
    let before = loaded.pool_bytes();
    let mut fresh = ListHandle::EMPTY;
    for i in 0..200u32 {
        loaded.push(&mut fresh, &i.to_be_bytes()).unwrap();
    }
    assert_eq!(loaded.pool_bytes(), before);

    let corrupt = |m: &[u8], p: &[u8]| match ChunkPool::load(cfg, m, p) {
        Err(Error::Corrupt(what)) => what,
        other => panic!("expected Corrupt, got {other:?}"),
    };
    assert_eq!(
        corrupt(&meta[..7], &bytes),
        "chunk meta shorter than its header"
    );
    assert_eq!(
        corrupt(&meta[..meta.len() - 1], &bytes),
        "chunk meta length mismatch"
    );
    assert_eq!(
        corrupt(&meta, &bytes[..bytes.len() - 64]),
        "chunk pool length mismatch"
    );
    // used > payload
    let mut m = meta.clone();
    m[8] = 61;
    let what = corrupt(&m, &bytes);
    assert!(
        what == "chunk used bytes exceed the payload size"
            || what == "free chunk has nonzero used bytes",
        "got {what}"
    );
    // Out-of-bounds link inside a chunk.
    let mut b = bytes.clone();
    b[0..4].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
    assert_eq!(corrupt(&meta, &b), "chunk link out of bounds");
    // Free-list cycle: point the free head's chunk at itself.
    let free_head = u32::from_le_bytes(meta[4..8].try_into().unwrap());
    assert_ne!(free_head, u32::MAX, "sample must have free chunks");
    let mut b = bytes.clone();
    let at = free_head as usize * 64;
    b[at..at + 4].copy_from_slice(&free_head.to_le_bytes());
    assert_eq!(corrupt(&meta, &b), "chunk free-list contains a cycle");
    // Free chunk with nonzero used.
    let mut m = meta.clone();
    m[8 + free_head as usize] = 5;
    assert_eq!(corrupt(&m, &bytes), "free chunk has nonzero used bytes");
    // Chunk count sentinel and an out-of-bounds free head need crafted
    // metas (unreachable from any real dump).
    let mut m = vec![0u8; 8];
    m[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(corrupt(&m, &[]), "chunk count overflows the index space");
    let mut m = meta.clone();
    m[4..8].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
    assert_eq!(corrupt(&m, &bytes), "chunk free-list head out of bounds");
    // Pool over the configured ceiling.
    let tiny = ChunkPoolCfg::new().with_max_bytes(64);
    assert_eq!(
        ChunkPool::load(tiny, &meta, &bytes).unwrap_err(),
        Error::Corrupt("chunk pool exceeds the configured ceiling")
    );
}

#[test]
fn interner_roundtrip_and_corruption() {
    let cfg = BlobHeapCfg::new();
    let mut it = Interner::new(cfg);
    let ids: Vec<_> = (0..300)
        .map(|i| it.intern(&format!("term-{i}")).unwrap())
        .collect();
    let (mut index, mut pool, mut table) = (Vec::new(), Vec::new(), Vec::new());
    it.dump_index(&mut index);
    it.dump_pool(&mut pool);
    it.dump_table(&mut table);

    let mut loaded = Interner::load(cfg, &index, &pool, &table).unwrap();
    assert_eq!(loaded.len(), it.len());
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(loaded.resolve(id), format!("term-{i}"));
        // Stored table works: re-interning returns the same id.
        assert_eq!(loaded.intern(&format!("term-{i}")).unwrap(), id);
    }
    // New terms extend the loaded interner.
    let fresh = loaded.intern("fresh").unwrap();
    assert_eq!(fresh.0 as usize, ids.len());
    // Canonical re-dump (before the mutations above).
    let reloaded = Interner::load(cfg, &index, &pool, &table).unwrap();
    let (mut i2, mut p2, mut t2) = (Vec::new(), Vec::new(), Vec::new());
    reloaded.dump_index(&mut i2);
    reloaded.dump_pool(&mut p2);
    reloaded.dump_table(&mut t2);
    assert_eq!((i2, p2, t2), (index.clone(), pool.clone(), table.clone()));

    let corrupt = |i: &[u8], p: &[u8], t: &[u8]| match Interner::load(cfg, i, p, t) {
        Err(Error::Corrupt(what)) => what,
        other => panic!("expected Corrupt, got {other:?}"),
    };
    // Non-UTF-8 term bytes.
    let mut p = pool.clone();
    p[0] = 0xFF;
    assert_eq!(
        corrupt(&index, &p, &table),
        "interned term is not valid UTF-8"
    );
    // Table shape violations.
    assert_eq!(
        corrupt(&index, &pool, &table[..7]),
        "interner table shorter than its header"
    );
    let mut t = table.clone();
    t[0..4].copy_from_slice(&24u32.to_le_bytes());
    assert_eq!(
        corrupt(&index, &pool, &t),
        "interner table size is not a power of two"
    );
    assert_eq!(
        corrupt(&index, &pool, &table[..table.len() - 4]),
        "interner table length mismatch"
    );
    let mut t = table.clone();
    t[4..8].copy_from_slice(&5u32.to_le_bytes());
    assert_eq!(
        corrupt(&index, &pool, &t),
        "interner length disagrees with its heap"
    );
    // Load factor: table claiming 16 slots for 300 terms.
    let mut t = Vec::new();
    t.extend_from_slice(&16u32.to_le_bytes());
    t.extend_from_slice(&300u32.to_le_bytes());
    t.resize(8 + 16 * 4, 0);
    assert_eq!(
        corrupt(&index, &pool, &t),
        "interner table over the load factor"
    );
    // Entry out of bounds / duplicated / miscounted.
    let entry_at = |t: &[u8]| {
        (8..t.len())
            .step_by(4)
            .find(|&at| u32::from_le_bytes(t[at..at + 4].try_into().unwrap()) != 0)
            .unwrap()
    };
    let mut t = table.clone();
    let at = entry_at(&t);
    t[at..at + 4].copy_from_slice(&1000u32.to_le_bytes());
    assert_eq!(
        corrupt(&index, &pool, &t),
        "interner table entry out of bounds"
    );
    let mut t = table.clone();
    let at = entry_at(&t);
    let first = t[at..at + 4].to_vec();
    let mut next = at + 4;
    while u32::from_le_bytes(t[next..next + 4].try_into().unwrap()) == 0 {
        next += 4;
    }
    t[next..next + 4].copy_from_slice(&first);
    assert_eq!(
        corrupt(&index, &pool, &t),
        "interner table stores an id twice"
    );
    let mut t = table.clone();
    let at = entry_at(&t);
    t[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        corrupt(&index, &pool, &t),
        "interner table entry count disagrees with len"
    );
}
