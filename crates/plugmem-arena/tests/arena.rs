//! Boundary and contract tests for `Arena` — every invariant from
//! `specs/01-arena.md` that applies to the v1 layer is pinned here.

use plugmem_arena::{Arena, ArenaCfg, Error, PAGE_BYTES, ShardMode, Slot, key};

/// 5-byte test record: 4-byte big-endian key + 1-byte payload.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rec {
    id: u32,
    val: u8,
}

impl Slot for Rec {
    const SIZE: usize = 5;
    const KEY_LEN: usize = 4;
    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id);
        out[4] = self.val;
    }
    fn read(bytes: &[u8]) -> Self {
        Rec {
            id: key::read_u32(bytes),
            val: bytes[4],
        }
    }
}

fn rec_key(id: u32) -> [u8; 4] {
    let mut k = [0u8; 4];
    key::write_u32(&mut k, id);
    k
}

fn ordered_arena(shards: usize) -> Arena<Rec> {
    Arena::new(ArenaCfg::new(shards, ShardMode::Ordered)).unwrap()
}

// --- construction ---------------------------------------------------------

#[test]
fn bad_shard_counts_are_rejected() {
    for shards in [0usize, 3, 6, 1000] {
        let err = Arena::<Rec>::new(ArenaCfg::new(shards, ShardMode::Uniform)).unwrap_err();
        assert_eq!(err, Error::BadShardCount { got: shards });
    }
    // Powers of two, including a single shard, are fine.
    for shards in [1usize, 2, 64, 4096] {
        assert!(Arena::<Rec>::new(ArenaCfg::new(shards, ShardMode::Uniform)).is_ok());
    }
}

/// `[u8; 0]`: SIZE == KEY_LEN == 0 — invalid layout.
#[test]
fn zero_size_slot_is_rejected() {
    let err = Arena::<[u8; 0]>::new(ArenaCfg::default()).unwrap_err();
    assert_eq!(
        err,
        Error::BadSlot {
            size: 0,
            key_len: 0
        }
    );
}

/// A slot larger than one page cannot exist.
struct Huge;
impl Slot for Huge {
    const SIZE: usize = PAGE_BYTES + 1;
    const KEY_LEN: usize = 4;
    fn write(&self, out: &mut [u8]) {
        out[0] = 1;
    }
    fn read(_bytes: &[u8]) -> Self {
        Huge
    }
}

/// KEY_LEN exceeding SIZE is nonsense.
struct KeyTooLong;
impl Slot for KeyTooLong {
    const SIZE: usize = 4;
    const KEY_LEN: usize = 8;
    fn write(&self, out: &mut [u8]) {
        out[0] = 1;
    }
    fn read(_bytes: &[u8]) -> Self {
        KeyTooLong
    }
}

#[test]
fn invalid_slot_layouts_are_rejected() {
    assert_eq!(
        Arena::<Huge>::new(ArenaCfg::default()).unwrap_err(),
        Error::BadSlot {
            size: PAGE_BYTES + 1,
            key_len: 4
        }
    );
    assert_eq!(
        Arena::<KeyTooLong>::new(ArenaCfg::default()).unwrap_err(),
        Error::BadSlot {
            size: 4,
            key_len: 8
        }
    );
    // Exercise the trait impls directly so their bodies are covered even
    // though the arena refuses to host these types.
    let mut buf = [0u8; PAGE_BYTES + 1];
    Huge.write(&mut buf);
    let _ = Huge::read(&buf);
    KeyTooLong.write(&mut buf[..4]);
    let _ = KeyTooLong::read(&buf[..4]);
}

#[test]
fn default_cfg_is_uniform_1024() {
    let cfg = ArenaCfg::default();
    assert_eq!(cfg.shards, 1024);
    assert_eq!(cfg.mode, ShardMode::Uniform);
    assert_eq!(cfg.max_bytes, usize::MAX);
}

// --- empty arena ----------------------------------------------------------

#[test]
fn empty_arena_behaves() {
    let mut a = ordered_arena(64);
    assert!(a.is_empty());
    assert_eq!(a.len(), 0);
    assert_eq!(a.pool_bytes(), 0);
    assert_eq!(a.iter().count(), 0);
    assert_eq!(a.iter().len(), 0);
    assert_eq!(a.get(&rec_key(1)), None);
    assert!(!a.contains(&rec_key(1)));
    assert!(!a.remove(&rec_key(1)));
    assert_eq!(a.get_slot(&rec_key(1)), None);
    assert_eq!(a.payload_mut(&rec_key(1)), None);
}

// --- basic lifecycle ------------------------------------------------------

#[test]
fn insert_get_remove_roundtrip() {
    let mut a = ordered_arena(64);
    assert!(a.insert(&Rec { id: 10, val: 1 }).unwrap());
    assert!(a.insert(&Rec { id: 20, val: 2 }).unwrap());
    assert!(a.insert(&Rec { id: 15, val: 3 }).unwrap());
    assert_eq!(a.len(), 3);
    assert!(!a.is_empty());

    assert_eq!(a.get(&rec_key(10)), Some(Rec { id: 10, val: 1 }));
    assert_eq!(a.get(&rec_key(15)), Some(Rec { id: 15, val: 3 }));
    assert_eq!(a.get(&rec_key(20)), Some(Rec { id: 20, val: 2 }));
    assert_eq!(a.get(&rec_key(11)), None);

    assert!(a.remove(&rec_key(15)));
    assert!(!a.contains(&rec_key(15)));
    assert!(!a.remove(&rec_key(15)), "double remove reports absence");
    assert_eq!(a.len(), 2);
}

#[test]
fn duplicate_insert_is_soft_and_keeps_payload() {
    let mut a = ordered_arena(64);
    assert!(a.insert(&Rec { id: 7, val: 1 }).unwrap());
    assert!(!a.insert(&Rec { id: 7, val: 99 }).unwrap());
    assert_eq!(a.len(), 1);
    assert_eq!(a.get(&rec_key(7)).unwrap().val, 1, "payload untouched");
}

#[test]
fn reinsertion_after_remove() {
    let mut a = ordered_arena(64);
    assert!(a.insert(&Rec { id: 7, val: 7 }).unwrap());
    assert!(a.remove(&rec_key(7)));
    assert!(a.is_empty());
    assert!(a.insert(&Rec { id: 7, val: 8 }).unwrap());
    assert_eq!(a.get(&rec_key(7)).unwrap().val, 8);
    // The page allocated by the first insert is reused, not re-allocated.
    assert_eq!(a.pool_bytes(), PAGE_BYTES);
}

// --- sorted order & shifting ---------------------------------------------

#[test]
fn shard_stays_sorted_under_any_insert_order() {
    // A single shard forces every key into one page: pure shift coverage
    // (front, middle, back inserts).
    let mut a = ordered_arena(1);
    for id in (0..100u32).rev() {
        assert!(a.insert(&Rec { id, val: id as u8 }).unwrap());
    }
    let ids: Vec<u32> = a.iter().map(|r| r.id).collect();
    assert_eq!(ids, (0..100).collect::<Vec<_>>());
}

#[test]
fn removal_boundaries_first_middle_last() {
    let mut a = ordered_arena(1);
    for id in 1..=5u32 {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    assert!(a.remove(&rec_key(1))); // first
    assert!(a.remove(&rec_key(3))); // middle
    assert!(a.remove(&rec_key(5))); // last
    let ids: Vec<u32> = a.iter().map(|r| r.id).collect();
    assert_eq!(ids, [2, 4]);
}

#[test]
fn ordered_mode_iterates_globally_ascending() {
    let mut a = ordered_arena(64);
    // Spread keys across the full u32 range so multiple shards are hit.
    let mut ids: Vec<u32> = (0..500u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
    for &id in &ids {
        assert!(a.insert(&Rec { id, val: 0 }).unwrap());
    }
    let got: Vec<u32> = a.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    assert_eq!(
        got, ids,
        "Ordered mode: byte order == numeric order globally"
    );
}

#[test]
fn uniform_mode_spreads_sequential_keys() {
    // Sequential ids are the worst case for prefix sharding; Fibonacci
    // hashing must spread them over many pages instead of one hot shard.
    let mut a = Arena::<Rec>::new(ArenaCfg::new(64, ShardMode::Uniform)).unwrap();
    for id in 0..1000u32 {
        assert!(a.insert(&Rec { id, val: 0 }).unwrap());
    }
    let pages = a.pool_bytes() / PAGE_BYTES;
    assert!(pages > 1, "sequential keys must not pile into one shard");
    // Everything is still findable, and iteration sees every record once.
    assert_eq!(a.iter().count(), 1000);
    for id in 0..1000u32 {
        assert!(a.contains(&rec_key(id)));
    }
}

// --- capacity -------------------------------------------------------------

#[test]
fn shard_full_is_a_typed_error() {
    let mut a = ordered_arena(1);
    let slots = Arena::<Rec>::slots_per_page(); // 4096 / 5 = 819
    assert_eq!(slots, PAGE_BYTES / Rec::SIZE);
    for id in 0..slots as u32 {
        assert!(a.insert(&Rec { id, val: 0 }).unwrap());
    }
    let err = a
        .insert(&Rec {
            id: slots as u32,
            val: 0,
        })
        .unwrap_err();
    assert_eq!(err, Error::ShardFull { shard: 0, slots });
    // A duplicate of an existing key still answers Ok(false), not ShardFull.
    assert!(!a.insert(&Rec { id: 0, val: 1 }).unwrap());
    assert_eq!(a.len(), slots);
}

#[test]
fn max_bytes_is_enforced_before_allocation() {
    let cfg = ArenaCfg::new(1, ShardMode::Ordered).with_max_bytes(PAGE_BYTES - 1);
    let mut a = Arena::<Rec>::new(cfg).unwrap();
    let err = a.insert(&Rec { id: 1, val: 0 }).unwrap_err();
    assert_eq!(
        err,
        Error::CapacityExceeded {
            max_bytes: PAGE_BYTES - 1
        }
    );
    assert_eq!(a.pool_bytes(), 0, "failed insert allocated nothing");
    assert!(a.is_empty());
}

// --- payload mutation -----------------------------------------------------

#[test]
fn payload_mut_updates_only_payload() {
    let mut a = ordered_arena(64);
    a.insert(&Rec { id: 42, val: 1 }).unwrap();
    let payload = a.payload_mut(&rec_key(42)).unwrap();
    assert_eq!(payload.len(), Rec::SIZE - Rec::KEY_LEN);
    payload[0] = 200;
    assert_eq!(a.get(&rec_key(42)), Some(Rec { id: 42, val: 200 }));
    // Raw slot view agrees: key bytes intact, payload changed.
    let slot = a.get_slot(&rec_key(42)).unwrap();
    assert_eq!(key::read_u32(slot), 42);
    assert_eq!(slot[4], 200);
}

// --- key contract ---------------------------------------------------------

#[test]
#[should_panic(expected = "key length must equal Slot::KEY_LEN")]
fn wrong_key_length_panics() {
    let a = ordered_arena(64);
    let _ = a.get(&[0u8; 3]);
}

#[test]
fn byte_array_slots_have_set_semantics() {
    // The [u8; N] impl: whole value is the key (mirrors the original test
    // version of this structure, which stored 32-byte ids).
    let mut a = Arena::<[u8; 32]>::new(ArenaCfg::new(256, ShardMode::Ordered)).unwrap();
    let mut id = [0u8; 32];
    id[0] = 0xAB;
    id[31] = 0xCD;
    assert!(a.insert(&id).unwrap());
    assert!(!a.insert(&id).unwrap());
    assert!(a.contains(&id));
    assert_eq!(a.get(&id), Some(id));
    assert!(a.remove(&id));
    assert!(a.is_empty());
}

// --- iterator contract ----------------------------------------------------

#[test]
fn iterator_is_exact_size_and_into_iter_works() {
    let mut a = ordered_arena(64);
    for id in 0..10u32 {
        a.insert(&Rec {
            id: id * 1000,
            val: 0,
        })
        .unwrap();
    }
    let mut it = a.iter();
    assert_eq!(it.len(), 10);
    assert_eq!(it.size_hint(), (10, Some(10)));
    it.next();
    assert_eq!(it.len(), 9);
    // `&arena` is iterable directly.
    let n = (&a).into_iter().count();
    assert_eq!(n, 10);
}

// --- key helpers ----------------------------------------------------------

#[test]
fn key_helpers_roundtrip_and_preserve_order() {
    let mut buf = [0u8; 12];
    key::write_u64(&mut buf, 0xDEAD_BEEF_0123_4567);
    assert_eq!(key::read_u64(&buf), 0xDEAD_BEEF_0123_4567);
    key::write_u32(&mut buf, 0xCAFE_BABE);
    assert_eq!(key::read_u32(&buf), 0xCAFE_BABE);
    key::write_pair(&mut buf, 77, 5);
    assert_eq!(key::read_pair(&buf), (77, 5));

    // The reason big-endian is a hard rule: byte order == numeric order.
    let pairs = [(0u64, 1u32), (1, 0), (1, 1), (2, 0), (u64::MAX, u32::MAX)];
    let mut encoded: Vec<[u8; 12]> = pairs
        .iter()
        .map(|&(hi, lo)| {
            let mut k = [0u8; 12];
            key::write_pair(&mut k, hi, lo);
            k
        })
        .collect();
    let numeric_sorted = encoded.clone();
    encoded.sort_unstable(); // byte-wise sort
    assert_eq!(encoded, numeric_sorted);
}

// --- misc surface ---------------------------------------------------------

#[test]
fn debug_output_is_a_summary() {
    let mut a = ordered_arena(64);
    a.insert(&Rec { id: 1, val: 1 }).unwrap();
    let s = format!("{a:?}");
    assert!(s.contains("Arena"));
    assert!(s.contains("len: 1"));
    assert!(s.contains("slot_size: 5"));
}

#[test]
fn error_display_messages_are_stable() {
    // Error texts are part of the public contract (wrappers show them to
    // humans and agents), so pin them.
    assert_eq!(
        Error::ShardFull {
            shard: 3,
            slots: 819
        }
        .to_string(),
        "shard 3 is full (819 slots per page)"
    );
    assert_eq!(
        Error::CapacityExceeded { max_bytes: 4095 }.to_string(),
        "arena capacity exceeded: pool would grow past 4095 bytes"
    );
    assert_eq!(
        Error::BadShardCount { got: 3 }.to_string(),
        "shard count must be a non-zero power of two, got 3"
    );
    assert!(
        Error::BadSlot {
            size: 0,
            key_len: 0
        }
        .to_string()
        .starts_with("invalid slot layout")
    );
}

#[test]
fn cfg_accessor_reflects_construction() {
    let cfg = ArenaCfg::new(8, ShardMode::Ordered).with_max_bytes(1 << 20);
    let a = Arena::<Rec>::new(cfg).unwrap();
    assert_eq!(*a.cfg(), cfg);
}

// --- counters (feature-gated) ---------------------------------------------

#[cfg(feature = "counters")]
mod counters {
    use super::*;

    #[test]
    fn counters_observe_work_and_reset() {
        let mut a = ordered_arena(1);
        assert_eq!(a.counters(), plugmem_arena::Counters::default());

        a.insert(&Rec { id: 10, val: 0 }).unwrap();
        let after_first = a.counters();
        assert_eq!(after_first.pages_allocated, 1);
        assert_eq!(after_first.bytes_shifted, 0, "append shifts nothing");

        // Inserting before an existing key shifts exactly one slot.
        a.insert(&Rec { id: 5, val: 0 }).unwrap();
        let after_front = a.counters();
        assert_eq!(after_front.bytes_shifted, Rec::SIZE as u64);
        assert!(after_front.cmp_ops > 0);

        // Lookups count comparisons too (Cell works through &self).
        let before = a.counters().cmp_ops;
        assert!(a.contains(&rec_key(10)));
        assert!(a.counters().cmp_ops > before);

        // Removing the first record shifts the tail back over it.
        assert!(a.remove(&rec_key(5)));
        assert_eq!(a.counters().bytes_shifted, 2 * Rec::SIZE as u64);

        a.reset_counters();
        assert_eq!(a.counters(), plugmem_arena::Counters::default());
    }
}
