//! Boundary and contract tests for `Arena` — every v2 invariant from
//! `specs/01-arena.md` is pinned here: sorted shard chains, page splits
//! (including the single-slot degenerate case), free-list recycling,
//! range scans, capacity errors, and the key/payload contracts.

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

fn ordered_arena(shards: usize) -> Arena<'static, Rec> {
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

// --- capacity: splits, chains, free-list ----------------------------------

#[test]
fn full_page_splits_instead_of_failing() {
    let mut a = ordered_arena(1);
    let slots = Arena::<Rec>::slots_per_page(); // 4096 / 5 = 819
    assert_eq!(slots, PAGE_BYTES / Rec::SIZE);
    // Fill one page exactly, then keep going: the page must split.
    for id in 0..(slots as u32) * 3 {
        assert!(
            a.insert(&Rec {
                id,
                val: (id % 251) as u8
            })
            .unwrap()
        );
    }
    assert_eq!(a.len(), slots * 3);
    assert!(a.pool_bytes() > PAGE_BYTES, "chain grew beyond one page");
    // Everything findable, iteration sorted and complete.
    for id in 0..(slots as u32) * 3 {
        assert_eq!(a.get(&rec_key(id)).unwrap().val, (id % 251) as u8);
    }
    let ids: Vec<u32> = a.iter().map(|r| r.id).collect();
    assert_eq!(ids, (0..(slots as u32) * 3).collect::<Vec<_>>());
    // A duplicate of an existing key still answers Ok(false).
    assert!(!a.insert(&Rec { id: 0, val: 1 }).unwrap());
}

#[test]
// Heavy shift workload: minutes under the miri interpreter; the same
// code paths are exercised by the small tests above.
#[cfg_attr(miri, ignore)]
fn descending_and_interleaved_inserts_split_correctly() {
    // Descending order forces every insert to position 0 — splits happen at
    // the front repeatedly; then a second pass fills the gaps (odd ids) so
    // mid-page splits occur too.
    let mut a = ordered_arena(1);
    let n = 3000u32;
    for id in (0..n).step_by(2).rev() {
        assert!(a.insert(&Rec { id, val: 0 }).unwrap());
    }
    for id in (1..n).step_by(2) {
        assert!(a.insert(&Rec { id, val: 1 }).unwrap());
    }
    let ids: Vec<u32> = a.iter().map(|r| r.id).collect();
    assert_eq!(ids, (0..n).collect::<Vec<_>>());
}

#[test]
// Heavy shift workload: minutes under the miri interpreter; the same
// code paths are exercised by the small tests above.
#[cfg_attr(miri, ignore)]
fn stress_against_btreemap_with_splits_and_removals() {
    // Deterministic xorshift workload big enough to force many splits and
    // page recyclings in a single shard; lockstep with a BTreeMap.
    use std::collections::BTreeMap;
    let mut a = ordered_arena(1);
    let mut model: BTreeMap<u32, u8> = BTreeMap::new();
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..30_000 {
        let r = rng();
        let id = (r >> 32) as u32 % 8192;
        match r % 3 {
            0 | 1 => {
                let inserted = a.insert(&Rec { id, val: id as u8 }).unwrap();
                assert_eq!(inserted, !model.contains_key(&id));
                model.insert(id, id as u8);
            }
            _ => {
                assert_eq!(a.remove(&rec_key(id)), model.remove(&id).is_some());
            }
        }
        assert_eq!(a.len(), model.len());
    }
    let got: Vec<u32> = a.iter().map(|r| r.id).collect();
    let want: Vec<u32> = model.keys().copied().collect();
    assert_eq!(got, want);
}

#[test]
// Heavy shift workload: minutes under the miri interpreter; the same
// code paths are exercised by the small tests above.
#[cfg_attr(miri, ignore)]
fn emptied_pages_are_recycled_through_the_free_list() {
    let mut a = ordered_arena(1);
    let n = Arena::<Rec>::slots_per_page() as u32 * 4;
    for id in 0..n {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    let peak = a.pool_bytes();
    // Remove everything: every page empties and is unlinked.
    for id in 0..n {
        assert!(a.remove(&rec_key(id)));
    }
    assert!(a.is_empty());
    // Reinsert the same volume: recycled pages serve it — the pool must not
    // grow beyond its previous peak.
    for id in 0..n {
        a.insert(&Rec { id, val: 1 }).unwrap();
    }
    assert_eq!(a.len(), n as usize);
    assert!(
        a.pool_bytes() <= peak,
        "free-list reuse keeps the pool bounded"
    );
}

#[test]
// Heavy shift workload: minutes under the miri interpreter; the same
// code paths are exercised by the small tests above.
#[cfg_attr(miri, ignore)]
fn removing_a_middle_page_keeps_the_chain_intact() {
    let mut a = ordered_arena(1);
    let slots = Arena::<Rec>::slots_per_page() as u32;
    // Three pages worth of records.
    for id in 0..slots * 3 {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    // Carve out a contiguous middle band — some page in the middle of the
    // chain empties and must be unlinked without breaking its neighbors.
    for id in slots..slots * 2 {
        assert!(a.remove(&rec_key(id)));
    }
    for id in 0..slots {
        assert!(a.contains(&rec_key(id)));
    }
    for id in slots * 2..slots * 3 {
        assert!(a.contains(&rec_key(id)));
    }
    let ids: Vec<u32> = a.iter().map(|r| r.id).collect();
    assert!(ids.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(ids.len(), (slots * 2) as usize);
}

/// Slot bigger than half a page: pages hold exactly one record, exercising
/// the degenerate split path (chain of single-slot pages).
struct Big {
    id: u32,
    fill: u8,
}
impl Slot for Big {
    const SIZE: usize = 2100;
    const KEY_LEN: usize = 4;
    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id);
        out[4..].fill(self.fill);
    }
    fn read(bytes: &[u8]) -> Self {
        Big {
            id: key::read_u32(bytes),
            fill: bytes[4],
        }
    }
}

#[test]
fn single_slot_pages_chain_and_stay_sorted() {
    assert_eq!(Arena::<Big>::slots_per_page(), 1);
    let mut a = Arena::<Big>::new(ArenaCfg::new(1, ShardMode::Ordered)).unwrap();
    // Insert in an order that hits both degenerate-split branches: appending
    // after (pos == 1) and inserting before (pos == 0).
    for id in [5u32, 3, 8, 1, 9, 7] {
        assert!(a.insert(&Big { id, fill: id as u8 }).unwrap());
    }
    let ids: Vec<u32> = a.iter().map(|b| b.id).collect();
    assert_eq!(ids, [1, 3, 5, 7, 8, 9]);
    for id in [1u32, 3, 5, 7, 8, 9] {
        assert_eq!(a.get(&rec_key(id)).unwrap().fill, id as u8);
    }
    assert!(a.remove(&rec_key(5)));
    let ids: Vec<u32> = a.iter().map(|b| b.id).collect();
    assert_eq!(ids, [1, 3, 7, 8, 9]);
}

// --- range scans ----------------------------------------------------------

#[test]
fn range_scans_ordered_mode() {
    let mut a = ordered_arena(64);
    // Multi-shard spread plus a dense run inside one shard.
    let ids: Vec<u32> = (0..2000u32).map(|i| i * 7919).collect();
    for &id in &ids {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    let from = rec_key(1_000_000);
    let to = rec_key(9_000_000);
    let got: Vec<u32> = a.range(&from, &to).map(|r| r.id).collect();
    let want: Vec<u32> = {
        let mut v: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| (1_000_000..9_000_000).contains(&id))
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(got, want);

    // `from` inclusive, `to` exclusive.
    let exact_from = rec_key(want[0]);
    let exact_to = rec_key(*want.last().unwrap());
    let inclusive: Vec<u32> = a.range(&exact_from, &exact_to).map(|r| r.id).collect();
    assert_eq!(inclusive.first(), Some(&want[0]), "from is inclusive");
    assert_eq!(
        inclusive.last(),
        Some(&want[want.len() - 2]),
        "to is exclusive"
    );

    // Empty and inverted ranges yield nothing.
    assert_eq!(a.range(&rec_key(2), &rec_key(3)).count(), 0);
    assert_eq!(a.range(&to, &from).count(), 0);
    // Full range covers everything.
    assert_eq!(a.range(&rec_key(0), &rec_key(u32::MAX)).count(), ids.len());
}

#[test]
fn range_spans_page_chains_within_a_shard() {
    let mut a = ordered_arena(1);
    let n = Arena::<Rec>::slots_per_page() as u32 * 3;
    for id in 0..n {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    let got: Vec<u32> = a
        .range(&rec_key(100), &rec_key(n - 100))
        .map(|r| r.id)
        .collect();
    assert_eq!(got, (100..n - 100).collect::<Vec<_>>());
}

#[test]
fn range_starting_past_the_covering_page_continues_or_ends() {
    // `from` is larger than every key in the page that covers it: the scan
    // must hop to the next chain page (or finish cleanly when there is none).
    let mut a = ordered_arena(1);
    for id in 0..10u32 {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    // Past everything: covering page is the only page, scan ends empty.
    assert_eq!(a.range(&rec_key(50), &rec_key(100)).count(), 0);

    // Now force a chain of pages and a gap between them: remove the tail of
    // the first page so `from` falls in the gap — the covering page is the
    // first one, but the matching records start in the second.
    let mut a = ordered_arena(1);
    let slots = Arena::<Rec>::slots_per_page() as u32;
    for id in 0..slots * 2 {
        a.insert(&Rec { id, val: 0 }).unwrap();
    }
    // First page now covers [0..half); second page starts somewhere >= half.
    // Remove a band right before the second page's first key.
    let second_first = a.iter().map(|r| r.id).nth(a.len() / 2).unwrap();
    for id in second_first - 5..second_first {
        a.remove(&rec_key(id));
    }
    let got: Vec<u32> = a
        .range(&rec_key(second_first - 3), &rec_key(second_first + 3))
        .map(|r| r.id)
        .collect();
    assert_eq!(got, (second_first..second_first + 3).collect::<Vec<_>>());
}

#[test]
#[should_panic(expected = "range scans require ShardMode::Ordered")]
fn range_on_uniform_mode_panics() {
    let a = Arena::<Rec>::new(ArenaCfg::new(64, ShardMode::Uniform)).unwrap();
    let _ = a.range(&rec_key(0), &rec_key(10));
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
        Error::CapacityExceeded { max_bytes: 4095 }.to_string(),
        "capacity exceeded: pool would grow past 4095 bytes"
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
    assert_eq!(
        Error::BlobTooLarge {
            len: 9,
            max_blob: 8
        }
        .to_string(),
        "blob of 9 bytes exceeds the configured max_blob of 8 bytes"
    );
    assert_eq!(
        Error::ValueTooLarge { len: 61 }.to_string(),
        "value of 61 bytes exceeds the chunk payload of 60 bytes"
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
    fn splits_and_chain_steps_are_counted() {
        let mut a = ordered_arena(1);
        let n = Arena::<Rec>::slots_per_page() as u32 + 1;
        for id in 0..n {
            a.insert(&Rec { id, val: 0 }).unwrap();
        }
        let c = a.counters();
        assert_eq!(c.splits, 1, "filling one page past capacity splits once");
        // Looking up a key in the second page walks one chain step.
        a.reset_counters();
        assert!(a.contains(&rec_key(n - 1)));
        assert_eq!(a.counters().chain_steps, 1);
        // First-page lookups take no chain steps.
        a.reset_counters();
        assert!(a.contains(&rec_key(0)));
        assert_eq!(a.counters().chain_steps, 0);
    }

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
