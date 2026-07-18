//! Property tests: the arena must behave exactly like a reference
//! `BTreeMap<u32, u8>` under arbitrary operation sequences. This is the
//! safety net that lets the implementation carry its one `unsafe` (and any
//! future measured optimization): if a shortcut breaks semantics, some
//! random sequence here finds it and proptest shrinks it to a minimal
//! counterexample.

use std::collections::BTreeMap;

use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, key};
use proptest::prelude::*;

/// Same 5-byte record as the boundary tests: 4-byte BE key + 1-byte payload.
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

/// One step of the workload.
#[derive(Debug, Clone)]
enum Op {
    Insert(u32, u8),
    Remove(u32),
    Get(u32),
    Update(u32, u8),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    // A small key space (0..512) makes duplicates, removals of existing
    // keys and updates actually happen instead of being measure-zero.
    let k = 0u32..512;
    prop_oneof![
        (k.clone(), any::<u8>()).prop_map(|(id, v)| Op::Insert(id, v)),
        k.clone().prop_map(Op::Remove),
        k.clone().prop_map(Op::Get),
        (k, any::<u8>()).prop_map(|(id, v)| Op::Update(id, v)),
    ]
}

/// Runs a workload against the arena and the model in lockstep, checking
/// every intermediate answer, then compares the final contents.
fn run_model(mode: ShardMode, shards: usize, ops: Vec<Op>) {
    let mut arena = Arena::<Rec>::new(ArenaCfg::new(shards, mode)).unwrap();
    let mut model: BTreeMap<u32, u8> = BTreeMap::new();

    for op in ops {
        match op {
            Op::Insert(id, val) => {
                // v2 semantics: full pages split, so an insert only ever
                // reports "new" or "duplicate" (soft), never a capacity
                // error at page granularity.
                let inserted = arena.insert(&Rec { id, val }).unwrap();
                let model_inserted = !model.contains_key(&id);
                assert_eq!(inserted, model_inserted, "insert({id})");
                if model_inserted {
                    model.insert(id, val);
                }
            }
            Op::Remove(id) => {
                assert_eq!(arena.remove(&rec_key(id)), model.remove(&id).is_some());
            }
            Op::Get(id) => {
                let got = arena.get(&rec_key(id)).map(|r| r.val);
                assert_eq!(got, model.get(&id).copied(), "get({id})");
                assert_eq!(arena.contains(&rec_key(id)), model.contains_key(&id));
            }
            Op::Update(id, val) => {
                let updated = match arena.payload_mut(&rec_key(id)) {
                    Some(payload) => {
                        payload[0] = val;
                        true
                    }
                    None => false,
                };
                assert_eq!(updated, model.contains_key(&id));
                if updated {
                    model.insert(id, val);
                }
            }
        }
        assert_eq!(arena.len(), model.len());
    }

    // Final contents must agree as multisets (and iteration must visit each
    // record exactly once).
    let mut got: Vec<(u32, u8)> = arena.iter().map(|r| (r.id, r.val)).collect();
    got.sort_unstable();
    let want: Vec<(u32, u8)> = model.iter().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(got, want);
}

proptest! {
    /// Uniform sharding: the general-purpose lookup configuration.
    #[test]
    // proptest's harness calls into the OS (cwd for failure
    // persistence), which miri's isolation forbids; UB-paths are covered
    // by the boundary tests.
    #[cfg_attr(miri, ignore)]
    fn behaves_like_btreemap_uniform(ops in proptest::collection::vec(op_strategy(), 1..400)) {
        run_model(ShardMode::Uniform, 64, ops);
    }

    /// Ordered sharding, several shards: iteration must additionally be
    /// globally key-ascending (checked implicitly: BTreeMap order == sorted).
    #[test]
    // proptest's harness calls into the OS (cwd for failure
    // persistence), which miri's isolation forbids; UB-paths are covered
    // by the boundary tests.
    #[cfg_attr(miri, ignore)]
    fn behaves_like_btreemap_ordered(ops in proptest::collection::vec(op_strategy(), 1..400)) {
        let mut arena = Arena::<Rec>::new(ArenaCfg::new(16, ShardMode::Ordered)).unwrap();
        let mut model: BTreeMap<u32, u8> = BTreeMap::new();
        for op in &ops {
            if let Op::Insert(id, val) = op
                && arena.insert(&Rec { id: *id, val: *val }).unwrap_or(false)
            {
                model.insert(*id, *val);
            }
        }
        // Ordered mode promise: iter() == ascending key order, no sorting.
        let got: Vec<(u32, u8)> = arena.iter().map(|r| (r.id, r.val)).collect();
        let want: Vec<(u32, u8)> = model.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(got, want);
    }

    /// A single shard is the stress case for in-page shifting: every key
    /// lands in one page, so inserts/removes constantly move memory.
    #[test]
    // proptest's harness calls into the OS (cwd for failure
    // persistence), which miri's isolation forbids; UB-paths are covered
    // by the boundary tests.
    #[cfg_attr(miri, ignore)]
    fn behaves_like_btreemap_single_shard(ops in proptest::collection::vec(op_strategy(), 1..300)) {
        run_model(ShardMode::Ordered, 1, ops);
    }
}
