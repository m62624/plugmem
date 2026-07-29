//! Deterministic performance gates.
//!
//! Wall-clock numbers lie in CI; these tests assert ceilings on the
//! `counters`-feature work counters instead. The workload is fixed-seed,
//! so the counter values are identical on every machine — an algorithmic
//! regression (an accidentally quadratic path, a degenerated chain walk)
//! fails the same way everywhere, while hardware noise cannot.
//!
//! Ceilings were set from measured values with a ×1.2 margin (the
//! rule) and may only be lowered; raising one is an explicit decision that
//! must be recorded in. Measured 2026-07-18 on the 100k workload
//! below; the assert messages print the measured value on failure.
//!
//! Runs only with `--features counters` (the counters are compiled out
//! otherwise); CI executes the feature-matrix test job.

#![cfg(feature = "counters")]

use plugmem_arena::{Arena, ArenaCfg, BlobHeapCfg, Interner, ShardMode, Slot, key};

/// The engine's hot record shape: 16-byte slot, 12-byte composite key.
#[derive(Clone, Copy)]
struct Rec16 {
    hi: u64,
    lo: u32,
    val: u32,
}

impl Slot for Rec16 {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 12;
    fn write(&self, out: &mut [u8]) {
        key::write_pair(out, self.hi, self.lo);
        out[12..16].copy_from_slice(&self.val.to_be_bytes());
    }
    fn read(bytes: &[u8]) -> Self {
        let (hi, lo) = key::read_pair(bytes);
        Self {
            hi,
            lo,
            val: u32::from_be_bytes(bytes[12..16].try_into().unwrap()),
        }
    }
}

/// Deterministic pseudo-random u64 stream (xorshift64, fixed seed).
fn keys(n: usize) -> Vec<u64> {
    let mut s = 0xB0B5_1234_5678_9ABCu64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect()
}

const N: usize = 100_000;

#[test]
fn arena_insert_work_is_bounded() {
    let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Uniform)).unwrap();
    for &k in &keys(N) {
        a.insert(&Rec16 {
            hi: k,
            lo: 0,
            val: 1,
        })
        .unwrap();
    }
    let c = a.counters();
    // Measured @100k uniform-random, 1024 shards: cmp_ops 528_063,
    // bytes_shifted 38_980_112, pages_allocated 1024. At ~98 records per
    // shard every shard fits its first page, so chains and splits must
    // not appear at all — those two are exact, not margins.
    assert!(c.cmp_ops <= 634_000, "cmp_ops regressed: {}", c.cmp_ops);
    assert!(
        c.bytes_shifted <= 46_800_000,
        "bytes_shifted regressed: {}",
        c.bytes_shifted
    );
    assert!(
        c.pages_allocated <= 1_100,
        "pages_allocated regressed: {}",
        c.pages_allocated
    );
    assert_eq!(
        c.chain_steps, 0,
        "chains formed under a one-page-per-shard load"
    );
    assert_eq!(
        c.splits, 0,
        "splits happened under a one-page-per-shard load"
    );
}

/// Same corpus over few shards: long chains and splits are the point here
/// — this is the regime the chain/split machinery must stay linear in.
#[test]
fn arena_chain_and_split_work_is_bounded() {
    let mut a = Arena::<Rec16>::new(ArenaCfg::new(64, ShardMode::Uniform)).unwrap();
    for &k in &keys(N) {
        a.insert(&Rec16 {
            hi: k,
            lo: 0,
            val: 1,
        })
        .unwrap();
    }
    let c = a.counters();
    // Measured @100k over 64 shards (~1560 records, ~7 pages per shard):
    // cmp_ops 738_847, bytes_shifted 136_900_992, pages_allocated 512,
    // chain_steps 176_276 (~1.8 hops per insert), splits 448.
    assert!(c.cmp_ops <= 887_000, "cmp_ops regressed: {}", c.cmp_ops);
    assert!(
        c.bytes_shifted <= 164_300_000,
        "bytes_shifted regressed: {}",
        c.bytes_shifted
    );
    assert!(
        c.pages_allocated <= 620,
        "pages_allocated regressed: {}",
        c.pages_allocated
    );
    assert!(
        c.chain_steps <= 212_000,
        "chain_steps regressed: {}",
        c.chain_steps
    );
    assert!(c.splits <= 540, "splits regressed: {}", c.splits);
}

#[test]
fn arena_get_work_is_bounded() {
    let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Uniform)).unwrap();
    let ks = keys(N);
    for &k in &ks {
        a.insert(&Rec16 {
            hi: k,
            lo: 0,
            val: 1,
        })
        .unwrap();
    }
    a.reset_counters();
    const LOOKUPS: usize = 10_000;
    let mut found = 0u32;
    for &k in ks.iter().step_by(N / LOOKUPS) {
        let mut kb = [0u8; 12];
        key::write_pair(&mut kb, k, 0);
        found += u32::from(a.contains(&kb));
    }
    assert_eq!(found as usize, LOOKUPS);
    let c = a.counters();
    // Measured @10k hits over 100k records: cmp_ops 57_702 — ~5.8
    // comparisons per lookup, zero chain hops (one page per shard).
    assert!(
        c.cmp_ops <= 69_000,
        "lookup cmp_ops regressed: {}",
        c.cmp_ops
    );
    assert_eq!(
        c.chain_steps, 0,
        "lookup chain hops appeared: {}",
        c.chain_steps
    );
    assert_eq!(c.bytes_shifted, 0, "lookups must not move bytes");
}

#[test]
fn arena_range_scan_work_is_bounded() {
    let mut a = Arena::<Rec16>::new(ArenaCfg::new(1024, ShardMode::Ordered)).unwrap();
    let ks = keys(N);
    for &k in &ks {
        a.insert(&Rec16 {
            hi: k,
            lo: 0,
            val: 1,
        })
        .unwrap();
    }
    let mut sorted = ks.clone();
    sorted.sort_unstable();
    a.reset_counters();
    let (mut from, mut to) = ([0u8; 12], [0u8; 12]);
    key::write_pair(&mut from, sorted[N / 2], 0);
    key::write_pair(&mut to, sorted[N / 2 + 1000], 0);
    assert_eq!(a.range(&from, &to).count(), 1000);
    let c = a.counters();
    // Measured for a 1000-record window: cmp_ops 6, chain_steps 0 — seek
    // cost only; the scan itself is pointer-walking, not searching.
    assert!(
        c.cmp_ops <= 8,
        "range seek cmp_ops regressed: {}",
        c.cmp_ops
    );
    assert!(
        c.chain_steps <= 8,
        "range chain_steps regressed: {}",
        c.chain_steps
    );
}

#[test]
fn interner_probe_work_is_bounded() {
    let mut it = Interner::new(BlobHeapCfg::new());
    for i in 0..30_000u32 {
        it.intern(&format!("term-{i}")).unwrap();
    }
    // Measured for a 30k-term build (including rehashes): 141_938 probes
    // — ~4.7 per intern amortized at load factor <= 0.7.
    let build = it.probes();
    assert!(build <= 170_000, "build probes regressed: {build}");
    it.reset_probes();
    for i in 0..30_000u32 {
        it.intern(&format!("term-{i}")).unwrap();
    }
    // Measured for 30k warm hits: 42_532 probes — ~1.4 per hit.
    let hits = it.probes();
    assert!(hits <= 51_000, "hit probes regressed: {hits}");
}
