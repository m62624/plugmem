//! A self-contained tour of `plugmem-arena`, written to be lifted into your
//! own project: run it with `cargo run -p plugmem-arena --example basic`.
//!
//! The scenario: an append-heavy event log with a composite time-ordered key
//! (`[timestamp | event_id]`) — the same shape the plugmem engine uses for
//! its temporal index. It shows the three things that make this container
//! different from a `BTreeMap`:
//!
//! 1. all state is one flat byte pool (persisting it is a `memcpy` — the
//!    structure was designed for wasm environments, where you hand the
//!    linear-memory bytes to the host and you are done);
//! 2. keys are big-endian, so byte order == numeric order and ordered
//!    iteration works on raw bytes;
//! 3. costs are page-local and observable (see the `counters` feature).

use plugmem_arena::{Arena, ArenaCfg, Error, ShardMode, Slot, key};

/// One log record, 16 bytes: 12-byte key `[timestamp | event_id]` followed
/// by a 4-byte payload. Fixed-size records are the deal you make with the
/// arena: in exchange you get binary search over raw bytes and zero
/// per-record allocations.
#[derive(Debug, PartialEq)]
struct Event {
    /// Unix-milliseconds timestamp — the *leading* (most significant) key
    /// component, so records sort chronologically.
    ts: u64,
    /// Disambiguates events sharing a timestamp — the trailing key component.
    id: u32,
    /// Payload: any fixed-size data; here, a severity value.
    severity: u32,
}

impl Slot for Event {
    const SIZE: usize = 16;
    const KEY_LEN: usize = 12;

    fn write(&self, out: &mut [u8]) {
        // Big-endian, most significant component first: this is what makes
        // `arena.iter()` chronological without any comparator code.
        key::write_pair(out, self.ts, self.id);
        key::write_u32(&mut out[12..], self.severity);
    }

    fn read(bytes: &[u8]) -> Self {
        let (ts, id) = key::read_pair(bytes);
        Event {
            ts,
            id,
            severity: key::read_u32(&bytes[12..]),
        }
    }
}

/// Builds the 12-byte lookup key for an event.
fn event_key(ts: u64, id: u32) -> [u8; 12] {
    let mut k = [0u8; 12];
    key::write_pair(&mut k, ts, id);
    k
}

fn main() -> Result<(), Error> {
    // `Ordered` mode: the shard of a key is its top bits, so iterating shard
    // 0, 1, 2, ... visits keys in ascending order. Use `Uniform` instead
    // when you only ever look records up by key (it spreads sequential ids
    // evenly, but gives up global iteration order).
    let cfg = ArenaCfg::new(64, ShardMode::Ordered).with_max_bytes(1 << 20);
    let mut log = Arena::<Event>::new(cfg)?;

    // Insert out of chronological order on purpose.
    for (ts, id, severity) in [
        (1_700_000_300_000, 1, 2),
        (1_700_000_100_000, 1, 5),
        (1_700_000_200_000, 1, 1),
        (1_700_000_100_000, 0, 4), // same ts as another event: id breaks the tie
    ] {
        let inserted = log.insert(&Event { ts, id, severity })?;
        assert!(inserted);
    }

    // Duplicate keys are rejected softly: `Ok(false)`, nothing overwritten.
    let dup = log.insert(&Event {
        ts: 1_700_000_100_000,
        id: 1,
        severity: 9,
    })?;
    assert!(!dup);
    assert_eq!(log.len(), 4);

    // Point lookup by the byte key.
    let k = event_key(1_700_000_200_000, 1);
    assert_eq!(log.get(&k).unwrap().severity, 1);

    // In-place payload update: `payload_mut` hands out only the bytes
    // *after* the key prefix, so sorted order cannot be corrupted.
    key::write_u32(log.payload_mut(&k).unwrap(), 7);
    assert_eq!(log.get(&k).unwrap().severity, 7);

    // Ordered iteration falls out of the big-endian keys: no comparator,
    // just bytes.
    println!("chronological log:");
    for event in &log {
        println!(
            "  ts={} id={} severity={}",
            event.ts, event.id, event.severity
        );
    }
    let times: Vec<(u64, u32)> = log.iter().map(|e| (e.ts, e.id)).collect();
    assert!(times.windows(2).all(|w| w[0] < w[1]));

    // Removal shifts within one 4 KiB page — bounded, cache-local work.
    assert!(log.remove(&k));
    assert_eq!(log.len(), 3);

    // Capacity is a typed error, never a panic: this arena was capped at
    // 1 MiB, a 4 GiB one would get `Error::CapacityExceeded` on the page
    // allocation that crosses the limit.
    println!("pool bytes in use: {}", log.pool_bytes());

    Ok(())
}
