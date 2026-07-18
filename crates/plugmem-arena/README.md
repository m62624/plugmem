# plugmem-arena

Flat byte-pool storage structures: a sharded sorted arena, an append-only
blob heap, chunked lists, and a string interner — `no_std + alloc`,
zero-copy-persistable, **tuned for WebAssembly first**.

This crate is the storage foundation of the plugmem engine, but nothing in
it knows about facts, vectors or LLMs. If you need a compact,
allocation-frugal sorted container whose in-memory representation *is* its
serialized form, you can lift it into your own project as-is.

## Philosophy

1. **State is flat bytes.** A container is one contiguous byte pool plus a
   few small metadata arrays. No per-element allocations, no pointer graphs.
   Persisting a container is a `memcpy`; loading it back is bounds-checking
   the metadata and adopting the bytes.
2. **Costs are local and visible.** Every operation touches one 4 KiB page.
   Worst cases are small, fixed and *measured* — the optional `counters`
   feature exposes deterministic work counters used as CI performance gates.
3. **wasm is the primary target.** The design fits 32-bit address spaces
   (`u32` ids and offsets), builds on `wasm32v1-none` (no OS, no threads),
   and its one `unsafe` exists precisely because of a measured 12x win on
   the wasm allocation path (see below).

## The four structures

| Structure | Shape | Typical use |
|---|---|---|
| `Arena<T: Slot>` | sorted fixed-size records, sharded chains of 4 KiB pages | record stores, ordered indexes |
| `BlobHeap` | append-only variable-length blobs, dense `u32` ids | texts, names, raw vectors |
| `ChunkPool` | many small growable lists over 64-byte chunks | posting lists, adjacency lists |
| `Interner` | string → dense `u32` (heap + flat hash table) | terms, tags, entity names |

Keys are **big-endian**: byte-wise comparison equals numeric comparison, so
binary search, ordered iteration and range scans run on raw bytes.

## Quick start

```rust
use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, key};

/// A fixed-size record: 4-byte big-endian key + 1-byte payload.
struct Rec { id: u32, level: u8 }

impl Slot for Rec {
    const SIZE: usize = 5;
    const KEY_LEN: usize = 4;
    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.id);
        out[4] = self.level;
    }
    fn read(bytes: &[u8]) -> Self {
        Rec { id: key::read_u32(bytes), level: bytes[4] }
    }
}

let mut arena = Arena::<Rec>::new(ArenaCfg::new(64, ShardMode::Ordered))?;
arena.insert(&Rec { id: 7, level: 3 })?;
arena.insert(&Rec { id: 1, level: 9 })?;

// Ordered mode: iteration and range scans in global key order.
let ids: Vec<u32> = arena.iter().map(|r| r.id).collect();
assert_eq!(ids, [1, 7]);
# Ok::<(), plugmem_arena::Error>(())
```

See `examples/basic.rs` for a walkthrough and the crate docs for the full
API, including page-split mechanics, the free-list, and `range()` scans.

## Performance

![benchmark chart](assets/bench-matrix.svg)

Honest framing first: these are different *classes*. The arena's class is
"ordered map over flat memory with incremental updates" — that is what
`BTreeMap` is compared against. `HashMap` gives up ordering and `sorted Vec`
gives up incremental inserts (it is bulk-built here), which is why they win
their columns; neither offers what the arena exists for — **the memory
image is the snapshot** (persist = `memcpy`, load = adopt bytes), across
native and wasm.

One thread, 100k records, 16-byte slots, deterministic seeded keys, best of
3 (2026-07-18):

| insert ns/elem | native | wasmtime | wasmer |
|---|---|---|---|
| **plugmem Arena (Ordered)** | **65.5** | **90.4** | **82.2** |
| std BTreeMap (same class) | 93.6 | 129.9 | 111.4 |
| std HashMap (no ordering) | 20.9 | 28.3 | 28.1 |
| sorted Vec (bulk build) | 17.9 | 20.1 | 20.6 |

| point lookup ns/op | native | wasmtime | wasmer |
|---|---|---|---|
| **plugmem Arena (Uniform)** | **61.2** | **78.6** | **76.0** |
| std BTreeMap (same class) | 86.3 | 91.9 | 95.2 |
| std HashMap (no ordering) | 10.2 | 13.4 | 14.0 |
| sorted Vec (bulk build) | 31.2 | 35.6 | 36.6 |

Ordered range scans decode ~230M records/s natively (4.3 ns per 16-byte
record, fully deserialized). Memory sits at 42 B/elem after random-order
inserts (split pages average ~70% full; a compaction pass can rebuild to
near 16 B/elem) versus ~20 B for `BTreeMap` — that is the trade the
snapshot-as-memcpy property buys.

### Reproduce it yourself

The workload is deterministic and the harness is dependency-free. Install
the wasm runtimes you want to compare ([wasmtime](https://wasmtime.dev),
[wasmer](https://wasmer.io)) — the runner never downloads anything — then:

```text
rustup target add wasm32-wasip1
cargo run --release -p plugmem-bench-matrix
```

This builds the `bench_repro` example for native and `wasm32-wasip1`, runs
it on every runtime found, and prints the merged per-metric tables plus a
machine-readable TSV. Missing runtimes are skipped with an install hint.
Criterion micro-benchmarks (statistical, native): `cargo bench -p
plugmem-arena`.

## The one `unsafe`

Page allocation without zeroing (`Vec::reserve` + `set_len`). Kept because
it was *measured*, not assumed: zeroing freshly grown pages made the wasm
allocation path **12x slower** (wasmtime, 32k pages: 3889 µs zeroed vs
316 µs uninit), while on native x86-64 the difference is noise. The safety
invariant is local — bytes of a page beyond the occupied slot count are
never read — and the whole crate passes **miri**. Bounds-check elimination
(`get_unchecked`) was measured on the same harness and rejected: ≤1% on
native, *slower* under wasm. `Arena` deliberately implements neither
`Clone` nor `PartialEq` (a byte-wise clone or compare would read the
uninitialized page tails); `BlobHeap` and friends, which hold no
uninitialized bytes, derive both.

## Related crates

Crates solving neighboring problems, and why this one is different:

- [`btree-slab`](https://crates.io/crates/btree-slab), [`scapegoat`](https://crates.io/crates/scapegoat) —
  ordered maps over slab/arena backing (the closest class). Node-linked
  internally; the backing is an allocation strategy, not a serialization
  format — no snapshot-as-memcpy.
- [`sorted-vec`](https://crates.io/crates/sorted-vec) — flat sorted arrays;
  O(n) per random insert (whole-tail shift). The arena bounds every shift
  to one 4 KiB page.
- [`indexmap`](https://crates.io/crates/indexmap), [`slotmap`](https://crates.io/crates/slotmap) —
  fast dense-storage maps, but insertion-ordered / unordered: no key-sorted
  iteration or range scans.
- [`heapless`](https://crates.io/crates/heapless) — static capacity,
  no growth; a different point on the embedded spectrum.

## Feature flags

- `std` *(default)* — convenience only; the crate is fully functional as
  `no_std + alloc` (built and gated on `wasm32v1-none`).
- `counters` — deterministic work counters on every structure (key
  comparisons, bytes shifted, pages allocated, chain steps, splits, interner
  probes). Zero cost when disabled.

## Testing

75 boundary tests + 6 proptest reference models (`BTreeMap`, `Vec<Vec<u8>>`,
per-list model, `HashMap` bijection), ≥90% line coverage with hand-audited
analyzer artifacts, clippy-clean in all four feature combinations, miri on
the full suite, and `wasm32v1-none` builds as a hard gate.

## License

MIT.
