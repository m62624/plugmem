# Local guide: `plugmem-arena`

## Role

`plugmem-arena` is the low-level, generic storage layer. It knows nothing about facts, embeddings, language models, clocks, files, or threads. `plugmem-core` builds the memory engine on top of these containers.

The crate is `#![no_std]` with `alloc`. Its serialized representation is deliberately close to its in-memory representation: flat byte pools plus small metadata arrays, rather than a graph of heap objects.

## Public structures

- `Arena<T: Slot>` — sorted fixed-size records distributed over 4 KiB pages. `ShardMode::Ordered` preserves global ordered iteration; `ShardMode::Uniform` spreads keys by hash for more uniform point lookup.
- `BlobHeap` — append-only variable-length byte storage with dense `BlobId` handles. Use it for text, names, and opaque payloads.
- `ChunkPool` — growable lists made from 64-byte chunks. A `ListHandle` points to a chain; each chunk has link metadata and a fixed payload.
- `Interner` — string-to-dense-`TermId` mapping backed by a flat hash table and byte heap.
- `Slot` — the serialization contract for fixed-size arena records. `SIZE`, `KEY_LEN`, `write`, and `read` must agree exactly.

Keys are encoded in big-endian order. The byte ordering of a key must equal the numeric ordering expected by binary search and iteration.

## The single library `unsafe`

The only default `unsafe` in the library is in `Arena::alloc_page` (`src/arena.rs`). A newly grown page is reserved and its length is set without zeroing it. This was measured because zeroing fresh pages was much slower on the wasm target.

The safety invariant is strict:

1. A page is read only through `counts[page] * T::SIZE` initialized bytes.
2. Every slot is fully written before its page count is incremented.
3. Searches, reads, iteration, ranges, key extraction, and shifts never inspect the uninitialized tail.
4. Snapshot writers emit only initialized page prefixes.

Do not add whole-pool byte reads, `Clone`, or `PartialEq` implementations that could inspect uninitialized tails. The overlay/grown-tail path must not introduce a second `unsafe`. If the allocation invariant changes, update the safety comment and the property/performance tests together.

`get_unchecked` was measured and rejected: it provided negligible native benefit and was worse under wasm. Prefer safe indexing everywhere else.

## Allocation discipline

The second load-bearing property of this crate, and as much a contract as the single `unsafe`. The crate exists because a graph of heap objects is the wrong shape for this workload; a change that quietly reintroduces one has undone the crate whether or not it is faster.

- **Zero allocations per operation.** `insert`, `get`, `remove`, `range` and iteration must not allocate. Reusable buffers (`scratch`, `split_buf`) are grown once and kept; growth happens only at a new size maximum.
- **Only amortized growth.** Every collection is a `Vec` that doubles. Nothing is allocated per record, per page or per call.
- **The published number is the gate.** `examples/bench_repro` reports `allocs` — allocator calls to build the corpus — and it appears on `arena-allocs-1m.svg` next to `std` containers. Building 1M records costs a few dozen calls. A change that raises it needs a stated reason, not just green tests; re-run the example and compare on the same machine.
- **New metadata rides inside an existing collection**, never in a new parallel one. A second `Vec` alongside an existing one doubles that structure's reallocations for no structural benefit — this is why the page directory stores each page's first key inside its own entry rather than in a companion vector.
- **Flat only.** `Vec` of plain data. No `Box`, no `BTreeMap`, no `HashMap`, no `Rc`, in data or in metadata.

## Runtime metadata

`prev`, `tails` and `dir` (the page directory, including each page's cached first key) are **derived state**: rebuilt by the load walk, absent from the on-disk image, and free to change shape without touching the format. The rule they impose is that every mutation path keeps them true — page allocation, splits, page recycling and the load walk itself. A stale directory misdirects a lookup instead of failing, so the invariant is asserted in debug builds after each insert and remove; keep that assertion alive when adding a mutation path.

## 32-bit targets

`usize` is 32 bits on wasm32, and this crate reads untrusted images. Arithmetic on lengths and offsets taken from an image must be provably in range there too: compute in `u64` and compare against a real slice length before casting down, or use `checked_*`. The existing loaders (`blob.rs`, `chunk.rs`, `interner.rs`) show the pattern and say so in comments — follow it rather than reasoning about 64-bit sizes.

## Shard counts come from the caller

The arena validates that a shard count is a non-zero power of two and nothing more; how many shards an arena *should* have is `plugmem-core`'s decision (`memory/shards.rs`), because only it knows how much will be stored. Two things follow for this crate. The per-shard cost is real and paid up front — `heads`, `tails` and `dir_at` are one vector each — so keep it flat and keep it small. And a shard count read from an untrusted image becomes the length of those vectors, which is why the ceiling that bounds it lives in the caller's `Config::validate`: an arena handed an absurd count would simply try to allocate it.

## Borrowing and overlays

Containers can own their bytes or borrow an immutable base with an owned grown tail. The borrowed base must never be resized or cloned merely to append. Handles and metadata must continue to refer to the same logical byte offsets in both modes.

Capacity is governed by the relevant `*_Cfg` and `max_bytes`; return `Error::CapacityExceeded` rather than allowing arithmetic to wrap. Preserve the distinction between a missing key/blob and malformed storage.

## Tests and checks

Important coverage is split across `tests/arena.rs`, `arena_overlay.rs`, `borrowed.rs`, `blob.rs`, `chunk.rs`, `chunk_overlay.rs`, `interner.rs`, `image.rs`, `prop.rs`, `perf_gates.rs`, and optional `serde.rs`.

Useful commands:

```bash
cargo test -p plugmem-arena
cargo test -p plugmem-arena --all-features
cargo run --release -p plugmem-arena --example basic
cargo run --release -p plugmem-arena --example overlay
cargo run --release -p plugmem-arena --example bench_repro -- 100000
cargo bench -p plugmem-arena --bench storage
```

Keep arena measurements separate from full-engine measurements. The reproducibility example compares the containers with standard-library baselines and can also exercise wasm runtimes; record the runtime and corpus size with every result.
