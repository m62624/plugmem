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
