# Local guide: `plugmem-core`

## Role and constraints

`plugmem-core` is the memory engine over `plugmem-arena`. It is `#![no_std]`, uses `alloc`, and has `#![forbid(unsafe_code)]`. Do not introduce `std`, OS I/O, a clock, global state, threads, or unsafe byte casts into this crate. Timestamps are supplied by callers and bytes leave through `Storage`.

The engine is single-threaded at this layer. Concurrency, file locking, mmap, and embedding belong to `plugmem-host`.

## Main modules

- `memory.rs` — `Memory`, remember/revise/forget/link, journal application, stats, and orchestration.
- `memory/recall.rs` — `RecallQuery`, candidate sources, scratch buffers, filtering, fusion, and result rendering data.
- `memory/maintain.rs` — purge/compaction and HNSW rebuild policy.
- `memory/persist.rs` — loading and writing the arena-backed engine image.
- `model.rs` — fixed-size fact/entity/edge/temporal records and flags.
- `index/bm25.rs` and `index/postings.rs` — lexical search and postings.
- `index/vecpool.rs` — fixed-stride quantized vectors and flat search.
- `index/hnsw.rs` — graph index, search scratch, validation, and persistence.
- `tokenizer.rs` — Unicode segmentation/normalization without `std`.
- `journal.rs`, `snapshot.rs`, and `storage.rs` — deterministic operations, snapshots, and the storage trait.
- `metadata.rs` — canonical key/value metadata encoding.

## Write semantics

`RememberInput` carries host time, text, optional subject entity, tags, links, an optional vector, validity start, and opaque metadata. `remember` indexes the same fact into all applicable structures and journals the operation. `revise` closes the predecessor and creates a successor; `forget` tombstones a fact until maintenance physically purges it; `link` creates/upserts typed entity edges.

Journal ids are authoritative. Replay must use the same internal apply path as live mutation, remain deterministic, and be safe for a re-applied tail. Do not create a second replay implementation.

Metadata keys are sorted and duplicate keys are rejected/canonicalized before being stored as one opaque blob. The engine does not interpret metadata values.

## Vectors and recall

`Config::dim == 0` disables vector storage. When dimension is non-zero, input `f32` vectors are quantized into the vector pool; replay reconstructs the quantized representation rather than relying on nondeterministic floating-point state.

Flat vector search scans the vector pool. `Config::flat_to_hnsw` selects the regime; the default is 24,000 slots. `remember` does not incrementally rebuild HNSW. Maintenance builds/rebuilds the graph and keeps newer slots in the flat tail when appropriate. Any performance statement must identify flat versus HNSW, dimension, `ef`, result count, and whether it measures the complete recall pipeline.

Recall may combine BM25, vector, entity graph, tags, and temporal sources. The final result uses filtering, RRF-style fusion, recency, closed-fact policy, and token budget. A source benchmark is not a mixed recall benchmark. Reuse `RecallScratch` for repeated queries instead of allocating per query.

## Configuration and persistence

`Config` is serialized into the engine image and validated on open. Changes to field order, widths, magic/version, or validation rules are persistence-format changes: update snapshot/journal tests and the host recovery path together.

The core does not open paths or call fsync. `MemStorage` is useful for unit tests; file-backed behavior must be tested through `plugmem-host`.

## Tests and checks

Run focused tests when changing a subsystem, then the full package:

```bash
cargo test -p plugmem-core
cargo test -p plugmem-core --all-features
cargo test -p plugmem-core --test zero_alloc
cargo run --release -p plugmem-core --example bench_ops
cargo run --release -p plugmem-core --example measure_remember
cargo run --release -p plugmem-core --example recall_quality
cargo bench -p plugmem-core --bench engine
```

Relevant regression suites include `journal*`, `snapshot`, `persist`, `maintain`, `recall`, `vectors`, `hnsw_engine`, `metadata`, `tokenizer`, `zero_alloc`, and `perf_gates`. Do not relax `#![forbid(unsafe_code)]` or bypass zero-allocation/performance gates to make a test pass.
