# Local guide: `plugmem-core`

## Role and constraints

`plugmem-core` is the memory engine over `plugmem-arena`. It is `#![no_std]`, uses `alloc`, and has `#![forbid(unsafe_code)]`. Do not introduce `std`, OS I/O, a clock, global state, threads, or unsafe byte casts into this crate. Timestamps are supplied by callers and bytes leave through `Storage`.

The engine is single-threaded at this layer. Concurrency, file locking, mmap, and embedding belong to `plugmem-host`.

## Main modules

- `memory.rs` — `Memory`, remember/revise/forget/link, journal application, stats, and orchestration.
- `memory/recall.rs` — `RecallQuery`, candidate sources, scratch buffers, filtering, fusion, and result rendering data.
- `memory/maintain.rs` — policy-driven no-op, purge/compaction, text reindex and HNSW optimization.
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

Similar-detection compares the new fact's term set against those of the entity's recent facts. It does not recover a candidate's term set by re-reading and re-tokenizing its text: the per-document BM25 record carries a summary of that set (distinct-term count plus one hashed bit per term), and the summary bounds the overlap from above. **Any prefilter here must be a strict upper bound.** The hints are part of the answer, so a bound that can undershoot silently drops a conflict the caller was supposed to see. The property test recomputes the expected hints from the raw Jaccard of the tokenized texts — keep it honest rather than relaxing it, and keep the bound gated on the tokenizer version, since a stale index may hold terms today's tokenizer would not produce.

## Vectors and recall

`Config::dim == 0` disables vector storage. When dimension is non-zero, input `f32` vectors are quantized into the vector pool; replay reconstructs the quantized representation rather than relying on nondeterministic floating-point state.

Flat vector search scans the vector pool. `Config::flat_to_hnsw` selects the regime; the default is 24,000 slots. `remember` does not incrementally rebuild HNSW. Maintenance advances/rebuilds the graph and keeps newer slots in the flat tail when appropriate; `Auto` uses a bounded insertion budget. Any performance statement must identify flat versus HNSW, dimension, `ef`, result count, and whether it measures the complete recall pipeline.

Recall may combine BM25, vector, entity graph, tags, and temporal sources. The final result uses filtering, RRF-style fusion, recency, closed-fact policy, and token budget. A source benchmark is not a mixed recall benchmark. Reuse `RecallScratch` for repeated queries instead of allocating per query.

The lexical scan is O(Σ df) in decodes and is meant to be. What it must not have is a *random* lookup per posting: document lengths come from a flat array, partial scores accumulate by merging sorted runs rather than probing a map, and the admission predicate — which costs a fact-record lookup — is asked only about documents in contention for the top `k` and only after the cheap tag test. The `decoded`, `scored` and `admitted` counters gate that split; treat a change that raises `admitted` above `k` on an unfiltered query as a regression, not a detail.

The flat length array is derived state, capped to a dense id space and trusted only below the first id it declined — a snapshot is untrusted input and nothing range-checks the ids inside its stored records. Ids above that mark are answered from the arena. Keep memory bounds like this one on the *size of the cache*, never on the correctness of the answer.

## Configuration and persistence

`Config` is serialized into the engine image and validated on open. Changes to field order, widths, magic/version, or validation rules are persistence-format changes: update snapshot/journal tests and the host recovery path together.

The five `shards_*` fields are **engine-managed state, not settings**. `memory/shards.rs` derives them from what the database holds; `open` adopts whatever the file records rather than comparing; `maintain` moves them. Three properties keep that safe, and each has a test that fails if you break it:

- **the rule is a pure function of engine state.** The journal records only that a maintenance pass ran, not the layout it chose, so replay recomputes it. A rule that consulted the clock, the caller's config or anything outside the engine would make a replay diverge from the run it replays.
- **the answer must not depend on pointer width.** The products overflow a 32-bit `usize` at populations a database can reach, so the arithmetic is `u64` and the clamp precedes the one cast. A wasm32 host that computed a different layout would replay the same journal into a different file.
- **a pass may only claim the layout it actually built.** `cfg.shards_*` changes next to `install`, never on a path that rebuilds no arena, and every arena sharded by a given count must be rebuilt together — that is why a compaction now also rebuilds `by_name`, which used to ride through untouched. Claiming more leaves the config describing a shape the file does not have, and the loader then reads those arenas wrongly: corruption, not inefficiency.

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

The snapshot and journal parsers are also fuzzed: `fuzz/` holds `cargo-fuzz`
targets that feed both untrusted files arbitrary bytes and then exercise the
accessors that trust the load. Changing the load path, a record layout or the
section set means running them — the load path is what makes this crate's
contract panics sound. See `fuzz/README.md`.
