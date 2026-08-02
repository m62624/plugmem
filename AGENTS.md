# AGENTS.md

## Project overview

`plugmem` is a Rust embedded memory and knowledge engine. It stores facts, temporal state, tags, entities, relationships, text indexes, and vectors. The core is designed to be usable in `no_std` environments; the host layer adds filesystem persistence, journals, snapshots, memory mapping, recovery, and embedding integration.

The main dependency flow is:

```text
plugmem-arena -> plugmem-core -> plugmem-host -> plugmem-cli / plugmem-mcp / plugmem-napi
                         ^
                 plugmem-testgen and benchmark tools
```

## Workspace crates

### `crates/plugmem-arena`

Low-level `no_std` storage primitives:

- flat byte arenas and handle-based access;
- `Arena`, `BlobHeap`, `ChunkPool`, and `Interner`;
- packed byte/string storage;
- mmap overlays, copy-on-write behavior, and read-only views.

Arena benchmarks describe the storage primitive itself. Do not present them as measurements of the complete memory engine.

### `crates/plugmem-core`

The main `no_std` engine. It contains:

- `Memory` and the remember/recall orchestration;
- facts, `FactId`, bitemporal state, intervals, and closing facts;
- entities, edges, and graph traversal;
- tags and inverted indexes;
- tokenization and BM25/full-text search;
- fixed-stride quantized vector storage;
- flat vector search and optional HNSW indexing;
- temporal filters, RRF fusion, recency, and token budgets;
- journal operation encoding and snapshots.

Important source areas include:

- `src/lib.rs` — public API and core types;
- `src/memory.rs` — write and recall orchestration;
- `src/vector.rs` — vector storage, signatures, flat search, and HNSW;
- `src/search.rs`, `src/bm25.rs`, `src/temporal.rs`, and `src/graph.rs` — recall components;
- `src/journal.rs` and `src/snapshot.rs` — persistence formats;
- `examples/` — executable examples and focused experiments.

For vector storage, the slot stride is the fixed header plus signature bytes plus the quantized vector dimension. With dimension 384, the raw vector slot is 440 bytes before other engine structures. Flat search scans all slots, so its latency is primarily controlled by the number of stored vectors rather than by the requested result count. HNSW is built by maintenance/configuration paths; `remember` does not automatically rebuild the graph on every write.

### `crates/plugmem-host`

The `std` integration layer:

- filesystem and in-memory storage implementations;
- immutable generation snapshots and manifests;
- append-only journals, locks, and recovery;
- mmap read-only and overlay access;
- embedders and host-level maintenance.

The host wrapper can create embeddings when an embedder is configured. Core vector behavior should be tested separately from embedding model latency and external process overhead.

### `crates/plugmem-cli`

Command-line interface over the host/core APIs. It provides database opening, remember/recall, maintenance, snapshot, recovery, and diagnostic commands.

### `crates/plugmem-mcp`

MCP adapter/server. It exposes memory operations through the host API and should not duplicate storage logic.

### `crates/plugmem-napi`

Node.js N-API bindings. Treat it as a boundary layer; measure FFI overhead separately from core and host performance.

### `crates/plugmem-testgen`

Utilities for generating deterministic test data and scenarios. It is not part of the production storage path.

### `tools/bench-matrix`

Benchmark matrix runner and result collection. Results must identify the component, data shape, configuration, and units being measured.

### `tools/bench-charts`

TSV-to-SVG benchmark chart generator. Treat `baseline.tsv` as checked-in measurement data, not as an automatically verified source of truth. Check units, labels, duplicate rows, and ordering when changing it.

### `tools/wasm-probe`

Wasm build and compatibility probe for the core.

## Main data paths

The write path is broadly:

```text
remember(input)
  -> apply_remember
  -> tokenize/intern, BM25, tags/entities, vectors, and temporal indexes
  -> journal operation
```

The recall path is broadly:

```text
RecallQuery
  -> tag/entity/temporal filters
  -> BM25 candidates and/or flat vector/HNSW candidates
  -> graph candidates
  -> RRF fusion, recency, and token budget
```

`RecallQuery` supports text, vectors, tags, entities, temporal constraints, result count, token budget, closed-fact inclusion, and HNSW search parameters. A vector-only benchmark is not equivalent to a mixed text/vector/graph recall.

## Verification commands

Use release mode for performance work and record the data shape, configuration, hardware, compiler version, warm-up policy, repetitions, units, and correctness checks.

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
git diff --check
git status --short
```

For a targeted package or example, prefer a focused command first, then run the broader workspace checks when practical. If a check is skipped, state that explicitly in the handoff.

## Engineering rules

- Read the relevant crate and its tests before changing behavior.
- Preserve existing user changes; inspect `git status` and `git diff` before editing.
- Keep arena measurements separate from full engine and host measurements.
- Keep embedding/model latency separate from storage and recall latency.
- Verify benchmark units and conversions (`ns`, `us`, `ms`, per-item, total time, and throughput).
- Do not infer persistent memory usage from peak allocator activity without checking retained storage, temporary allocations, journal bytes, mmap mappings, and process RSS.
- Do not compare against another database without matching data, distance metric, index type, recall target, persistence mode, concurrency, and hardware.
- Use `apply_patch` for manual file edits.
- Avoid destructive commands such as `git reset --hard`, broad deletion, or overwriting unrelated work unless explicitly requested.
