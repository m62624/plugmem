# plugmem

Embedded, data-agnostic long-term memory engine for LLM applications —
facts, entities and relations with temporal awareness on one side, lexical
and vector recall on the other, in a single in-process store. Think
"SQLite, not a server": you link it, point it at a file, and it works — on
native and inside WebAssembly runtimes.

**Status: early development.** The design is settled (see `specs/` — being
translated to English as the implementation stabilizes) and the storage
foundation is built and measured; the engine layers are landing next.

## Workspace

| Crate | What it is | State |
|---|---|---|
| [`plugmem-arena`](crates/plugmem-arena) | flat byte-pool storage structures (`no_std`, wasm-first) | **done**: tested, measured, documented |
| `plugmem-core` | the engine: data model, indexes, recall, snapshots (`no_std`) | in progress |
| `plugmem-host` | OS-side glue: files, locking, embedder clients | stub |
| `plugmem-cli` / `plugmem-mcp` | command-line and MCP surfaces | stub |
| `plugmem-wasm` | wasm bindings for non-Rust hosts | stub |
| `plugmem-testgen` | deterministic corpus generator for tests and benches | stub |
| `tools/bench-matrix` | dependency-free cross-runtime benchmark runner | done |

## Principles

- **Fast is a number, not an adjective.** Every performance claim is backed
  by a benchmark in-repo; cross-runtime results (native / wasmtime /
  wasmer) are reproducible with one command:
  `cargo run --release -p plugmem-bench-matrix`.
- **The memory image is the snapshot.** State lives in flat byte pools;
  persisting is a `memcpy`, loading is validation plus adoption.
- **Embedded first.** Single-threaded core, `no_std + alloc`, built and
  gated on `wasm32v1-none`; OS concerns stay in thin host layers.
- **Quality rails**: ≥90% line coverage (goal 100%), property tests against
  reference models, miri, clippy across the full feature matrix, and
  deterministic work-counter gates instead of wall-clock CI benchmarks.

## License

MIT.
