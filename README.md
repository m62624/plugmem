# plugmem

Temporal memory for LLM agents, as an embeddable library. plugmem stores
short **facts** — with a subject entity, tags, an optional embedding and
two time axes — and answers a query with a ranked, token-budgeted block
ready to paste into a prompt. It runs in-process, keeps a whole database
in one snapshot file plus an append-only journal, and needs no server and
no disk of its own — the same engine runs on native, in WebAssembly, and
in the browser.

It is **not** a vector database. A vector is one of four recall sources —
lexical (BM25), vector, entity graph and time — fused with reciprocal-rank
fusion; tags act as filters. What plugmem is actually about:

- **bitemporal facts** — `revise`/`forget`, "what was true *then*" vs now,
  revision chains kept intact, physical erasure on `maintain`;
- **an entity graph** — typed edges between entities, relational knowledge,
  not just nearest-neighbor vectors;
- **conflict surfacing on `remember`** — a new fact comes back with the
  live facts it may duplicate or contradict; the engine never merges on
  its own, the caller decides;
- **a compact rendered block** — the result is selected greedily under a
  token budget, ready for the prompt;
- **embeddable everywhere** — single-threaded `no_std + alloc` core,
  zero-allocation recall after warm-up, one file, no server; built and
  tested on `wasm32v1-none` and a real 32-bit wasm runtime in CI.

**Status: early development.** The design is settled (see `specs/`) and the
storage, engine and host layers are built and tested; the snapshot format
is not frozen (pre-1.0).

## Which crate do I need?

- **An embedded database** — `plugmem-core`. The engine itself: data model,
  indexes, recall, snapshot/journal. `no_std + alloc`, no I/O, no clock, no
  threads. Bring your own storage (a browser, a wasm host, your own file
  layer) through a small `Storage` trait.
- **The "point it at a file" experience** — `plugmem-host`. Adds the OS
  side (`std`): a file-backed store with an exclusive lock, fsync and
  auto-snapshot policy, a read-only mmap open, and embedding providers over
  HTTP. Re-exports the engine, so this crate alone is enough.

| Crate | What it is | State |
|---|---|---|
| [`plugmem-arena`](crates/plugmem-arena) | flat byte-pool storage structures (`no_std`, wasm-first) | done: tested, measured |
| [`plugmem-core`](crates/plugmem-core) | the engine: data model, indexes, recall, snapshots (`no_std`) | done |
| [`plugmem-host`](crates/plugmem-host) | OS glue: files, locking, mmap read-only, embedder clients (`std`) | done |
| [`plugmem-wasm`](crates/plugmem-wasm) | wasm bindings for non-Rust hosts | in progress |
| `plugmem-cli` / `plugmem-mcp` | command-line and MCP surfaces | in progress |
| `plugmem-testgen` | deterministic corpus generator for tests and benches | done |

## Principles

- **Fast is a number.** Every performance claim has a benchmark in-repo;
  cross-runtime results (native / wasmtime / wasmer) reproduce with one
  command.
- **The memory image is the file format.** State lives in flat byte pools;
  a snapshot is a `memcpy`, loading is validation plus adoption, and the
  same file opens byte-identically on native, wasm32 and wasm64.
- **Embedded first.** Single-threaded `no_std + alloc` core; OS concerns
  stay in the thin `plugmem-host` layer.
- **Untrusted input never panics.** Arbitrary bytes can produce any typed
  `Error`, never a panic or UB; after a load every stored id is
  range-checked.

## License

MIT.
