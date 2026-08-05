# 00 — Overview

The entry point to every spec. Where this overview and a topic spec disagree,
the topic spec wins.

## What it is

**plugmem** is an embedded bitemporal memory and retrieval engine for
local-first applications and agents. The core is a pure library with no server
and no I/O assumptions (the SQLite model); everything else is a thin wrapper in
one workspace. The store is data-agnostic (it holds opaque bytes), and the API
offers `remember / recall / revise / forget`, built-in bitemporality, an entity
graph, structured ranked results, and an optional compact rendered block for
prompt or other bounded-context consumers.

The same engine ships four ways:

1. **Binaries** (the `plugmem-cli` CLI, the `plugmem-mcp` server) — installed via
   cargo-dist (shell/PowerShell/`.msi`/Homebrew + `cargo binstall`).
2. **Rust API** (crates.io) — embed the library directly.
3. **npm package** (`plugmem`) — a native Node.js addon (napi-rs) that embeds the
   host in-process, with TypeScript types.
4. **PyPI package** (`plugmem`) — a CPython extension module (PyO3) that embeds
   the host in-process, with generated type stubs.

The core is `no_std`, so it also builds and runs on WebAssembly; wasm is a
portability target for the core (and the way the snapshot format is proven
pointer-width independent), **not** the npm distribution mechanism.

## Principles

Violating any of these is grounds to stop a review.

1. **Embedded-first.** The core is a library. A server, if one ever appears, is a
   wrapper.
2. **`no_std + alloc` core.** The library crates build for `wasm32v1-none` (checked
   in CI). The core knows nothing of files, threads, network, or **time** —
   timestamps are passed in by the host on every call.
3. **Flat arenas.** All state is byte pools plus small tables (no `Box`/`Rc`/`HashMap`
   in the data). Consequence: **the in-memory image *is* the snapshot format**, so a
   load is a `memcpy` plus validation.
4. **Single-threaded, no compromise.** No background threads in the core; all upkeep
   is an explicit `maintain()`. Concurrency is a wrapper option, not a foundation.
   SIMD yes (including wasm simd128); threads no.
5. **Translators live outside the core.** Fact extraction is the calling agent (per
   `SKILL.md`); embeddings are an `Embedder` trait in the host. The core takes ready
   vectors. Without an embedder the system is still complete (BM25 + tags + graph +
   time).
6. **A performance contract, not a slogan.** Budgets are enforced by deterministic
   work counters as CI gates, with full coverage of the library crates.

## Workspace

| Crate | std? | Role |
|---|---|---|
| `plugmem-arena` | no_std | Flat structures: sharded byte arena, blob heap, chunked list, string interner. The substrate. |
| `plugmem-core` | no_std | The engine: data model, indexes (BM25, time, graph, vectors), hybrid recall, snapshot/journal, the API verbs, the `Storage` trait. |
| `plugmem-host` | std | Native host layer: `FileStorage`, the `Embedder` trait and its clients, read-only mmap, locking, cross-process concurrency. |
| `plugmem-cli` | std | The `plugmem-cli` binary: commands over core + host. |
| `plugmem-mcp` | std | The MCP server (stdio JSON-RPC) over core + host; the skill is embedded via `include_str!`. |
| `plugmem-napi` | std | The native Node addon (napi-rs) over host; the `Plugmem` class mirrors `Database`; the npm package is `plugmem`. `publish = false` (ships to npm, not crates.io). |
| `plugmem-py` | std | The CPython extension (PyO3) over host; its surface mirrors `plugmem-napi` rather than host directly (see `07-wrappers.md`); the PyPI package is `plugmem`. `publish = false` (ships to PyPI, not crates.io). |
| `plugmem-testgen` | std | Internal: a deterministic corpus generator for tests and benches. `publish = false`. |

Edition 2024, MIT. All crate versions are one number, inherited from
`[workspace.package]` and bumped together. Library crates set `dist = false`;
only the two binaries are released by cargo-dist.

**Field inheritance.** `version`, `edition`, `authors`, `license`, `repository`
and `homepage` come from `[workspace.package]`; each crate sets only its own
`name`, `description`, `publish`, `[lib] crate-type` and `[[bin]]`.

**Dependencies go through `[workspace.dependencies]`.** Versions are pinned once in
the root; a crate writes `name = { workspace = true }` and may only narrow features,
never set a version. `plugmem-arena` and `plugmem-core` are pulled with
`default-features = false`; the consumer turns on `std`, so the
`--no-default-features --target wasm32v1-none` gate always reflects the real
no_std slice.

**Features.** `std` (default on arena/core; off = pure `no_std + alloc`) and
`counters` (deterministic work counters, zero-cost when off) propagate down the
chain (`plugmem-core/std = ["plugmem-arena/std"]`). The host adds `serde`, `config`
and `counters` as its own axes.

**Quality gates, from the first commit.** `missing_docs = "deny"` (plus
`unsafe_op_in_unsafe_fn` and `rustdoc::broken_intra_doc_links`) is set in
`[workspace.lints]` and inherited by every crate — a missing doc comment fails the
build. Code, rustdoc and these specs are English-only. Coverage runs through
`tarpaulin.toml`, scoped to the library crates (binaries and the napi bridge are
excluded, with the reason in the config comments).

## Capacity and performance contract

The ceiling is dictated by wasm32: linear memory ≤ 4 GiB, our budget **≤ 2 GiB**.
64-bit builds (native and Wasm 3.0 memory64) carry the same format bytes with
larger limits; the capacity classes and their proofs live in `09-portability.md`.

| Parameter | Value |
|---|---|
| Design center | **100k facts** — everything is instant |
| Guaranteed ceiling | **1M facts on wasm32** (with quantized vectors) |
| Cost per fact | ~0.36 KB without vectors; ~0.82 KB (+i8 384d); ~1.2 KB (+i8 768d) |
| Snapshot | ≈ the in-RAM image size (they are the same bytes) |
| `recall` @100k | < 1 ms worst case, hundreds of µs typical |
| `remember` | < 500 µs including full similar-detection; excludes embedding compute |
| Cold start | file read time + < 5 ms init |
| Allocations in recall | **0** (an invariant, checked by a test) |
| Vectors | i8 quantization is the default; binary signatures pre-filter; f32 is not stored |
| Flat → HNSW | threshold ~24k vectors (tuned by a bench) |

10M+ facts is out of scope for v1 (the territory of a future native server).

## Spec map

| # | File | What it fixes |
|---|---|---|
| 00 | `00-overview.md` | this document, plus the workspace layout |
| 01 | `01-arena.md` | `plugmem-arena`: arena, blob heap, chunked list, interner |
| 02 | `02-data-model.md` | ids, slot layouts, temporality, revisions, invariants, physical deletion |
| 03 | `03-snapshot.md` | snapshot format, journal, the `Storage` trait, corrupt-input safety, read-only mmap |
| 04 | `04-recall.md` | tokenizer, BM25, time, graph, vectors, HNSW, hybrid recall |
| 05 | `05-api.md` | the core public API, `Config`, `Embedder`, verb semantics |
| 06 | `06-host.md` | `plugmem-host`: `FileStorage`, `Database`, embedders, concurrency |
| 07 | `07-wrappers.md` | CLI, MCP, napi, `SKILL.md`, delivery |
| 08 | `08-performance.md` | budgets, counters, benches, coverage, fuzz/miri, dependency pins, CI gates |
| 09 | `09-portability.md` | WebAssembly 2.0/3.0: memory64, capacity classes, cross-target equivalence |
| 10 | `10-workspace.md` | the optional workspace: many memories in one directory, named and described |
| 11 | `11-file-format.md` | the normative byte layout: on-disk files, container, section catalogue, record slots, journal |

## Design choices

- The extractor is the calling agent; the engine makes no LLM calls. Zep/Graphiti-class
  graph memory is reached mechanically (bitemporality, conflict hints on `remember`);
  the judgement stays with the agent.
- One store, several indexes. "Vector + graph" is not two services but two entry
  points into one fact graph.
- No serialization: a snapshot is the image of the arenas.
- The embedder and the LLM work together **outside** the core; on wasm both are host
  callbacks.
- Sharding / multi-instance is not built into the core: at the capacity above it does
  not pay (orchestration costs more than the work). The doors are left in the wrappers —
  read replicas over the zero-copy snapshot, and rayon behind a feature flag for batch
  work (`maintain`, embeddings).
