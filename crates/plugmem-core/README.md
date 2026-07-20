# plugmem-core

`plugmem-core` is an embedded **temporal-memory engine for LLM agents**,
as a library inside your process — no server, no cloud. An agent talks to
it in four verbs — `remember / recall / revise / forget` — and it answers
with a ranked, token-budgeted context block ready to paste into a prompt.
It keeps a whole database in one snapshot file plus an append-only
journal; storage is flat byte arenas, so the memory image *is* the file
format (loading is a bounds-check plus adopt, replay is deterministic to
the byte, and the same file opens on native, wasm32 and wasm64 unchanged).

It is **not** a vector database. A vector is one of four recall sources,
fused with rank-based scoring:

- **Lexical** — [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) with the
  Robertson idf over a Unicode
  ([UAX #29](https://unicode.org/reports/tr29/)) tokenizer;
- **Vector** — symmetric **int8-quantized** cosine search, a flat
  two-phase scan below a threshold and an
  [HNSW](https://arxiv.org/abs/1603.09320) graph above it;
- **Graph** — an entity graph with typed edges, expanded breadth-first
  from query anchors;
- **Temporal** — range scans over a `recorded_at`-ordered index;

the four are merged by [reciprocal-rank
fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) with a recency
boost (tags act as filters, not a source). On top of that: **bitemporal
facts** (`revise`/`forget`, "what was true *then*", revision chains,
physical erasure), and **conflict surfacing** — a new `remember` returns
the live facts it may duplicate or contradict, and the engine never merges
on its own; the caller decides.

**This crate is the engine itself:** `no_std + alloc`, zero I/O, no clock,
no threads. Bytes enter and leave through a five-method `Storage` trait,
timestamps arrive as parameters, and embeddings are computed by the caller
— which is what lets the same engine run natively, in `wasm32v1-none`, or
anywhere else Rust compiles.

## Which crate do you need?

| You want | Depend on |
|---|---|
| point it at a file and go — OS locking, fsync and auto-snapshot policy, a zero-copy read-only mmap open, automatic embeddings over HTTP (OpenAI/Ollama/LM Studio/vLLM/llama.cpp) | [`plugmem-host`](../plugmem-host) (`std`; re-exports this engine) |
| the engine alone with your own storage (a browser, a wasm host, custom persistence), `no_std`, full control | **this crate** |
| the flat byte structures underneath (sorted page arenas, blob heap, chunk pool, interner) for your own storage project | [`plugmem-arena`](../plugmem-arena) (`no_std`) |
| no Rust at all: a CLI, an MCP server for agents, an npm package | `plugmem-cli` / `plugmem-mcp` / `plugmem-wasm` — in progress, not published yet |

## Who this is for

Agents and applications that need a **personal memory**: tens of
thousands to a million facts about a user, a project, a codebase — with
temporal reasoning ("what was true then"), revision history, a
relationship graph, and hybrid retrieval, all inside the process and
inside a single file. It is not a horizontally scalable search cluster
and does not try to be one; the capacity passport is deliberately sized
to the 32-bit wasm address space (≤ 2 GiB, design center 100k facts,
ceiling 1M). On 64-bit hosts — native or WebAssembly 3.0
[memory64](https://github.com/WebAssembly/memory64) — the same code and
the same file format carry larger limits; see
[Targets and WebAssembly](#features-and-targets).

## Quick start

```rust
use plugmem_core::{Config, MemStorage, Memory, RecallQuery, RememberInput};

let mut store = MemStorage::new(); // file-backed storage lives in plugmem-host
let mut mem = Memory::new(Config::default()).unwrap();

let out = mem.remember(&mut store, RememberInput {
    entity: Some("user"),
    tags: &["pref"],
    ..RememberInput::text(1_784_000_000_000, "prefers tokio with pinned versions")
}).unwrap();
// out.similar lists live facts this one may duplicate or contradict —
// the engine never merges on its own; the caller decides.

let res = mem.recall(RecallQuery::text(1_784_000_100_000, "which runtime?")).unwrap();
println!("{}", res.rendered); // a compact, ranked block for the prompt

mem.snapshot(&mut store, 1_784_000_200_000).unwrap(); // full image + journal reset
```

## Data model

A **fact** is one short statement with an optional subject **entity**,
tags, an optional embedding, and two time axes (a simplified bitemporal
model):

- `recorded_at` — when the memory learned it (immutable);
- `valid_from / valid_to` — when it was/is true.

`revise` closes the old fact's validity interval and records the
successor; the old version is kept — "lived in Moscow (2023 → 2025)"
stays answerable through `as_of` queries. `forget` tombstones a fact
immediately; the next `maintain` removes it physically, and its id is
burned, never reissued. Entities form a graph through typed **edges**
(`works_at`, `depends_on`, …) with a provenance link back to the fact
that justified them.

| Verb | Effect |
|---|---|
| `remember` | new fact + indexes + similar-fact hints (Jaccard term overlap, vector cosine) |
| `recall` | hybrid ranked retrieval, zero allocations after warm-up |
| `revise` | close the predecessor, record the successor, keep the chain |
| `forget` | immediate tombstone; physical purge at `maintain` |
| `link` | upsert a typed edge between entities |
| `maintain` | the one O(base) verb: purge, compaction, index rebuilds, HNSW build |
| `snapshot` | full image + journal reset |

## Retrieval

Four sources feed one ranked result:

- **Lexical** — BM25 with the Robertson idf over delta-encoded
  ([LEB128](https://en.wikipedia.org/wiki/LEB128)) posting lists; the
  tokenizer does NFKC normalization,
  [UAX #29](https://unicode.org/reports/tr29/) word segmentation, Latin
  diacritic folding and CJK bigrams. A stop-frequency guard drops query
  terms whose posting lists would dominate the cost.
- **Vector** — embeddings are stored as symmetric int8 quantizations of
  the L2-normalized vector (f32 is never persisted). Below a configured
  threshold, search is a two-phase flat scan: a Hamming prefilter over
  1-bit sign signatures, then an exact quantized-cosine rescore of the
  best candidates. Above the threshold, `maintain` builds an
  [HNSW](https://arxiv.org/abs/1603.09320) graph (Malkov & Yashunin)
  with the neighbor-selection heuristic and early-stopped beam search;
  vectors added since the last build sit in a flat tail that is scanned
  exactly and merged.
- **Graph** — bounded breadth-first expansion from entity anchors over
  the edge arenas, with hard budgets on entities, edges, candidates and
  examined posting entries (a hub entity cannot blow the query up).
- **Temporal** — range scans over a `recorded_at`-ordered index.

Sources are fused with [reciprocal rank
fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) (Cormack, Clarke
& Buettcher) — rank-based, so the sources need no score calibration —
plus an exponential recency boost. Selection is greedy under `k` and a
token budget, and the result includes both structured facts and a
rendered block ready to paste into a prompt.

## Storage and durability

All state lives in flat byte structures (sorted page arenas, blob heaps,
chunked lists, an interner) from `plugmem-arena`. The consequence: **the
memory image is the file format**. A snapshot is the concatenation of
each structure's sections, 64-byte aligned, with
[xxh3](https://github.com/Cyan4973/xxHash) checksums per section and
over the whole file; loading is bounds-checking the metadata and
adopting the bytes — no per-record parsing.

Between snapshots, every mutation appends one framed record to a
journal. Replay is deterministic to the byte: quantization is a pure
function, HNSW levels are a pure function of the fact id, and `maintain`
re-executes identically — so `snapshot → crash → replay` and the
uninterrupted engine produce the same file. A torn journal tail (crash
mid-append) is detected and dropped; any other inconsistency is a typed
error. Snapshots are canonical: save → load → save is byte-identical.

The loader treats every input as untrusted: arbitrary bytes can produce
any `Error` but never a panic or undefined behavior, and after a
successful load every stored id is range-checked, every chunk chain
walked, every invariant the hot path relies on re-established.

Each database carries a `db_uuid` (minted by the host at creation) so
external holders of fact ids can tell "same database" from "a different
one".

## Performance

Deterministic work counters (`cmp_ops`, `postings_decoded`,
`dist_evals`, allocation counts — behind the `counters` feature) act as
CI gates: a complexity regression fails the same way on any machine.
`recall` and `get` perform **zero allocator calls** after warm-up,
enforced by a counting-allocator test.

Per-source recall latency (single thread, native). The chart is rendered
by [`plugmem-bench-charts`](../../tools/bench-charts) from the
`bench_ops` example's output — the same plotters pipeline as the arena
charts:

![recall source latency](assets/recall-latency.svg)

The composite recall paths and the write side, which the chart does not
break out:

| Operation (single thread, native) | Latency |
|---|---|
| tags + time-range recall @ 100k | ~230 µs |
| hybrid recall (text + hub entity anchor) @ 100k | ~470 µs |
| `remember` (tokenize, index, quantize d384, similar-detect, journal) | ~72 µs mean |
| one-time HNSW build inside `maintain` | ~1.6 ms/vector |

Reproduce: `cargo bench -p plugmem-core` (Criterion; a separate target,
never run under `cargo test`) for the full statistical suite, or
`cargo run --release -p plugmem-core --example bench_ops` for the chart's
`#TSV` rows. Corpora come from `plugmem-testgen` — seeded, so every run
measures the same workload.

## Limits, stated plainly

- Single-threaded, single-writer. Concurrency belongs to the embedding
  process (the [`plugmem-host`](../plugmem-host) crate serializes access
  to a file).
- ≤ 2 GiB of state by default (the 32-bit wasm budget); ids are `u32`;
  texts ≤ 4 KiB; dimensions ≤ 4096. On 64-bit builds `max_bytes` may be
  raised past 4 GiB — such a database then opens only on 64-bit hosts
  (a typed error, not corruption, on 32-bit ones).
- Vector search is quantized (int8) — exact f32 scores are never
  computed — and approximate above the HNSW threshold (recall@10 ≥ 0.9
  against brute force is a test gate, not a proof).
- The tokenizer does no stemming or lemmatization.
- `maintain` is O(database) and the first one past the HNSW threshold
  pays the graph build; call it on your schedule, not on a hot path.
- The snapshot format is not yet frozen (pre-1.0): a new version may
  require re-importing, not migrating.

## Features and targets

The crate is `no_std + alloc` unconditionally — there is no `std` feature
(it builds and is gated on `wasm32v1-none` in CI).

- `counters` — the deterministic work counters; zero cost when off.

### WebAssembly 2.0 and 3.0

One source, one file format, two address-space classes:

- **wasm32** (WebAssembly 2.0 class; `wasm32v1-none`,
  `wasm32-unknown-unknown`) — the default and the portability baseline:
  runs in every engine and browser. The full contract test suite runs on
  a real 32-bit target:
  `cargo test -p plugmem-core --target wasm32-wasip1` under wasmtime.
- **wasm64** (WebAssembly 3.0
  [memory64](https://github.com/WebAssembly/memory64)) — lifts the 4 GiB
  linear-memory ceiling. The crate builds for `wasm64-unknown-unknown`
  unchanged (nightly `-Zbuild-std=core,alloc`; the target is tier 3).
  Engine support today: wasmtime and 3.0-era browsers run it; wasmer
  does not yet.

Snapshots are pointer-width independent by construction — every codec
writes fixed-width little-endian fields — and this is verified, not
assumed: the same scenario produces **byte-identical snapshots** on
native x86-64, wasm32 and wasm64 (and with `+simd128` enabled). A
database is portable across all of them as long as its configured
limits fit the host's address space; "migrating" from a 32-bit to a
64-bit deployment is opening the same file. Determinism is preserved on
purpose: the engine stays inside the Wasm 3.0 *deterministic profile*
(integer distances, total-order float comparisons, no relaxed SIMD in
any state-affecting path).

Crates are published to crates.io under these names once the format
freezes; until then, use the git repository.

## License

MIT.
