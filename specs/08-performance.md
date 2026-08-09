# 08 — Performance, quality, dependencies and CI

"Fast" is a number in a table and a gate in CI, not an adjective in a README. This
covers budgets, how they are measured, the coverage mandate, fuzz/miri, the pinned
dependencies, and the CI.

## Latency budgets (contract)

Corpora: S = 10k facts, M = 100k (design center), L = 1M (the wasm32 ceiling).
Single-threaded; native = a modern x86-64 desktop, wasm = wasmtime on the same machine.
p50 / p99 after warm-up.

| Operation | M native | M wasm | L native | Note |
|---|---|---|---|---|
| `remember` (no embed) | < 500 µs / 2 ms | < 1.5 ms | < 1 ms | includes full similar-detection and the journal without fsync; `skip_similar` < 100 µs |
| `recall` structural | < 600 µs / 1 ms | < 1.5 ms | < 1 ms | k=8, budget 512 |
| `recall` + vector (Flat) | < 500 µs | < 1.5 ms | — | at/below threshold |
| `recall` + vector (HNSW) | < 500 µs | < 1.5 ms | < 1 ms | ef=64 |
| `get(id)` | < 2 µs | < 5 µs | < 2 µs | |
| `list_tags` | bounded by page + overlay runs | same | same | default 64, hard max 256; no fact scan |
| cold open | file read + < 5 ms | memcpy + < 5 ms | + < 20 ms | validation is O(metadata) |
| `snapshot()` | < 50 ms | < 100 ms | < 500 ms | no fsync |
| `maintain` full | < 1 s | < 3 s | < 10 s | the only O(database) operation |

Memory budget: the "RAM image / useful data" inflation ≤ 1.6, checked by a test on
corpus M. Representative measured numbers (native, testgen corpus): structural recall
@100k ~70 µs (a Zipf hub anchor still makes the graph source spend its full
`GRAPH_EXAMINE_CAP` of 2048 postings — the price of the declared caps, not a complexity
regression); tags+range @100k ~17 µs; flat 24k×384 k=8 ~267 µs; HNSW 30k×384 ef64
~3.3 ms; BM25 3-terms-of-10k ~8.5 µs. Ceilings only tighten; loosening one is a
deliberate edit to this file.

Those figures come from one machine and are not comparable to earlier editions of
this file: the vector and graph rows moved with the hardware, not with the code,
which is why only rows measured in the same session may be compared. The lexical
and tag rows moved with the code — see `04-recall.md`.

### What the derived shard layout cost and bought

Measured as an A/B on one machine (the only comparison worth making — a committed
baseline drifts between machines and between days), 100k operations, base commit
against the branch that made the layout follow the data:

| | base | derived layout | |
|---|---:|---:|---|
| pool after `maintain` | 66.4 MB | 44.5 MB | **−33 %** |
| snapshot on disk | 71.8 MB | 47.1 MB | **−34 %** |
| reopen | 18.1 ms | 14.5 ms | −20 % |
| read-only open | 18.3 ms | 13.4 ms | −27 % |
| `maintain` | 214 ms | 197 ms | −8 % |
| streaming load | 66,618 ops/s | 60,589 ops/s | **−9 %** |
| `recall` text-only p50 | 24.1 µs | 25.3 µs | +5 % |
| `recall` entity p50 | 68.0 µs | 76.9 µs | +13 % |

The load figure is the honest cost and it is a **cold-import** cost: that run
starts empty and climbs through the size classes, paying a rebuild at each
crossing. A database already at its layout pays none of it. The gains are the
other way round — they are permanent, and at the sizes a personal memory
actually is they are far larger than 33 %: a thousand facts went from 14.1 MB
to 0.43 MB.

The recall rows are the real trade. Fewer shards mean deeper per-shard page
directories, and a few extra cache-friendly compares per lookup show up as a
few percent. That is the price of not paying a million-fact floor on a
thousand-fact database.

The **first `maintain` past the HNSW threshold is out of the < 1 s budget on purpose**:
the graph build is ~1.6 ms/vector, done once when `flat_to_hnsw` is crossed (or > 10%
dead nodes); steady-state maintain carries the graph through a remap (O(edges),
milliseconds) plus a tail insert. Wrapper policy must know "the first maintain past the
threshold is long".

## Zero-allocation invariant

`recall` and `get` after warm-up: **0 allocator calls** (scratch is reused; scratch
grows only on new size maxima). Mechanism: a global counting allocator in the test
harness, a reference scenario on corpus M, assert 0. `remember`: ≤ 8 allocations
(amortized pool growth is counted separately).

Tag discovery does not add a string table. Its checkpointed base is one compact
8-byte `(TermId, count)` record per active tag; names remain in the common
interner. Writes update a small sorted buffer (at most 64 entries), then merge it
into binary-counter runs. This makes catalogue maintenance amortized logarithmic
in the number of tags changed since the checkpoint rather than O(all tags) per
fact. `list_tags` merges the base and those few runs until its bounded page is
full. `remove_tag` is intentionally O(affected facts): preserving temporal
history requires one successor per affected current fact.

## Deterministic work counters (CI gates)

Wall-clock lies in CI; the gates are on counters (feature `counters`, off in release
benches): `cmp_ops`, `bytes_shifted`, `pages_walked`, `splits` (arena);
`postings_decoded` (BM25); `dist_evals`, `sig_words_scanned` (vectors); `edges_walked`,
`entities_visited` (graph); `alloc_count`, `alloc_bytes` (the test allocator). The gate
tests use a fixed-seed testgen corpus → reference operations → assert "counter ≤
ceiling". Ceilings are set at first implementation with a ×1.2 margin and then guard
complexity: an accidentally quadratic pass fails the PR identically on any hardware.

## Corpus generator (`plugmem-testgen`)

Deterministic by seed (no unseeded randomness in tests — a repo-wide law). Parameters:
N facts; a 30k-term vocabulary, Zipf s≈1.07; text length normal μ=150; entities ~5% of
N with a power-law popularity (hubs exist!); 1–4 tags from a Zipf pool of 500; ~3
edges/fact; a time axis over two years, denser toward "today"; revisions on 10% of facts
(chains of 2–4); vectors from 64 Gaussian clusters on the unit sphere plus noise (it
exercises both recall semantics and quantization). API: `Corpus::generate(seed, size)
-> impl Iterator<Item = Op>` — operations, run through the public API.

## Coverage and correctness — mandate

- **Coverage**: the target is 100% executable lines for `plugmem-arena` and
  `plugmem-core`; the **hard gate is ≥ 90% both for Rust core and for the
  aggregate measured Rust workspace**. Tarpaulin
  mis-attributes some non-executable lines (struct-literal fields, macro-call
  lines, const-folded branches) — those are audited by hand in review (that the
  surrounding statement is exercised), never dismissed silently. N-API and
  Python are exercised through their real Node/CPython runtimes, including
  event-loop/GIL progress tests; they do not carry a synthetic line-coverage
  percentage. Always pair the suites with every language's formatter/linter and
  `cargo clippy` with zero warnings before committing.
- **Property tests (proptest)** — per structure, equivalence to a reference model
  (listed in specs 01–04).
- **miri** — all of plugmem-arena plus the core unsafe paths, on any PR touching unsafe.
  Bulk shift stress tests are `#[cfg_attr(miri, ignore)]` (the UB paths are covered by
  small tests of the same branches); proptest is also miri-ignored (its harness touches
  the OS, which miri's isolation forbids).
- **Fuzz (cargo-fuzz, nightly)**: the snapshot loader, journal replay, the tokenizer,
  RecallQuery parsing. The loader oracle is **no panic at load or on later access**.
  Load is always structural + lazy (trust/sparse default): structural damage (header,
  section table, ref ranges) → `Err` at load; content damage (text/vec bytes) passes
  load, then a full access sweep (`get` all facts, `recall`, vector search,
  `snapshot_bytes`, `verify`, `scrub`) must finish without panic. Two on-demand checks
  catch latent damage: `verify()` (content — text UTF-8, fact↔slot bijection, metadata
  — **and the graph's cross-references**: both edge mirrors, a current edge against its
  open version, and every open version reachable as a current edge) and `scrub()` (byte
  — per-section + file_hash xxh3), each → `Corrupt` naming the bad section. Any
  panic/UB from fuzz is a P0 bug.

  The graph checks live in `verify` rather than in the loader deliberately: each is a
  random lookup per edge, which on a million-record graph is most of an open, and none
  of them is a bounds check. **A successful open means nothing in the image can make a
  read unsafe — it does not mean the graph agrees with itself.** Rendering therefore
  skips an edge whose endpoints do not resolve instead of unwrapping them.
- **Cross-target**: the whole core test suite runs natively and under wasmtime;
  `cargo build --target wasm32v1-none` for the libraries is a gate.

## Dependencies

The core (`plugmem-arena`, `plugmem-core`) must live in `wasm32v1-none`; every core
dependency is validated by a real probe build for that target before it enters
Cargo.toml (an empty `#![no_std]` crate + the candidate + `cargo build --target
wasm32v1-none`).

| Crate | Config | Role |
|---|---|---|
| `xxhash-rust` | `xxh3` only | section/journal checksums, interner hash |
| `thiserror` | no-default | errors |
| `hashbrown` | no-default | **query scratch only** (BM25 accumulator etc.); forbidden in persistent data |
| `unicode-segmentation` | no-default | UAX #29 word boundaries |
| `unicode-normalization` | no-default | NFKC + canonical decomposition |
| `libm` | no-default | float math (BM25/recency) in no_std |
| `serde` | optional, no-default, `derive`+`alloc` | behind feature `serde`: Serialize/Deserialize of public data types for the JSON wrappers |

The unicode tables cost ~163 KiB in the wasm binary (~16% of the 1 MiB artifact budget)
— accepted, the price of Lucene-class segmentation. No off-the-shelf HNSW crate builds
for `wasm32v1-none` (all pull `getrandom`/std), which is why HNSW is implemented inside
plugmem-core over our own arenas — needed anyway for a zero-copy graph snapshot.

Wrapper/dev crates (std allowed): `clap` (cli), `ureq` (host HTTP), `serde_json`
(mcp, cli `--json`), `memmap2` (host read-only mmap), `napi`/`napi-derive` (napi),
`proptest`/`criterion`/`insta` (dev). All versions live in `[workspace.dependencies]`;
the lock file is committed; MSRV is pinned at the first release (`core::error::Error`
needs ≥ 1.81).

## CI

A single required check (`CI passed`) aggregates every job, so branch protection never
needs editing:

1. **lint + test**, fanned out over Linux/Windows/macOS: `cargo fmt --check`; clippy
   (deny warnings) in the default, `counters` and `serde` feature combos; `cargo test
   --workspace` in the same three combos.
2. **no_std build**: `cargo build -p plugmem-arena -p plugmem-core --target
   wasm32v1-none` (default, `counters`, `serde`).
3. **core suite on wasm32**: the full contract suite on `wasm32-wasip1` under both
   wasmtime and wasmer, each with/without `counters` (this caught a real 32-bit journal
   allocation bug native CI cannot see).
4. **snapshot equivalence**: the wasm-probe run natively, on wasm32 and on wasm64
   (memory64, built on nightly with build-std) must print a byte-identical snapshot hash
   — one green run proves the format is pointer-width independent.
5. **skill structure**: `skill/SKILL.md` exists, carries a well-formed
   `<!-- skill-version -->` marker and the wasm-strip fence, and fits the Agent Skills
   frontmatter limits.
6. **napi npm build**: build the addon (`npx napi build`) and smoke-test it through Node.
7. **cargo-dist plan**: validate the release plan.

Coverage (tarpaulin) and hardening (miri, a leak-growth gate) run in their own
workflows. The release pipeline (cargo-dist) needs one secret set by hand:
`HOMEBREW_TAP_TOKEN` (a fine-grained PAT with Contents: write on
`m62624/homebrew-plugmem`). Both npm and crates.io publishing use OIDC trusted
publishing (no long-lived registry token); crates.io's `release.yml` job gets
`id-token: write` and exchanges it through `rust-lang/crates-io-auth-action@v1`.
`GITHUB_TOKEN` is provided automatically.

Trusted publishing is configured separately on crates.io for each published
workspace crate: `plugmem-arena`, `plugmem-core`, `plugmem-host`,
`plugmem-cli`, and `plugmem-mcp`. Each entry trusts owner `m62624`, repository
`plugmem`, and workflow filename `release.yml` (the file lives under
`.github/workflows/`); no Cargo manifest setting is needed. The first version
of each crate must already exist on crates.io before its trusted publisher can
be added.
