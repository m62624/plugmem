# AGENTS.md

## README link policy

- The repository-root `README.md` is the workspace README. It may use relative
  links to files, crates, assets, and sections inside this repository.
- A crate-local `crates/<crate>/README.md` is package documentation: Cargo can
  publish it to crates.io, and readers may encounter it outside the checkout.
  Keep it self-contained and do not link to the workspace with paths such as
  `../../README.md` or to sibling crate files with `../...` paths.
- In crate-local READMEs, link public Rust APIs and sibling crates through their
  absolute `https://docs.rs/<crate>/latest` URLs. Link repository-only material
  (settings files, benchmarks, source files, and markdown documents from another
  crate) through absolute `https://github.com/m62624/plugmem/...` URLs.
- A crate-local README may refer to an SVG or other asset shipped by that same
  crate with a relative path such as `assets/chart.svg`. Cargo packages the
  README and the crate's `assets/` directory together, so this remains valid in
  the workspace, on GitHub, and on crates.io. Do not use relative paths to
  workspace files or sibling crates.
- Workspace-only branding, generated top-level documentation assets, and legacy
  reference material belong under the root `assets/` directory. Keep legacy
  opaque references under `assets/references/`; they are design/test material,
  not published crate assets. Root `README.md` may link to these files with
  relative paths such as `assets/logo.png`.

## Benchmark documentation

- Benchmark SVGs are generated artifacts. Keep their source data and renderer
  in `tools/bench-charts`; do not hand-draw a chart that the tool cannot
  reproduce.
- When a benchmark compares corpus sizes, the chart must be generated from the
  same tool input and the README must state the workload, platform, units, and
  whether embeddings are enabled.
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

The crate has two contracts, not one: its single audited `unsafe`, and its allocation budget — zero allocations per operation, amortized growth only, and a published allocator-call count that a change may not raise without saying why. Both are spelled out in `crates/plugmem-arena/AGENTS.md`; treat them as equally binding.

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

For vector storage, the slot stride is the fixed header plus signature bytes plus the quantized vector dimension. With dimension 384, the raw vector slot is 440 bytes before other engine structures. Flat search scans all slots, so its latency is primarily controlled by the number of stored vectors rather than by the requested result count. HNSW is advanced by maintenance/configuration paths with bounded work in `Auto`; `remember` does not automatically rebuild the graph on every write.

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

Node.js N-API bindings. A boundary layer only: its surface equals `plugmem-host`'s (see the wrapper-parity rule below), the engine logic stays in host, and FFI overhead is measured separately from core and host performance.

### `crates/plugmem-testgen`

Utilities for generating deterministic test data and scenarios. It is not part of the production storage path.

### `fuzz`

`cargo-fuzz` targets for the two untrusted files, the snapshot and the journal.
Its own workspace (nightly plus a sanitizer runtime), excluded from the main
one. The seed corpus is committed real images — without seeds the fuzzer never
gets past the magic number. CI runs each target briefly on every pull request;
a real campaign is a manual long run. See `fuzz/README.md`.

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

## Wrapper parity

A language binding — today `plugmem-napi`, and whatever follows it — exposes the whole of `plugmem-host`'s surface.

- An omission is allowed only when the construct is **unrepresentable** in the target
  language, never because it is awkward to write. "A resumable iterator is inconvenient
  across this boundary" is not a reason; it is the work.
- Format concerns are not host verbs and do not live in a wrapper's Rust. JSONL
  import/export belongs to the CLI, or to a pure module in the target language shipped
  in the same package.
- Idioms the target language demands are not divergence: a Python context manager or a
  JavaScript promise has no counterpart to drift from.
- A divergence between wrappers that does not follow from the above is a bug, not a
  decision. Two verbs are worth naming because they were once excluded on this exact
  mistaken ground: `scrub` and `recover`.

`plugmem-mcp` is judged differently, and deliberately. It is not a language binding: its
caller is a model reading a tool list, where every additional tool spends context and
adds a way to go wrong, and where a path argument is chosen by the model rather than by
a programmer. Its measure is "what an agent should be handed", not "what host has". So
the salvage verbs stay out of it — but that is the *only* licence the difference grants.
Anything an agent legitimately uses must be whole: `plugmem_export` returning facts
without edges is the same defect as the equivalent gap in a binding, not an exercise of
this exemption.

The corollary for configuration: every wrapper reaches `Config` through the one shared
loader (`Settings::load`), so a setting that is not in `settings_help` is a setting no
wrapper can offer. Adding a knob means adding it there, not in four places.

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

## Release versioning

- Keep the workspace, Cargo manifests, `Cargo.lock`, and npm metadata at the
  current development version between releases. The `skill/SKILL.md` marker is
  the one version string a human edits by hand.
- Do not manually bump package versions in a feature or performance PR. The
  release workflow derives the release version from the pushed `vX.Y.Z` tag,
  updates the workspace and npm versions on its release-candidate branch, and
  opens the synchronization PR back to `main`.
- Skill content may be updated in any normal PR, and so may the
  `<!-- skill-version: X.Y.Z -->` marker — bump it to the version you are
  heading for in whichever PR before that release is a natural place for it.
  No rule reserves it for a release branch.
- The marker exists so the skill cannot quietly fall behind the engine.
  `skill/SKILL.md` is what an agent reads instead of the source, so a release
  that changed verbs, flags or behavior has to be checked against it: read the
  skill, confirm it still describes what shipped, update it if not, then set
  the marker. The release gate compares the marker with the tag-derived
  package version and fails when they differ — that failure is the reminder to
  do the reading, not a versioning chore. Regular CI only checks the marker is
  well formed and never compares it with the development version, so a bumped
  marker sits on `main` until the release catches up with it.
- Verify the release workflow when changing this policy: `.github/workflows/release.yml`
  is the source of truth for tag parsing, version synchronization, and the
  skill-version gate.
