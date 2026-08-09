# plugmem-host

> ⚠️ Experimental. plugmem is mostly an AI-built experiment — written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

`plugmem-host` is the `std` host layer for the plugmem
[temporal-memory engine](https://docs.rs/plugmem-core/latest). It supplies
what the `no_std` engine does not own — files, locking, and network — so from
this one crate a Rust program gets `remember / recall / revise / forget` plus
graph `link`/`unlink`, backed by durable storage. It re-exports the engine, so
**this one crate is all a Rust program needs.**

**No embedding model is required.** Of the four recall sources, only the vector
one needs an embedder; text, graph and time work with nothing but the database.
Unlike the CLI and the MCP server, a Rust caller has two ways to supply one —
an `[embedder]` section in `config.toml`, or `DatabaseBuilder::embedder` — and
a third option, passing a vector per call, which needs neither. See
[Embedding is optional](#what-you-get) below.

## Which crate do I need?

**Writing Rust and just want a working memory? This is the crate — it has
everything.** The others are for narrower needs.

| You want | Use | Why |
|---|---|---|
| **A memory in a Rust program** — the common case | **`plugmem-host`** (this crate, `std`) | Everything included: files, locking, read-only mmap, HTTP embedders, integrity, cross-process concurrency. One dependency — it re-exports the engine. |
| A memory in Rust with **no `std`** or **your own storage** (browser, wasm host, custom file layer) | [`plugmem-core`](https://docs.rs/plugmem-core/latest) (`no_std`) | The engine only. You bring the `Storage` trait, the clock, file I/O and embedding — so you also manage when the file opens and how memory loads. |
| Just the **flat byte-pool containers** | [`plugmem-arena`](https://docs.rs/plugmem-arena/latest) (`no_std`) | The storage substrate, engine-agnostic. |
| A memory from a **terminal or shell script** | [`plugmem-cli`](https://docs.rs/plugmem-cli/latest) (`plugmem`) | One local database, no server; `plugmem repl` keeps the engine open for host speed. |
| A memory for an **agent, local-first app, or non-Rust program** | [`plugmem-mcp`](https://docs.rs/plugmem-mcp/latest) | Long-lived stdio JSON-RPC; language-independent. In Rust, embed this crate instead. |
| A memory in **JavaScript / TypeScript** (Node) | [`plugmem-napi`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-napi) | The engine as a native Node addon (napi-rs), in-process; on npm as `plugmem`. |
| A memory in **Python** | [`plugmem-py`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-py) | The engine as a CPython extension (PyO3), in-process; on PyPI as `plugmem`. |

## Configuration

The shared `config.toml` loader and platform-aware database paths are documented
in the [full settings reference](https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md).
[`plugmem-cli`](https://docs.rs/plugmem-cli/latest),
[`plugmem-mcp`](https://docs.rs/plugmem-mcp/latest),
[`plugmem-napi`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-napi) and
[`plugmem-py`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-py) use the same settings
catalogue and database-path precedence; only their explicit override syntax
differs.

The `config` feature is enabled by default for this native host crate, so
`Settings::load` and `read_config` are available without extra feature flags.
Applications that construct `Config` programmatically and want the smallest
host dependency surface can use `default-features = false`; this does not alter
the `plugmem-core` or WASM build.

## What recall does

Recall is not a vector lookup — it fuses four sources by reciprocal-rank fusion
with a recency boost (tags filter; they are not a source):

| Source | Algorithm | What it finds |
|---|---|---|
| **Lexical** | [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (Robertson idf) over a Unicode ([UAX #29](https://unicode.org/reports/tr29/)) tokenizer | exact terms / keyword overlap |
| **Semantic** | int8-quantized cosine — a flat two-phase scan below a threshold, an [HNSW](https://arxiv.org/abs/1603.09320) graph above | meaning / nearest neighbours |
| **Graph** | entity graph with typed edges, breadth-first from query anchors; depth defaults to `Config::graph_depth` and can be overridden per recall | relational knowledge |
| **Temporal** | range scans over a `recorded_at`-ordered index; bitemporal validity | "what was true *then*", time windows |

## Two clocks

Every fact carries two timestamps, and the distinction is the reason the
temporal source exists at all:

- **`valid_from` / `valid_to`** — when the statement was true.
- **`recorded_at`** — when the memory learned it. Stamped by the engine.

One timestamp cannot hold both, and the host passes `now` explicitly, so the
whole sequence is expressible:

```rust,no_run
# use plugmem_host::{Config, Database, RecallQuery, RememberInput};
# let (db, _) = Database::builder(Config::default()).open("agent.plugmem")?;
const JAN: u64 = 1_767_225_600_000;
const MAR_5: u64 = 1_772_668_800_000;
const MAR_10: u64 = 1_773_100_800_000;

// In January the memory learns where kim lives.
let first = db.remember(RememberInput {
    entity: Some("kim"),
    ..RememberInput::text(JAN, "lives in Moscow")
})?;

// On March 10th it learns the address changed. `revise` closes the old
// interval at this instant; it does not delete the record.
db.revise(
    first.id,
    RememberInput {
        entity: Some("kim"),
        ..RememberInput::text(MAR_10, "lives in Berlin")
    },
)?;

// Now: Berlin.
db.recall(RecallQuery {
    entities: &["kim"],
    ..RecallQuery::text(MAR_10, "kim")
})?;

// As of March 5th: Moscow. Still valid then, and already recorded.
db.recall(RecallQuery {
    entities: &["kim"],
    as_of: Some(MAR_5),
    ..RecallQuery::text(MAR_10, "kim")
})?;

// The graph depth belongs to this question: widen the walk for a neighbourhood
// query, or use `Some(0)` when only kim's own facts should answer.
let _around_kim = db.recall(RecallQuery {
    entities: &["kim"],
    graph_depth: Some(3),
    ..RecallQuery::text(MAR_10, "kim")
})?;
# Ok::<(), plugmem_host::HostError>(())
```

`as_of` moves **both** clocks: a fact answers only if it was valid at that
instant *and* had already been recorded by then. The second half is the one
people trip over. An `as_of` earlier than a fact's `recorded_at` does not see
it, because the memory genuinely knew nothing then — answering with today's
knowledge would be the wrong answer to "what did I hold".

`valid_from` is the other half of the model: a statement that became true before
you heard of it. Recording on March 10th that the move happened on March 1st —
`valid_from: Some(MAR_1)` on the revision — closes the old interval at March 1st
instead of at March 10th. A query as of March 5th then finds *neither*: Moscow
had stopped being true, and Berlin was not yet known. That is not a hole in the
model, it is the honest answer for that instant, and it is exactly what one
timestamp cannot express.

The old record stays on disk either way. `forget` is the destructive verb: use
it when a fact was wrong, not when it changed.

Edges are temporal the same way, so an `as_of` traversal walks the graph as it
stood then — through relationships that have since been unlinked.

## What `plugmem-host` adds

The retrieval above lives in the engine; this crate adds the OS side:

- **File-backed storage** — atomic snapshots (tmp + fsync + rename),
  an append-only journal with a configurable fsync policy, crash
  recovery (a torn journal tail is detected and dropped on open);
- **OS locking, one writer** — an advisory lock per database file. A second
  read-write open is refused with a typed `HostError::Locked` rather than
  corrupting the file silently. Readers do **not** take that lock;
- **A read-only mmap open, concurrent with the writer** —
  `Database::open_readonly` pins the current published snapshot generation and
  maps it, borrowing the mapped pages so a large read-mostly database resides
  only in the pages a query touches. Generations are immutable — a checkpoint
  writes a new one and never rewrites an old one — so **a writer and any number
  of readers run at the same time**, across threads and across processes,
  sharing the OS page cache. This is the WAL/MVCC arrangement, not "N readers
  *or* one writer": a reader sees state as of the checkpoint it pinned
  (snapshot isolation) and calls `refresh()` to adopt a newer one;
- **A write path that does not clone the file** — `Database::open`
  memory-maps the snapshot and the engine borrows it as an *overlay*:
  mutations land in a small owned overlay (an appended tail plus per-page
  copy-on-write), so opening a multi-gigabyte database to append one fact
  no longer copies the whole image into RAM. A snapshot materializes the
  base + overlay into a fresh file and re-maps it. Validation is lazy (the
  SQLite model): an open range-checks the record metadata and nothing else, so
  the large text, vector and per-fact-metadata pools stay non-resident until a
  query touches them — a measured open residents well under half of a
  text-heavy image. What that guarantees is precise and worth stating: **no
  stored id can make a read unsafe. It does not mean the data agrees with
  itself.** The default open never checksums the image either, so it stays
  sparse; corruption is caught when the bad record is read (never a panic), or
  on demand with `verify()` (content *and* graph consistency) and `scrub()`
  (byte-level container integrity) — see **Integrity & recovery** below;
- **Maintenance policy** — auto-snapshot and optional auto-`maintain`,
  run inline (no background threads);
- **Embedding providers** — one HTTP client for the `/v1/embeddings`
  shape shared by OpenAI, Ollama, LM Studio, vLLM and llama.cpp-server.

Without an embedder it is fully functional — lexical, tags, graph and
time still answer; vectors are an addition, not a requirement.

## Example

```rust,no_run
use plugmem_host::{Config, Database, OpenAiCompatEmbedder, RecallQuery, RememberInput};

let mut cfg = Config::default();
cfg.dim = 768;

let (db, _report) = Database::builder(cfg)
    // one client covers OpenAI, Ollama, LM Studio, vLLM, llama.cpp-server:
    .embedder(Box::new(OpenAiCompatEmbedder::new(
        "http://localhost:11434/v1/embeddings", // Ollama's full embeddings endpoint
        "nomic-embed-text",
        768,
    )))
    .open("agent.plugmem")?;

// The text is embedded automatically (outside the database lock),
// quantized and indexed:
db.remember(RememberInput::text(1_784_000_000_000, "prefers tokio"))?;

// Bulk load: the whole batch is embedded in one round-trip and the
// journal is fsynced once (not per fact) — the write path for `import`:
db.remember_many(vec![
    RememberInput::text(1_784_000_050_000, "uses pinned versions"),
    RememberInput::text(1_784_000_060_000, "lives in Berlin"),
])?;

// The query text is embedded too; recall fuses lexical, vector, graph
// and temporal evidence into one ranked, token-budgeted block:
let out = db.recall(RecallQuery::text(1_784_000_100_000, "which runtime?"))?;
println!("{}", out.rendered);
# Ok::<(), plugmem_host::HostError>(())
```

## Benchmarks

```text
cargo run -p plugmem-host --example edge_lifecycle
cargo run -p plugmem-host --example maintain_modes
cargo run --release -p plugmem-host --example bench_database -- 100000 --diagnose-recall | tee database-benchmark-100k.tsv
cargo run --release -p plugmem-host --example bench_database -- 1000000 --diagnose-recall | tee database-benchmark-1m.tsv
cat database-benchmark-100k.tsv database-benchmark-1m.tsv > database-benchmark-scale.tsv
cargo run -p plugmem-bench-charts -- database-benchmark-scale.tsv --force
cargo run --release -p plugmem-host --example bench_edges -- 100000 | tee edge-benchmark-100k.tsv
cargo run --release -p plugmem-host --example bench_edges -- 1000000 | tee edge-benchmark-1m.tsv
cat edge-benchmark-100k.tsv edge-benchmark-1m.tsv > edge-benchmark-scale.tsv
cargo run -p plugmem-bench-charts -- edge-benchmark-scale.tsv --force
cargo bench -p plugmem-host
```

The committed `assets/database-*.svg` charts are generated from the database
runs; `assets/edge-lifecycle-*.svg` charts are generated from `bench_edges`.
![Recall latency at 5k, 100k and 1M operations](assets/database-recall-scale.svg)
![Edge lifecycle graph recall at 100k versus 1M edges](assets/edge-lifecycle-recall-100k-1m.svg)

For the same-workload comparison between 100k and 1M, see the
[measured scale table on GitHub](https://github.com/m62624/plugmem#measured-scale).
For the public Rust API, see the [plugmem-host documentation on docs.rs](https://docs.rs/plugmem-host/latest/).
The database example uses deterministic synthetic facts and `dim=0`; it does
not call an embedding service or use a network connection.

Native builds are 64-bit, so a host process reads every capacity class
of the shared file format: databases sized for the 32-bit wasm budget
(≤ 2 GiB, the default) and databases with larger limits alike. Opening,
reading, scanning and checkpointing go through the mmap overlay, whose
clean pages the OS can reclaim — so those work on a database larger than
RAM. A *rebuild* (`maintain` and `recover`) is disk-first too: it streams
the two big pools (vectors, text) through temp files and keeps only the
metadata and the HNSW graph resident, so peak RAM tracks the record count,
not the image size. The residual limit is the graph itself — a database
whose graph exceeds RAM is a further tier. The per-structure byte costs and
pool limits (what a fact, an edge or a vector weighs, and where each tops
out) are tabulated in
[`plugmem-core`](https://docs.rs/plugmem-core/latest)'s *Capacity — what
weighs what*. The snapshot format is pointer-width independent — a file
written here opens unchanged in a wasm32 or wasm64 build of the core, as
long as its configured limits fit that host.

## Files

One local database rooted at `agent.plugmem` consists of:

| file | role |
|---|---|
| `agent.plugmem` | the manifest naming the current snapshot generation |
| `agent.plugmem.snap.<N>` | immutable snapshot generation — the engine's memory image |
| `agent.plugmem.journal` | append-only journal since the current generation |
| `agent.plugmem.lock` | advisory lock file |

Snapshot writes are atomic (tmp + fsync + rename + directory fsync): a
reader observes the old image or the new one, never a torn file.
Journal appends are fsynced per operation by default
(`FsyncPolicy::EachOp`); `OnSnapshot` trades the crash-window of the
journal tail for speed. On open, the journal is replayed over the
snapshot deterministically; a torn tail from a crash mid-append is
detected, dropped and reported.

## Workspaces (optional)

**Default: one `Database`, one local database layout.** `Workspace` is for a process serving many
independent memories — a directory of named databases, opened on demand and
pooled, plus an optional registry of what each is for.

```rust,no_run
use plugmem_host::{DbName, IfMissing, Settings};

let ws = Settings::load(None)?.open_workspace("/srv/bot".as_ref())?;
let db = ws.get(&DbName::parse("chat-42")?, now_ms(), IfMissing::Create)?;
# fn now_ms() -> u64 { 0 }
# Ok::<(), Box<dyn std::error::Error>>(())
```

A name is `[a-z0-9][a-z0-9_-]*` and cannot represent a path, so it resolves to
exactly one named database inside the directory. The pool bounds how many stay open;
`close_idle` releases the rest, which matters because an open writer holds the
file's exclusive lock — the timeout is a liveness setting, not a memory one.

The core is untouched by any of this: one `Memory` is still one database.
Rust callers may intentionally keep the `Database` returned by `get`; its RAII
`Drop` makes that ownership explicit. Garbage-collected language wrappers must
instead keep only the name and call `Workspace::lease` inside each verb. A
lease pins one pool entry for that operation, so active entries are never
evicted or swept; if every `max_open` slot is active, another name receives
`WorkspaceError::AtCapacity` immediately. `release(name)` closes one inactive
pooled handle without invalidating logical foreign-language references.
See [`specs/10-workspace.md`](https://github.com/m62624/plugmem/blob/main/specs/10-workspace.md).

## Concurrency model

The engine is single-writer by design. The host runs a WAL/MVCC-style
versioned layout, so **one writer and any number of readers run at the same
time** — across threads *and* processes — without a reader ever blocking the
writer or seeing a torn image.

- **One writer.** `Database::open` takes an *exclusive* writer lock; a second
  writer gets `HostError::Locked` immediately (a typed refusal, not silent
  corruption). The lock dies with the process, even on a crash. Readers do
  **not** take this lock, so they never contend with the writer.
- **Many readers, concurrent with the writer.** A checkpoint never overwrites
  a live file: it writes a new immutable snapshot *generation* and repoints a
  tiny manifest. `Database::open_readonly` pins the current generation with a
  *shared* lock and maps it — so it coexists with a live writer and reads a
  consistent snapshot "as of the last checkpoint" (it does not see writes made
  after it opened; reopen to advance). Readers across threads or processes
  share one copy of a generation in the OS page cache. The writer reclaims a
  superseded generation only once no reader still pins it, so disk stays
  bounded by the longest-lived reader.
- **One process, many threads or agents.** A `Database` is a
  `Clone + Send + Sync` handle; clone it freely. The read verbs
  (`recall`/`get`/`stats`/`export`/`verify`) take a *shared* guard and run
  **concurrently**; the write verbs take an *exclusive* guard and serialize
  — against each other and against readers (an `RwLock` over the engine, the
  same reader/writer discipline as the file lock, one level down). At
  microsecond engine calls neither is a bottleneck. `ReadOnlyDatabase` is
  `Send + Sync` too, and its reads are lock-free — a fan-out of reader
  threads over one mapped snapshot. (Each reader thread keeps its own recall
  scratch, so concurrent `recall`s never contend.)
- **Many files.** Fully independent databases: separate locks, separate
  mutexes, natural parallelism. Two models each with their own memory
  file never contend; two models sharing one memory clone one handle.
- **Network stays outside the lock.** Embedding calls (the slow,
  external part) run before the mutex is taken, so an agent waiting on
  its embedding provider does not stall the others.

The typical shape is **build occasionally, read a lot**: one writer
snapshots the memory (on a schedule or in a maintenance window), then many
read-only consumers query it in parallel.

```rust,no_run
use plugmem_host::{Config, Database, RecallQuery};

// Many readers over one checkpointed file — zero-copy, shared page cache.
// (A read-only open needs a published generation: checkpoint once, first. It
// then reads that generation and ignores any later journal, so a reader sees
// the memory as of the last checkpoint.)
let ro = Database::open_readonly("agent.plugmem", Config::default())?;
let out = ro.recall(RecallQuery::text(1_784_000_100_000, "which runtime?"))?;
println!("{}", out.rendered);
# Ok::<(), plugmem_host::HostError>(())
```

## Memory-mapped opens and disk-first rebuilds

Both `open` and `open_readonly` memory-map the snapshot; the engine borrows an
overlay over the mapping rather than reading it into a heap copy. Opening is an
`mmap` plus a bounds-check regardless of file size; only dereferenced pages
fault in, and the OS may evict clean ones. Resident memory tracks the working
set, not the file size.

A read-write handle appends new records to a small owned tail; the mapped base
is never rewritten in place (the append-only structures avoid copy-on-write). A
checkpoint streams the fresh image to a temp file and renames it atomically,
unmapping first (Windows will not rename a mapped file).

Read-only handles take a shared lock and map the same snapshot, so several
share one copy in the OS page cache. Open read-write only to mutate.

`maintain` and `recover` run disk-first: the two large pools (vectors, text)
stream through a temp `Scratch` file (a sibling of the database, mapped on
freeze, deleted on drop), keeping only metadata and the HNSW graph resident.
Peak RAM is proportional to record count, not content size. On `no_std` (no
files) the engine uses the in-RAM rebuild instead.

## Integrity & recovery

The default open trusts the file (like SQLite): it does not checksum the
whole image, so a large database opens sparse. Integrity is on demand, in
three layers of increasing cost, and corruption is never a panic — the
accessors tolerate bad bytes, these turn latent damage into an explicit
error or a repaired file.

| call | checks | cost |
|---|---|---|
| `verify()` | everything an open defers — stored text is valid UTF-8, metadata blobs decode, the fact↔vector-slot bijection holds, and the graph agrees with itself (both edge mirrors, a current edge against its open version, every open version reachable as a current edge) | one linear pass over the text + vector pools, plus a lookup per edge |
| `scrub()` | *byte-level* container integrity — each section's stored xxh3 and the whole-file hash (the ZFS-scrub model) | resumable; on either handle |
| `recover()` | *salvage* — drop the content-corrupt facts, rebuild, write a clean copy | rebuilds in RAM ≈ image size |

**`scrub()` — the bitrot detector.** A resumable iterator over the mapped
snapshot: each `next()` hashes up to a slice budget, so you pace it
yourself (run to completion, or a slice at a time on a background thread,
pausing/cancelling between slices). It holds a shared lock for its whole
life, reads the map linearly (pages fault in, get hashed, stay
reclaimable — it never residents the whole file), and reports the first
mismatch, naming the damaged section.

A scrub reads the *file*, not a handle's view of it, so it is on both
handles. Use the writer's when you already hold one: a second read-only
open would map the whole image again and take a second lock to hash bytes
whose path you already know.

```rust,no_run
use plugmem_host::{Config, Database};

let (db, _) = Database::open("agent.plugmem", Config::default())?;
// Verify every container byte, a slice at a time.
for step in db.scrub()? {
    let progress = step?; // Err(Corrupt) names the first damaged section
    // e.g. report progress.done_bytes / progress.total_bytes to a UI
    let _ = progress;
}
# Ok::<(), plugmem_host::HostError>(())
```

It needs a *published* generation — checkpoint once — but not a clean
journal: it checks the container as it stands, and the journal describes
the generation that has not been written yet.

**`recover()` — Tier 2 salvage.** For *content* corruption (bad text
bytes, a broken vector bijection): it opens the source, drops the facts
that fail `verify()`'s per-fact checks, compacts the survivors and their
indexes, and writes a fresh image to a new file — **leaving the source
untouched** as evidence. It returns a
`RecoverReport { kept, dropped_text, dropped_vector }`. It is **disk-first**
(the source opens as an mmap overlay and the two big pools stream through
temp files), so peak RAM tracks the record count — a database larger than
RAM can be recovered, as long as its graph fits.

```rust,no_run
use plugmem_host::{Config, Database};

// now = a millisecond timestamp; dst must differ from src.
let report = Database::recover("agent.plugmem", "agent.recovered.plugmem",
                               Config::default(), 1_784_000_000_000)?;
println!("kept {}, dropped {} text + {} vector",
         report.kept, report.dropped_text, report.dropped_vector);
# Ok::<(), plugmem_host::HostError>(())
```

**What recover does not do.** *Structural* damage — a snapshot that will
not even parse — is not salvageable here: the source fails to open and
recover returns the typed error; restore from a backup instead (Tier 0).

**Recovery layers (first release).** Most recovery is not salvage at all:

- **Tier 0 — restore.** A snapshot is one atomic file (tmp + fsync +
  rename); back it up and copy it back. `scrub()` tells you *when* to.
  This covers the overwhelming majority of cases.
- **Tier 1 — regenerate.** Re-ingest from your upstream source (logs,
  documents) into a fresh database.
- **Tier 2 — `recover()`.** Content-corruption salvage, as above.

## Maintenance policy

Configured through the builder, executed inside the same critical
section as the operation that triggered it — there are no background
threads, matching the engine's own philosophy:

| knob | default | effect |
|---|---|---|
| `snapshot_every_ops` | 1024 | full snapshot + journal reset after N mutations |
| `snapshot_journal_bytes` | 4 MiB | …or when the journal outgrows this |
| `maintain_every_forgets` | off | optional auto-`maintain` (physical purge) |
| — | always on | re-shard when the layout no longer fits the data |

`maintain` is policy-driven. The default `Auto` path first checks whether
anything is pending; with no tombstones, stale text index or vector tail to
optimize, it returns a no-op report without rewriting the snapshot. When work
is pending, host maintenance stays disk-first: text and vector pools stream
through scratch files, ordinary BM25 compaction filters existing postings, and
HNSW work is bounded unless a full rebuild is explicitly requested.

`maintain_with_options` selects the policy explicitly. No mode ever drops a
fact revision or an edge version — what the heavier modes buy is bytes and
index freshness, never less history.

| mode | what it does | cost |
|---|---|---|
| `Auto` | only pending work; no-op when there is none | bounded |
| `Compact` | purge tombstones, compact storage and indexes | O(live records) |
| `ReindexText` | rebuild BM25 by re-tokenizing stored text | O(text) |
| `OptimizeVectors` | build or advance the vector graph | O(vectors) |
| `Full` | rebuild everything, fully optimize vectors, repack the edge arenas | O(database) |

`Full` is the only mode that repacks the edge arenas. Relinking many relations
fragments them — the incoming mirror is keyed by the far endpoint, so
interleaved runs keep splitting pages in half — and rewriting them in key
order packs the pages again. Measured over 200 relations relinked 1000 times
(200k retained versions): 31.9 MB → 23.4 MB, 59 ms, every version kept.

The same selection is exposed by `plugmem maintain --mode <mode>`, the
`mode` argument of the MCP `plugmem_maintain` tool, and `maintain(mode?)` in
the Node bindings.

## Embedders

The `Embedder` trait is two methods (`dim`, batched `embed`).
`OpenAiCompatEmbedder` speaks the `/v1/embeddings` shape that OpenAI,
Ollama, LM Studio, vLLM and llama.cpp-server share — there is no
provider-specific client because there is no provider-specific
protocol. The dimension is configured explicitly (no startup probe);
a server answering with a different one is a typed error. Tests run
against a local mock — no network in CI.

`embed` takes **`&self`**, and the trait requires `Send + Sync`, because an
embedder is a client of a remote service rather than a piece of mutable state.
That is what lets every caller share one instance with no lock in front of it:
`SharedEmbedder` is a plain refcount, a `Database` holds its embedder unlocked,
and concurrent verbs sit in the provider at the same time instead of queueing
behind one HTTP request. An implementation that does need mutable state (a
cache, a rate-limit budget) brings its own interior mutability, which is the
only place that knows what may overlap.

## Feature flags

- `serde` — `Serialize`/`Deserialize` on the public data types
  (`FactSnapshot`, `ExportedFact`, `RecoverReport`, `FsyncPolicy`), forwarding
  to `plugmem-core/serde`. Off by default. `HostError` is deliberately not
  covered — it wraps `std::io::Error`, which is not serializable.
- `counters` — deterministic work counters for the perf gates, forwarded to
  `plugmem-core/counters`. **Do not enable it in normal use — it is a
  single-threaded measurement build only.** The arena's counter `Cell`s are
  not `Sync`, so with `counters` on the engine lock falls back from an
  `RwLock` to a `Mutex`: the read verbs then **serialize** instead of running
  concurrently. The public API is identical either way — only the internal
  lock (and thus read concurrency) changes. Leave it off to keep concurrent
  readers; reach for it only when measuring operation counts in a specific
  scenario.

## License

MIT.
