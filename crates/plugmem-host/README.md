# plugmem-host

> ⚠️ Experimental. plugmem is mostly an AI-built experiment — written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

`plugmem-host` is the `std` host layer for the plugmem
[temporal-memory engine](https://docs.rs/plugmem-core/latest). It supplies
what the `no_std` engine does not own — files, locking, and network — so from
this one crate a Rust program gets `remember / recall / revise / forget` backed
by durable storage. It re-exports the engine, so **this one crate is all a Rust
program needs.**

## Which crate do I need?

**Writing Rust and just want a working memory? This is the crate — it has
everything.** The others are for narrower needs.

| You want | Use | Why |
|---|---|---|
| **A memory in a Rust program** — the common case | **`plugmem-host`** (this crate, `std`) | Everything included: files, locking, read-only mmap, HTTP embedders, integrity, cross-process concurrency. One dependency — it re-exports the engine. |
| A memory in Rust with **no `std`** or **your own storage** (browser, wasm host, custom file layer) | [`plugmem-core`](https://docs.rs/plugmem-core/latest) (`no_std`) | The engine only. You bring the `Storage` trait, the clock, file I/O and embedding — so you also manage when the file opens and how memory loads. |
| Just the **flat byte-pool containers** | [`plugmem-arena`](https://docs.rs/plugmem-arena/latest) (`no_std`) | The storage substrate, engine-agnostic. |
| A memory from a **terminal or shell script** | `plugmem-cli` (`plugmem`) | One file, no server; `plugmem repl` keeps the engine open for host speed. |
| A memory for an **LLM agent** or a **non-Rust program** | `plugmem-mcp` | Long-lived stdio JSON-RPC; language-independent. In Rust, embed this crate instead. |
| A memory in **JavaScript / TypeScript** (Node) | `plugmem-napi` | The engine as a native Node addon (napi-rs), in-process; on npm as `plugmem`. |

## Configuration

The shared `config.toml` loader and platform-aware database paths are documented
in the [full settings reference](SETTINGS.md). `plugmem-cli`, `plugmem-mcp` and
`plugmem-napi` use the same settings catalogue and database-path precedence;
only their explicit override syntax differs.

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
| **Graph** | entity graph with typed edges, breadth-first from query anchors | relational knowledge |
| **Temporal** | range scans over a `recorded_at`-ordered index; bitemporal validity | "what was true *then*", time windows |

## What `plugmem-host` adds

The retrieval above lives in the engine; this crate adds the OS side:

- **File-backed storage** — atomic snapshots (tmp + fsync + rename),
  an append-only journal with a configurable fsync policy, crash
  recovery (a torn journal tail is detected and dropped on open);
- **OS locking** — an advisory lock per database file: read-write opens
  take it exclusively, read-only opens take it *shared*, so a conflicting
  opener is refused with a typed `HostError::Locked` rather than corrupting
  silently (the model is SQLite-like: N readers **or** one writer);
- **A read-only mmap open** — `Database::open_readonly` maps the snapshot
  and lets the engine borrow the mapped pages, so a large read-mostly
  database residents only the pages a query touches instead of loading
  the whole file. It holds a shared lock, so **many readers map the same
  file at once** — across threads or processes — sharing the OS page cache;
- **A write path that does not clone the file** — `Database::open`
  memory-maps the snapshot and the engine borrows it as an *overlay*:
  mutations land in a small owned overlay (an appended tail plus per-page
  copy-on-write), so opening a multi-gigabyte database to append one fact
  no longer copies the whole image into RAM. A snapshot materializes the
  base + overlay into a fresh file and re-maps it. Validation is lazy (the
  SQLite model): an open checks only the record metadata, so the large text,
  vector and per-fact-metadata pools stay non-resident until a query touches
  them — a measured open residents well under half of a text-heavy image. The default open
  trusts the file and never checksums the whole image (the SQLite model),
  so it stays sparse; corruption is caught when the bad record is read
  (never a panic), or on demand with `verify()` (content) and `scrub()`
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
        "http://localhost:11434/v1", // Ollama's OpenAI-compatible endpoint
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

One database at `agent.plugmem` is:

| file | role |
|---|---|
| `agent.plugmem` | the snapshot — the engine's memory image, verbatim |
| `agent.plugmem.journal` | append-only journal since the last snapshot |
| `agent.plugmem.lock` | advisory lock file |

Snapshot writes are atomic (tmp + fsync + rename + directory fsync): a
reader observes the old image or the new one, never a torn file.
Journal appends are fsynced per operation by default
(`FsyncPolicy::EachOp`); `OnSnapshot` trades the crash-window of the
journal tail for speed. On open, the journal is replayed over the
snapshot deterministically; a torn tail from a crash mid-append is
detected, dropped and reported.

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
// (A read-only open requires an empty journal: snapshot/checkpoint first.)
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
| `verify()` | *content* consistency — stored text is valid UTF-8, the fact↔vector-slot bijection holds | one linear pass over the text + vector pools |
| `scrub()` | *byte-level* container integrity — each section's stored xxh3 and the whole-file hash (the ZFS-scrub model) | resumable; a read-handle op |
| `recover()` | *salvage* — drop the content-corrupt facts, rebuild, write a clean copy | rebuilds in RAM ≈ image size |

**`scrub()` — the bitrot detector.** A resumable iterator over the mapped
snapshot: each `next()` hashes up to a slice budget, so you pace it
yourself (run to completion, or a slice at a time on a background thread,
pausing/cancelling between slices). It holds a shared lock for its whole
life, reads the map linearly (pages fault in, get hashed, stay
reclaimable — it never residents the whole file), and reports the first
mismatch, naming the damaged section.

```rust,no_run
use plugmem_host::{Config, Database};

let ro = Database::open_readonly("agent.plugmem", Config::default())?;
// Verify every container byte, a slice at a time.
for step in ro.scrub()? {
    let progress = step?; // Err(Corrupt) names the first damaged section
    // e.g. report progress.done_bytes / progress.total_bytes to a UI
    let _ = progress;
}
# Ok::<(), plugmem_host::HostError>(())
```

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

`maintain` is O(database) and the first pass beyond the HNSW threshold
pays the vector-graph build (~1.6 ms per vector) — which is why it is
explicit by default: call `db.maintain(now)` on your schedule.

## Embedders

The `Embedder` trait is two methods (`dim`, batched `embed`).
`OpenAiCompatEmbedder` speaks the `/v1/embeddings` shape that OpenAI,
Ollama, LM Studio, vLLM and llama.cpp-server share — there is no
provider-specific client because there is no provider-specific
protocol. The dimension is configured explicitly (no startup probe);
a server answering with a different one is a typed error. Tests run
against a local mock — no network in CI.

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
