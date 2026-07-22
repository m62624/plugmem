# plugmem-host

`plugmem-host` is the `std` host layer for the plugmem
[temporal-memory engine](https://docs.rs/plugmem-core/latest): point it at a file path and go.
It supplies the things the `no_std` engine deliberately does not own —
files, locking, and network — so that from this one crate an agent gets
`remember / recall / revise / forget` backed by durable storage. The
retrieval itself (BM25, int8-quantized vectors with
[HNSW](https://arxiv.org/abs/1603.09320), an entity graph, temporal
range scans, all fused by rank) lives in the engine; this crate adds:

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
  SQLite model): an open checks only the metadata, so the large text and
  vector pools stay non-resident until a query touches them — a measured
  open residents well under half of a text-heavy image. The default open
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

## Which crate do you need?

| You want | Depend on |
|---|---|
| point it at a file path and go — durability, locking, read-only mmap, auto-embedding | **this crate** (`std`; re-exports the engine's types) |
| the engine alone with your own storage (a browser, a wasm host, custom persistence) — BM25/HNSW/graph/time included, no files or network | [`plugmem-core`](https://docs.rs/plugmem-core/latest) (`no_std`) |
| the flat byte data structures underneath | [`plugmem-arena`](https://docs.rs/plugmem-arena/latest) (`no_std`) |
| no Rust at all: a CLI, an MCP server for agents, an npm package | `plugmem-cli` / `plugmem-mcp` / `plugmem-wasm` — in progress, not published yet |

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

The engine is single-writer by design, and the host orchestrates that
honestly instead of hiding it — the SQLite model: **N concurrent readers,
or one writer, never both at once.**

- **One writer.** `Database::open` takes an *exclusive* OS advisory lock;
  a second writer (or a live reader) gets `HostError::Locked` immediately
  — a typed refusal instead of silent corruption. The lock dies with the
  process, even on a crash.
- **Many readers.** `Database::open_readonly` takes a *shared* lock, so
  any number of read-only handles map the same snapshot at once — across
  threads or processes. Shared excludes exclusive, so no writer can change
  the file under a live reader (which is exactly what makes the mmap
  safe). With the zero-copy mmap, the readers also share one copy of the
  file in the OS page cache — a read-mostly database serves a fan-out of
  agents cheaply. When you need to write, drop the readers and open
  read-write.
- **One process, many threads or agents.** A `Database` is a
  `Clone + Send + Sync` handle; clone it freely. Every verb serializes
  on an internal mutex — at microsecond engine calls that is a queue,
  not a bottleneck. `ReadOnlyDatabase` is `Send + Sync` too.
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

## License

MIT.
