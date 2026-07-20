# plugmem-host

The native host layer for the [plugmem](../plugmem-core) memory engine:
point it at a file and go — the SQLite-style experience. It supplies the
three things the `no_std` core deliberately does not own — files,
locking, and network access to embedding providers.

If you need the engine itself (your own storage, wasm, tighter control),
depend on [`plugmem-core`](../plugmem-core) directly; this crate is a
convenience shell around it. Non-Rust surfaces (CLI, MCP server,
npm/wasm package) are the next roadmap stage.

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

Without an embedder the database is fully functional — BM25, tags,
entity graph and time still answer; vectors are an addition, not a
requirement.

Native builds are 64-bit, so a host process reads every capacity class
of the shared file format: databases sized for the 32-bit wasm budget
(≤ 2 GiB, the default) and databases with larger limits alike. The
snapshot format is pointer-width independent — a file written here opens
unchanged in a wasm32 or wasm64 build of the core, as long as its
configured limits fit that host (see the core README, "WebAssembly 2.0
and 3.0").

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
honestly instead of hiding it:

- **One file, one process.** `Database::open` takes an exclusive OS
  advisory lock; a second process (or a second `Database` on the same
  path) gets `HostError::Locked` immediately — a typed refusal instead
  of silent corruption. The lock dies with the process, even on a
  crash.
- **One process, many threads or agents.** `Database` is a
  `Clone + Send + Sync` handle; clone it freely. Every verb serializes
  on an internal mutex — at microsecond engine calls that is a queue,
  not a bottleneck.
- **Many files.** Fully independent databases: separate locks, separate
  mutexes, natural parallelism. Two models each with their own memory
  file never contend; two models sharing one memory clone one handle.
- **Network stays outside the lock.** Embedding calls (the slow,
  external part) run before the mutex is taken, so an agent waiting on
  its embedding provider does not stall the others.

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
