# 06 — plugmem-host: the file layer and access orchestration

The host is part of the Rust experience ("point at a file and go"), not a wrapper for
non-Rust callers (those are CLI/MCP/napi, `07-wrappers.md`).

## Crate contents

1. `FileStorage` — a `plugmem_core::Storage` over files.
2. `Database` — the main type: the engine + files + a maintenance policy behind one
   lock; a clonable, thread-safe handle (`Arc`).
3. `ReadOnlyDatabase` — a read-only, zero-copy mmap handle (`recall`/`get`/`stats`,
   plus `scrub` and cross-process freshness).
4. `Embedder` — the trait from `05-api.md` plus `OpenAiCompatEmbedder` and
   `NullEmbedder`.
5. `HostError` — the layer's typed errors.

## Concurrency model

The core is a single-threaded single-writer by construction (`00-overview.md`). The
host does not "improve" this — it orchestrates it honestly at three levels:

| Level | Mechanism | Behavior |
|---|---|---|
| **One database, many processes** | an exclusive OS advisory lock on `<base>.lock` (std `File::try_lock`, no new dependency) | a second **writer** gets `HostError::Locked` immediately, no wait. A loud refusal instead of silent corruption |
| **One database, many threads/agents of one process** | `Database` = `Arc<Inner>`, engine state behind an `RwLock`; the handle is `Clone + Send + Sync` | reads overlap; writes serialize. At microsecond core operations the short write queue is not a bottleneck |
| **Different databases** | independent `Database`s | full parallelism: different mutexes, different lock files. No "database manager" type is needed — open as many `Database`s as needed |

**Network outside the lock.** The expensive external step — computing an embedding over
HTTP — happens **before** the mutex is taken. While one agent waits on the embedder,
others read and write. The critical section holds only microsecond core calls (plus
fsync per policy).

## FileStorage

Layout for `base = "agent.plugmem"`: `agent.plugmem` (snapshot), `agent.plugmem.journal`
(append-only journal), `agent.plugmem.lock` (advisory lock file), `agent.plugmem.tmp`
(atomic-write temp), and immutable generation images `agent.plugmem.snap.<N>`.

- `write_snapshot`: tmp → `fsync(tmp)` → `rename` over the snapshot → `fsync` the
  directory (unix). A reader never sees a half file.
- `append_journal`: `O_APPEND` + `write_all`; fsync per policy.
- `clear_journal`: truncate + fsync.
- fsync policy (`FsyncPolicy`): `EachOp` (default) | `OnSnapshot` (faster; the loss
  window is the journal tail on an OS crash).
- The lock is taken in `open` and lives until `Drop` (closing the fd releases it even
  on an abnormal exit — an OS advisory-lock property).

### Versioned MVCC (writer + N readers, cross-process)

A read-write `open` takes an **exclusive** writer lock on `*.lock` — only against other
writers (a second writer → `Locked` at once). **Readers take no writer lock:**
`open_readonly` (see `03-snapshot.md`) **pins the current generation** by taking a
*shared* lock on the `*.snap.<N>` file itself and maps it. Generations are immutable (a
checkpoint writes a new one and never overwrites an old one), so **a writer and N
readers run at the same time** — cross-process, no blocking, no torn read. This is a
WAL/MVCC model, not "N readers OR one writer". GC of an old generation is pin-aware:
the writer removes it only when an exclusive `try_lock` on it succeeds (no reader is
pinning it). A reader sees state "as of the last checkpoint" (snapshot isolation;
reopen to see newer).

## Database — the maintenance policy

`Database::open(path, Config)` / a `Builder`:

| Knob | Default | Meaning |
|---|---|---|
| `fsync` | `EachOp` | see above |
| `snapshot_every_ops` | 1024 | auto-`snapshot` after N mutations |
| `snapshot_journal_bytes` | 4 MiB | …or when the journal grows past this |
| `maintain_every_forgets` | `None` | optional auto-`maintain` after N forgets (checked at a snapshot boundary) |

An auto-snapshot runs **inside the same critical section** as the operation that
triggered it — the next caller just waits a little longer on the mutex; there are no
background threads (the core principle). `maintain` is explicit by default: the first
maintain past the HNSW threshold pays the graph build (~1.6 ms/vector,
`08-performance.md`) — the process owner's decision, not a silent pause.

`Database::reembed` / `reembed_with` are a distinct explicit boundary. They
checkpoint a frozen source generation, release the state lock for every bounded
provider batch and snapshot write, rebuild the vector pool and HNSW, then
publish one new generation atomically. Reads continue against the source;
writes fail immediately with `HostError::ReembedBusy`. Provider or staging
failure leaves the source generation current. `maintain`, including automatic
maintenance after a write, has no code path into reembed.

`Database`'s verbs mirror the core
(`remember/remember_guarded/recall/revise/forget/link/get/stats/maintain/checkpoint`) with two
conveniences: **auto-embedding** (if an `Embedder` is configured, a `remember` without a
vector embeds the text and a `recall` with text and no vector embeds the query — both
before the lock; an explicitly passed vector is always honored; `NullEmbedder`/no
embedder = a purely structural database, complete by design) and input ownership
(methods take `&str`/slices like the core, no extra copies).

`remember_guarded` performs automatic embedding before the write guard, then
holds that guard across the bounded similarity check and conditional insert.
This prevents two callers from both passing a read-only preflight. A blocked
outcome calls neither persistence bookkeeping nor automatic maintenance and
therefore leaves the journal, ids, terms and indexes unchanged. Ordinary
`remember` remains a complete atomic write that always stores.

**Integrity and recovery.** Beyond the mirror verbs the host adds `verify()` (content
consistency), `scrub()` on **either** handle (byte-level container integrity, a
resumable iterator holding its own shared lock on the generation it pins — it reads the
file, not a handle's view of it, and needs a published generation but not a clean
journal), and `recover(src, dst, cfg, now)` (drop content-broken
facts, write a clean copy, src untouched). `recover` and `maintain` are **disk-first**:
the large pools (vectors, text) stream through a temp `Scratch`, keeping only metadata
and the HNSW graph in RAM, so a rebuild is not bounded by the image size. Open is
trust/sparse by default (the image is not checksummed).

## Embedder

The trait is as in `05-api.md`. **There is no native Ollama client:** Ollama, LM
Studio, vLLM, llama.cpp-server, OpenAI, OpenRouter and so on all speak one
OpenAI-compatible `/v1/embeddings`, so one `OpenAiCompatEmbedder` covers them all (for
Ollama, `endpoint_url = "http://localhost:11434/v1/embeddings"`). `OpenAiCompatEmbedder::new(endpoint_url,
model, dim)` + `.with_api_key(...)`: `dim` is given explicitly (determinism and no
startup network call; a mismatch with the server's response is a typed error). HTTP is
`ureq` (blocking, small — the core is synchronous, no async is needed), JSON is
`serde_json`. A built-in local embedder (candle, e5-small) is v1.1.

The built-in embedder's readable `space_id` defaults to its model name and can
be overridden with `OpenAiCompatEmbedder::with_space_id` or
`[embedder].space_id`. The id is a local declaration, never a network probe,
and is persisted beside the vectors. Opening an existing database with new target
settings adopts the stored dimension for normal reads; automatic embedding
then returns `VectorSpaceMismatch` until the caller explicitly reembeds.

**When it cannot be reached.** A provider that refuses the connection, times
out or answers with something other than the documented shape used to fail the
verb, and that was the only option: `remember` and a text `recall` both embed,
so a stopped provider took every write and every meaning-based read with it —
while the database itself was perfectly usable, exactly as it is when no
embedder was ever configured.

`[embedder].on_error` decides which happens:

- `fail` (the default, and what every release before 0.12 did) propagates
  `HostError::Embed`.
- `degrade` carries on **without** the vector: the fact is stored, the query is
  answered from the lexical, tag, graph and time sources, and the embedder is
  *suspended* so the next call does not pay the same failure again. Nothing is
  lost that a later `reembed` cannot restore from the stored text — a fact
  without a vector is the state every fact has in a database written with no
  embedder.

The suspension retries by itself: `[embedder].retry_after_ms` unset means one
second, doubling up to `retry_max_ms` (60s), reset by the first success; `0`
means never (the host calls `resume_embedder`); any other value is that fixed
interval. There is no timer and no background probe — the retry rides on the
next call that wanted an embedding anyway.

Three failures are deliberately **not** degraded, under either policy:

- `VectorSpaceMismatch`. The provider answered, and its answer belongs to a
  different semantic space; storing it would mix two spaces in one index, and
  no later repair can tell the halves apart.
- `reembed` while suspended. It refuses rather than publishing a generation
  with a fraction of its vectors, and says "suspended" rather than "you
  configured none".
- An explicit `Database::suspend_embedder()`. It outranks any retry timer,
  because it was a decision rather than an observation; `resume_embedder()`
  undoes it.

`[embedder].timeout_ms` (10s by default, `0` = wait indefinitely) bounds one
request end to end. Without it, degrading is late by however long a hung server
hangs — which is the failure it is most needed for, since a provider that is
simply not listening refuses the connection immediately.

All of it lives in `EmbedderGate` rather than in `Database`, because there are
two callers and they must not drift: a writer embeds inside its verbs, and every
wrapper over a zero-copy `ReadOnlyDatabase` embeds its own query (the reader
carries no provider by design). `Database::embedder_state()` — and the gate's
own `state()` — reports `Absent`, `Active` or `Suspended { retry_at }`, which is
what lets a surface tell a person that their memory is running without
meaning-based ranking and will try again by itself.

**No lock sits in front of it.** `embed` takes `&self` (`05-api.md`), so a
`Database` holds its embedder unlocked and `SharedEmbedder` — the handle a
workspace clones into every memory so a hundred chats do not open a hundred HTTP
clients — is a plain refcount. Concurrent verbs are inside the provider at the
same time, which is the arrangement a batched remote service is built for. The
round trip still happens **outside** the engine lock: a write embeds before it
takes the state lock, and a read-only handle embeds in its wrapper because the
engine cannot embed into a zero-copy mapping.

## Errors

```rust
#[non_exhaustive]
pub enum HostError {
    Locked { path },      // the database is busy (another process/handle)
    Io { path, source },  // a file operation
    Engine(plugmem_core::Error),
    Embed(String),        // embedder transport / response format
    ReembedBusy,          // frozen source; retry this write after reembed
}
```

## Test plan (mandate ≥ 90%)

- FileStorage: roundtrip open→remember→drop→reopen (journal replay); checkpoint →
  snapshot on disk, empty journal; the tmp file does not survive; a torn journal tail →
  open with `truncated_tail`.
- Lock: a second `open` of the same file → `Locked` (across processes and within one);
  after the first `drop`, it opens.
- Policy: `snapshot_every_ops = N` → empty journal after N mutations;
  `maintain_every_forgets` fires.
- Embedder: a mock server on `std::net::TcpListener` (no network in CI): a correct
  response → vectors; wrong dimension / bad JSON / non-200 → an `Embed` error;
  auto-embedding end-to-end.
- Concurrency: several threads on clones of one `Database` — all operations succeed, the
  fact count converges, no deadlocks. Cross-process: external readers coexist with a
  churning writer, never `Locked`, never torn (the MVCC invariant tests).
- Reembed: provider work holds no state lock; readers progress, writes fail
  fast, failure preserves the old generation, success survives reopen, and
  `Auto` never calls the model after a configured-space change.
- All temp files are in unique subdirectories of `std::env::temp_dir`, cleaned up.
