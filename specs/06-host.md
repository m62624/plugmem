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
| **One file, many processes** | an exclusive OS advisory lock on `<base>.lock` (std `File::try_lock`, no new dependency) | a second **writer** gets `HostError::Locked` immediately, no wait. A loud refusal instead of silent corruption |
| **One file, many threads/agents of one process** | `Database` = `Arc<Inner>`, engine state behind a `Mutex`; the handle is `Clone + Send + Sync` | all verbs serialize on the mutex. At microsecond core operations that is hundreds of thousands of ops/s — a queue, not a bottleneck |
| **Different files** | independent `Database`s | full parallelism: different mutexes, different lock files. No "database manager" type is needed — open as many `Database`s as files |

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

`Database`'s verbs mirror the core
(`remember/recall/revise/forget/link/get/stats/maintain/checkpoint`) with two
conveniences: **auto-embedding** (if an `Embedder` is configured, a `remember` without a
vector embeds the text and a `recall` with text and no vector embeds the query — both
before the lock; an explicitly passed vector is always honored; `NullEmbedder`/no
embedder = a purely structural database, complete by design) and input ownership
(methods take `&str`/slices like the core, no extra copies).

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
Ollama, `base_url = "http://localhost:11434/v1"`). `OpenAiCompatEmbedder::new(base_url,
model, dim)` + `.with_api_key(...)`: `dim` is given explicitly (determinism and no
startup network call; a mismatch with the server's response is a typed error). HTTP is
`ureq` (blocking, small — the core is synchronous, no async is needed), JSON is
`serde_json`. A built-in local embedder (candle, e5-small) is v1.1.

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
- All temp files are in unique subdirectories of `std::env::temp_dir`, cleaned up.
