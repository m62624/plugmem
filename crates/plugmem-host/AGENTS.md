# Local guide: `plugmem-host`

## Role

`plugmem-host` is the native `std` layer around `plugmem-core`. It owns filesystem paths, generation files, manifest publication, append-only journals, locks, mmap read-only access, settings, embedders, checkpointing, verification, and salvage recovery.

The public writer is `Database`; the zero-copy reader is `ReadOnlyDatabase`. Core logic must remain in `plugmem-core`; this crate coordinates it with durable storage and process-level concurrency.

## File and generation model

`FileStorage` keeps the published snapshot generations immutable and appends mutations to a journal. A manifest points to the current generation. Checkpoint writes a fresh image to a temporary generation, flushes according to `FsyncPolicy`, publishes it atomically, and clears the journal. Opening performs crash cleanup for unpublished/orphan staging files.

Default checkpoint thresholds are 1024 operations or a 4 MiB journal, unless configured through `DatabaseBuilder`/`Settings`. `maintain_every_forgets` is optional and triggers policy-driven maintenance after tombstones accumulate. The default `Auto` path is a cheap no-op when nothing is pending; tombstone compaction stays disk-first.

Never overwrite a published generation in place. Preserve the temp-write, fsync, rename, manifest, and garbage-collection ordering when changing persistence code.

## Writer, reader, and mmap rules

`Database` provides read-your-writes through its in-process engine and shared/exclusive locking. Readers can run concurrently with a writer according to the lock policy. `ReadOnlyDatabase` pins an immutable checkpoint generation and maps it without copying the large pools; it requires a checkpointed database with an empty journal at open.

Read-only handles are snapshot-isolated. They do not move to a newer generation automatically; call `refresh()` and check the returned boolean. `generation()` is the freshness marker. Do not promise read-your-writes from a read-only handle.

The only unsafe operations here are the inherent `memmap2::Mmap::map` calls. `self_cell` keeps the mmap owner alive while the borrowed `Memory` points into it. The file is pinned/locked and the generation is immutable, so a concurrent truncate cannot invalidate the mapping. Keep unsafe blocks narrow and preserve the lifetime/lock proof in the adjacent safety comment.

## Integrity and recovery

- `verify()` checks content-level invariants such as valid stored text and fact/vector consistency.
- `scrub()` checks stored section hashes and the whole snapshot incrementally with a byte budget; it is a resumable integrity scan.
- `recover()` salvages content-corrupt facts into a different destination, compacts survivors, and leaves the source untouched.
- Structural snapshot damage is not a salvage case; return the typed error and restore from backup.

Do not conflate `verify`, `scrub`, and `recover`. Tests should cover torn journal tails, corrupted content, bad structure, generation publication, and destination safety.

## Workspaces

`Workspace` is an optional layer over many `Database` handles in one directory. Three invariants hold it together, and a change that breaks any of them is a change to reject:

**The core does not know workspaces exist.** One `Memory` is one database; `no_std`, `forbid(unsafe_code)` and the file format stay exactly as they are. A step that needs `plugmem-core` or a format field is not this feature.

**A name cannot represent a path.** `DbName` admits only `[a-z0-9][a-z0-9_-]*`, so traversal is unconstructible rather than filtered, and resolution is a join. Windows device names (`con`, `nul`, `com1`, …) are refused on every platform: Windows resolves them as devices in any directory and with any extension, and a workspace directory is a thing people copy between machines. Lowercase-only is not style — case-insensitive filesystems would make `Work` and `work` one file.

**The directory is the truth; the registry is a rebuildable index.** Each database describes itself in a fact on the reserved entity, so `reindex` derives the registry from the data. Losing the registry costs search, never data. `verify` reports disagreement and repairs nothing — a workspace is a directory a person can edit, and guessing at their intent is how a consistency check becomes a data-loss bug.

Two behaviours that look like oversights and are not: the pool lock is held across an open (so two callers cannot race and have the loser told it is busy by its own process), and a handle handed out outlives its pool entry (the lock goes when the last `Arc` clone does). Both are documented on the types.

The idle timeout exists for **liveness**, not memory: an open writer holds the exclusive lock, so without the sweep a long-running server makes its databases permanently unreachable from the CLI.

## Embedders and settings

`Embedder` is a `Send` trait. `NullEmbedder` disables automatic vectors; `OpenAiCompatEmbedder` talks to an OpenAI-compatible HTTP endpoint and must be kept outside the engine lock where possible. Network/model latency is not storage latency.

The optional `config` feature parses TOML settings for engine, embedder, maintenance, and shared wrapper configuration. The optional `counters` feature is for deterministic performance measurements; it changes synchronization behavior and must not be enabled for normal concurrent-reader use.

## Verification commands

```bash
cargo test -p plugmem-host
cargo test -p plugmem-host --all-features
cargo test -p plugmem-host --test concurrency
cargo run --release -p plugmem-host --example integrity
cargo run -p plugmem-host --example edge_lifecycle
cargo run -p plugmem-host --example maintain_modes
cargo run --release -p plugmem-host --example recall_ollama
cargo run --release -p plugmem-host --example bench_database -- 100000 --diagnose-recall
cargo run --release -p plugmem-host --example bench_edges -- 100000
cargo bench -p plugmem-host --bench integrity
```

When changing persistence, also run the core snapshot/journal tests and inspect the generated files with `verify`/`scrub`. Use a temporary directory for tests; never point destructive recovery tests at a user database.
