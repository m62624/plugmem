# 03 — Snapshot, journal, the Storage trait

Persistence is "an image of memory plus a journal of operations". The core does not
know where the bytes live: every contact with the world goes through `Storage`. The
native wrappers provide files; a wasm host provides callbacks.

## Persistence model

- **Snapshot** — a full image of engine state: the arena sections (see the contracts
  in `01-arena.md`) concatenated, 64 B aligned, with a header and a section table. A
  load is one in-memory buffer → metadata validation → the structures wrap their own
  slices. No parsing, no per-element allocation.
- **Journal** — an append-only log of the mutating operations since the snapshot,
  giving durability without rewriting the image on every remember.
- A load is snapshot + journal replay. `snapshot()` = serialize the current state →
  `write_snapshot` → `clear_journal`. When to snapshot is a wrapper policy (e.g.
  journal > 4 MB or > 1000 entries, or an explicit command).

## The Storage trait (no_std)

```rust
pub trait Storage {
    type Error: core::fmt::Debug;
    /// None — no database yet (first run).
    fn read_snapshot(&mut self) -> Result<Option<Vec<u8>>, Self::Error>;
    /// Atomic image replacement (tmp + rename in the file implementation).
    fn write_snapshot(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
    /// All journal bytes (empty is fine).
    fn read_journal(&mut self) -> Result<Vec<u8>, Self::Error>;
    /// Append one journal entry (the operation's durability point).
    fn append_journal(&mut self, entry: &[u8]) -> Result<(), Self::Error>;
    fn clear_journal(&mut self) -> Result<(), Self::Error>;
}
```

The core calls `append_journal` synchronously at the end of each mutating operation.
The fsync policy is the implementation's business (`FileStorage`: `fsync_each` /
`fsync_snapshot_only`; default each — durability beats microseconds and the write
volumes are tiny). Implementations: `plugmem-host::FileStorage` (two files,
`<name>.plugmem` + `<name>.journal`), `plugmem-core::MemStorage` (tests, ephemeral
databases), and a wasm bridge to JS callbacks (see `07-wrappers.md`).

**Locking: one database, one process.** On open `FileStorage` takes an exclusive
advisory lock (flock/LockFileEx on a lock file beside the database) and holds it for
the handle's lifetime. A second writer gets `Error::Locked` ("database is locked by
another process") with no wait. Read-only handles instead take a *shared* lock, so
N readers OR one writer (the SQLite model; see the read-only path below and
`06-host.md`).

## Snapshot format

Everything is little-endian except the keys inside arena slots (those are BE — they
are data the format does not interpret).

```
Header (64 B):
  magic       u32   = 0x504C_474D ("PLGM")
  version     u16   = 1            // the format, not the crate version
  flags       u16   // bit 0: a vector section is present; reserved
  section_cnt u16
  reserved    [u8; 6]
  config_len  u32   // length of the Config block
  file_xxh3   u64   // hash of the whole file with this field zeroed (0 = none)
  created_at  u64   // informational
  engine_ver  [u8; 24] // semver string, informational

Config block: a fixed binary struct (see 05-api.md: dim, quantization, per-arena
  shards, max_*, hnsw params, rrf/recency, db_uuid — the database lineage id;
  ENCODED_LEN = 188). Everything that changes how bytes are interpreted lives here.

Section table: section_cnt x 32 B:
  kind   u16   // enum: ArenaFactsPool, ArenaFactsMeta, Blobs, Interner, Postings,
               // Temporal, EdgesOut, EdgesIn, Vectors, Sigs, Hnsw, Metas...
  align  u16   // = 64
  offset u64   // from file start, multiple of 64
  len    u64
  xxh3   u64   // per-section hash

Sections: contiguous, each 64-aligned.
```

The container hash covers the **whole file with the hash field zeroed** (so `flags`
and everything else is under the checksum, not just value-validation). The layout is
strictly **canonical**: sections in table order, all padding zero, no trailing bytes,
so dump → load → dump is byte-identical (tested per structure and for the container).
Uninitialized page tails and reused ChunkPool chunks are zeroed on dump (canonical +
no stale-data leak; miri confirms dump never reads uninitialized memory).

Rules:

1. An unknown `kind` on read → `UnsupportedSection` (v1 promises no forward-compat;
   see migrations).
2. A section expected by config but missing → `Corrupt`.
3. Each section's `len`/`offset` is bounds-checked against the buffer **before** any
   access; overlapping sections are rejected (checked by sorting).

## Validating untrusted input

A snapshot can come from anywhere (a wasm host, a foreign file). The loader contract:

- **never** panic or UB on any bytes — only `Err(LoadError)`;
- validation cost is O(metadata), not O(data): header, section table, arena metadata
  (chain cycles, bounds, counts), ChunkPool handles; slot *contents* are not checked
  and are validated lazily at the core level (a bad BlobId gives a typed error on
  access);
- the line between what a load checks and what it defers is **safety, not
  correctness**: a load range-checks every stored id, so no accessor can index past
  its structure; anything that only makes the data *disagree with itself* — the two
  edge mirrors, a current edge against its open history version, stored text, the
  vector bijection — is `verify()`'s job. Each of those costs a random lookup per
  record, which is most of the cost of opening a large database, and being wrong
  about one produces a wrong answer, never an unsafe read;
- **a size taken from the file may not become an allocation size.** Ids inside stored
  records are not range-checked, so any derived array sized from one is capped by
  what the data justifies and falls back to a lookup past the cap. `usize` is 32 bits
  on wasm32, so this is also where an offset computation has to be provably in range
  there — compute in `u64` and compare against a real slice length before casting.
  The config block's shard counts are the sharpest case: each is handed straight to an
  arena as the length of its per-shard vectors, so `validate` bounds them by
  `MAX_SHARDS` — a ceiling derived from the 32-bit page arithmetic and the per-arena
  metadata budget, and set high enough that no database a pool can hold ever reaches it;
- **open trusts the file by default** (the SQLite model): the container xxh3 is not
  read on open, so a large database opens sparse. Integrity is on demand: `scrub()` is
  byte-level (per-section + file_hash xxh3, resumable), `verify()` is content-level
  (text UTF-8, fact↔slot bijection, metadata well-formedness);
- a loader fuzz target is mandatory (see `08-performance.md`).

## Journal

An entry:

```
[len u32][xxh3_32 u32][op u8][payload ...]   // len = 1 + payload
```

Ops (payload is a compact binary format, strings as length-prefixed UTF-8):

| op | payload |
|---|---|
| 1 Remember | now, valid_from?, entity?, text, tags[], links[], metadata[], vector? (already i8-quantized — quantization happens before the journal, for determinism), assigned FactId |
| 2 Revise | now, target FactId, + the Remember fields, assigned FactId |
| 3 Forget | now, FactId |
| 4 Link | now, src entity, rel, dst entity, provenance? |
| 5 Maintain | now (a marker: replay after it knows a purge happened) |

Replay is strictly sequential; the ids in entries are authoritative (assigned at
execution), so replay is deterministic and idempotent on re-applying the tail
(entries with id ≤ the current max are skipped). A torn tail entry (a crash
mid-write) truncates the journal at the last valid entry with a warning in the load
report; a torn **non**-tail entry is `Corrupt`.

## Read-only mmap (zero-copy)

A read-only open maps the file and borrows the large pool sections instead of copying
them, so the OS pages in only what is actually touched — an 8 GiB database no longer
needs 8 GiB of RAM to open. The mechanism is one change in arena: each big `pool` is a
`Cow<'a, [u8]>` (which lives in `alloc::borrow`, so **arena stays no_std with zero new
dependencies**). The owned path (`new`/`open`/write/journal/snapshot/maintain) is
`Cow::Owned` and byte-for-byte unchanged (`Memory<'static>`); the borrowed path
(`open_readonly`) is `Cow::Borrowed(&mmap[..])` with mutation forbidden at the handle.
Only content-sized pools are borrowed (arena pages, blob pools, chunk pools, vector
slots); small metadata (page indices, tables) rebuilds owned, cheaply. `memmap2` is
the one new dependency and lives **only in plugmem-host**. The reader pins and maps
the current published generation and needs no journal, which is snapshot isolation —
it sees state as of the last checkpoint. Cross-process coexistence with a live writer
is in `06-host.md`.

## Versioning and migrations

- `version` in the header is the format version; minor engine releases do not change
  the format.
- A format change = version+1 plus an explicit migrator (`load_v(N)` → build state →
  `dump_v(N+1)`); no dual-read in the main path. The CLI gets a `migrate` command; a
  library user gets `migrate(bytes) -> Result<Vec<u8>>` behind a `migrate` feature so
  old loaders are not pulled into wasm by default.
- Before 1.0 the format may break freely; it freezes at the first public tag.

## Test plan

- Roundtrip: dump → load → dump — the second bytes equal the first (canonical format,
  a determinism test).
- Golden files: pinned snapshots in testdata; every future build loads each (catches
  an accidental format break).
- Corruption: systematic damage to each header/table/metadata field (a bitflip
  matrix) → always `Err`, never a panic; plus fuzz.
- Journal: a crash "between appends" (a break at each byte of the tail entry) →
  load with truncation; replay after a snapshot equals direct execution (property:
  random op sequences, compare states).
- Storage contract: one test suite run over both MemStorage and FileStorage.
