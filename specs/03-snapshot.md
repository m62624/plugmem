# 03 — Snapshot, journal, the Storage trait

Persistence is "an image of memory plus a journal of operations". The core does not
know where the bytes live: every contact with the world goes through `Storage`. The
native wrappers provide files; a wasm host provides callbacks.

This document is the **model**: why persistence is shaped this way and what the
rules are. The **byte layout is normative in `11-file-format.md`** — every
offset, every section kind, every journal op. The sketches here are orientation;
where the two disagree, 11 wins.

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
volumes are tiny). Implementations: `plugmem-host::FileStorage`,
`plugmem-core::MemStorage` (tests, ephemeral databases), and a wasm bridge to JS
callbacks (see `07-wrappers.md`).

`FileStorage` is **not two files.** The path the caller passes is a 24-byte
*manifest* naming the current snapshot generation; the image lives beside it as
`<name>.snap.<N>`, immutable and never rewritten, with the journal and the lock
file alongside. That is what lets a reader in another process map a stable
image while a writer keeps working — the MVCC arrangement in `06-host.md`. The
full on-disk layout, including the manifest's bytes, is `11-file-format.md` §1.

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
[header 64][config][pad][table: n x 32][pad][section][pad]...
```

The header carries the magic, format version, flags, section count, config-block
length, the file hash and two informational fields. The config block is a fixed
188-byte struct holding everything that changes how the following bytes are
*interpreted* (dim, per-arena shards, max_*, hnsw params, rrf/recency, and
`db_uuid`, the database lineage id). Each section-table entry carries a kind, the
alignment, the offset, the length and a per-section hash.

**The field offsets are in `11-file-format.md` §2**, along with the complete
section catalogue: the kinds are stable numbers 1..59 with retired ranges left as
gaps, and most structures contribute a small `meta`/`index` section plus a large
`pool` section, the small one first.

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
[len u32][check u32][op u8][payload ...]   // len = 1 + payload
```

`check` is the low 32 bits of xxh3-64 over `[op][payload]`. Six ops: Remember,
Revise, Forget, Link, Maintain, Unlink. Payloads are a compact binary format,
strings length-prefixed UTF-8 — the field order per op is in
`11-file-format.md` §9.

**The embedding is journaled as raw f32, before quantization.** Replay
re-quantizes with the same pure function and reproduces every vector slot byte
for byte; journaling the quantized form instead would tie the journal to the
quantizer's version. Metadata is likewise journaled as remembered and
re-canonicalized on replay.

Replay is strictly sequential; the ids in entries are authoritative (assigned at
execution), so replay is deterministic. A torn tail entry (a crash mid-write)
truncates the journal at the last valid entry with a warning in the load report;
a torn **non**-tail entry is `Corrupt`.

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
