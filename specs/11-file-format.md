# 11 — File format

The normative byte layout. `03-snapshot.md` says *why* persistence is shaped
this way; this document says *what is in the bytes*, down to the offset, so a
third-party reader can be written from it and a format change can be reviewed
against it.

Everything here is **format version 1**, which is not frozen. During public
testing, current binaries still open published legacy v1 shapes through the
explicit migrations in `memory/migrations.rs`; the next checkpoint writes the
current canonical v1 representation (`03-snapshot.md`, "Versioning and
migrations"). Where this document and the code disagree, the code is right and
this document is a bug.

Source of truth per section: `plugmem-core/src/snapshot.rs` (container),
`config.rs` (config block), `memory/persist.rs` (section catalogue),
`model.rs` (record slots), `journal.rs` (journal), `index/*` (index dumps),
`plugmem-arena/src/*` (structure dumps), `plugmem-host/src/storage.rs`
(on-disk layout and the manifest).

## 1. On disk

A database is **not one file**. The path the caller passes is the *manifest*;
the image lives beside it in immutable numbered generations. This is what lets
a reader in another process map a stable snapshot while a writer keeps working
(`06-host.md`).

| Path | Role |
|---|---|
| `agent.plugmem` | **manifest** — 24 bytes naming the current generation |
| `agent.plugmem.snap.<N>` | **generation N** — a full snapshot image, immutable, never rewritten |
| `agent.plugmem.journal` | append-only journal of operations since generation N |
| `agent.plugmem.lock` | advisory lock file (writer-vs-writer; readers take it shared) |
| `agent.plugmem.snap.<N>.tmp`, `agent.plugmem.manifest.tmp` | staging for the atomic writes |

A checkpoint writes generation `N+1` in full, fsyncs it, then repoints the
manifest (tmp + fsync + rename + directory fsync) and garbage-collects
superseded generations that nothing pins. A reader therefore only ever observes
a manifest naming a generation that already exists.

### 1.1 Manifest

24 bytes, little-endian, rewritten whole on every checkpoint.

| off | size | field |
|---|---|---|
| 0 | 4 | magic `0x504D_474C` (`"LGMP"` little-endian) |
| 4 | 2 | manifest version = 1 |
| 6 | 2 | reserved, zero |
| 8 | 8 | current generation number |
| 16 | 8 | FNV-1a over bytes `0..16` |

A manifest that is the wrong length, has the wrong magic, or fails its checksum
reads as "no snapshot yet" rather than as an error: a torn manifest is
indistinguishable from a database that has never checkpointed, and both are
recoverable by replaying the journal.

## 2. Snapshot container

```text
[header 64][config][pad → 64][table: n × 32][pad → 64][section 0][pad → 64][section 1]...
```

Every block is padded with zeros to a 64-byte boundary. All container fields are
little-endian. Arena *keys* inside slots are big-endian, because they are
compared as byte strings — the container does not interpret them.

The layout is strictly **canonical**: sections appear in table order, contiguous,
all padding zero, no trailing bytes. `dump → load → dump` is byte-identical, and
that is a test, not an aspiration.

### 2.1 Header (64 bytes)

| off | size | field |
|---|---|---|
| 0 | 4 | magic `0x504C_474D` (`"PLGM"`) |
| 4 | 2 | format version = 1 |
| 6 | 2 | flags — bit 0 `FLAG_VECTORS`: vector sections present; rest reserved zero |
| 8 | 2 | section count |
| 10 | 6 | reserved, zero |
| 16 | 4 | config block length |
| 20 | 8 | file xxh3 |
| 28 | 8 | `created_at`, unix ms (informational) |
| 36 | 24 | engine version, UTF-8, zero-padded (informational) |
| 60 | 4 | reserved, zero |

The file hash covers **the whole file with the hash field (offset 20) zeroed**,
so `flags`, the section table and the config block are all under it. It is
**not** verified on open — opening trusts the file, the SQLite model, so a large
database opens sparse. Integrity is on demand: `scrub()` checks these hashes in
resumable slices, `verify()` checks content-level agreement.

### 2.2 Config block

`ENCODED_LEN = 188` bytes, immediately after the header, length in the header at
offset 16. Everything that changes how the following bytes are *interpreted*
lives here. All size fields are fixed-width `u64` so the block is identical on
32- and 64-bit builds; a value that overflows this platform's `usize` is
`ConfigMismatch` ("this host is too small"), not corruption.

| off | size | fields |
|---|---|---|
| 0 | 14 × 8 | `dim`, `max_bytes`, `max_text`, `max_blob`, `shards_facts`, `shards_entities`, `shards_edges`, `shards_temporal`, `shards_postings`, `hnsw_m`, `hnsw_m0`, `hnsw_ef_construction`, `hnsw_ef_search`, `flat_to_hnsw` — each `u64` |
| 112 | 10 × 4 | `bm25_k1`, `bm25_b`, `w_bm25`, `w_vec`, `w_graph`, `w_time`, `w_recency`, `graph_decay`, `similar_cos`, `similar_jaccard` — each `f32` |
| 152 | 3 × 4 | `rrf_k`, `half_life_days`, `graph_depth` — each `u32` |
| 164 | 16 | `db_uuid` — `u128`, the database lineage id |
| 180 | 8 | reserved, must be zero |

The shard counts are the sharpest untrusted-input case in the file: each is
handed straight to an arena as the length of its per-shard vectors, so `validate`
bounds them by `MAX_SHARDS` before anything is allocated.

### 2.3 Section table

`section_count` entries of 32 bytes each.

| off | size | field |
|---|---|---|
| 0 | 2 | kind (see §3) |
| 2 | 2 | alignment, always 64 |
| 4 | 4 | reserved, zero |
| 8 | 8 | offset from file start, multiple of 64 |
| 16 | 8 | length in bytes, before padding |
| 24 | 8 | xxh3 of the section body |

Note the reserved `u32` at offset 4: the offset field is 8-aligned inside the
entry, not packed against `alignment`.

Rules on read: an unknown `kind` is `UnsupportedSection` (v1 promises no
forward compatibility); a section the config requires but the file omits is
`Corrupt`; every `offset`/`len` is bounds-checked in `u64` arithmetic before any
access, and overlapping sections are rejected.

## 3. Section catalogue

Kinds are stable numbers, not an enum whose order may shift. Gaps are retired
layouts kept unused so an old file is diagnosed rather than silently
misread (`memory/migrations.rs::legacy_kind`): **9–12** and **46–49** were the
edge sections before the time-ordered history layout, **26–27** the per-document
BM25 records before they carried the term-set summary.

Most structures contribute a **pair**: a small `meta`/`index` section and a large
`pool` section. The small one comes first, because a reader wants it first, and
because only the pools are borrowed zero-copy from an mmap.

| Kind | Section | Contents |
|---|---|---|
| 1, 2 | `FACTS_META`, `FACTS_POOL` | the `FactRecord` arena (§4.1) |
| 3, 4 | `AUX_META`, `AUX_POOL` | the `FactAux` arena (§4.2) |
| 5, 6 | `ENTITIES_META`, `ENTITIES_POOL` | the `EntityRecord` arena (§4.3) |
| 7, 8 | `BY_NAME_META`, `BY_NAME_POOL` | the `EntityByName` arena (§4.4) |
| 13, 14 | `TEMPORAL_META`, `TEMPORAL_POOL` | the `TemporalSlot` arena |
| 15, 16 | `TEXTS_INDEX`, `TEXTS_POOL` | fact text blob heap (§5.2) |
| 17, 18 | `TERMS_INDEX`, `TERMS_POOL` | interner string heap (§5.2) |
| 19 | `TERMS_TABLE` | interner hash table (§5.4) |
| 20, 21 | `TAG_LISTS_META`, `TAG_LISTS_POOL` | per-fact tag lists, a `ChunkPool` (§5.3) |
| 22, 23 | `BM25_HANDLES_META`, `BM25_HANDLES_POOL` | BM25 term → handle arena (§6) |
| 24, 25 | `BM25_CHUNKS_META`, `BM25_CHUNKS_POOL` | BM25 posting chunks |
| 28, 29 | `TAGS_HANDLES_META`, `TAGS_HANDLES_POOL` | tag → facts handle arena |
| 30, 31 | `TAGS_CHUNKS_META`, `TAGS_CHUNKS_POOL` | tag → facts chunks |
| 32, 33 | `ENTFACTS_HANDLES_META`, `ENTFACTS_HANDLES_POOL` | entity → facts handle arena |
| 34, 35 | `ENTFACTS_CHUNKS_META`, `ENTFACTS_CHUNKS_POOL` | entity → facts chunks |
| 36 | `ENGINE_STATE` | the id counters and BM25 corpus stats (§7) |
| 37 | `VEC_POOL` | quantized vector slots (§8) |
| 38 | `HNSW_META` | `[entry u32][indexed u32]` |
| 39 | `HNSW_LEVEL0` | level-0 neighbor blocks, `u32` values |
| 40, 41 | `HNSW_UPPER_META`, `HNSW_UPPER_POOL` | upper-level handle arena |
| 42, 43 | `HNSW_LISTS_META`, `HNSW_LISTS_POOL` | upper-level list pool |
| 44, 45 | `METAS_INDEX`, `METAS_POOL` | fact metadata blob heap |
| 50, 51 | `EDGES_OUT_META`, `EDGES_OUT_POOL` | current edges keyed `(src, rel, dst)` (§4.5) |
| 52, 53 | `EDGES_IN_META`, `EDGES_IN_POOL` | the same edges keyed `(dst, rel, src)` |
| 54, 55 | `EDGE_HIST_OUT_META`, `EDGE_HIST_OUT_POOL` | edge history keyed `[a \| valid_from \| edge]` (§4.6) |
| 56, 57 | `EDGE_HIST_IN_META`, `EDGE_HIST_IN_POOL` | the mirrored history |
| 58, 59 | `BM25_DOCLEN_META`, `BM25_DOCLEN_POOL` | per-document BM25 records with the term-set summary |
| 60 | `TAG_CATALOG` | sorted `[TermId u32 LE, current_fact_count u32 LE]` entries |

The eight edge sections (50–57) are written together. All absent means an empty
database or one written before the history layout, which `migrate_edges`
rebuilds; present but incomplete is corruption.

Sections 37–43 are the vector layer and appear only when `FLAG_VECTORS` is set.

`TAG_CATALOG` is derived: facts, their tag lists and tag postings remain the
source of truth. A v1 image written before kind 60 is valid; `migrations.rs`
rebuilds the catalogue on open and the next checkpoint persists it. Zero-count
entries are omitted. The shared interner may retain unused tag strings as
historical residue.

## 4. Record slots

Slots live inside arena pool sections. Every offset below is the previous
field's offset plus its width, so no field can be moved by editing one number.

A **Uniform** arena shards by hashing the key; an **Ordered** arena shards by key
prefix and keeps pages in ascending key order, which is what makes a prefix range
scan a traversal.

### 4.1 `FactRecord` — 48 bytes, Uniform

| off | size | field |
|---|---|---|
| 0 | 4 | `id` (key) |
| 4 | 4 | `entity` |
| 8 | 2 | `flags` (tombstone / closed / has-vector) |
| 10 | 2 | `kind` (reserved, 0 in v1) |
| 12 | 4 | `text` (blob id into `TEXTS_*`) |
| 16 | 4 | `vector` (slot index into `VEC_POOL`) |
| 20 | 4 | `revises` (predecessor fact, `NONE` sentinel) |
| 24 | 8 | `recorded_at` — the knowledge axis |
| 32 | 8 | `valid_from` — the truth axis start |
| 40 | 8 | `valid_to` — truth axis end, or `VALID_TO_OPEN` (`u64::MAX`) |

### 4.2 `FactAux` — 20 bytes, Uniform

`[id 4 | ListHandle 12 | meta 4]`. Split out so the hot 48-byte record stays
hot: tags and metadata are touched only by tag-filtered queries, `show`/`export`
and `maintain`.

### 4.3 `EntityRecord` — 24 bytes, Uniform

| off | size | field |
|---|---|---|
| 0 | 4 | `id` (key) |
| 4 | 4 | `name` (blob id) |
| 8 | 4 | `name_term` (interned term) |
| 12 | 8 | `created_at` |
| 20 | 4 | `flags` (reserved, 0 in v1) |

### 4.4 `EntityByName` — 8 bytes, Ordered

The whole slot is the key: `[name_term BE | id BE]`. The normalized name is
unique, so a prefix scan on `name_term` yields at most one record.

### 4.5 `EdgeSlot` — 28 bytes, Ordered

Key `[a BE | rel BE | b BE]`, payload `fact | edge | valid_from`. Stored twice
in two mirrored arenas — out keys by `(src, rel, dst)`, in by `(dst, rel, src)` —
so neighbor traversal in either direction is a prefix range scan. The slot
carries the identity of its open `EdgeHistorySlot` version, so closing an edge
addresses that record directly instead of searching the triple's versions.

### 4.6 `EdgeHistorySlot` — 48 bytes, Ordered

Key `[a BE | valid_from BE | edge BE]`: **time-ordered per entity**, not grouped
by relation. At most one version of a triple is valid at any instant, so grouping
by triple would force a walk through every version of every triple to find the
few that answer an `as_of(t)`. Ordering by `valid_from` lets the traversal start
at `t`, walk backwards through the versions that most recently became true, and
stop when it has enough.

## 5. Structure dumps

### 5.1 Arena — `meta` + `pool`

`meta`: `[shards u32][pages u32][free_head u32][total u64][mode u8][reserved 3]`,
then `heads` (`shards × u32`), `next` (`pages × u32`), `counts` (`pages × u16`).
`mode` is 0 Uniform, 1 Ordered.

`pool`: each page contributes its initialized prefix (`counts[page] × SIZE`
bytes) then zero padding to the page size. The uninitialized page tails are
never read, and zero-filling them makes the image canonical and leak-free (miri
confirms a dump never reads uninitialized memory).

Load validates in O(pages): exact section lengths, config agreement, per-page
counts within a page, every page reachable exactly once via a visited bitmap (no
cycles, no shared or orphan pages), chain pages non-empty and free pages empty,
chain pages ascending by key range and sitting in the shard their first key maps
to, and the record total matching the page counts.

### 5.2 Blob heap — `index` + `pool`

`index`: `[blobs u32]` then one `len u32` per blob. Offsets are **not** stored —
blobs are contiguous in push order, so the lengths alone reconstruct the index,
one redundancy less to validate.

`pool`: a straight copy of the logical pool, every byte initialized blob content.
An overlay heap dumps byte-identically to an owned heap holding the same blobs.

The interner's string heap uses this same pair of layouts.

### 5.3 Chunk pool — `meta` + `pool`

`meta`: `[chunks u32][free_head u32]` then one `used u8` per chunk. Free chunks
are written `used = 0` regardless of their stale in-memory value.

`pool`: each chunk contributes its link, its used payload prefix, and zero
padding to the chunk size; free chunks contribute link plus zeros. This
canonicalizes the stale bytes that recycling leaves behind — identical logical
state, identical bytes.

### 5.4 Interner hash table

`[slots u32][len u32]` then `slots × u32` entries: `0` is empty, otherwise
`TermId + 1`. The table is persisted rather than rebuilt on load, because
rebuilding is O(terms × hash) and the cold-start budget is tighter than the 4
bytes per slot.

## 6. Posting lists

One shape serves BM25, tag → facts and entity → facts: an arena of per-key
handle slots plus a `ChunkPool` holding the lists.

Per-key slot, 24 bytes, Uniform: `[key u32 | ListHandle 12 | count u32 | last u32]`.
`count` doubles as the document frequency for BM25; `last` is the highest id in
the list, which is the delta base.

Entries are appended in ascending fact-id order (ids are monotonic, so appends
keep lists sorted for free) and encoded as varint deltas: `[delta]` for plain id
lists, `[delta][tf u8]` for BM25. Each entry is pushed as one `ChunkPool` value,
so entries never straddle a chunk boundary and decoding walks whole entries per
chunk slice.

Load fully decodes every posting list: well-formed varints, ascending ids, and
agreement between the decoded count, the stored `count` and the stored `last`.

## 7. Engine state

`ENGINE_STATE` is 40 bytes:

| off | size | field |
|---|---|---|
| 0 | 4 | `next_fact` |
| 4 | 4 | `next_entity` |
| 8 | 8 | `bm25_docs` |
| 16 | 8 | `bm25_total_len` |
| 24 | 4 | `tokenizer_version` |
| 28 | 4 | reserved |
| 32 | 8 | `next_edge` |

Two shorter historical lengths are still accepted and identify older images: a
section shorter than 40 bytes predates edge versioning, and the loader records
that so `migrate_edges` knows to rebuild.

## 8. Vector slots

`VEC_POOL` is one contiguous run of fixed-stride slots — deliberately not an
arena or a blob heap, because a flat search reads every live slot and perfect
locality matters more than sorted lookup. Slots are append-only; dead ones are
dropped by `maintain`, never reused in place.

With `words = ceil(dim / 64)`, one slot is `8 + 8·words + dim` bytes:

| off | size | field |
|---|---|---|
| 0 | 4 | `fact` — the owning `FactId` |
| 4 | 4 | `scale` — f32 quantization scale |
| 8 | 8·words | `sig` — sign signature, bit `i` set iff `q[i] >= 0` |
| 8+8·words | dim | `q` — the i8 components |

Quantization is symmetric i8 over the L2-normalized vector
(`scale = max|x/‖x‖| / 127`, `q = round((x/‖x‖) / scale)`), so a quantized cosine
is `scale_a · scale_b · Σ qa·qb`. It is a pure function of the input, which is
what lets journal replay re-quantize and reproduce every slot byte for byte.

f32 vectors are never stored in the snapshot. The signature is the first half of
the two-phase search: a Hamming prefilter by popcount over `u64` words, then an
exact quantized-cosine rescore of only the best `max(4·k, 64)` candidates.

## 9. Journal

Framing, op-agnostic:

```text
[len u32 LE][check u32 LE][op u8][payload ...]      len = 1 + payload
```

`check` is the **low 32 bits of xxh3-64** over `[op][payload]`.

Strings in payloads are `[len u32 LE][bytes]`, UTF-8, checked on read. A vector
is `[count u32 LE][count × f32 LE]`, with the bounds computed in `u64` before any
allocation — on a 32-bit target `count * 4` in `usize` can wrap and slip a
hostile count past the check, and `with_capacity` on an unchecked count aborts a
wasm32 process.

| op | operation | payload |
|---|---|---|
| 1 | Remember | `now`, `valid_from` (already resolved — replay never re-derives it), `entity?`, `text`, `tags[]`, `links[]` as `(rel, target)`, `vector`, `metadata[]`, `revises` (`NONE`), `assigned` FactId |
| 2 | Revise | as op 1, with `revises` naming the predecessor |
| 3 | Forget | `now`, FactId |
| 4 | Link | `now`, `src`, `rel`, `dst`, `provenance` FactId (`NONE` if absent) |
| 5 | Maintain | `now`, `mode u8`, `max_hnsw_inserts u32` (`u32::MAX` = unlimited) |
| 6 | Unlink | `now`, `src`, `rel`, `dst` |
| 7 | RemoveTag | `now`, `tag` |

**The vector is journaled as raw f32, before quantization.** Replay re-quantizes
with the same pure function and reproduces every slot byte for byte; journaling
the quantized form instead would make the journal depend on the quantizer's
version. Metadata is likewise journaled as remembered and re-canonicalized
(sorted, deduped) on replay.

Ids inside entries are authoritative — assigned at execution, not re-derived —
so replay is deterministic.

### 9.1 Torn tails

There is exactly one writer and appends are sequential, so a crash can only
leave a *prefix* of the last record. That gives a clean rule:

- a frame extending past the end of the buffer is the torn tail: the scan
  succeeds, drops it, and reports `truncated_tail` in the open report;
- a complete frame with a bad checksum that ends exactly at the buffer end is
  also a torn tail (a torn write inside the final record's payload looks like
  this);
- anything else — a bad checksum mid-stream, a `len` of 0 — is `Corrupt`.

## 10. What a load checks, and what it defers

The line is **safety, not correctness**.

A load range-checks every stored id — facts' blob, entity and revision
references, edge endpoints, temporal and by-name entries — so no accessor can
index past its structure afterwards. That is what keeps the engine's panicking
accessors (`get`, `resolve`, contract-violation panics by design) sound on
arbitrary bytes. Chunk chains are walked with shared visited maps for cycles,
double-claims and orphans; posting lists are fully decoded; text and term pools
are UTF-8-checked.

Anything that only makes the data *disagree with itself* is `verify()`'s job,
not the loader's: the two edge mirrors against each other, a current edge against
its open history version, stored text, the vector bijection. Each of those costs
a random lookup per record — most of the cost of opening a large database — and
being wrong about one produces a wrong answer, never an unsafe read.

Validation cost is O(metadata), not O(data). Checksums are not verified at open
at all; `scrub()` does that on demand.

The loader must **never** panic or exhibit UB on any input — only `Err`. A
loader fuzz target enforces it (`08-performance.md`).

## 11. Writing without materializing the image

A snapshot is written twice over in the API, and the difference matters for
memory.

`SnapshotWriter::finish` builds the whole file in one `Vec<u8>`. It is the
simple path, used by tests and small images.

Production streams instead. `build_prefix` lays out the header, config block and
section table from per-section `(kind, len, hash)` metadata gathered in a first
pass, byte for byte identically to `finish`. The caller then writes each section
body followed by its zero padding through a `SnapshotSink`, feeding a running
xxh3 as it goes, and finally patches the header's file-hash field at
`FILE_HASH_OFFSET` — the one non-sequential write in the whole operation. Peak
memory is the prefix plus the largest single piece, not the image.

This is why RAM during a checkpoint is proportional to the number of records
rather than to the size of their content.

## 12. Zero-copy read

A read-only open maps the generation file and **borrows** the large pool
sections instead of copying them, so the OS pages in only what is touched. The
mechanism is one change in arena: each big pool is a `Cow<'a, [u8]>`, which
lives in `alloc::borrow` and so keeps arena `no_std` with no new dependency.

The owned path (`new`/`open`/write/journal/snapshot/maintain) is `Cow::Owned` and
byte-for-byte unchanged; the borrowed path is `Cow::Borrowed(&mmap[..])` with
mutation forbidden at the handle. Only content-sized pools are borrowed — arena
pages, blob pools, chunk pools, vector slots; the small metadata (page indices,
tables) rebuilds owned, cheaply.

A reader pins the generation it maps, needs no journal, and therefore sees state
as of that checkpoint: snapshot isolation. `refresh()` re-reads the manifest and
re-maps only when the writer has published a newer generation.
