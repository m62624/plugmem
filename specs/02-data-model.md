# 02 — Data model

What plugmem-core stores, and in which bytes. All arena keys are big-endian (see
`01-arena.md`); all timestamps are `u64`, unix milliseconds UTC, **passed in by the
host** (`now` is a parameter of every mutating call; the core knows nothing of
time).

## Model entities

- **Fact** — the unit of memory: a short text ("prefers tokio over strict
  versions"), an optional subject Entity, tags, temporality, and an optional
  vector. This is what an agent stores with `remember`.
- **Entity** — a graph node: a named object ("user", "the plugmem project",
  "Barsik"). Created lazily on first mention.
- **Edge** — a typed entity → entity edge (`rel` is an interned term: "works_at",
  "owns", …) with a provenance reference to the fact that grounds it.
- **Tag** — an interned label on a fact (free namespace: "pref", "health",
  "project:plugmem" — conventions are set by `SKILL.md`, not the core).

The core is agnostic: it does not know what a "user" or a "project" is — all
names/types/tags are interned host strings.

### Current tag catalogue

Facts and their per-fact tag lists are authoritative. A derived catalogue keeps
one `(TermId, current fact count)` entry for every tag used by at least one
current, non-tombstoned fact. It lets a caller discover tags without scanning
every fact or confusing tags with text/entity/relation terms in the shared
interner.

`list_tags` returns bounded pages in raw UTF-8 lexical order. Prefix matching is
exact and case-sensitive. Its opaque cursor includes the database identity,
prefix and a fingerprint of the current catalogue; when a write changes any tag
count, continuing an old scan returns `StaleCursor` so a caller restarts instead
of silently skipping or duplicating a tag.

The snapshot stores the merged catalogue as 8 bytes per tag and reuses the
interner's strings. Between checkpoints, changed counts live in a 64-entry
sorted buffer and logarithmically many sorted runs. A write therefore never
re-sorts the whole tag set; a page is a bounded k-way merge of the snapshot base
and those runs.

`remove_tag(tag)` is a bulk temporal update, not string deletion. Every current
fact carrying the tag is closed and replaced by an otherwise-identical successor
without it. Current recall and the catalogue stop showing the tag immediately;
historical `as_of` queries retain the old classification and all facts remain.
The interned string may remain as harmless historical residue.

## IDs

All ids are `u32`, monotonically increasing, and **never reused** (the journal is
replayable; holes after a purge are normal). `0` is a valid id; "no value" is
`u32::MAX` (`NONE`). Type-safe newtypes: `FactId`, `EntityId`, `TermId`, `BlobId`.
The 4.29-billion ceiling per kind is far above the 1M capacity contract.

Id monotonicity rests on a persistent counter `next_fact` (in the ENGINE_STATE
section), **not** on records staying in the arena — the fact arena is a map keyed
by id, so a missing record is just a hole in the numbering. Reusing an id via a
free list is rejected on purpose: an external saved reference would then silently
point at a *different* fact, and silent corruption is worse than a loud NotFound.

## Slot layouts

### FactRecord — `Arena<FactRecord>`, Uniform, 48 B slot, KEY_LEN 4

| off | size | field | note |
|---|---|---|---|
| 0 | 4 | `id` (key, BE) | |
| 4 | 4 | `entity` | EntityId or NONE |
| 8 | 2 | `flags` | bit 0 tombstone, 1 closed (has valid_to), 2 has_vector, 3–15 reserved |
| 10 | 2 | `kind` | reserved, 0 in v1 (fact typing is expressed by tag conventions from `SKILL.md`) |
| 12 | 4 | `text` | BlobId of the text (UTF-8) |
| 16 | 4 | `vector` | slot index in the vector arena, or NONE |
| 20 | 4 | `revises` | FactId of the predecessor in the revision chain, or NONE |
| 24 | 8 | `recorded_at` | when it was recorded |
| 32 | 8 | `valid_from` | when it became true (default = recorded_at) |
| 40 | 8 | `valid_to` | until when; `u64::MAX` = open ("true now") |

A fact's tags and metadata live in a cold sidecar `Arena<FactAux>` (Uniform, **20 B**
slot): `id` 4 + a compact `ListHandle` 12 (tags) + `meta` 4 (BlobId of the metadata,
or NONE). They are kept out of the hot 48-byte `FactRecord` so it stays hot. The
tag→facts inverted index is in `04-recall.md`.

### Fact metadata — a dedicated blob heap `metas`, one blob per fact

An optional key→value map (UTF-8 strings): pointers/attributes (a URI to the real
payload in another store, a mime type, an external key). The engine **never
interprets** the content. It is stored as one opaque blob in `metas` (a mirror of
`texts` but a separate, cold pool — not resident on an mmap'd database until
`show`/`export`), referenced by `FactAux.meta`. The canonical encoding of one blob:

```text
[count: u32 LE]
count times, keys STRICTLY ascending (raw UTF-8 bytes), no duplicates:
  [klen: u32 LE][key UTF-8]
  [vlen: u32 LE][val UTF-8]
```

The order is decided in exactly one place — at encode time (`core::metadata::encode`
sorts the keys and rejects duplicates) — so every reader (core, host, wrappers) hands
back the same order and snapshot/replay is byte-identical. An empty map means no blob
(`meta = NONE`). `verify` decodes each blob (UTF-8, ascending, unique); `faulty_facts`
attributes a `FactFault::Metadata`.

### EntityRecord — Arena, Uniform, 24 B slot, KEY_LEN 4

| off | size | field |
|---|---|---|
| 0 | 4 | `id` (key, BE) |
| 4 | 4 | `name` (BlobId, the canonical name as entered) |
| 8 | 4 | `name_term` (TermId of the normalized name — for lookup) |
| 12 | 8 | `created_at` |
| 20 | 4 | `flags` + reserved |

Name → EntityId resolves through the interner (`name_term`) plus an
`Arena<EntityByName>` (Ordered, key `[name_term BE | id]`, 8 B slot). The
**normalized name is unique**: remember/link do lookup-or-create by `name_term`, so
`entity(name)` is deterministic. Semantic duplicates ("user" vs "the user") are the
agent's concern (similar hints; an alias mechanism is v2).

### Edge — two mirror arenas, Ordered, 28 B slot, KEY_LEN 12

The **current** graph: the edges open right now.

- an out-arena, key `[src BE 4 | rel BE 4 | dst BE 4]`;
- an in-arena, key `[dst BE 4 | rel BE 4 | src BE 4]`;
- payload in both: `fact` 4 (provenance, may be NONE), `edge` 4, `valid_from` 8.

Walking neighbours is a range scan over the `src` (or `src+rel`) prefix — a linear
read. An edge is unique by (src, rel, dst); a repeated link closes the open version
and opens a new one.

`edge` and `valid_from` are the key tail of this edge's open history version, so
closing an edge addresses that record directly instead of searching a triple's
versions for the open one — which is what makes `unlink` a point lookup.

### Edge history — two mirror arenas, Ordered, 48 B slot, KEY_LEN 16

Every version an edge has ever had, closed or open. Nothing here is ever deleted:
history *is* the feature, and no `maintain` mode drops a version.

- an out-arena, key `[a BE 4 | valid_from BE 8 | edge BE 4]`;
- an in-arena, the same with the endpoints swapped;
- payload: `rel`, `b`, `fact`, `flags`, `kind`, `recorded_at`, `valid_to`.

The key is **time-ordered within an entity**, not grouped by triple. That is what
makes `as_of` sublinear: walking backwards from `as_of` yields an entity's versions
newest-first — those that most recently became true, and so the ones most likely to
still be valid — and the walk stops as soon as the caller has enough. The range
enforces `valid_from ≤ as_of`; `as_of < valid_to` is the other half of the test.

The exception, measured and documented: an instant at which the entity had **no**
valid edge has nothing to stop at, so proving the absence reads every version that
had already begun. Answering that sublinearly needs an interval index, not an
ordering.

The current arenas and the history arenas are one fact stored twice. That they
agree — both mirrors, a current edge against its open version, every open version
reachable as a current edge — is checked by `verify()`, not by the loader
(`08-performance.md`).

### Time index — Arena, Ordered, 12 B slot, KEY_LEN 12

Key `[recorded_at BE 8 | fact_id BE 4]`, empty payload. Range queries answer "what
was recorded in this interval"; validity filtering (valid_from/valid_to) is done per
candidate against the fact record (O(1) per candidate).

## Temporality — semantics

The model is bitemporal, in a simplified form:

- `recorded_at` — the knowledge axis: when memory learned of it. Immutable.
- `valid_from / valid_to` — the truth axis: when the fact was/remains true.

Rules:

1. A new fact: `valid_from = input or now`, `valid_to = MAX` (open).
2. **`revise(old, new)`**: the old fact's `valid_to := new.valid_from` and its
   `closed` flag is set; the new fact gets `revises = old`. The old fact is **not
   deleted** — "lived in Moscow" stays true for its interval. The revision chain is
   singly linked and acyclic (constructively: `new.revises` is a just-created id, so
   a cycle is impossible; the journal replay checks it too).
3. **`forget(id)`**: sets the tombstone flag; the fact leaves recall immediately and
   is physically purged at the next `maintain` (its record, vector slot, postings and
   provenance edges). This is a right to be forgotten, not a revision.
4. An `as_of(t)` query: a fact is live when `valid_from ≤ t < valid_to` and
   `recorded_at ≤ t` and it is not a tombstone. The default recall is `as_of(now)`,
   i.e. only open/current versions (closed ones surface with `include_closed`).

## Physical deletion (maintain)

`maintain` rebuilds the arenas and, in doing so, **does not re-insert** the
`FactRecord`/`FactAux` of a tombstoned fact — there is no leftover "shell". Because
`next_fact` is unchanged, the id stays burned forever and replay stays deterministic
(the skip rule `assigned < next_fact` does not depend on a record existing; a
Maintain entry always follows the Forget of a fact, and a Revise/Forget of an
already-purged id can never appear in a valid journal — the live verb would have
returned NotFound and not been journaled).

A reference to a purged ancestor (`revises`, edge provenance) keeps the burned
number rather than being scrubbed to NONE; a `get` on it returns `None`, exactly as
for a tombstone, which preserves observation-equivalence between maintained and
unmaintained runs.

What remains after `maintain` is only slow-growing residue: dead interner terms
(Zipf) and orphan entities (kept — an entity with no live fact is still knowledge).
Neither has a reclaim path in v1. Rotation (auto-`forget` by TTL/tags, then a
`maintain` trigger) is a wrapper policy, not core; because the file *is* the state
image, a checkpoint after a purge shrinks the file with no disk fragmentation.

## Database identity

`Config::db_uuid` (u128, host-supplied since the core has no RNG; `0` = an unnamed
database) is stored in the snapshot's config block and identifies the database
lineage — it survives `maintain` and re-saves. On `open`, a caller's `0` adopts the
stored uuid; a nonzero value must match the stored one or the open fails with
`ConfigMismatch("stored db_uuid differs")`. It is visible in `Stats::db_uuid`. A
different `db_uuid` means a different database whose ids are not comparable.

## Model invariants (debug_assert + journal replay)

1. Every reference (entity, text, vector, revises, edges, tags) points at an existing
   record; a tombstoned fact is cascaded away in maintain, and until then references
   to it are allowed but recall filters them.
2. `valid_from ≤ valid_to`; a closed fact has `valid_to < MAX`.
3. The `revises` chain is acyclic; the head has `valid_to = MAX` or is a tombstone.
4. A slot's key prefix is not mutated after it is written to the arena.
5. A fact's text ≤ `cfg.max_text` (default 4096 B); tags ≤ 32; edges per link ≤ 16 —
   guards against junk inserts, returned as typed errors, never panics.
6. The derived tag catalogue equals counts recomputed from current facts; `verify`
   checks this without putting a full scan on normal open or paging.

## Test plan

- Unit: slot layouts — write/read roundtrip of each field, byte-for-byte comparison
  against reference buffers (these pin the format: break the layout, break the test).
- Temporality semantics: a scenario table (create → revise → revise → as_of at five
  points; forget the head/middle of a chain; include_closed).
- Property: random remember/revise/forget sequences vs a reference model over a
  `Vec<RefFact>` with naive filtering — the live sets match.
- Invariants 1–5 over fuzz sequences of valid operations — none is violated; invalid
  inputs give typed errors.
