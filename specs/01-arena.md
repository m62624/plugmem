# 01 — plugmem-arena: flat storage structures

The foundation crate. `no_std + alloc`, no dependencies beyond `thiserror`
(core-compatible) and an xxh3 hash. Everything else in the project is built on the
four structures here.

## Philosophy

1. A structure's state is flat byte buffers plus small metadata arrays. No
   pointers, no per-element allocation.
2. The in-memory image *is* the persistence format: each structure hands out its
   sections as `&[u8]` and rebuilds from them without parsing (see `03-snapshot.md`).
3. Generic code is type-erased down to bytes: containers work over slot-slices and
   typing is a thin trait on top, so monomorphization does not bloat the wasm binary.
4. All cost is visible: operations are page-local, the worst case is bounded and
   counted, and the structures export work counters (feature `counters`).

## 1. `Arena<T: Slot>` — a sharded sorted arena

A shard is a chain of range-partitioned pages: each page holds a sorted run of
keys, and pages in a chain ascend by range — "the leaf level of a B+-tree".

The interior level is a flat **page directory**: `dir`, one entry per chain page
holding the page id and a copy of that page's first key, laid out shard after
shard. Locating a page is a binary search over it, never a walk. The assumption
that chains stay short does not hold — an `Ordered` arena whose keys share their
leading bytes (every edge of a hub entity, every timestamp of a real clock)
concentrates in one shard, whose chain then grows with the record count, and a
walk there is O(records) per operation.

Caching the key inside the entry rather than reading it from the pool is what
makes the search cheap: the pool is the size of the data, the directory is 20
bytes per 4 KiB page (≈0.5%), so it stays in cache while the pool does not.
The directory is **derived state** — rebuilt by the load walk, absent from the
on-disk image — alongside `prev` and `tails`.

### How many shards

The arena takes the count; the engine decides it, and **not from configuration**
(`04-recall.md` for where the rule lives). There is no correct constant, because
the two costs pull opposite ways:

- a shard's directory is sorted, so an insert lands mid-directory and memmoves
  the entries above it — more pages per shard, more memmove;
- every *touched* shard owns at least one whole page, so too few records per
  shard means buying 4 KiB for a handful of bytes.

Measured, since the balance is not obvious: sweeping 8…4096 shards over fixed
100k- and 1M-fact corpora moved write throughput by less than run-to-run noise —
flat even at 1465 pages per shard — while resident bytes grew monotonically with
the shard count, by 52 % at 100k facts across the range. A directory entry is 20
bytes, so even a thousand of them is one short `memmove`; the page floor is the
cost that actually bites. The engine therefore aims at **64 pages per shard**,
two orders of magnitude below where the sweep still measured nothing on the
write side, and holds the floor near 5 % of payload.

The count is bounded above by `MAX_SHARDS`, which is not a preference but a
safety limit: a shard count arrives from an untrusted snapshot and becomes the
length of the arena's per-shard vectors, so it is an allocation size taken from
a file (`03-snapshot.md`).

### Trait

```rust
pub trait Slot: Clone {
    /// Slot size in bytes (compile time).
    const SIZE: usize;
    /// Key-prefix length; sort and search compare only this.
    const KEY_LEN: usize; // KEY_LEN <= SIZE
    fn write(&self, out: &mut [u8]);          // out.len() == SIZE
    fn read(bytes: &[u8]) -> Self;            // bytes.len() == SIZE
}
```

Keys are written **big-endian** (numbers, timestamps) — a project invariant: a
byte-wise `cmp` on keys equals comparing the values, so range scans run over raw
bytes. Helpers: `key_u32`, `key_u64`, `key_pair(u64, u32)` and their readers.

For slots whose size is only known at runtime (vectors: `dim` from config) there is
a raw `DynArena` — the same code with `slot_size`/`key_len` in header fields;
`Arena<T>` is a thin typed wrapper over it.

### Layout

```rust
pub struct RawArena {
    pool: Vec<u8>,          // PAGE_BYTES pages, contiguous
    heads: Vec<u32>,        // shard -> first page of its chain (NONE = !0)
    next: Vec<u32>,         // page -> next page in the chain (NONE = !0)
    counts: Vec<u16>,       // page -> occupied slots
    free_head: u32,         // singly-linked free list of empty pages (via next)
    total: u64,             // element count
    cfg: ArenaCfg,          // slot_size, key_len, shards (power of 2), max_bytes, shard_mode
}
```

`PAGE_BYTES = 4096`. `slots_per_page = PAGE_BYTES / slot_size` (min 1; slots
larger than 4096 B are rejected with `Error::SlotTooLarge`). A fixed 4 KB page
stays L1-friendly at any slot size, and `u32` page indices keep the pool ceiling
well above the capacity contract.

### Sharding — two modes

- `Uniform` — for lookup arenas (facts keyed by id): shard = fibonacci hash of the
  key, giving even fill; order across shards is meaningless.
- `Ordered` — for range arenas (the time index, edges): shard = the top
  `log2(shards)` bits of the key, giving a global "shard after shard" order so a
  range scan crosses shards in sequence.

### Operations and their cost

- `insert(slot) -> Result<bool>`: shard → binary search the shard's run of the
  page directory for the page whose range covers the key → binary search in the
  page → memmove the page tail (≤ 4 KB) → write. A full page **splits**: a new
  page (from the free list or the pool end) takes the upper half, the chain and
  the directory are relinked, and the insert retries into the right half. A
  duplicate key returns `Ok(false)`.
  Appending *past* the last key does not split in half — that would leave both
  pages permanently half full, since nothing sorts into the lower one again.
  The record takes a fresh page and the full one stays full, so a monotonic
  load (which is every id and timestamp this crate stores) packs pages
  completely.
- `get(key)` / `contains`, `find_by` / `find_slice_by` / `find_slice_mut_by`: the
  same descent and binary search, O(log pages + log slots). `find_slice_mut_by`
  may mutate only payload bytes — the key prefix is immutable (debug_assert).
- Every mutation that can change which record sits first on a page refreshes
  that directory entry's cached key. A stale entry misdirects a lookup rather
  than failing, so the entry a search lands on is re-checked against its page
  under `debug_assertions`.
- `remove(key) -> bool`: shift within the page; a page that empties is unlinked to
  the free list. Half-empty neighbours are not merged (memory is reclaimed by the
  `maintain` compaction, which rebuilds the arena).
- `range(from_key..to_key)` — an iterator over slot-slices; `Ordered` only.
- `iter()` — all elements, shard by shard (in key order for `Ordered`).
- Capacity: `pool.len()` never exceeds `cfg.max_bytes` → `Error::CapacityExceeded`.
  No panics on overflow.

### Snapshot contract

The arena emits four sections: `pool`, `heads`, `next+counts` (page metadata), and
a `header` (`ArenaCfg` + `free_head` + `total`). On load, the metadata is
bounds-checked (every page index < page_count, counts ≤ slots_per_page, chains
acyclic — an O(pages) bitmap check); slot *contents* are not validated (they are
the owner's data). Malformed metadata is `Error::Corrupt`, never a panic.

### unsafe policy

The one default `unsafe` is allocating uninitialized pages
(`reserve + set_len` instead of zero-filling): measured ~12× faster on the page
allocation path under wasm (our target), free insurance natively. It carries a
strict invariant ("read only up to `count`"), a `// SAFETY:` comment, and miri
confirmation. Any other `unsafe` — including bounds-check elision, which measured
*slower* under wasm because the runtime bounds-checks anyway — is added only in its
own commit with a bench on a real arena. Functional correctness is always pinned by
a safe version first. The whole crate runs under miri in CI.

## 2. `BlobHeap` — variable length (texts, names)

```rust
pub struct BlobHeap {
    pool: Vec<u8>,            // append-only bytes
    index: Vec<(u32, u32)>,   // BlobId (= position in index) -> (offset, len)
}
pub struct BlobId(pub u32);
```

- `push(&[u8]) -> Result<BlobId>` (bounded by `max_bytes`, blob ≤ `max_blob`),
  `get(BlobId) -> &[u8]`.
- No deletion: blobs live while a record references them. Memory comes back via the
  `maintain` compaction — walk live references, rewrite the pool, build an
  old→new `BlobId` redirect table, owners update their references (the compaction
  protocol is in `05-api.md`).
- Snapshot: two sections, as-is.

## 3. `ChunkedList` — growing lists over a flat pool

For the inverted indexes and adjacency lists: many small growing lists with no
per-list allocation.

```rust
pub struct ChunkPool {
    pool: Vec<u8>,      // CHUNK_BYTES = 64 chunks
    free_head: u32,
    cfg: ...,
}
/// The list handle is held by the owner (an index), not the pool.
pub struct ListHandle { head: u32, tail: u32, len: u32, tail_used: u8 }
```

A 64 B chunk is `[next: u32][payload: 60 B]`. Operations: `push` (a value's bytes
never straddle a chunk boundary when the value is ≤ 60 B — always true for the
varint pairs stored here), `iter` (sequential read), `free` (chunks to the free
list). Compaction (relaying a list's chunks contiguously for read locality) is part
of `maintain`.

## 4. `Interner` — string → u32

```rust
pub struct Interner {
    heap: BlobHeap,                 // string bytes; BlobId == TermId
    table: Vec<u32>,                // open addressing: 0 = empty, else TermId+1
    mask: u32, len: u32,            // table size (power of 2), occupancy
}
pub struct TermId(pub u32);
```

- `intern(&str) -> Result<TermId>`: xxh3 → linear probing → byte compare via the
  heap; a miss does `heap.push` + a table write. Load factor ≤ 0.7, rehash ×2
  (amortized; a rehash allocates one table, not per element).
- `resolve(TermId) -> &str` — `heap.get`, O(1).
- Snapshot: the heap sections plus the table, as-is (the table is small — 4 B per
  slot — and stored rather than rebuilt; cold start beats a few megabytes at the
  ceiling).
- Never-forget: terms are not deleted (the vocabulary grows slowly; dictionary
  compaction is out of scope for v1).

## Test plan (mandate: 100% crate coverage)

1. **Property tests (proptest)** — model equivalence:
   - `Arena` ≡ `BTreeMap<Vec<u8>, Vec<u8>>` over random insert/remove/get/range
     sequences (both shard modes, slots 4–4096 B, KEY_LEN 4–32);
   - `Interner` ≡ `HashMap<String, u32>` plus the intern/resolve bijection;
   - `ChunkPool` ≡ `Vec<Vec<Vec<u8>>>`.
2. **Boundaries**: an empty arena; one element; an exactly-full page; a split at
   every insert position (first/middle/last slot); a cascade of splits; delete to an
   empty page and reuse from the free list; duplicates; `max_bytes` at the edge and
   over; a slot of exactly 4096 and 4097 B; a zero-length blob; an interner rehash
   at the load-factor boundary.
3. **Snapshot load**: dump → load → structural equivalence; malformed metadata (a
   cycle in a chain, count > slots_per_page, an out-of-range index) → `Error::Corrupt`,
   not a panic (see the fuzz plan in `08-performance.md`).
4. **BE order**: for `Ordered`, iteration is strictly ascending by key over random
   u32/u64/(u64,u32) keys.
5. **miri** over the whole crate (the unsafe paths must execute in tests).

## Counters and benches

Counters (feature `counters`, zero cost otherwise): `cmp_ops`, `bytes_shifted`,
`pages_walked`, `splits`, `probes` (interner). Targets on the reference workloads:
insert ≤ 300 ns, get ≤ 150 ns @100k (native). The full criterion matrix, the gate
ceilings and the recorded numbers live in `08-performance.md`.
