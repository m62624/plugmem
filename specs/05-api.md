# 05 — The core public API

`plugmem-core`: types, verbs, Config, errors, the Embedder contract. The API is
synchronous (a single-threaded core), `no_std + alloc`. Every `now` is a u64 unix
millisecond timestamp from the host.

## Entry point

```rust
pub struct Memory<'a> { /* arenas, indexes, config, scratch */ }

impl<'a> Memory<'a> {
    /// A new empty database.
    pub fn new(cfg: Config) -> Result<Memory<'static>, Error>;
    /// Load: snapshot + journal replay from Storage. A None snapshot => new(cfg).
    pub fn open<S: Storage>(store: &mut S, cfg: Config)
        -> Result<(Memory<'static>, OpenReport), Error>;
    /// Load from ready bytes (the wasm path: the host already fetched blob + journal).
    pub fn from_bytes(snapshot: Option<&[u8]>, journal: &[u8], cfg: Config)
        -> Result<(Memory<'static>, OpenReport), Error>;
    /// Load borrowing the mapped bytes (the read-only mmap path; see 03-snapshot.md).
    pub fn from_bytes_borrowed(snapshot: &'a [u8], cfg: Config)
        -> Result<(Memory<'a>, OpenReport), Error>;

    pub fn remember<S: Storage>(&mut self, s: &mut S, input: RememberInput<'_>)
        -> Result<RememberOutcome, Error>;
    /// Bulk write: journaled as a sequence of Remember; similar-detection is
    /// skippable (skip_similar) for import speed.
    pub fn remember_batch<S: Storage>(&mut self, s: &mut S,
        inputs: &[RememberInput<'_>], skip_similar: bool)
        -> Result<Vec<RememberOutcome>, Error>;
    /// &self — all mutable scratch is caller-owned (RecallScratch), so many
    /// readers can recall one engine at once.
    pub fn recall(&self, q: RecallQuery<'_>) -> Result<RecallResult, Error>;
    pub fn recall_into(&self, q: RecallQuery<'_>, s: &mut RecallScratch,
        out: &mut RecallResult) -> Result<(), Error>;
    pub fn revise<S: Storage>(&mut self, s: &mut S, target: FactId, input: RememberInput<'_>)
        -> Result<RememberOutcome, Error>;
    pub fn forget<S: Storage>(&mut self, s: &mut S, now: u64, id: FactId) -> Result<bool, Error>;
    pub fn remove_tag<S: Storage>(&mut self, s: &mut S, now: u64, tag: &str)
        -> Result<RemoveTagReport, Error>; // revises current facts; keeps history
    pub fn list_tags(&self, query: TagQuery<'_>) -> Result<TagPage, Error>;
    pub fn link<S: Storage>(&mut self, s: &mut S, input: LinkInput<'_>) -> Result<(), Error>;
    /// Close the current edge of (src, rel, dst). Its history version stays,
    /// so `as_of` before the close still sees it. `Ok(false)` when no such
    /// edge is open.
    pub fn unlink<S: Storage>(&mut self, s: &mut S, input: UnlinkInput<'_>)
        -> Result<bool, Error>;
    /// The integrity an open defers: text, metadata, the vector bijection and
    /// the graph's cross-references (`03-snapshot.md`, `08-performance.md`).
    pub fn verify(&self) -> Result<(), Error>;

    /// All upkeep. Explicit, no background.
    pub fn maintain<S: Storage>(&mut self, s: &mut S, now: u64) -> Result<MaintainReport, Error>;
    /// Full image + journal clear.
    pub fn snapshot<S: Storage>(&mut self, s: &mut S, now: u64) -> Result<(), Error>;
    /// Image to bytes (the wasm path; the host clears the journal).
    pub fn snapshot_bytes(&self, now: u64) -> Vec<u8>;

    pub fn stats(&self) -> Stats;   // #[non_exhaustive]
    pub fn get(&self, id: FactId) -> Option<FactView<'_>>;
    /// A fact's metadata into the caller's buffer, in canonical (ascending) order;
    /// tolerant of bad bytes (hides, never panics). The host collects a BTreeMap,
    /// CLI/MCP/napi an object (Record<string,string>), same order.
    pub fn metadata_of(&self, id: FactId, out: &mut Vec<(&str, &str)>) -> bool;
    /// &mut self — name normalization uses the tokenizer scratch.
    pub fn entity(&mut self, name: &str) -> Option<EntityId>;
}
```

`recall` takes `&self`: it mutates no data, and all mutable buffers (term/score
vectors, the fusion map, its **own tokenizer and name buffer**) are in a caller-owned
`RecallScratch`. This is the key to concurrent reads — N readers recall one engine in
parallel, each with its own `RecallScratch` (on the native host, one per thread).
`recall` is the convenience wrapper that makes the scratch and result itself; a hot
loop owns a `RecallScratch` and calls `recall_into` for zero-alloc. Synchronization
(`RwLock`/thread-local) lives in the host; the core stays `no_std` and takes only an
explicit `&mut RecallScratch`. The `'a` lifetime is `'static` for every owned
constructor; only `from_bytes_borrowed` ties it to the mapped buffer.

## Inputs / outputs

```rust
pub struct RememberInput<'a> {
    pub now: u64,
    pub text: &'a str,                       // <= cfg.max_text
    pub entity: Option<&'a str>,             // subject; created lazily
    pub tags: &'a [&'a str],                 // <= 32
    pub links: &'a [(&'a str, &'a str)],     // (rel, target_entity), <= 16; edges subject->target
    pub vector: Option<&'a [f32]>,           // len == cfg.dim; quantized internally
    pub valid_from: Option<u64>,             // default now
    pub metadata: Option<&'a [(&'a str, &'a str)]>, // key->value, any order; the engine
                                             // canonicalizes (sort+dedup) and stores it opaquely
}

pub struct RememberOutcome {
    pub id: FactId,
    pub entity: Option<EntityId>,
    /// Hints to the agent: similar / potentially conflicting live facts.
    /// The engine NEVER revises on its own — the agent decides.
    pub similar: Vec<Similar>,               // <= 8, descending score
}
pub struct Similar {
    pub id: FactId,
    pub score: f32,
    pub reason: SimilarReason,               // LexicalOverlap | VectorCosine
}

pub struct TagQuery<'a> {
    pub prefix: Option<&'a str>,             // exact, case-sensitive
    pub cursor: Option<&'a str>,             // opaque continuation token
    pub limit: usize,                        // 0 = 64; maximum 256
}
pub struct TagSummary { pub name: String, pub count: u32 }
pub struct TagPage { pub items: Vec<TagSummary>, pub next_cursor: Option<String> }
pub struct RemoveTagReport { pub affected: u32 }
```

Similar detection (cheap, from the ready indexes): live facts of the same entity
(term overlap > 0.5 Jaccard on top terms) ∪ vector neighbours with cos > 0.85
(threshold in Config). This is the key to Graphiti-class behavior with no LLM inside:
the engine finds, the agent decides (`revise` / keep both / `forget`). Full detection
(vector included) fits the remember budget ≤ 500 µs — against an embedder call the
engine is invisible anyway.

Comparing term sets exactly needs both of them, and only the new fact's is at hand.
The candidate's is **not** recovered by re-reading and re-tokenizing its text — that
was nine tenths of a write. The per-document BM25 record carries a summary of its
term set (`04-recall.md`), and the summary bounds the overlap from above: a term
whose bit is clear is absent, and Jaccard rises with the intersection, so a bound at
or below the threshold settles the question. Only a candidate that survives the bound
is read. **Any prefilter here must be a strict upper bound** — the hints are part of
the answer, so one that can undershoot silently drops a conflict the caller was meant
to see. The bound is trusted only while the index was built by the current tokenizer;
a stale index falls back to reading every candidate.

```rust
pub struct RecallQuery<'a> {
    pub now: u64,
    pub text: Option<&'a str>,
    pub vector: Option<&'a [f32]>,
    pub tags: &'a [&'a str],
    pub entities: &'a [&'a str],
    pub as_of: Option<u64>,                  // default now
    pub range: Option<(u64, u64)>,           // recorded_at window (episodic)
    pub k: usize,                            // default 8, <= 64
    pub token_budget: Option<usize>,         // default 512
    pub include_closed: bool,
    pub ef: Option<usize>,                   // HNSW ef_search override
}

pub struct RecallResult {
    pub facts: Vec<RecalledFact>,            // id, score, sources bitmask, text-ref,
                                             // recorded_at, valid interval, entity, tags
    pub edges: Vec<RecalledEdge>,            // edges traversed by the graph source
    pub rendered: String,                    // a ready block for the prompt
    pub truncated: bool,                     // hit the budget
}
```

### The `rendered` format (a contract, pinned by tests)

```
## memory
- [f42] user: prefers tokio (2025-11; active) #pref
- [f17] user: lived in Moscow (2023-01 -> 2025-06; closed) #location
- links: user -works_on-> plugmem
```

One line per record, stable order (by score), ISO month dates, `active`/`closed`
markers, the id for a later revise/forget. An empty result → an empty string (not
"nothing found" — don't spend tokens).

## Config (full, with defaults)

| Field | Default | Note |
|---|---|---|
| `dim` | 0 | 0 = vector layer off |
| `max_bytes` | 2 GiB | ceiling for **each** pool, not their sum; the default is the wasm32 passport |
| `max_text` | 4096 | bytes |
| `max_blob` | 64 KiB | |
| `shards_facts / entities / edges / temporal / postings` | the floor | **engine-managed**, see below |
| `bm25_k1 / b` | 1.2 / 0.75 | |
| `rrf_k` | 60 | |
| `w_bm25 / w_vec / w_graph / w_time` | 1.0 | RRF weights |
| `w_recency / half_life_days` | 0.25 / 180 | |
| `graph_depth / graph_decay` | 2 / 0.5 | `graph_depth` is the *default*; a recall overrides it per call, uncapped (the walk is bounded by its entity/edge caps) |
| `similar_cos / similar_jaccard` | 0.85 / 0.5 | |
| `hnsw_m / m0 / ef_construction / ef_search` | 16 / 32 / 200 / 64 | `m`/`m0` shape the stored graph, so a mismatch on open is `ConfigMismatch`; `ef_search` is the default a recall's `ef` overrides |
| `flat_to_hnsw` | 24_000 | threshold, tuned by a bench |
| `db_uuid` | 0 | u128 database lineage id: host-minted at creation, 0 = unnamed; on open 0 adopts the stored one, nonzero must match (`ConfigMismatch`); printed in `stats()` |

**The shard counts are not a setting.** They describe how a file is laid out, and
the engine derives them from how much it holds (`01-arena.md` for the rule and the
measurement behind it). A new database starts on the floor, `open` **adopts**
whatever the snapshot records — the caller's values are ignored, because the loader
needs the stored ones to read the arenas at all — and `maintain` moves the layout as
the data grows or shrinks, eagerly upward and lazily downward so a database near a
boundary does not rebuild itself repeatedly. There is no `config.toml` key; the
current layout is reported by `stats()`. Setting the fields in Rust affects only a
database being created, and the next maintenance pass overrules it.

**Integrity is not config.** Open trusts the file by default (trust/sparse, the SQLite
model) and does not checksum the image. Byte integrity is on demand via `scrub()`
(resumable); content integrity via `verify()`. Config is stored in the snapshot; on
`open` the given config is checked — an incompatible `dim` against a non-empty
database is `Error::ConfigMismatch` (changing dim = reindexing, a separate CLI
utility).

## Errors

```rust
#[non_exhaustive]
pub enum Error {
    CapacityExceeded { what: &'static str },
    TooLarge { what: &'static str, len: usize, max: usize },
    DimMismatch { got: usize, want: usize },
    NotFound(FactId),
    AlreadyClosed(FactId),          // revise over a closed fact
    ConfigMismatch(&'static str),   // invalid Config or incompatible with the database
    Corrupt(&'static str),          // snapshot/journal
    UnsupportedVersion(u16),        // a snapshot of an unknown format version
    Invalid(&'static str),          // a structural input violation not about size
    StaleCursor,                    // catalogue changed; restart tag pagination
    Storage(String),               // debug-render of a Storage error (the core stays
                                    // generic and Clone; the wrapper logs the source)
    Arena(plugmem_arena::Error),   // #[from]
}
```

The enum is `#[non_exhaustive]`, derives `Debug + Clone + PartialEq + Eq` and
`thiserror::Error`. A panic in the core is a bug by definition (pinned by fuzz and the
review policy).

## Verb semantics — summary

| Verb | Journal | Effect |
|---|---|---|
| remember | yes | a new open fact + indexes + similar hints |
| revise | yes | close target (valid_to = new.valid_from), a new fact with revises=target |
| forget | yes | tombstone immediately (recall hides it), physical purge in maintain |
| remove_tag | yes | revise every current fact carrying the tag; facts and historical tag state survive |
| link | yes | opens an edge version for (src,rel,dst); a repeat closes the open one and opens a new one |
| unlink | yes | closes the current edge; the version stays and `as_of` still sees it |
| recall | no | a pure query |
| list_tags | no | bounded lexical page of current tags and current-fact counts |
| maintain | yes (marker) | physical deletion of tombstoned records (ids burned; see 02-data-model.md), satellite rebuild, stat recompute, folding the vector tail into HNSW. Policy-driven (`auto` / `compact` / `reindex-text` / `optimize-vectors` / `full`); **no mode ever drops a fact revision or an edge version** |
| snapshot | clears | full image + clear_journal |

`maintain` and the explicit bulk `remove_tag` can cost O(database); point verbs are
microseconds (budgets in `08-performance.md`). Wrappers call maintenance on their own policy (CLI — a command
plus auto after N ops; MCP — when idle; wasm — the host decides).

## The Embedder layer (`plugmem-host`, std)

```rust
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    /// Batch is in the signature on purpose: providers are far cheaper batched.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError>;
}
```

**`&self`, not `&mut self`, and that is the whole contract.** An embedder is a
client of a remote service, not a piece of mutable state, so the trait does not
demand exclusive access. It used to, and the cost was structural: a `&mut self`
method reachable only through an exclusive borrow forced every sharer to put a
`Mutex` in front of it, and that mutex spanned the HTTP round trip, so concurrent
callers queued one request at a time. With `&self` the sharer is a plain
refcount (`SharedEmbedder`) and the provider sees the concurrency it was built
for.

An implementation that genuinely needs mutable state — a cache, a rate-limit
budget — brings its own interior mutability. That is the right place for it:
only that implementation knows what may overlap and what must not, so it can
guard the few bytes that need guarding instead of the whole call.

Implementations: `OpenAiCompatEmbedder` (`/v1/embeddings` — also covers Ollama via its
compatible endpoint) and `NullEmbedder` (dim 0). Neither needs a guard: an
`OpenAiCompatEmbedder` holds a `ureq::Agent`, itself a `Send + Sync`
connection-pool handle. The HTTP client is `ureq`; errors flow through the host's
`HostError`. A built-in local embedder is a v1.1 item behind a `local-embed`
feature (target: a quantized `multilingual-e5-small`, CPU or GPU by choice) — the
core is untouched, the Embedder contract is ready.

The wrapper calls `embed` before `remember`/`recall` (if an embedder is configured) and
puts the vector in the input; the core does not know about Embedder. On wasm the JS
wrapper does the same through a host callback (see `07-wrappers.md`).

## Test plan

- Contract tests per verb (the tables from `02-data-model.md` plus similar scenarios).
- `rendered` — golden tests (the format is a contract).
- Errors: each Error variant is reachable by a test.
- Zero-alloc recall: a counting allocator in the test harness, 0 allocations on a
  reference recall after warm-up (see `08-performance.md`).
- Embedder implementations: against a local mock HTTP (no network in CI).
