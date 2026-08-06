# plugmem

> ⚠️ Experimental. plugmem is mostly an AI-built experiment — written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

An embeddable bitemporal memory database for local-first applications and
agents, embedded in your Node process. It stores short facts and answers a
query with ranked facts and edges plus an optional bounded rendered block.

File-backed on disk, no server, no daemon. It links into your process the way
SQLite does: the engine is [`plugmem-host`](https://docs.rs/plugmem-host/latest)
compiled to a native addon through [napi-rs](https://napi.rs), so there is no
WebAssembly copy of the file in RAM and no 4 GiB ceiling. Runs on Node, Deno and
Bun.

**Contents:** [Install](#install) · [Quick start](#quick-start) ·
[What it stores](#what-it-stores) · [Two clocks](#two-clocks) ·
[How recall works](#how-recall-works) · [API](#api) · [Errors](#errors) ·
[Configuration](#configuration-and-embeddings) ·
[Async](#async-and-the-event-loop) · [Many memories](#many-memories-in-one-directory) ·
[What it is not for](#what-it-is-not-for)

## Install

```console
$ npm install plugmem
```

That pulls a meta package which, through `optionalDependencies`, installs only
the prebuilt binary for your platform: one of `plugmem-{linux-x64-gnu,
linux-arm64-gnu, darwin-x64, darwin-arm64, win32-x64-msvc, win32-arm64-msvc}`.
No toolchain, no build step.

## Quick start

```typescript
import { Plugmem } from "plugmem";

const db = await Plugmem.open("agent.plugmem");

await db.remember({ text: "the user prefers tokio", entity: "user", tags: ["pref"] });
await db.remember({ text: "the release ships on friday", entity: "release" });

const res = await db.recall({ query: "tokio", k: 5 });
console.log(res.rendered);   // paste this into the prompt
// - [f0] user: the user prefers tokio (2026-08; active) #pref

db.close();
```

The query is `"tokio"` and not `"which runtime?"` for a reason worth knowing up
front: with no embedder configured, recall matches on **words**, and "runtime"
appears nowhere in that fact, so the more natural question returns nothing.
Reach it through the graph instead with `entities: ["user"]`, or configure an
[embedder](#configuration-and-embeddings) and the meaning matches too. Only one
of the four sources needs a model — see [How recall works](#how-recall-works).

`Plugmem.open` is a static method, not a constructor, because opening replays a
journal and maps a snapshot — work proportional to the file — and a JavaScript
constructor has no way to hand that to a worker thread. Everything is typed:
`index.d.ts` is generated from the Rust, so a TypeScript host gets real
autocomplete on arguments and results.

## What it stores

One **fact** is one statement. It carries:

| Field | Meaning |
|---|---|
| `text` | the statement itself, and what lexical search indexes |
| `entity` | the subject, by name — created on first mention, shared across facts |
| `tags` | filters, not ranking: a query asking for a tag requires it |
| `metadata` | an opaque `Record<string,string>`. The engine stores and returns it and never looks inside — use it for a URI to the real payload elsewhere, a mime type, an external key |
| `vector` | an optional embedding; supply your own, or let a configured embedder produce it |
| `validFrom` | when the statement became true |

Entities are joined by **typed edges**: `link({ src: "ann", rel: "hires", dst: "bob" })`.
An edge can name the fact it follows from (`provenance`), so a later reader can
answer "why does the memory think this" instead of trusting a bare relationship.

Facts are never rewritten in place. `revise` closes the old one and chains a
successor; `forget` tombstones a fact and the next `maintain` erases it from
disk. `unlink` closes an edge the same way `revise` closes a fact.

## Two clocks

This is the part worth reading, because it is the one thing that behaves
differently from every other store.

Every fact carries **two** timestamps, not one:

- **`validFrom` / `validTo`** — when the statement was true.
- **`recordedAt`** — when the memory learned it. Set by the engine, never by you.

They are different questions, and one timestamp cannot hold both.

Unlike the Rust library, this binding reads the system clock on every call — you
never pass `now` — so `recordedAt` is always the moment of the write:

```typescript
await db.remember({ text: "lives in Moscow", entity: "kim" });
const between = Date.now();
await db.revise(0, { text: "lives in Berlin", entity: "kim" });

(await db.recall({ entities: ["kim"] })).rendered;
// - [f1] kim: lives in Berlin (2026-08; active)

(await db.recall({ entities: ["kim"], asOf: between })).rendered;
// - [f0] kim: lives in Moscow (2026-08 → 2026-08; closed)
```

`revise` closed the first fact's interval rather than deleting it, which is why
the second query has something to answer with.

`asOf` moves **both** clocks: a fact answers only if it was valid at that
instant *and* had already been recorded by then. The second half is the one
people trip over — an `asOf` earlier than a fact's `recordedAt` sees nothing,
because the memory genuinely knew nothing then. Answering with today's knowledge
would be the wrong answer to "what did I hold".

`validFrom` is the other half: a statement that became true before you heard of
it. Recording today that someone moved a week ago closes the previous interval a
week ago rather than now, so a query as of three days back finds **neither** —
the old fact had stopped being true, and the new one was not yet known. That is
not a hole in the model; it is the honest answer for that instant, and it is
what a single timestamp cannot express.

Two more queries over the same axes:

```typescript
await db.recall({ range: [from, to] });          // what did I record in this window
await db.recall({ query: "kim", closed: true }); // include closed revisions
```

Use `revise` when something changed and `forget` only when a fact was simply
wrong: `forget` destroys the "what was true then" answer, `revise` keeps it.

Edges are temporal too, so `asOf` walks the graph as it stood then — through
relationships that have since been unlinked.

## How recall works

Not a vector lookup. Four sources run and are fused by
[reciprocal-rank fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
with a recency boost; tags filter and are not a source:

| Source | What it finds | Needs an embedder |
|---|---|---|
| **Lexical** — [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) over a Unicode ([UAX #29](https://unicode.org/reports/tr29/)) tokenizer | exact terms, keyword overlap | no |
| **Graph** — typed edges walked from the query's anchor entities | relational knowledge | no |
| **Temporal** — range scans over the `recordedAt` index, plus the validity test | "what was true then", time windows | no |
| **Semantic** — int8-quantized cosine, a flat scan below a threshold and an [HNSW](https://arxiv.org/abs/1603.09320) graph above | meaning, nearest neighbours | **yes** |

The sources compose. A query with no `query` string still answers from tags,
entities and time. **Without an embedder the system is complete** — the other
three sources need no model, no network and no API key.

The result carries both a `rendered` block, selected greedily under a token
budget and ready to paste, and the structured `facts`/`edges` behind it:

```typescript
const res = await db.recall({
  query: "release plans",
  entities: ["ann"],       // graph anchors
  tags: ["work"],          // filter: a fact must carry all of these
  k: 10,                   // cap the number of facts
  tokenBudget: 400,        // cap the size of the block — your context budget
  graphDepth: 3,           // how far to walk from the anchors (default 2)
});

res.rendered;   // string, prompt-ready
res.facts;      // { id, score, entity, recordedAt, validFrom, validTo, sources }[]
res.edges;      // { src, rel, dst, provenance }[] — what the graph walked
res.truncated;  // true if selection stopped at k or the budget with more left
```

`remember` returns the new fact's id **plus any live facts it may duplicate or
contradict**. The engine never merges or deletes on its own — it surfaces the
tension and your code decides:

```typescript
const out = await db.remember({ text: "the user prefers async-std", entity: "user" });
for (const s of out.similar) {
  // s.id, s.score, s.reason: "LexicalOverlap" | "VectorCosine"
  // → revise it, forget it, or keep both. Your call.
}
```

## API

Every method wraps the identically-named verb of the Rust `Database`; this layer
only moves arguments and results across the boundary.

**Writing** — all return promises:

| Method | Does |
|---|---|
| `remember(args)` | store a fact; resolves with its id and similar facts |
| `rememberMany(args[])` | store a batch: one embedding round-trip, one journal sync |
| `revise(id, args)` | close a fact and record its successor |
| `forget(id)` | tombstone a fact; resolves with whether it was live |
| `link(args)` | upsert a typed edge, optionally with `provenance` |
| `unlink(args)` | close the current edge; resolves with whether one was open |

**Reading** — synchronous ones touch mapped memory and return in microseconds:

| Method | Does |
|---|---|
| `recall(args?)` | ranked, fused, token-budgeted result (async) |
| `get(id)` | one fact's full card, or `null` (sync) |
| `tagsOf(id)` | that fact's tags (sync) |
| `stats()` | engine size counters (sync) |
| `path()` | the file this handle resolved to (sync) |
| `export()` | every open fact as one array (async, unbounded — see below) |
| `exportPage(cursor?)` | the same data in bounded pages of 128 (async) |
| `exportEdges(onBatch)` | every current edge, streamed in batches (async) |
| `configWarnings()` | anything in `config.toml` nothing claimed (sync) |

**Upkeep** — all async, all on a worker thread:

| Method | Does |
|---|---|
| `maintain(mode?)` | `"auto"` (default), `"compact"`, `"reindex-text"`, `"optimize-vectors"`, `"full"`. No mode ever drops a revision or an edge version |
| `checkpoint()` | flush the journal into a fresh snapshot |
| `verify()` | full content-integrity sweep; rejects on the first inconsistency |
| `scrub(options?)` | start a resumable byte-level check of the snapshot |
| `recover(src, dst, options?)` | module function: salvage a damaged file into a clean copy |

**Read-only handles** (`{ readOnly: true }`) observe another process's writer
over a published snapshot. The read verbs answer, the write verbs throw, and two
more appear: `generation()` (the pinned snapshot number) and `refresh()` (adopt
the writer's latest checkpoint, returning whether a newer one existed).

`close()` releases the file and its lock; every verb afterwards throws, and
calling it twice is a no-op.

### Bringing your own embedding

`remember`, `revise`, `rememberMany` and `recall` all take an optional `vector`
whose length must equal the configured `dim`. Given one, it **replaces** the
embedder for that call — nothing is sent to the provider:

```typescript
const own = await myEmbedder(text);
await db.remember({ text, vector: own });
const res = await db.recall({ query: text, vector: own });
```

Use it for vectors you already have, for a model that is not an OpenAI-shaped
HTTP endpoint, or for a deterministic test with no network.

### Backing up: facts are only half of it

`export`/`exportPage` dump facts. An **edge** is a statement between two
entities — `kim -works_on-> plugmem` — and belongs to no single fact, so a dump
of facts alone loses the graph. `exportEdges` is the other half.

It streams: the walk runs on a worker and hands your callback one batch at a
time, so memory stays flat whether the graph has ten edges or ten million. When
a callback is slower than the walk, the *worker* waits — never the event loop.

```javascript
const edges = [];
const count = await db.exportEdges((batch) => edges.push(...batch));
```

`count` is `2` here, and `edges` is complete the moment the promise resolves —
no extra tick needed:

```json
[
  { "src": "kim", "rel": "works_on", "dst": "plugmem", "provenance": 0 },
  { "src": "kim", "rel": "reports_to", "dst": "ann" }
]
```

`provenance` is the fact the edge follows from, when it was recorded with one.
It is **absent** rather than zero when there is none, so it can never be
mistaken for fact `0` — as the second edge above shows.

### Checking a file has not rotted

`verify()` and `scrub()` ask different questions, and neither replaces the other:

- **`verify()`** — does the *content* agree with itself? Text is valid UTF-8,
  each vector belongs to its fact, both directions of every edge match.
- **`scrub()`** — are the *bytes* the ones that were written? It recomputes each
  section's checksum and the whole-file hash. This is what catches a flipped bit
  that the structure happily accepts.

A scrub is paced by you rather than run in one go, so it stays affordable on a
live database — the model ZFS uses. Each step checks up to a budget's worth of
bytes and returns:

```javascript
const scrub = await db.scrub();          // default budget: 1 MiB per step
let step;
while ((step = await scrub.next()) !== null) {
  // step.doneBytes of step.totalBytes — progress through the snapshot file
}
```

`next()` returns a promise because a step reads from disk, not because hashing
is slow: over a memory-mapped file the bytes are paged in as they are read, so a
step is I/O of whatever length your storage takes. On the JS thread that would
freeze the process.

Two things to know. **Holding the object holds a lock** on the snapshot
generation it is scanning, so run it to completion or `close()` it. And it is
one-shot: after it returns `null`, or throws, `active()` is `false` and you ask
the database for another.

```javascript
const partial = await db.scrub({ budget: 16 * 1024 });
await partial.next();     // { doneBytes: 16384, totalBytes: <the file's size> }
partial.close();          // released; further next() calls return null
```

Damage rejects with `PLUGMEM_ENGINE` naming what failed its checksum.

### Repairing a damaged file

`recover` is a module function, not a method: it works on **paths**, and takes
the source's exclusive lock, so close your handle first.

```javascript
import { recover } from "plugmem";

const report = await recover("memory.plugmem", "repaired.plugmem");
// { kept: 1, droppedText: 0, droppedVector: 0, droppedMetadata: 0 }
```

**The source is never written.** It stays exactly as it was, as evidence; this
produces a repaired copy beside it, and swapping them is your decision. `dst`
must therefore be a different path — passing the same one throws.

The three `dropped` counts are the damage: each is a fact the source could not
produce intact. All zero means the image was content-clean and this was a
compaction. Memory stays proportional to the record count rather than the file,
so a database far larger than RAM can be recovered.

It handles **content** damage — the kind `verify()` reports. A snapshot whose
container will not parse at all is not salvageable here; that is what a backup
is for.

### The one thing the Rust library has and this does not

`import`. The JSONL dump format is defined by
[`plugmem-cli`](https://docs.rs/plugmem-cli/latest), not by the engine — there is
no `import` verb to mirror. A Node program holding records already has
`rememberMany` and `link`, which is what an importer is made of.

## Errors

Every failure plugmem itself decides carries a stable `code`, so a program
branches on it instead of on wording:

```js
try {
  db = await Plugmem.open("agent.plugmem");
} catch (err) {
  if (err.code === "PLUGMEM_LOCKED") retryLater();
  else throw err;
}
```

`PLUGMEM_LOCKED`, `PLUGMEM_NEEDS_CHECKPOINT`, `PLUGMEM_CONFIG` and
`PLUGMEM_OPEN` come from opening; `PLUGMEM_INVALID_ARG` and
`PLUGMEM_INVALID_NAME` from an argument that was refused; `PLUGMEM_CLOSED`,
`PLUGMEM_READ_ONLY`, `PLUGMEM_WRITER_ONLY` and `PLUGMEM_BUSY` from calling a
verb the handle cannot serve; `PLUGMEM_ENGINE` from the engine itself, carrying
its own message.

The code is there whether the verb threw or the promise rejected — the two are
the same contract, so nothing has to be handled twice.

An argument that would shape an answer is refused rather than dropped: `range`
must be exactly `[from, to]`, and `range`, `asOf` and `validFrom` must each be a
finite, non-negative instant. Silently ignoring one produces an answer computed
without it, indistinguishable from a correct one.

## Configuration and embeddings

Settings resolve from an explicit `config` path, then `$PLUGMEM_CONFIG`, then
the platform config directory, then defaults. The database path resolves from an
explicit argument, then `$PLUGMEM_DB`, then `[database].path`, then the platform
data directory.

```typescript
const db = await Plugmem.open(undefined, { config: "./plugmem.toml" });
```

```toml
# plugmem.toml
[database]
path = "/path/to/memory.plugmem"

[engine]
dim = 768                     # embedding size (0 = vectors off)

[recall]                      # optional — every key has a tuned default
w_vec = 2.0                   # trust meaning over keywords in this memory
half_life_days = 30           # and treat anything older than a month as stale

[embedder]                    # optional — omit for lexical/tag/graph/time only
enabled = true                # false keeps settings but makes no embedder calls
url   = "http://localhost:11434/v1/embeddings"
model = "nomic-embed-text"
api_key_env = "OPENAI_API_KEY" # env var holding the bearer token

[maintenance]
fsync = "each_op"             # or "on_snapshot": faster, loses the journal tail on an OS crash
```

`[engine]` is what a database is *built* with; changing one of those on an
existing file is refused. `[recall]` and `[index]` are the opposite — reopening
with different weights is how you change the ranking, so tune them freely. All
of them are in the [full settings reference](https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md).

### When a key is misspelled

Unknown keys and sections do not stop anything, but they are not swallowed
either — a misspelled `w_vec` changes no behaviour, and silence would leave you
believing you had tuned something. **Read them once after opening**, because a
native addon has nowhere sensible to print:

```javascript
const db = await Plugmem.open("agent.plugmem", { config: "./plugmem.toml" });
for (const warning of db.configWarnings()) console.warn(warning);
// unknown setting [recall].w_vector — did you mean `w_vec`?
```

With an `[embedder]`, a text-only `remember`/`recall` embeds automatically, and
the provider's HTTP call happens outside the engine lock. The `dim` open option
sets the embedding size when there is no config; if the config built an
embedder, its dimension governs and `dim` must agree.

The host uses one `OpenAiCompatEmbedder` implementation for OpenAI, Ollama,
LM Studio, vLLM and other OpenAI-compatible servers. `url` is the complete
embeddings endpoint exactly as provided (nothing is appended), and `model` is
the model name understood by that server. Set `enabled = false` to keep the
settings without creating or calling the embedder; `$PLUGMEM_EMBEDDER_ENABLED`
overrides it with `true` or `false`.

A read-only handle cannot embed inside the engine — writing into a zero-copy
mapping is exactly what read-only exists to avoid — so this binding embeds the
query itself before the read. A text `recall` reaches the vector source in both
modes.

The [full settings reference](https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md)
lists every field and the OS-specific paths.

## Async and the event loop

Node runs all JavaScript on one thread, so a native call that waits on an
embedder's HTTP round trip or an fsync would freeze every timer, socket and
callback in the process. Anything that can do that runs on a libuv worker and
returns a promise instead.

Promises: `Plugmem.open`, `remember`, `rememberMany`, `revise`, `recall`,
`forget`, `link`, `unlink`, `export`, `exportPage`, `verify`, `maintain`,
`checkpoint`, and every `Workspace` verb except `closeIdle`, `openCount` and
`close`.

Synchronous: `path`, `get`, `stats`, `tagsOf`, `generation`, `refresh`, `close`.
These touch mapped memory and return in microseconds, where a promise would be
pure ceremony.

Arguments are still checked on your thread: a refused one **throws** at the call
site rather than rejecting later, so a mistake in your code and a failure in the
engine never arrive the same way.

### Two costs a promise does not hide

**The worker pool is shared, and it has four threads by default.** libuv runs
`fs`, `dns.lookup`, `zlib` and `crypto.pbkdf2` on the same pool this addon uses.
The event loop stays free either way, but four concurrent plugmem tasks fill the
default pool and everything else queues behind them. Measured on one machine, a
4 MiB `fs.readFile` in the same process:

| | `fs.readFile` |
|---|---|
| idle pool | 1.7 ms |
| 4 plugmem tasks in flight | 29 577 ms |
| the same, `UV_THREADPOOL_SIZE=8` | 1.4 ms |

plugmem's tasks are unusually long — `maintain('full')` is minutes on a large
memory — so raise `UV_THREADPOOL_SIZE` if the process does anything else with
libuv while maintenance runs.

The pool is also the ceiling on **concurrent embedding**. With an `[embedder]`
configured, each `remember`/`recall` occupies one worker for its HTTP round
trip, so at the default four, four is as parallel as it gets. Against a mock
provider with a fixed 100 ms latency, 16 concurrent recalls took 404 ms on the
default pool and 101 ms at `UV_THREADPOOL_SIZE=16` — the same 16 requests, four
waves or one. If a process issues many concurrent recalls against a remote
provider, size the pool for that, not for the CPU.

**`export()` builds its whole result on your thread.** The scan is on a worker,
but every fact becomes a JavaScript object during the promise's resolution, and
that part is main-thread work by definition. On 100 000 facts it holds the
thread for about 244 ms of the call's 289 ms. `exportPage()` over the same
memory holds it for **0 ms**, in 128-fact pages:

```ts
let cursor: number | undefined;
do {
  const page = await db.exportPage(cursor);
  for (const fact of page.facts) await destination.write(fact);
  cursor = page.nextCursor;
} while (cursor !== undefined);
```

Each promise owns exactly one page and resolves only after its native scan
completed; no database lock is held while your loop body runs. A writer may
change between pages, so do not mutate it during a snapshot-style dump — a
read-only handle pages one immutable checkpoint and is stable.

### Concurrency

A `Plugmem` handle is safe to use from anywhere in your process. Reads run
concurrently; writes serialize behind the engine's lock for the microseconds
they take. A second *process* opening the same file for writing is refused with
`PLUGMEM_LOCKED` rather than corrupting it, while any number of read-only
handles map the same file at once — a writer and its readers coexist across
processes, sharing the OS page cache.

## Many memories in one directory

**Default: one logical memory backed by a local database layout.** `Plugmem.open(path)` and nothing here
applies.

The problem this solves: a process serving many conversations, tenants or
projects wants each to have its **own** memory — nothing from one surfacing in
another — without managing a pile of file paths by hand. Give a name, get a
memory:

```ts
import { Workspace, type DbEntry } from "plugmem";

const ws = new Workspace("/srv/memories");

// The same `Plugmem` class comes back, so a named memory has exactly the verbs
// a path-opened one has. A first write to an unused name creates it.
const chat = await ws.open("chat-42");
await chat.remember({ text: "prefers tokio" });

// Another name is another memory. They cannot see each other.
const other = await ws.open("chat-99");
(await other.recall({ query: "tokio" })).facts.length;   // 0
```

Memories are **independent by design**: nothing searches across them and no
entity links between them. A fact filed under the wrong name is not merely
misplaced, it is unreachable from the other memory.

If you do not know the name, ask what each memory is for. Descriptions are
searchable, and so are owners, even though an owner is stored as a graph edge
rather than as text:

```ts
await ws.describe("chat-42", { description: "release planning", owner: "ann" });

const hits: DbEntry[] = await ws.find("release planning");  // → [{ db: "chat-42", … }]
const byOwner: DbEntry[] = await ws.find("ann");            // → the same memory
```

A name is `[a-z0-9][a-z0-9_-]*` and **cannot express a path**, so it resolves to
exactly one named database inside the directory — traversal is not filtered out, it is
unconstructible. `ws.open(name, false)` refuses a name that does not exist yet,
which is what a read should do so a typo is diagnosed rather than answered with
an empty result.

**Who may reach which memory is not this package's job.** The name comes from
your code, so the policy belongs there.

### Running it

| Method | Does |
|---|---|
| `open(name, create?)` | open (default: create if missing) and hand back a `Plugmem` |
| `list()` | every memory in the directory, from the filesystem — including undescribed ones |
| `entries()` | every described memory, from the registry |
| `find(query, k?)` | memories whose description or owner best matches |
| `describe(name, args)` | record what a memory is for; revises rather than duplicating |
| `archive(name)` | label it archived, keeping its description. Nothing is moved or deleted |
| `reindex()` | rebuild the registry from the memories' own descriptions |
| `verify()` | report disagreements between registry and directory; repairs nothing |
| `closeIdle()` | close memories unused past the idle timeout (sync) |
| `openCount()` | how many are open right now (sync) |
| `close()` | close every pooled memory and the registry |

`closeIdle()` matters more than it looks. An open memory holds its file's
exclusive lock, so a long-running process that never lets go makes its memories
unreachable from anything else on the machine. Call it on a timer — that is what
the idle timeout is for, liveness rather than memory. The pool bounds how many
stay open at once (`maxOpen`, default 16, least-recently-used closed to make
room):

```ts
const ws = new Workspace("/srv/memories", { maxOpen: 16, idleTimeoutMs: 60_000 });
setInterval(() => ws.closeIdle(), 30_000);
```

A `Plugmem` handed out by `open()` is **not** closed by `ws.close()`: it is its
own handle holding its own lock until you close it or it is garbage collected.

`verify()` reports and never repairs, because a workspace is a directory a
person can edit, and guessing at their intent is how a consistency check loses
data.

## What it is not for

plugmem is for local-first application and agent memory: one process, one local database,
no service to operate. Its design centre is around 100 000 active facts on one machine, and
the benchmarks track 1M-operation profiles to show how the same engine behaves
under heavier local load.

It is **not** a vector database and not built for multi-million vector
workloads, cluster sharding, multi-tenant serving or managed nearest-neighbour
search. For those, use a dedicated system — [Qdrant](https://qdrant.tech),
[Milvus](https://milvus.io), [Weaviate](https://weaviate.io),
[Pinecone](https://www.pinecone.io) or
[pgvector](https://github.com/pgvector/pgvector).

## Other ways in

The same engine ships five ways. This package is the Node one.

| You are | Use |
|---|---|
| writing JavaScript / TypeScript for Node | **this package** |
| writing Python | [`plugmem`](https://pypi.org/project/plugmem/) on PyPI |
| writing Rust | [`plugmem-host`](https://docs.rs/plugmem-host/latest) — the engine in your process |
| an agent, or another language | [`plugmem-mcp`](https://docs.rs/plugmem-mcp/latest) — a stdio JSON-RPC sidecar |
| a person at a terminal | [`plugmem-cli`](https://docs.rs/plugmem-cli/latest) |

**Working with an LLM agent?** There is a companion
[skill](https://github.com/m62624/plugmem/blob/main/skill/SKILL.md) describing
the remember/recall loop, the contradiction workflow and the verbs. This package
ships it: `skill()` returns the text and `skillVersion()` the version it was
written against.

## License

MIT. Source: <https://github.com/m62624/plugmem>
