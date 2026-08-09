# plugmem

> ⚠️ Experimental. plugmem is mostly an AI-built experiment — written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

An embeddable bitemporal memory database for local-first applications and
agents, embedded in your Python process. It stores short facts and answers a
query with ranked facts and edges plus an optional bounded rendered block.

File-backed on disk, no server, no daemon. The
[`plugmem-host`](https://docs.rs/plugmem-host/latest) engine is compiled to a
CPython extension module through [PyO3](https://pyo3.rs) and linked directly
into the process, so the data lives in mapped files rather than in the
interpreter's heap. Every call releases the GIL for the duration of the work.

**No embedding model is required.** Three of the four retrieval sources — text,
graph and time — need nothing but the database. An embedder is optional and
adds the fourth; see [Do you need an embedder?](#do-you-need-an-embedder) for
what changes when you add one and what you give up without it.

**Contents:** [Install](#install) · [Quick start](#quick-start) ·
[Do you need an embedder?](#do-you-need-an-embedder) ·
[What it stores](#what-it-stores) · [Two clocks](#two-clocks) ·
[How recall works](#how-recall-works) · [API](#api) · [Errors](#errors) ·
[Configuration](#configuration-and-embeddings) ·
[Threads and the GIL](#threads-and-the-gil) · [Typing](#typing) ·
[Many memories](#many-memories-in-one-directory) ·
[What it is not for](#what-it-is-not-for)

## Install

```console
$ pip install plugmem
```

Prebuilt wheels cover Linux, macOS and Windows on x86-64 and arm64. One wheel
per platform serves every CPython from 3.10 on — it is built against the stable
ABI, which does not change between versions — plus a separate wheel for the
free-threaded 3.14t build, which has an ABI of its own. No toolchain, no build
step.

## Quick start

```python
import plugmem

db = plugmem.Plugmem.open("agent.plugmem")

db.remember("the user prefers tokio", entity="user", tags=["pref"])
db.remember("the release ships on friday", entity="release")

res = db.recall("tokio", k=5)
print(res.rendered)

page = db.list_tags(prefix="pre", limit=64)
print(page.items)            # [TagSummary(name='pref', count=1)]
# db.remove_tag("pref")      # global: revises every current fact carrying it

db.close()
```

```
## memory
- [f0] user: the user prefers tokio (2026-08; active) #pref
```

`Plugmem` is also a context manager, which is the usual way to write it:

```python
with plugmem.Plugmem.open("agent.plugmem") as db:
    db.remember("the deploy target is fly.io", entity="release")
```

`open` is a static method rather than a constructor because it takes the file's
exclusive lock, replays the journal and maps the snapshot — work proportional
to what is on disk, and worth naming rather than hiding behind `Plugmem(...)`.

## Do you need an embedder?

No, and it is worth being precise about what that costs, because the answer
decides how you should write your queries.

**Without one**, three sources answer: BM25 over the text, the entity graph,
and time. That is a working memory with no model, no API key, no network call
and no per-query cost. What you lose is matching by *meaning*: BM25 needs
shared words, so the query above finds the fact because both say "tokio".
Ask it the way a person would —

```python
db.recall("which runtime?", k=5)   # → no facts: no word in common
```

— and you get nothing back, because "runtime" appears nowhere in "the user
prefers tokio". Anchor on an entity (`entities=["user"]`) or use the words the
fact uses, and it answers.

**With one**, a fourth source runs: each fact and each query is embedded, and
cosine similarity finds the fact whose *meaning* is close even when no word
matches. `"which runtime?"` then reaches "the user prefers tokio". The cost is
a provider round trip per write and per text query, an API key, and a `dim`
that is fixed for the life of the database.

You can also skip the provider and pass vectors yourself — see
[Bringing your own embedding](#bringing-your-own-embedding) — which is the
route for a local model or one that is not an OpenAI-shaped HTTP endpoint.

Sensible default: start without one. Tag and anchor your facts, see whether
lexical recall is enough for your queries, and add an embedder when you catch
yourself wishing a query had understood a synonym.

## What it stores

A **fact** is one short statement plus the things that make it findable and
datable:

```python
db.remember(
    "the user prefers tokio",
    entity="user",                       # the subject
    tags=["pref", "runtime"],            # filters
    links=[("works_on", "plugmem")],     # typed edges from the subject
    metadata={"src": "chat-2026-08-05"}, # opaque to the engine
    valid_from=1_767_225_600_000,        # when it became true (unix ms)
)
```

`metadata` is a string-to-string map the engine never interprets. It is where a
URI to the real payload goes, or a mime type, or a key in your own system — the
fact stays short and searchable while the bulk stays wherever you keep bulk.

`remember` returns the new id and any live facts that look like duplicates or
contradictions:

```python
out = db.remember("the user prefers async-std", entity="user")
for hint in out.similar:
    print(hint.id, hint.score, hint.reason)   # 0 0.87 LexicalOverlap
```

The engine never merges on its own. You decide: `revise` if it changed,
`forget` if it was wrong, or nothing if both are true at once.

## Two clocks

This is what separates plugmem from a store that overwrites. Every fact carries
two independent intervals:

- **recorded_at** — when this memory *learned* it. Immutable.
- **valid_from / valid_to** — when the statement was *true in the world*.

`revise` closes the old interval instead of deleting the old row:

```python
JAN = 1_767_225_600_000   # 2026-01-01
JUL = 1_782_864_000_000   # 2026-07-01

berlin = db.remember("the user lives in berlin", entity="user", valid_from=JAN)
db.revise(berlin.id, "the user lives in lisbon", entity="user", valid_from=JUL)

db.recall("lives", entities=["user"])
# → [the user lives in lisbon]

db.recall("lives", entities=["user"], closed=True)
# → [the user lives in berlin, the user lives in lisbon]
```

The Berlin fact is still there, with `valid_to` now set to `JUL` — the instant
its successor took over. Nothing was overwritten, so "where did the user live
in March" remains answerable.

`as_of` asks the bitemporal question, and it filters on **both** axes:

```python
db.recall("lives", entities=["user"], as_of=FEBRUARY)
# → []
```

Empty, and that is the correct answer rather than a bug. Both facts were
*recorded* today, so as of February this memory did not know either of them. A
memory that answered would be claiming knowledge it did not have. Ask as of an
instant the memory had already reached and you get whatever was true then.

## How recall works

Four sources run and are fused with reciprocal-rank fusion, then a recency
boost is applied. Tags filter; they are not a source.

| Source | What it matches | Needs an embedder |
|---|---|---|
| lexical | BM25 over the fact text | no |
| graph | facts reachable from the anchors in `entities` | no |
| time | facts inside the `range` window | no |
| vector | cosine over embeddings | yes |

```python
res = db.recall(
    "who owns the deploy",
    tags=["ops"],                  # a filter, not a source
    entities=["release"],          # graph anchors
    range=(FROM_MS, TO_MS),        # window over recorded_at
    k=8,                           # facts to return
    token_budget=512,              # size of `rendered`
    graph_depth=2,                 # hops from the anchors, this call only
)

for fact in res.facts:
    print(fact.id, fact.score, fact.sources)
for edge in res.edges:
    print(edge.src, edge.rel, edge.dst)
print(res.rendered)                # the bounded block
print(res.truncated)               # True if something was left out
```

`res.facts` and `res.edges` are the structured answer; `res.rendered` is a
convenience for callers that want a block of text under a token budget. Neither
is more real than the other.

`graph_depth` is per call because how wide a net to cast belongs to the
question: "what is this person's stated preference" wants fewer hops than "what
is known around this person". There is no ceiling — the walk is bounded by its
own entity and edge caps.

## API

Everything is synchronous. See [Threads and the GIL](#threads-and-the-gil) for
why that is the right shape and not a limitation.

| Verb | Does |
|---|---|
| `Plugmem.open(path=None, *, dim=None, read_only=False, config=None)` | open or create; resolves `PLUGMEM_DB`, then `[database].path`, then the platform data path |
| `remember(text, *, entity, tags, links, metadata, valid_from, vector)` | store one fact |
| `remember_many(facts)` | store a batch — one journal write, one embedding round trip |
| `revise(id, text, ...)` | close a fact's interval and record the successor |
| `recall(query=None, *, tags, entities, as_of, range, k, closed, token_budget, ef, graph_depth, vector)` | the ranked answer |
| `forget(id)` | tombstone a fact; `maintain` purges it later |
| `remove_tag(tag)` | remove a tag from every current fact while preserving facts/history |
| `link(src, rel, dst, *, provenance)` / `unlink(src, rel, dst)` | open or close a typed edge |
| `get(id)` / `tags_of(id)` / `stats()` | one fact's card, its tags, engine counters |
| `list_tags(*, prefix=None, cursor=None, limit=0)` | bounded lexical page of current tags and counts |
| `export()` / `export_page(cursor)` / `export_edges(on_batch)` | dump facts, dump them in pages, stream edges |
| `verify()` / `scrub(budget=None)` | logical check; byte-level check |
| `maintain(mode="auto")` / `checkpoint()` | housekeeping; publish a snapshot |
| `reembed(batch_size=128)` | explicitly recompute every retained vector with the configured model and publish atomically; `maintain("auto")` never invokes it |
| `generation()` / `refresh()` | read-only handles: which snapshot, and move to the newest |
| `config_warnings()` / `path()` / `close()` | config typos, the resolved file, release it |

Module level: `version()`, `about()`, `settings_help()`, `skill()`,
`skill_full()`, `skill_version()`, `recover(src, dst)`, and the
`export_pages(db)` generator.

### Bringing your own embedding

Pass `vector` and nothing is sent to a provider — it replaces the embedder for
that call. This is the route for a local model, or one that is not an
OpenAI-shaped HTTP endpoint:

```python
db = plugmem.Plugmem.open("agent.plugmem", dim=384)
db.remember("the user prefers tokio", entity="user", vector=my_model.encode(text))
db.recall(vector=my_model.encode("which runtime?"), k=5)
```

The length must equal the configured `dim`, which is fixed when the database is
created.

### Backing up: facts are only half of it

A fact names its own tags and metadata, but an **edge is a statement between
two entities** and outlives any single fact. A complete dump is both streams:

```python
facts = []
for page in plugmem.export_pages(db):      # bounded pages, not one big list
    facts.extend(page.facts)

edges = []
db.export_edges(edges.extend)              # called with a list at a time
```

`export_edges` hands over batches rather than one edge per call, because each
call has to reacquire the interpreter; the walk itself runs with the GIL
released. It returns the total.

### Checking a file has not rotted

`verify()` asks whether the indexes agree with the facts. `scrub()` asks
whether the bytes on disk are the bytes that were written — it recomputes the
stored checksums, which is what catches a flipped bit that the structure
happily accepts.

```python
with db.scrub() as scan:
    for progress in scan:
        print(f"{progress.done_bytes}/{progress.total_bytes}")
```

It is paced by you rather than run to completion, so it is affordable on a live
database. **Holding the object holds a lock** on the snapshot generation it is
scanning, so the writer cannot recycle that file underneath it — finish the
scan or close it, which the `with` block does for you.

### Repairing a damaged file

```python
report = plugmem.recover("damaged.plugmem", "clean.plugmem")
print(report.kept, report.dropped_text, report.dropped_vector, report.dropped_metadata)
```

It reads the source fact by fact, writes what survives to a new file, and
reports what it had to drop. The source is left untouched as evidence. This is
not a repair for structural damage: a snapshot that will not parse cannot be
walked, and that case is a restore from backup.

### The one thing the Rust library has and this does not

`import` is not an engine verb — JSONL is a format the CLI defines. If you need
it, `remember_many` plus `link` is the whole of it, in a dozen lines of Python
you can shape to your own file.

## Errors

Every failure this binding decides raises a subclass of `PlugmemError` carrying
a stable `code`. The codes are the same strings the Node binding puts on a
thrown `Error`, so cross-language documentation stays one table.

```python
try:
    db = plugmem.Plugmem.open("agent.plugmem")
except plugmem.LockedError as e:
    print(e.code)          # PLUGMEM_LOCKED — another process holds the writer
```

| Class | `code` | Means |
|---|---|---|
| `LockedError` | `PLUGMEM_LOCKED` | another process holds the writer lock |
| `NeedsCheckpointError` | `PLUGMEM_NEEDS_CHECKPOINT` | `read_only` on a database nobody has checkpointed |
| `ConfigError` | `PLUGMEM_CONFIG` | the `config.toml` could not be read or is invalid |
| `OpenError` | `PLUGMEM_OPEN` | any other failure to open |
| `InvalidArgError` | `PLUGMEM_INVALID_ARG` | an argument refused before it reached the engine |
| `InvalidNameError` | `PLUGMEM_INVALID_NAME` | not a usable memory name |
| `ClosedError` | `PLUGMEM_CLOSED` | `close()` was already called |
| `ReadOnlyError` | `PLUGMEM_READ_ONLY` | a write verb on a read-only handle |
| `WriterOnlyError` | `PLUGMEM_WRITER_ONLY` | `generation`/`refresh` on a writer |
| `BusyError` | `PLUGMEM_BUSY` | another operation holds this handle |
| `EngineError` | `PLUGMEM_ENGINE` | the engine failed; the message is its own |

## Configuration and embeddings

Without a config, plugmem answers from text, tags, the graph and time. Add an
`[embedder]` section and `remember`/`recall` also embed, giving the vector
source something to work with — see
[Do you need an embedder?](#do-you-need-an-embedder) for the trade.

The file is resolved the same way on every surface — CLI, MCP server, Node and
Python: an explicit path, then `$PLUGMEM_CONFIG`, then the platform config
directory — `$XDG_CONFIG_HOME/plugmem/config.toml` on Linux,
`~/Library/Application Support/plugmem/config.toml` on macOS,
`%APPDATA%\plugmem\config\config.toml` on Windows.

```toml
# plugmem.toml
[database]
path = "~/.local/share/plugmem/agent.plugmem"

[engine]
dim = 1536

# Optional. Delete this section and everything still works, minus the vector
# source.
[embedder]
enabled = true
url = "https://api.openai.com/v1/embeddings"
model = "text-embedding-3-small"
space_id = "text-embedding-3-small@v1" # optional; defaults to model
api_key_env = "OPENAI_API_KEY"

[recall]
w_bm25 = 1.0        # weight of the lexical source
w_vec = 1.0         # weight of the vector source
w_graph = 0.7       # weight of the graph source
half_life_days = 30 # how fast the recency boost decays
graph_depth = 2     # default hops, overridable per call
```

```python
db = plugmem.Plugmem.open("agent.plugmem", config="plugmem.toml")
```

The host uses one `OpenAiCompatEmbedder` implementation for OpenAI, Ollama,
LM Studio, vLLM and other OpenAI-compatible servers. `url` is the complete
embeddings endpoint exactly as provided (nothing is appended), and `model` is
the model name understood by that server. `space_id` optionally identifies the
exact semantic space and defaults to `model`; it is never discovered over the
network. Set `enabled = false` to keep the
settings without creating or calling the embedder; `$PLUGMEM_EMBEDDER_ENABLED`
overrides it with `true` or `false`.

`plugmem.settings_help()` returns the whole catalogue — every section, key,
type, default and what it does — without opening anything.

### When a key is misspelled

A typo in a key used to change nothing, silently. Now it is reported:

```python
db = plugmem.Plugmem.open("agent.plugmem", config="plugmem.toml")
for warning in db.config_warnings():
    print(warning)
# [recall] unknown key `w_vector` — did you mean `w_vec`?
```

It is a value rather than a printed warning because a library has nowhere
sensible to print. Read it once after opening and log it your own way.

## Threads and the GIL

The Python API is deliberately synchronous. In ordinary Python code, call every
method directly. **Every verb releases the GIL while native work runs**, so
other Python threads can continue and a shared handle is safe to use from a
`ThreadPoolExecutor`.

Releasing the GIL does not move the call to another thread. If a synchronous
method is called directly from an `async def`, that event-loop thread remains
occupied until the method returns. Async applications should send operations
that can wait on I/O, an embedder, a lock, or database-sized work to
`asyncio.to_thread`:

```python
import asyncio

res = await asyncio.to_thread(db.recall, "tokio", k=5)
page = await asyncio.to_thread(db.list_tags, prefix="project:", limit=64)
report = await asyncio.to_thread(db.remove_tag, "obsolete")
vectors = await asyncio.to_thread(db.reembed, 128)
```

Use this practical split inside an async application:

| Call directly | Use `await asyncio.to_thread(...)` |
|---|---|
| `get`, `tags_of`, `stats`, `generation`, `path`, `config_warnings` | `open`, `remember`, `remember_many`, `revise`, `recall`, `forget`, `remove_tag`, `list_tags`, `link`, `unlink` |
| simple property/result access | `export`, `export_page`, `export_edges`, `verify`, `scrub`, `maintain`, `reembed`, `checkpoint`, `refresh`, `recover` |

The left column only performs bounded in-memory reads and normally returns in
microseconds. The right column may open files, wait for another process, call an
embedding provider, flush storage, or scan data. Using `to_thread` is the async
bridge; the binding does not add a second runtime or pretend native work can be
cancelled after it has started.

A handle is safe to share across threads. Reads genuinely overlap; `refresh`
and `close` take the handle exclusively, so a reader never observes it
half-swapped.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(lambda q: db.recall(q, k=4), queries))
```

Writes serialize inside the engine, which is a property of the engine and not
of this binding — one writer per file is the design. Embedding happens outside
that lock, so several `remember` calls do reach the provider at once.

### Reading while another process writes

```python
reader = plugmem.Plugmem.open("agent.plugmem", read_only=True)
print(reader.generation())      # which published snapshot this is
if reader.refresh():            # move to the newest one
    print("moved to", reader.generation())
```

A read-only handle maps a published snapshot without taking the writer's lock,
so it coexists with a live writer in another process. It needs the database to
have been checkpointed at least once; otherwise `NeedsCheckpointError`.

## Typing

The package ships `py.typed` and generated stubs, so an editor completes the
API and a type checker checks it. The stubs are generated from the same macros
the binding is written with and gated in CI against the Rust surface, so they
cannot describe a method that does not exist.

```python
res: plugmem.RecallResult = db.recall("tokio", k=5)
first: plugmem.RecalledFact = res.facts[0]
```

Results are frozen: they are what the engine said, and mutating a copy of that
is never what anyone means.

## Many memories in one directory

Optional. If you want one memory, point `Plugmem.open` at a file and skip this.

For a process serving many independent memories — one per conversation, per
tenant, per project — address them by name instead:

```python
ws = plugmem.Workspace("~/bot-data")

memory = ws.memory("conversation-42")      # opens nothing and holds no lock
memory.remember("the user prefers dark mode", entity="user")  # first write creates

ws.describe("conversation-42", "support thread about billing", owner="ann")
for entry in ws.find("billing"):
    print(entry.db, entry.description)
```

A name (`[a-z0-9][a-z0-9_-]*`) is not a path and cannot become one, so it
resolves to exactly one database inside the directory. `describe` is what makes
`find` useful when the caller does not know the name; `owner` is recorded as an
edge, so `find("ann")` returns what Ann owns even though no description
mentions her.

The workspace owns a bounded pool of real database handles. A
`WorkspaceMemory` owns only its name plus a weak reference to the workspace;
each verb obtains a scoped handle while the GIL is released. An inactive
least-recently-used database is closed to make room, while active verbs are
never evicted. If every `max_open` slot is active, a call for a different memory
raises `BusyError` immediately instead of waiting or opening a hidden extra
handle.

`release(name)` closes one inactive pooled handle without invalidating logical
references; their next verb reopens it. `close_idle()` closes entries idle past
the configured timeout. Both matter because a pooled writer holds that
database's exclusive OS lock. `Workspace.close()` invalidates every
`WorkspaceMemory`; garbage-collecting a logical reference closes nothing.

This does not change the ordinary API: `Plugmem.open(path)` is still an
explicitly owned database handle, and its `close()` releases that direct lock.

This replaces the old handle-returning call and is intentionally breaking:

```python
# before
memory = ws.open("conversation-42")
memory.close()

# now
memory = ws.memory("conversation-42")
ws.release("conversation-42")  # optional immediate release when inactive
```

`WorkspaceMemory` has no `close()` because it owns no native resource. All its
verbs are ordinary synchronous Python calls whose engine work runs without the
GIL.

## What it is not for

plugmem is for local-first application and agent memory: one process, one local
database, no service to operate. Its design centre is around 100 000 active
facts on one machine, and the benchmarks track 1M-operation profiles to show
how the same engine behaves under heavier local load.

It is **not** a vector database and not built for multi-million vector
workloads, cluster sharding, multi-tenant serving or managed nearest-neighbour
search. For those, use a dedicated system — [Qdrant](https://qdrant.tech),
[Milvus](https://milvus.io), [Weaviate](https://weaviate.io),
[Pinecone](https://www.pinecone.io) or
[pgvector](https://github.com/pgvector/pgvector).

## Other ways in

The same engine ships five ways. This package is the Python one.

| You are | Use |
|---|---|
| writing Python | **this package** |
| writing JavaScript / TypeScript for Node | [`plugmem`](https://www.npmjs.com/package/plugmem) on npm |
| writing Rust | [`plugmem-host`](https://docs.rs/plugmem-host/latest) — the engine in your process |
| an agent, or another language | [`plugmem-mcp`](https://docs.rs/plugmem-mcp/latest) — a stdio JSON-RPC sidecar |
| a person at a terminal | [`plugmem-cli`](https://docs.rs/plugmem-cli/latest) |

**Working with an LLM agent?** There is a companion
[skill](https://github.com/m62624/plugmem/blob/main/skill/SKILL.md) describing
the remember/recall loop, the contradiction workflow and the verbs. This
package ships it: `skill()` returns the text and `skill_version()` the version
it was written against.

## License

MIT. Source: <https://github.com/m62624/plugmem>
