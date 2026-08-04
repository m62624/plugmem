# plugmem-napi

> ⚠️ Experimental. plugmem is mostly an AI-built experiment — written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

`plugmem-napi` is the **native Node.js addon** for the plugmem
[temporal-memory engine](https://docs.rs/plugmem-core/latest) — it embeds
[`plugmem-host`](https://docs.rs/plugmem-host/latest) **in the Node process**
(real mmap, file locking, cross-process MVCC — the whole engine, unchanged) and
exposes it to **JavaScript / TypeScript** as a `Plugmem` class. It is published
to npm as **`plugmem`**.

Because it is native (not WebAssembly), there is no whole-file-in-RAM copy and no
4 GiB ceiling: the OS pages the snapshot in and out exactly as it does for the
Rust library. It loads in **Node, and any N-API host (Deno, Bun)**.

## Install

```console
$ npm install plugmem
```

`npm i plugmem` pulls the meta package, which through `optionalDependencies`
installs only the prebuilt binary for your platform — one of
`plugmem-{linux-x64-gnu, linux-arm64-gnu, darwin-x64, darwin-arm64,
win32-x64-msvc, win32-arm64-msvc}`. No toolchain, no build step.

## Which door is this?

plugmem is **embedded-first, like SQLite**. Pick the door for your language:

| You are… | Use | Why |
|---|---|---|
| **writing JavaScript / TypeScript for Node** | **`plugmem-napi`** (this, npm `plugmem`) | The engine *in your Node process*, native speed, typed for TS. |
| **writing Rust** | [`plugmem-host`](https://docs.rs/plugmem-host/latest) | The engine in your process, like linking SQLite. |
| **an agent, or another language** (Python, Go…) | [`plugmem-mcp`](https://docs.rs/plugmem-mcp/latest) | A long-lived stdio JSON-RPC sidecar; language-independent. |
| a person at a **terminal / script** | [`plugmem-cli`](https://docs.rs/plugmem-cli/latest) | The human door. |

So: **Node/TS → napi; Rust → host; an agent or other language → MCP; a human →
the CLI.**

## Usage (TypeScript)

Every argument and result is typed — napi generates `index.d.ts`, so a TS host
gets full autocomplete and checking:

```typescript
import { Plugmem } from "plugmem";

const db = new Plugmem("agent.plugmem");          // or { readOnly: true }

const out = await db.remember({
  text: "prefers tokio",
  entity: "user",
  tags: ["pref"],
  links: [{ rel: "works_at", entity: "acme" }],
  metadata: { source: "chat", uri: "s3://bucket/note.txt" }, // opaque; a pointer
});
out.id;                       // number
out.similar;                  // Similar[] — the engine surfaces conflicts, you decide

const res = await db.recall({ query: "runtime?", k: 5 });
res.rendered;                 // the prompt-ready block
res.facts;                    // RecalledFact[] — { id, score, entity, recordedAt, … }

const card = db.get(out.id);
card.metadata;                // Record<string,string> — keys sorted, {} when none

await db.revise(out.id, { text: "prefers async-std" });
await db.link({ src: "user", rel: "works_at", dst: "acme" });
db.unlink({ src: "user", rel: "works_at", dst: "acme" }); // closes the current edge

await db.checkpoint();        // async (see below)
db.close();                   // release the file + lock explicitly
```

## Configuration & embeddings

The constructor resolves settings **exactly like the CLI and MCP server**: an
explicit `config` path wins, else `$PLUGMEM_CONFIG`, else the platform config
directory, else all defaults. The database path is resolved as an explicit
constructor path, then `$PLUGMEM_DB`, then `[database].path`, then the platform
data directory. See the [full settings reference](https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md)
for all fields and OS-specific paths.

```typescript
const db = new Plugmem(undefined, { config: "./plugmem.toml" });
```

```toml
# plugmem.toml
[database]
path = "/path/to/memory.plugmem" # optional example

[engine]
dim = 768                     # embedding size (0 = vectors off)

[embedder]                    # optional — omit for lexical/tag/graph/time only
kind  = "ollama"              # or openai / lmstudio / vllm / llamacpp
url   = "http://localhost:11434/v1/embeddings"
model = "nomic-embed-text"
```

With an `[embedder]`, a text-only `remember`/`recall` **auto-embeds** — the
provider's HTTP call runs outside the engine lock. Without one, there is no
embedder and vector recall is skipped (lexical, tag, graph and time recall still
answer). The optional `dim` open option sets the embedding size when there is no
config; if the config configured an embedder, its dimension governs and `dim`
must agree. A `{ readOnly: true }` handle cannot auto-embed inside the engine —
embedding into a zero-copy mapping is what read-only exists to avoid — so this
binding embeds the query itself before the read, exactly as the CLI and the MCP
server do. A text `recall` therefore reaches the vector source in both modes.

## The verbs

Every method here is the identically-named `plugmem-host` `Database` verb; the
engine logic is entirely the host's.

**Writer** (default): `remember`, `rememberMany`, `recall`, `revise(id, args)`,
`forget(id)`, `link`, `unlink`, `get(id)`, `tagsOf(id)`, `stats`, `export`,
`exportPage(cursor?)`, `verify`, and the async maintenance verbs below.
**Read-only** (`{ readOnly: true }`, observing another process's writer):
`recall`, `get`, `tagsOf`, `stats`, `export`, `exportPage(cursor?)`, `verify`,
plus `generation()` (the pinned snapshot generation) and `refresh()` (advance
to the writer's latest checkpoint); the write verbs throw.
**Both**: `path()` — the file the handle resolved to, which is the only way to
learn it when the constructor was given no path.

### Errors

Every failure plugmem itself decides carries a stable `code`, so a program
branches on it instead of on wording:

```js
try {
  db = new Plugmem("agent.plugmem");
} catch (err) {
  if (err.code === "PLUGMEM_LOCKED") retryLater();
  else throw err;
}
```

`PLUGMEM_LOCKED`, `PLUGMEM_NEEDS_CHECKPOINT`, `PLUGMEM_CONFIG` and
`PLUGMEM_OPEN` come from opening; `PLUGMEM_INVALID_ARG` and
`PLUGMEM_INVALID_NAME` from an argument that was refused; `PLUGMEM_CLOSED`,
`PLUGMEM_READ_ONLY`, `PLUGMEM_WRITER_ONLY` and `PLUGMEM_BUSY` from calling a
verb the handle cannot serve. A failure *inside* the engine keeps napi's
`GenericFailure` and the host's message — napi-rs fixes the error type of an
async task, so a code there could not be delivered on `maintain` or
`checkpoint`, and one that appeared only on the synchronous half would mean
different things on different verbs.

An argument that shapes an answer is refused rather than dropped: `range` must
be exactly `[from, to]`, and `range`, `asOf` and `validFrom` must each be a
finite, non-negative instant. Silently ignoring one produced an answer computed
without it — indistinguishable from a correct one.

### What the host has and this does not

The list above is the whole supported `Plugmem` surface. The only host
operation intentionally kept out of this boundary is path-level recovery:

| host verb | boundary note |
|---|---|
| `recover` | salvaging a damaged file is a path-level operation on the disk the process is running on — [`plugmem-cli recover`](https://docs.rs/plugmem-cli/latest)'s job, like `import` and `scrub`. |
| `remember_many` | Exposed as async `rememberMany(items)`. It writes a batch with one embedding round-trip and resolves with outcomes in input order. |
| `export_each` | Exposed as pull-based `exportPage(cursor?)`. Each Promise returns at most 128 facts and releases the native read lock before JS processes them; this gives Node backpressure without a cross-thread callback. |
| `tags_of` | Exposed as synchronous `tagsOf(id)`, returning one fact's tags or an empty array. |

**No `import` verb** either — bulk-loading a `backup.jsonl` reads a file on disk,
which is the CLI's job. A Node host can use `rememberMany` for bounded batches
when it already owns the input records.

## Many memories in one directory (optional)

**Default: one memory, one file.** `new Plugmem(path)` and nothing here applies.

A process that serves many independent memories — one per conversation, per
tenant, per project — can point at a directory and address them by name:

```ts
import { Workspace, type DbEntry } from "plugmem";

const ws = new Workspace("/srv/memories", { maxOpen: 16, idleTimeoutMs: 60_000 });

// `open` hands back the same `Plugmem` class, so a named memory has exactly the
// verbs a path-opened one has. A first write to an unused name creates it.
await (await ws.open("chat-42")).remember({ text: "prefers tokio" });

// Do not know the name? Ask what each memory is for. Owners are searchable too,
// even though an owner is a graph edge rather than text.
await ws.describe("chat-42", { description: "release planning", owner: "ann" });
const hits: DbEntry[] = await ws.find("release planning"); // → [{ db: "chat-42", … }]
```

A name is `[a-z0-9][a-z0-9_-]*` and **cannot represent a path**, so it resolves
to exactly one file inside the directory — traversal is not filtered out, it is
unconstructible. `open(name, false)` refuses a name that does not exist yet,
which is what a read should do so a typo is diagnosed rather than answered with
an empty result.

The pool bounds how many stay open; `closeIdle()` releases the rest. Call it on
a timer: an open memory holds the file's **exclusive lock**, so a long-running
process that never let go would make its memories unreachable from anything else
on the machine. That is what the idle timeout is for — liveness, not memory.

Two things to know before building on it. Memories are **independent**: nothing
searches across them and no entity links between them, so a fact filed in the
wrong one is unreachable from the other rather than merely misplaced. And **who
may reach which memory is not this package's responsibility** — the name comes
from your code, so put the policy there.

Every `Workspace` verb that touches a database returns a promise — `open`,
`list`, `entries`, `find`, `describe`, `archive`, `reindex`, `verify` — because
the registry is itself a plugmem memory and a named memory is a real file being
opened. `closeIdle()` and `openCount()` stay synchronous.

`reindex()` and `verify()` return promises: they open and read every memory in
the directory, which is not work for the main thread.

## Async and concurrency

Operations with unbounded storage or batch work use napi-rs `AsyncTask`: they
return a **`Promise`** and run on Node's **libuv** worker pool, keeping the event
loop available for application code. This includes `rememberMany`,
`exportPage`, `maintain`, `checkpoint`, `reindex` and `verify`.

For a bounded export, call `exportPage()` once, process its `facts`, then pass
`nextCursor` to the next call until it is absent:

```ts
let cursor: number | undefined;
do {
  const page = await db.exportPage(cursor);
  for (const fact of page.facts) {
    await destination.write(fact);
  }
  cursor = page.nextCursor;
} while (cursor !== undefined);
```

The Promise is the completion boundary for that page: no callback remains
queued, and no database lock is held while the loop body runs. Each page has at
most 128 facts. A read-only handle pages one immutable checkpoint; when paging a
writer for a snapshot-style backup, do not mutate it between calls.

`rememberMany(items)` performs one batch embedding pass and one journal sync,
then resolves with outcomes in input order. A maintenance call may also return
a no-op report when there is nothing to purge, reindex or optimize.

`maintain(mode?)` takes `"auto"` (the default), `"compact"`, `"reindex-text"`,
`"optimize-vectors"` or `"full"`. No mode ever drops a fact revision or an edge
version; the heavier ones buy bytes and index freshness. `"full"` is the only
one that repacks the edge arenas, which a relink-heavy workload fragments.

### What is async, and why

`remember`, `revise`, `recall`, `rememberMany`, `exportPage`, `maintain` and
`checkpoint` return promises. Everything else — `get`, `stats`, `tagsOf`,
`forget`, `link`, `unlink`, `verify`, `export`, `path`, `generation`,
`refresh` — is synchronous.

The line is drawn at blocking work. Node runs all JavaScript on **one** thread,
so a native call that waits on an embedder's HTTP round trip or on an fsync
freezes every timer, socket and callback in the process for as long as it takes.
The verbs above can do exactly that — with an `[embedder]` configured,
`remember` and a text `recall` each cost a request to the provider — so they run
on a libuv worker and hand JavaScript a promise. The rest touch only mapped
memory and return in microseconds, where a promise would be pure ceremony.

Arguments are still checked on your thread: a refused one **throws** at the call
site rather than rejecting later, so a mistake in your code and a failure in the
engine never arrive the same way.

### Bringing your own embedding

`remember`, `revise`, `rememberMany` and `recall` all take an optional
`vector` — a precomputed embedding whose length must equal the configured `dim`:

```typescript
const own = await myEmbedder(text);          // your model, your pipeline
await db.remember({ text, vector: own });    // nothing is sent to `[embedder]`
const res = await db.recall({ query: text, vector: own });
```

Given one, it **replaces** the embedder for that call — the engine embeds only
when the field is absent. Use it for vectors you already have, for a model that
is not an OpenAI-shaped HTTP endpoint, or for a deterministic test with no
network. The CLI (`--vector`) and the MCP tools (`vector`) take the same thing.

`close()` releases the file and its lock; every verb afterwards throws, and it is
idempotent (the handle is also released on garbage collection, but `close()`
makes the moment explicit — e.g. before reopening the same file read-only).

## License

MIT.
