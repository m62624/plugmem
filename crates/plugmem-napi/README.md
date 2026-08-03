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

const out = db.remember({
  text: "prefers tokio",
  entity: "user",
  tags: ["pref"],
  links: [{ rel: "works_at", entity: "acme" }],
  metadata: { source: "chat", uri: "s3://bucket/note.txt" }, // opaque; a pointer
});
out.id;                       // number
out.similar;                  // Similar[] — the engine surfaces conflicts, you decide

const res = db.recall({ query: "runtime?", k: 5 });
res.rendered;                 // the prompt-ready block
res.facts;                    // RecalledFact[] — { id, score, entity, recordedAt, … }

const card = db.get(out.id);
card.metadata;                // Record<string,string> — keys sorted, {} when none

db.revise(out.id, { text: "prefers async-std" });
db.link({ src: "user", rel: "works_at", dst: "acme" });
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
must agree. A `{ readOnly: true }` handle never auto-embeds — pass a vector.

## The verbs

Every method here is the identically-named `plugmem-host` `Database` verb; the
engine logic is entirely the host's.

**Writer** (default): `remember`, `recall`, `revise(id, args)`, `forget(id)`,
`link`, `unlink`, `get(id)`, `stats`, `export`, `verify`, and the two **async**
verbs below. **Read-only** (`{ readOnly: true }`, observing another process's writer):
`recall`, `get`, `stats`, `export`, `verify`, plus `generation()` (the pinned
snapshot generation) and `refresh()` (advance to the writer's latest checkpoint);
the write verbs throw.

### What the host has and this does not

The list above is the whole surface — it is not yet all of `Database`, and the
gaps are named rather than left to be discovered:

| host verb | why it is not here |
|---|---|
| `recover` | salvaging a damaged file is a path-level operation on the disk the process is running on — [`plugmem-cli recover`](https://docs.rs/plugmem-cli/latest)'s job, like `import` and `scrub`. |
| `remember_many` | **not exposed yet.** It writes a batch with one embedding round-trip; without it, loading N facts from Node is N requests to the embedder rather than N/batch. Loop over `remember` until it lands. |
| `export_each` | **not exposed yet.** `export()` collects every fact into one array, so a very large memory exports with a RAM spike instead of a stream. |
| `tags_of` | **not exposed yet.** One fact's tags; `export()` carries them for every fact meanwhile. |

**No `import` verb** either — bulk-loading a `backup.jsonl` reads a file on disk,
which is the CLI's job. An agent remembers facts one at a time as the
conversation goes.

## Many memories in one directory (optional)

**Default: one memory, one file.** `new Plugmem(path)` and nothing here applies.

A process that serves many independent memories — one per conversation, per
tenant, per project — can point at a directory and address them by name:

```ts
import { Workspace, type DbEntry } from "plugmem";

const ws = new Workspace("/srv/memories", { maxOpen: 16, idleTimeoutMs: 60_000 });

// `open` hands back the same `Plugmem` class, so a named memory has exactly the
// verbs a path-opened one has. A first write to an unused name creates it.
ws.open("chat-42").remember({ text: "prefers tokio" });

// Do not know the name? Ask what each memory is for. Owners are searchable too,
// even though an owner is a graph edge rather than text.
ws.describe("chat-42", { description: "release planning", owner: "ann" });
const hits: DbEntry[] = ws.find("release planning");   // → [{ db: "chat-42", … }]
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

`reindex()` and `verify()` return promises: they open and read every memory in
the directory, which is not work for the main thread.

## Async and concurrency (no tokio)

`maintain()` and `checkpoint()` can do real disk I/O (compaction, HNSW work,
fsync), so they return a **`Promise`** and run on the **libuv** thread pool —
they never block the event loop. A maintenance call may also return a no-op
report when there is nothing to purge, reindex or optimize.

`maintain(mode?)` takes `"auto"` (the default), `"compact"`, `"reindex-text"`,
`"optimize-vectors"` or `"full"`. No mode ever drops a fact revision or an edge
version; the heavier ones buy bytes and index freshness. `"full"` is the only
one that repacks the edge arenas, which a relink-heavy workload fragments.

Every other verb is
microsecond-fast in memory and stays synchronous (a Promise there would be pure
overhead). There is no async runtime: the engine is CPU-bound, and the one thing
that can wait — a remote embedder's HTTP call — happens outside the engine lock.

`close()` releases the file and its lock; every verb afterwards throws, and it is
idempotent (the handle is also released on garbage collection, but `close()`
makes the moment explicit — e.g. before reopening the same file read-only).

## License

MIT.
