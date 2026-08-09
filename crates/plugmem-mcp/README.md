# plugmem-mcp

> ⚠️ Experimental. plugmem is mostly an AI-built experiment — written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

`plugmem-mcp` is the [Model Context Protocol](https://modelcontextprotocol.io)
server over the plugmem [temporal-memory engine](https://docs.rs/plugmem-core/latest)
— a thin, long-lived shell around
[`plugmem-host`](https://docs.rs/plugmem-host/latest) that exposes a memory to
**AI agents, local-first applications and any non-Rust program** as MCP tools
over stdio JSON-RPC. The
engine stays resident for the process's lifetime, so every call is host speed.

The installed binary is **`plugmem-mcp`**.

**No embedding model is required.** Of the four recall sources, only the vector
one needs an embedder; text, graph and time work with nothing but the database.
Configure `[embedder]` to add matching by meaning, or leave it out and match on
words, entities and time.

## Install

Prebuilt for **Linux, Windows and macOS (x64 & arm64)** on every tagged release.
**Pick one method — you don't need more than one; they install the same
`plugmem-mcp` binary.**

### Homebrew (macOS / Linux)

From the [`m62624/homebrew-plugmem`](https://github.com/m62624/homebrew-plugmem)
tap; `brew upgrade` / `brew uninstall` then manage it like any formula:

```console
$ brew install m62624/plugmem/plugmem-mcp
```

### Installer script (no Rust toolchain)

`latest` always points at the newest tag on the
[Releases page](https://github.com/m62624/plugmem/releases):

```console
# Linux / macOS  (POSIX sh)
$ curl --proto '=https' --tlsv1.2 -LsSf https://github.com/m62624/plugmem/releases/latest/download/plugmem-mcp-installer.sh | sh
```

```powershell
# Windows (PowerShell) — alternative to the .msi
> powershell -ExecutionPolicy Bypass -c "irm https://github.com/m62624/plugmem/releases/latest/download/plugmem-mcp-installer.ps1 | iex"
```

### Windows `.msi`

Download `plugmem-mcp-*.msi` from the
[Releases page](https://github.com/m62624/plugmem/releases). Double-click to
install; it registers in **"Add or remove programs"** for normal upgrades and
uninstalls.

### `cargo binstall`

[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) downloads the
prebuilt binary instead of compiling — it just works on every OS/arch above:

```console
$ cargo binstall plugmem-mcp
```

### From source

Needs a Rust toolchain. From crates.io:

```console
$ cargo install plugmem-mcp
```

…or from a local checkout of this repo:

```console
$ cargo install --path crates/plugmem-mcp
# or, to build without installing:
$ cargo build --release -p plugmem-mcp    # binary at target/release/plugmem-mcp
```

### Uninstall

`cargo uninstall plugmem-mcp` (for `cargo install`/`binstall`);
`brew uninstall plugmem-mcp` (Homebrew); "Add or remove programs" (`.msi`). The
shell/PowerShell installers ship no uninstaller — remove `~/.cargo/bin/plugmem-mcp`
and `~/.config/plugmem-mcp` (Windows: `%USERPROFILE%\.cargo\bin\plugmem-mcp.exe`
and `%LOCALAPPDATA%\plugmem-mcp`) by hand. See the
[workspace README](https://github.com/m62624/plugmem#install) for the full matrix.

## Which door is this? (read before reaching for MCP)

plugmem is **embedded-first** — the fastest, simplest path is to link the
engine into your process, not to talk to a server.

| You are… | Use | Why |
|---|---|---|
| **an agent, or a program in another language** (Python, Node, Go…) that wants a memory | **`plugmem-mcp`** (this binary) | Spawn the process, speak JSON-RPC on its stdin/stdout. Language-independent; the memory stays resident. |
| **writing Rust** | [`plugmem-host`](https://docs.rs/plugmem-host/latest) — embed it as a dependency | The engine runs *in your process*. Maximum speed, no pipe, no second process. **Don't** front your own Rust with MCP. |
| a person at a **terminal or shell script** | [`plugmem-cli`](https://docs.rs/plugmem-cli/latest) | The human/scripting door. **Not** the door for programmatic or cross-language access — that's MCP. |
| **JavaScript / TypeScript** (Node) | [`plugmem-napi`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-napi) | The engine as a native Node addon (napi-rs), in-process; on npm as `plugmem`. |
| **Python** | [`plugmem-py`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-py) | The engine as a CPython extension (PyO3), in-process; on PyPI as `plugmem`. |

So: **another language → MCP; Rust → embed the host lib; a human → the CLI.**
The MCP server's main consumer is the agent itself. And whichever door you use,
you are tending *your own* memory file — plugmem keeps no server of its own.

## What is (and isn't) MCP here

- A **sidecar process, not a daemon.** The host (Claude Desktop, an IDE, an
  agent runner) *spawns* `plugmem-mcp` and talks to it over stdin/stdout. It
  listens on no port and serves **one local database** for its lifetime. When the
  host goes away, so does the sidecar.
- **Many readers / many languages = many processes**, coordinated by plugmem's
  file-level MVCC (immutable snapshot generations + an advisory writer lock),
  *not* one network server. There is deliberately no network mode: that would
  add ports, auth and a connection pool for no embedded-use benefit.
- **Concurrent, on plain threads.** Independent requests run in parallel on a
  small worker pool; the only thing a request ever waits on is the embedder's
  HTTP call, kept off the shared lock. No async runtime — the engine's work is
  in-memory and fast, so threads are the whole story (see [Concurrency](#concurrency)).

## What recall does

Recall fuses four sources by reciprocal-rank fusion with a recency boost (tags
filter; they are not a source):

| Source | Algorithm | What it finds |
|---|---|---|
| **Lexical** | [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) over a Unicode ([UAX #29](https://unicode.org/reports/tr29/)) tokenizer | exact terms / keyword overlap |
| **Semantic** | int8-quantized cosine — flat below a threshold, an [HNSW](https://arxiv.org/abs/1603.09320) graph above | meaning / nearest neighbours |
| **Graph** | entity graph with current typed edges on the hot path; `as_of` walks edge history | relational knowledge |
| **Temporal** | range scans over a `recorded_at`-ordered index; bitemporal validity | "what was true *then*", time windows |

### Two clocks

The temporal source exists because a fact carries two timestamps, not one:
`valid_from`/`valid_to` for when the statement was true, and `recorded_at` for
when the memory learned it. The server reads the system clock on every call, so
`recorded_at` is the moment of the write.

`plugmem_revise` closes the old fact's interval instead of deleting it, so the
earlier state stays answerable:

```jsonc
{"name": "plugmem_remember", "arguments": {"text": "lives in Moscow", "entity": "kim"}}
// → fact 0

{"name": "plugmem_revise", "arguments": {"id": 0, "text": "lives in Berlin", "entity": "kim"}}
// → fact 1

{"name": "plugmem_recall", "arguments": {"entities": ["kim"]}}
// → f1, Berlin, active

{"name": "plugmem_recall", "arguments": {"entities": ["kim"], "as_of": <between the two>}}
// → f0, Moscow, closed
```

`as_of` moves **both** clocks: a fact answers only if it was valid at that
instant *and* had already been recorded by then. That second half matters for an
agent replaying old context — an `as_of` earlier than a fact's `recorded_at`
sees nothing, because the memory genuinely knew nothing then, and reporting
today's knowledge would be the wrong answer to "what did I hold".

`valid_from` is the other half: something that became true before the agent
heard of it. Recording today that a move happened last week closes the previous
interval last week rather than today, so a query as of three days ago finds
*neither* — the old fact had stopped being true and the new one was not yet
known. That is the honest answer for that instant.

`plugmem_forget` is the destructive verb: for a fact that was wrong, not one
that changed.

## Tools

Every tool is named `plugmem_*` (so it never collides with another server's
tools) and takes an optional `format` argument: `"json"` (default) returns
compact machine JSON; `"human"` pretty-prints it (and, for `plugmem_recall`,
returns the engine's prompt-ready block instead of the structured result).
Result payloads ride in the MCP `content[].text` field; a tool-level failure
sets `isError: true` so the model can read and react to it.

**Writer mode** (default — a read-write memory of its own):

| tool | what it does |
|---|---|
| `plugmem_remember` | store a fact (`text`, optional `entity`, `tags[]`, `links[]` of `{rel, entity}`, `valid_from`); returns the id + similar/conflicting facts |
| `plugmem_recall` | ranked, token-budgeted recall (`query`, `tags[]`, `entities[]`, `as_of`, `range [from,to]`, `k`, `closed`, `token_budget`, `ef`) |
| `plugmem_revise` | close fact `id`, record the successor (same args as remember + `id`) |
| `plugmem_forget` | tombstone fact `id` (purged at the next maintain) |
| `plugmem_tags` | bounded lexical page of current tags and counts (`prefix`, opaque `cursor`, `limit` up to 256) |
| `plugmem_remove_tag` | remove one tag from every current fact while preserving facts and historical tag state |
| `plugmem_link` | upsert a typed edge `src -rel-> dst`, optionally with `provenance`: the fact id the edge follows from, which graph recall returns |
| `plugmem_unlink` | close the current typed edge `src -rel-> dst` while preserving `as_of` history |
| `plugmem_show` | one fact's full card by `id` |
| `plugmem_stats` | engine size counters |
| `plugmem_export` | the whole memory as `{ facts, edges }` — an edge belongs to no single fact, so facts alone would lose the graph |
| `plugmem_maintain` | policy maintenance, or explicit vector replacement. `mode`: `auto` (default), `compact`, `reindex-text`, `optimize-vectors`, `full`, or `reembed`; only `reembed` calls the configured model, in bounded `batch_size` requests |
| `plugmem_checkpoint` | flush the journal into a fresh snapshot |
| `plugmem_verify` | the integrity check an open defers: content plus graph consistency |
| `plugmem_version` / `plugmem_about` | the running version; a pointer to the plugmem skill |

**Read-only mode** (`--read-only` — observe another process's writer over a
shared snapshot): `plugmem_recall`, `plugmem_show`, `plugmem_stats`, `plugmem_tags`,
`plugmem_export`, `plugmem_verify`, plus `plugmem_generation` (the pinned
snapshot generation) and `plugmem_refresh` (advance to the writer's latest
published checkpoint). Write tools are refused with a tool-level error.

The **fact id** on each `plugmem_recall` line (the `[fN]` in the human block, or
the `"id"` field in JSON) is how you address a fact in `plugmem_revise`,
`plugmem_forget` and `plugmem_show` — the usual "recall, then act" flow.

**No `import` tool — that's the CLI's job.** `plugmem_export` returns the facts
inline (no file needed), but bulk-**loading** from a `backup.jsonl` means reading
a file *on the server's disk* — which a sandboxed or remote server can't see.
So restoring/migrating a memory from a file is done with
[`plugmem-cli import`](https://docs.rs/plugmem-cli/latest) (it has the disk, and
streams the file in batches), not over MCP. An agent doesn't bulk-load anyway —
it remembers facts one at a time with `plugmem_remember` as the conversation goes.

## Usage

The host spawns the binary and wires its arguments once, in its MCP config:

```text
plugmem-mcp [--db PATH] [--config PATH] [--read-only] [--workers N]
```

- `--db` — the memory file (else `$PLUGMEM_DB`, else the platform data path).
- `--config` — a `config.toml` (else `$PLUGMEM_CONFIG`, else the XDG default).
- `--read-only` — observe another process's writer (requires a checkpointed
  database).
- `--workers N` — worker threads (else `[server].workers`, else half the cores).

A Claude Desktop / MCP-client config entry looks like:

```json
{
  "mcpServers": {
    "plugmem": {
      "command": "plugmem-mcp",
      "args": ["--db", "/home/me/agent.plugmem"]
    }
  }
}
```

A failure to start (bad config, or the file already locked by another writer) is
reported to stderr with a non-zero exit, so the spawning host sees the server
did not come up.

## Many memories from one server (optional)

**Default: one memory, and the tools have no `db` argument.** That is the right
shape for one process per conversation, and it is what everything above
describes. This section is for the other case.

### Which shape to run

| your situation | run |
|---|---|
| one agent, one memory | `--db FILE` — the default, nothing to configure |
| one server process per conversation or per user, each with its own memory | `--db <that memory>`, workspace or not. The process boundary already answers "which memory", so the model is never asked |
| one server process for many conversations | `--workspace DIR`, and pass `db` on every call |
| mostly one memory, occasionally a shared one | `--workspace DIR --db <the usual one>` — `db` is then optional and defaults to it |

Prefer a process per memory when you can. It is the shape with no way to address
the wrong memory, and its extra cost is small: a chat-sized memory opens in
milliseconds and holds well under a megabyte resident.

Reach for one process for many when spawning per conversation is not practical —
hundreds of live conversations, or a host that keeps one long-lived connection.

### How it works

Started with `--workspace DIR`, the server holds a directory of named memories
and every tool that touches one gains a `db` argument:

| started with | `db` in the tool schema |
|---|---|
| `--db FILE` | absent — nothing changes |
| `--workspace DIR --db NAME` | present, optional, defaults to `NAME` |
| `--workspace DIR` | present, required |

The argument disappearing when it has no answer to give is the point. In MCP the
*model* fills tool arguments, so a `db` field is a decision the model makes on
every call — while the process that spawned the server usually knew the answer
for certain. Where the answer is known, the question is not asked: it cannot be
got wrong and it costs no tokens.

There is no verb to switch memories. With a worker pool that would be shared
mutable state, and worker A switching to X while B switches to Y is a race that
writes to the wrong person's memory.

Two extra tools appear: `plugmem_workspace_list` and `plugmem_workspace_find`
(search the memories' descriptions — the way a model picks a `db` when it does
not know the name).

A write to an unused name creates that memory, which is how a new conversation
gets one without a registration step; a *read* of an unknown name is refused,
because such a read is a typo far more often than a new memory. `--no-create`
turns the write case off too.

`--read-only` has no workspace form: a read-only handle pins one immutable
snapshot generation, and a pool of pinned snapshots that silently age is worse
than not offering it.

**Access control is not this server's job.** `db` arrives from the caller, and
the harness that spawned this process sees the call before the server does — put
the policy there. `--allow <name>` (repeatable) is a convenience for a
single-tenant process, not a boundary. The simplest deployment avoids the
question altogether: one server per conversation with `--db <file>`.

## Concurrency

One reader thread pulls stdin lines into a channel; a pool of worker threads
drains it, dispatches, and writes replies under a single stdout lock (so lines
never interleave). Each worker holds a cheap handle — a writer clones the
`Database` (an `Arc` around the engine's `RwLock`), a reader shares one snapshot
behind its own `RwLock` for concurrent reads — so independent requests overlap.
The embedder carries no lock at all: `Embedder::embed` takes `&self`, so the
whole pool can be waiting on the provider at once, and the HTTP call never
touches the engine lock. Replies carry their JSON-RPC `id`, so a client
correlates them regardless of completion order.

The pool defaults to `max(1, available_parallelism() / 2)` — half the machine's
cores, leaving room for the agent, the OS and a local embedder rather than
monopolizing the box. Override it with `[server].workers` or `--workers`.

## Configuration

Optional `config.toml`, found by `--config PATH`, then `$PLUGMEM_CONFIG`, then
the platform config directory. The engine, database, embedder and maintenance
sections are the **same** shared loader the CLI uses; MCP adds one `[server]`
section. See the [full settings reference](https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md) for
all fields and OS-specific paths.

```toml
[database]
path = "/path/to/memory.plugmem" # optional example; --db and PLUGMEM_DB win

[server]
workers = 4            # worker threads (default: half the cores)

[engine]
dim = 768              # embedding size (0 = vectors off); what the database is
                       # *built* with — changing one later is refused

[recall]               # optional — every key has a tuned default
w_vec = 2.0            # weight of the vector source (0 turns it off)
half_life_days = 30    # age at which the recency discount has halved
                       # also: w_bm25, w_graph, w_time, w_recency, rrf_k,
                       # bm25_k1, bm25_b, graph_depth, graph_decay,
                       # hnsw_ef_search, similar_cos, similar_jaccard

[index]                # optional
flat_to_hnsw = 50000   # vectors before maintenance builds the HNSW graph

[embedder]             # optional — omit for lexical/tags/graph/time only
enabled = true         # false keeps settings but makes no embedder calls
url = "http://localhost:11434/v1/embeddings"
model = "nomic-embed-text"
space_id = "nomic-embed-text@v1" # optional; defaults to model
api_key_env = "OPENAI_API_KEY" # env var holding the bearer token

[maintenance]
snapshot_every_ops = 1024
maintain_every_forgets = 100
```

`plugmem_remember`, `plugmem_revise` and `plugmem_recall` also take an optional
`vector`: a precomputed embedding (an array of numbers whose length equals
`dim`) that **replaces** the configured embedder for that call — nothing is sent
to the provider. Arguments that narrow an answer are validated rather than
guessed: `range` must be exactly `[from, to]`, and `as_of` / `valid_from` must
each be a whole non-negative unix-millisecond number. A malformed one is a tool
error, not an answer quietly computed without it.

The embedder unlocks the **vector** recall source; without an active
`[embedder]`, recall still answers from lexical, tag, graph and temporal
evidence. The host uses one `OpenAiCompatEmbedder` implementation for
OpenAI-compatible OpenAI, Ollama, LM Studio, vLLM and llama.cpp-server
endpoints. Set `enabled = false` to keep the settings but make no embedder
calls. `$PLUGMEM_EMBEDDER_ENABLED` overrides `[embedder].enabled` with `true`
or `false`, and `api_key_env` names the environment variable containing the
bearer token.

`[recall]` and `[index]` are safe to change on an existing memory — reopening
with different weights is how the ranking changes — while `[engine]` is not.
A key nothing recognises is reported on **stderr** (stdout carries the JSON-RPC
framing, so nothing else can go there) rather than silently ignored:

```text
plugmem-mcp: unknown setting [recall].w_vector — did you mean `w_vec`?
```

The model can read the whole catalogue at runtime with `plugmem_settings_help`.

## License

MIT.
