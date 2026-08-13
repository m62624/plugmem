# plugmem-cli

> ⚠️ Experimental. plugmem is mostly an AI-built experiment, written with
> the help of a small local model (Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) and various
> Claude models, in roughly equal measure. Expect non-professional design
> choices, rough edges, broken behavior, or mistakes. Use it at your own risk.

`plugmem-cli` is the command-line surface over the plugmem
[temporal-memory engine](https://docs.rs/plugmem-core/latest). It is a thin shell around
[`plugmem-host`](https://docs.rs/plugmem-host/latest) that lets you (or an agent's
launcher) keep a logical memory in a local database from a terminal or a shell script. Each
one-shot command parses arguments, calls one engine verb, and prints the result
(a human report, `--json` for tooling); the interactive `plugmem repl` keeps the
engine open across commands for host speed. No memory logic lives here.

The installed binary is **`plugmem-cli`**.

**No embedding model is required.** Of the four recall sources, only the vector
one needs an embedder; text, graph and time work with nothing but the database.
Configure `[embedder]` to add matching by meaning, or leave it out and match on
words, entities and time.

## Install

Prebuilt for **Linux, Windows and macOS (x64 & arm64)** on every tagged release.
Choose one method. Each installs the same `plugmem-cli` binary.

### Homebrew (macOS / Linux)

From the [`m62624/homebrew-plugmem`](https://github.com/m62624/homebrew-plugmem)
tap; `brew upgrade` / `brew uninstall` then manage it like any formula:

```console
$ brew install m62624/plugmem/plugmem-cli
```

### Installer script (no Rust toolchain)

`latest` always points at the newest tag on the
[Releases page](https://github.com/m62624/plugmem/releases):

```console
# Linux / macOS  (POSIX sh)
$ curl --proto '=https' --tlsv1.2 -LsSf https://github.com/m62624/plugmem/releases/latest/download/plugmem-cli-installer.sh | sh
```

```powershell
# Windows (PowerShell) — alternative to the .msi
> powershell -ExecutionPolicy Bypass -c "irm https://github.com/m62624/plugmem/releases/latest/download/plugmem-cli-installer.ps1 | iex"
```

### Windows `.msi`

Download `plugmem-cli-*.msi` from the
[Releases page](https://github.com/m62624/plugmem/releases). Double-click to
install; it registers in **"Add or remove programs"** for normal upgrades and
uninstalls.

### `cargo binstall`

[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) downloads the
prebuilt binary instead of compiling and supports every OS/architecture listed
above:

```console
$ cargo binstall plugmem-cli
```

### From source

Needs a Rust toolchain. From crates.io:

```console
$ cargo install plugmem-cli
```

…or from a local checkout of this repo:

```console
$ cargo install --path crates/plugmem-cli
# or, to build without installing:
$ cargo build --release -p plugmem-cli    # binary at target/release/plugmem-cli
```

### Uninstall

`cargo uninstall plugmem-cli` (for `cargo install`/`binstall`);
`brew uninstall plugmem-cli` (Homebrew); "Add or remove programs" (`.msi`). The
shell/PowerShell installers ship no uninstaller — remove `~/.cargo/bin/plugmem-cli`
and `~/.config/plugmem-cli` (Windows: `%USERPROFILE%\.cargo\bin\plugmem-cli.exe`
and `%LOCALAPPDATA%\plugmem-cli`) by hand. See the
[workspace README](https://github.com/m62624/plugmem#install) for the full matrix.

## Which crate do I need?

The CLI is the **human/scripting door**. A Rust program links the library
instead; an agent or another language comes in over a protocol.

| You want | Use | Why |
|---|---|---|
| A memory from a **terminal or shell script** | **`plugmem-cli`** (this binary) | One local database, no server; `plugmem repl` keeps the engine open for host speed. |
| **A memory in a Rust program** — the common case | [`plugmem-host`](https://docs.rs/plugmem-host/latest) (`std`) | The engine plus files, locking, mmap, HTTP embedders, integrity, concurrency. |
| The engine with **no `std`** or **your own storage** | [`plugmem-core`](https://docs.rs/plugmem-core/latest) (`no_std`) | Engine only; you bring persistence. |
| A memory for an **agent, local-first app, or non-Rust program** | [`plugmem-mcp`](https://docs.rs/plugmem-mcp/latest) | Long-lived stdio JSON-RPC; language-independent — the door for programmatic / cross-language access (the CLI is the human one). |
| A memory in **JavaScript / TypeScript** (Node) | [`plugmem-napi`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-napi) | The engine as a native Node addon (napi-rs), in-process; on npm as `plugmem`. |
| A memory in **Python** | [`plugmem-py`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-py) | The engine as a CPython extension (PyO3), in-process; on PyPI as `plugmem`. |

## What recall does

Recall fuses four sources by reciprocal-rank fusion with a recency boost (tags
filter; they are not a source):

| Source | Algorithm | What it finds |
|---|---|---|
| **Lexical** | [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (Robertson idf) over a Unicode ([UAX #29](https://unicode.org/reports/tr29/)) tokenizer | exact terms / keyword overlap |
| **Semantic** | int8-quantized cosine — flat two-phase below a threshold, an [HNSW](https://arxiv.org/abs/1603.09320) graph above | meaning / nearest neighbours |
| **Graph** | entity graph with current typed edges on the hot path; `--as-of` walks edge history | relational knowledge |
| **Temporal** | range scans over a `recorded_at`-ordered index; bitemporal validity | "what was true *then*", time windows |

## Usage

```text
plugmem-cli [--db PATH] [--json] <command>
```

The database is chosen by, in order: `--db PATH`, the `PLUGMEM_DB`
environment variable, or the platform data path (created on first write). The
engine keeps no clock, so `now` comes from the system clock at each call.

| command | what it does |
|---|---|
| `remember <TEXT> [--guarded] [--entity E] [--tag T]… [--link REL:ENTITY]… [--meta KEY=VALUE]… [--valid-from TS] [--vector F32,…]` | store a fact; prints its id and any similar/conflicting facts. `--guarded` checks similarity and writes only if clear, without a race between those steps. The similarity check is scoped to `--entity`, so `--guarded` with no `--entity` has nothing to compare against and always stores; the run then prints that it stored without a check, and the JSON carries `"checked": false`. `--meta` is repeatable (opaque key→value, e.g. a URI; last value wins per key) |
| `recall [QUERY] [--tag T]… [--entity E]… [--as-of TS] [--range FROM TO] [-k N] [--closed] [--token-budget N] [--ef N] [--graph-depth N] [--vector F32,…]` | ranked, token-budgeted block; sources compose. Each line is `- [fN] text …` — `N` is the fact's id (see below). `--token-budget` caps the block (default 512), `--ef` widens the vector search beam, `--graph-depth` sets how far the graph walks from an anchor (default 2; `0` disables expansion) |
| `revise <ID> <TEXT> [same flags as remember]` | close the old fact, record the successor |
| `forget <ID>…` | tombstone one or more facts, batched under one write (purged physically at the next `maintain`) |
| `tags [--prefix P] [--cursor C] [--limit N]` | list one stable lexical page of current tags and counts; pass the returned cursor to continue |
| `remove-tag <TAG>` | remove a tag from every current fact while preserving facts and historical tag state |
| `link <SRC> <REL> <DST> [--provenance FACT_ID]` | upsert a typed edge between entities. `--provenance` records the fact the edge follows from, and graph recall returns it |
| `unlink <SRC> <REL> <DST>` | close the current typed edge while preserving `--as-of` history |
| `show <ID>` | one fact's full card — text, both time axes, state |
| `stats` | engine size counters |
| `maintain [--mode M]` | policy-driven maintenance: cheap no-op, tombstone compaction, text reindex or bounded HNSW work. `M` is `auto` (default), `compact`, `reindex-text`, `optimize-vectors` or `full`; only `full` repacks the edge arenas, and no mode drops history |
| `maintain --reembed [--batch-size N]` | explicitly recompute every retained fact with the configured embedder, rebuild HNSW and atomically publish the new vector space; never implied by `auto` |
| `checkpoint` | flush the journal into a fresh snapshot and clear it (leaves the database checkpointed) |
| `verify` | check integrity an open defers: text UTF-8, metadata, vector↔fact consistency, and that the edge graph agrees with itself; exit 2 on damage |
| `scrub` | check the snapshot's byte-level container checksums; exit 2 on the first damaged section |
| `recover <DST>` | salvage a content-corrupt database into a clean `DST`; the source (`--db`) is left untouched |
| `export` | dump the memory as JSONL to stdout: every open fact, then every open edge, one per line, each tagged with `kind`. Streamed, so a large memory never has to fit in RAM |
| `import <FILE> [--batch N]` | load a file written by `export`, streamed in batches — one embed round-trip and one fsync per batch. Edges are re-linked and their provenance retargeted to the ids this database assigns |

`--vector` takes a comma-separated embedding (`--vector 0.1,-0.2,…`, or
`--vector "$(cat vec.txt)"` for a real one). Given, it **replaces** the
configured embedder for that call — nothing is sent to the provider — and its
length must equal `[engine].dim`. Omit it and the engine embeds the text itself,
which is what you want unless the vector already exists or your model is not an
OpenAI-shaped HTTP endpoint.

**Fact ids.** A fact's id is how you address it in `forget`, `revise`, and
`show`. `remember` prints it on store (`remembered fact 3`), and `recall`
carries it on every line: the human block renders each fact as `- [fN] text …`
where `N` is the id (so `[f3]` → `forget 3`), and `--json` exposes the same
value as a plain `"id"` field on each fact. You don't guess an id — you read it
back from a `recall` (or `show`), then act on it, the usual "find, then change"
flow.

Read-only commands (`recall`, `show`, `stats`, `tags`, `export`) open the snapshot
**zero-copy over an mmap** (a shared lock, so several may run at once and
the whole file is not loaded) — falling back to a normal open if the
journal is un-checkpointed. `recall` uses that fast path only when no
embedder is configured, because embedding the query needs the read-write
handle. `scrub` also takes the shared mmap open, so it needs a checkpointed
database — run `checkpoint` (or `maintain`) first if the journal is dirty.

### Examples

```sh
# Remember, with a subject entity, a tag, and opaque metadata (a URI to the
# real payload in another store — the engine never interprets it):
plugmem-cli remember "prefers tokio with pinned versions" --entity user --tag pref \
    --meta source=chat --meta uri=s3://bucket/note.txt

# Recall — a ranked block bounded for a prompt or another context consumer:
plugmem-cli recall "which runtime"

# Bitemporal: correct a fact, then ask what was true earlier.
plugmem-cli remember "lives in Moscow" --entity user      # → fact 0
plugmem-cli revise 0 "lives in Berlin" --entity user
plugmem-cli recall "lives" --entity user --as-of 1700000000000   # → Moscow

# Machine-readable output for tooling / agents:
plugmem-cli --json recall "runtime" --tag pref
plugmem-cli --json stats

# Reclaim space held by forgotten facts. `recall` shows the id as `[fN]`;
# use that N to forget, then compact:
plugmem-cli recall "old runtime note"   # → - [f3] prefers deno … 
plugmem-cli forget 3                     # the 3 from [f3]
plugmem-cli maintain

# Move a memory to another file. Both halves stream, so size does not matter:
plugmem-cli export > dump.jsonl
plugmem-cli --db other.plugmem import dump.jsonl --batch 256
# → "imported 128 facts and 14 edges"
#
# What crosses: text, entity, tags, metadata, valid_from, edges and each
# edge's provenance. What does not: closed revisions and vectors. This is a
# portable knowledge dump, not a backup — for a backup, copy the files.

# Integrity & recovery (exit 2 on corruption — scriptable as a gate):
plugmem-cli verify                       # content consistency
plugmem-cli checkpoint && plugmem-cli scrub  # flush journal, then byte-level checksums
plugmem-cli recover agent.recovered.plugmem   # salvage into a clean copy
```

## Many memories in one directory (optional)

**Default: one logical memory backed by a local database layout.** `--db path/to/memory.plugmem` and nothing
here applies. Reach for this only when you keep several independent
memories — one per project, per client, per agent — and want to address them by
name instead of remembering where each file is.

Point the CLI at a directory and `--db` starts taking a name:

```console
$ export PLUGMEM_WORKSPACE=~/memories        # or --workspace, or [workspace].dir
$ plugmem-cli --db work remember "the release branch is integration/0.3.0"
$ plugmem-cli --db personal recall "release branch"   # sees nothing: separate memory
```

The rule is one line: a name has no separator and no dot, so `work` is a name
while `work.plugmem`, `./work` and `/srv/work` stay paths. A name resolves to
`<dir>/db/<name>.plugmem` and nowhere else — it is not a path and cannot become
one. A first write to an unused name creates that memory; there is no
registration step.

Memories are **independent**: no search spans them, and there is no way to link
an entity across them. That is the point (one memory answering for another is
what makes a shared store useless), and it is also the thing to decide up front
— a fact filed in the wrong memory is not merely misplaced, it is unreachable
from the other.

### The `workspace` command group

Administrative, and none of it is needed for everyday use:

| command | what it does |
|---|---|
| `workspace list` | every memory on disk, with its description when it has one |
| `workspace find <QUERY> [-k N]` | which memory is the one about… — searches descriptions, and a person's name finds what they own |
| `workspace describe <NAME> <TEXT> [--tag T]… [--owner WHO]` | say what a memory is for; creates it if absent |
| `workspace archive <NAME>` | label it archived (it stays where it is and stays openable) |
| `workspace reindex` | rebuild the registry from the memories' own descriptions |
| `workspace verify` | report where the registry and the directory disagree; exit `1` if they do |
| `workspace use <NAME>` | print the shell line that points **this terminal** at a memory |

The description lives in two places: inside the memory itself, and in a registry
(`<dir>/registry.plugmem`, an ordinary plugmem database). The directory is the
truth and the registry is only a searchable index over it — delete it and
`workspace reindex` rebuilds it from the memories; what you lose is search, not
data.

`workspace use` writes **nothing to disk**. It prints a line for the shell:

```console
$ eval "$(plugmem-cli workspace use work)"     # sh / bash / zsh
$ plugmem-cli workspace use work | Invoke-Expression   # PowerShell
```

The selection then lives in that terminal, so a second window is unaffected — a
state file shared by every window would let one of them silently redirect a
script running in another.

## Interactive mode (`repl`)

`plugmem repl` opens the database **once** and runs commands from stdin (one
per line, the same subcommand grammar), keeping the engine resident so each
command is host speed instead of a per-command reload. `help` lists the verbs,
`exit`/`quit` (or EOF) leaves, and the session checkpoints on exit. This is the
read-write session: it holds the writer lock and sees its own writes instantly
(read-your-writes), so there is never anything to "refresh".

```text
$ plugmem-cli repl
plugmem> remember "prefers tokio"
plugmem> recall runtime
plugmem> exit
```

`plugmem repl --read-only` is a **separate, observe-only** session for watching
a database that **another process** is writing. It opens a shared, zero-copy
mmap over the last published snapshot (it does not take the writer lock and does
not write), so only the read verbs run — `recall`, `show`, `stats`, `export`,
`verify`. It adds two cross-process freshness verbs:

| verb | what it does |
|---|---|
| `generation` | print the snapshot generation this session is pinned to (a number that a writer's checkpoint bumps) |
| `refresh` | advance to the writer's latest published checkpoint, if any — prints `refreshed → generation N` or `already current → generation N` |

**These two verbs exist only in `--read-only`.** A normal (writer) `repl` and
every one-shot command already see the newest data — read-your-writes, or a
fresh open per command — so refreshing there is meaningless and is not offered.
`--read-only` needs a published snapshot to map, so a database that has never
been checkpointed is refused: run `checkpoint` in the writing process first.
After that the writer may keep writing — a read-only session simply answers as
of the checkpoint it pinned, until you `refresh`.

```text
$ plugmem-cli repl --read-only        # in a second terminal, while a writer runs
plugmem(ro)> generation
generation 7
plugmem(ro)> recall runtime           # answers as of generation 7
plugmem(ro)> refresh                   # the writer has checkpointed since
refreshed → generation 9
plugmem(ro)> recall runtime           # now answers as of generation 9
plugmem(ro)> exit
```

## Configuration

Optional `config.toml`, found by `--config PATH`, then `$PLUGMEM_CONFIG`, then
the platform config directory (all optional — the CLI works with none).
Precedence overall is **explicit path/flag > environment > config file >
platform default**. See the [full settings reference](https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md)
for all fields and OS-specific paths.

```toml
[database]
path = "/path/to/memory.plugmem" # optional example; --db and PLUGMEM_DB win

[engine]
dim = 768              # embedding size (0 = vectors off); also max_bytes,
                       # max_text, max_blob. What the database is *built* with:
                       # changing one on an existing file is refused.

[recall]               # optional — every key has a tuned default
w_vec = 2.0            # weight of the vector source (0 turns it off)
half_life_days = 30    # age at which the recency discount has halved
                       # also: w_bm25, w_graph, w_time, w_recency, rrf_k,
                       # bm25_k1, bm25_b, graph_depth, graph_decay,
                       # hnsw_ef_search, similar_cos, similar_jaccard

[index]                # optional
flat_to_hnsw = 50000   # vectors before maintenance builds the HNSW graph
                       # also: hnsw_ef_construction

[embedder]             # optional — omit for lexical/tags/graph/time only
enabled = true         # false keeps settings but makes no embedder calls
url = "http://localhost:11434/v1/embeddings"
model = "nomic-embed-text"
space_id = "nomic-embed-text@v1" # optional; defaults to model
api_key_env = "OPENAI_API_KEY"   # env var holding the bearer token
on_error = "fail"      # or "degrade": answer without the vector instead
timeout_ms = 10000     # one request, end to end; 0 waits indefinitely
retry_after_ms = 0     # a suspended embedder: 0 = never on its own,
                       # unset = 1s doubling to retry_max_ms (60s)

[maintenance]
snapshot_every_ops = 1024
snapshot_journal_bytes = 4194304
maintain_every_forgets = 100     # optional auto-purge
batch_size = 128                 # facts per `import` batch (--batch overrides)
```

`import` streams the file in batches of `batch_size` (default 128; `--batch N`
overrides it): each batch is a single embedder round-trip and a single journal
fsync, so a bulk load with an embedder makes one HTTP call per batch instead of
one per fact, and the file is never fully read into memory. Larger batches mean
fewer round-trips but a bigger request body and more memory per batch.

The embedder is what unlocks the **vector** recall source: without an active
`[embedder]`, `remember`/`recall` still answer from lexical, tag, graph and
temporal evidence, but no embeddings are computed. The host creates its one
`OpenAiCompatEmbedder` implementation for any OpenAI-compatible server —
OpenAI, Ollama, LM Studio, vLLM or llama.cpp-server. Set `enabled = false` to
keep the URL/model settings without creating the client; the environment
variable `$PLUGMEM_EMBEDDER_ENABLED` overrides the config value with `true` or
`false`. `api_key_env` names the environment variable that contains the
bearer token.

When the provider cannot be reached — a stopped Ollama, a laptop that went
offline, a model that was unloaded — `on_error` decides what that costs.
`fail`, the default, propagates the error, which is what every release before
0.12 did. `degrade` carries on **without** the vector: the fact is stored, the
query is answered from the lexical, tag, graph and time sources, and the
embedder is suspended so the next call does not pay the same failure again. A
fact stored that way is not damaged — it is in the state every fact is in when
a memory is written with no embedder, and `reembed` fills the vectors in from
the stored text once the provider answers again. A suspension lifts by itself
after `retry_after_ms` (unset: one second, doubling to `retry_max_ms`, reset by
the first success). A vector-space mismatch is never degraded, and a `reembed`
refuses while the embedder is suspended rather than publishing half a vector
axis. Each of these keys has an environment override under the usual
precedence: `$PLUGMEM_EMBEDDER_ON_ERROR`, `$PLUGMEM_EMBEDDER_TIMEOUT_MS`,
`$PLUGMEM_EMBEDDER_RETRY_AFTER_MS`, `$PLUGMEM_EMBEDDER_RETRY_MAX_MS`.

`[recall]` and `[index]` are safe to change on an existing memory: reopening
with different weights is how you change the ranking, and the next `checkpoint`
records them in the file. Reach for them when a specific memory answers badly —
the defaults are tuned, and `w_bm25 = 0` (say) is mostly useful for asking what
one source alone thinks.

A key nothing recognises is **reported, not ignored**, on stderr. Refusing it
would mean an older binary could not read a newer config, but staying silent
would leave you believing you had tuned something:

```console
$ plugmem-cli stats
plugmem: unknown config section [engin] — did you mean `engine`?
plugmem: unknown setting [recall].w_vector — did you mean `w_vec`?
facts       0
...
```

Run `plugmem-cli help settings` for the complete catalogue with every default.

## Exit codes

Scriptable as a gate:

| code | meaning |
|---|---|
| `0` | success |
| `1` | a soft miss — the target fact does not exist (`show`), or the database is locked by another process |
| `2` | a usage error (bad arguments) or a runtime error (I/O, a corrupt image) |

## Lifecycle — open per command

Each invocation is a **short-lived process**: it opens the database file,
runs one command, and exits — the process *is* the session boundary, so
there is no explicit open/close and nothing to keep open between calls.
This is the usual model for command-line file-backed tools:
run → one operation → done. Two invocations that happen to overlap in that
brief window collide on the lock (the second gets exit `1`); back to back,
they never do.

Opening reads the snapshot into memory and replays the journal, so on a
large database each command pays that load. For a memory of tens of
thousands of facts it is milliseconds; if you need many operations against
a big memory without re-loading each time, that is the **long-lived
handle** case — embed [`plugmem-host`](https://docs.rs/plugmem-host/latest)'s
`Database` in your process (open once, call many verbs, all in RAM), or
run the MCP server, which keeps the memory resident. The CLI deliberately
does not: it trades a per-command load for a stateless, scriptable tool.

How much a memory weighs — the byte cost of a fact, an edge or a vector,
and where each structure tops out — is tabulated in
[`plugmem-core`](https://docs.rs/plugmem-core/latest)'s *Capacity — what weighs
what*; it applies verbatim to a file the CLI opens.

## Concurrency

One database is a single-writer resource: `plugmem-cli` takes an
exclusive lock for the length of the (short-lived) command, so a second
`plugmem-cli` — or an MCP server holding the same file — is refused with
exit `1` rather than corrupting it. See the
[host concurrency model](https://docs.rs/plugmem-host/latest).

## License

MIT.
