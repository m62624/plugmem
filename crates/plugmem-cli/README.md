# plugmem-cli

`plugmem-cli` is the command-line surface over the plugmem
[temporal-memory engine](https://docs.rs/plugmem-core/latest) — a thin shell around
[`plugmem-host`](https://docs.rs/plugmem-host/latest) that lets an agent (or you) keep a memory
in a single file from a terminal or a shell script. It parses arguments,
calls one engine verb, and prints the result: a human report by default,
`--json` for tooling. No memory logic lives here; retrieval (BM25,
int8-quantized vectors with [HNSW](https://arxiv.org/abs/1603.09320), an
entity graph, temporal range scans, fused by rank) is the engine's.

The installed binary is **`plugmem-cli`**.

## Install

Until the crates are published, build from the workspace:

```sh
cargo build --release -p plugmem-cli
# the binary is target/release/plugmem-cli
```

## Which crate do you need?

| You want | Use |
|---|---|
| a memory from the shell or a script — one file, no server | **this binary** |
| the same, but embedded in a Rust program | [`plugmem-host`](https://docs.rs/plugmem-host/latest) (`std`) |
| the engine alone, your own storage, `no_std` (browser/wasm) | [`plugmem-core`](https://docs.rs/plugmem-core/latest) |
| agents over a protocol instead of a shell | `plugmem-mcp` — in progress |

## Usage

```text
plugmem-cli [--db PATH] [--json] <command>
```

The database is chosen by, in order: `--db PATH`, the `PLUGMEM_DB`
environment variable, or `./plugmem.db` (created on first write). The
engine keeps no clock, so `now` comes from the system clock at each call.

| command | what it does |
|---|---|
| `remember <TEXT> [--entity E] [--tag T]… [--link REL:ENTITY]… [--valid-from TS]` | store a fact; prints its id and any similar/conflicting facts |
| `recall [QUERY] [--tag T]… [--entity E]… [--as-of TS] [--range FROM TO] [-k N] [--closed]` | ranked, token-budgeted block; sources compose |
| `revise <ID> <TEXT> [same flags as remember]` | close the old fact, record the successor |
| `forget <ID>` | tombstone a fact (purged physically at the next `maintain`) |
| `link <SRC> <REL> <DST>` | upsert a typed edge between entities |
| `show <ID>` | one fact's full card — text, both time axes, state |
| `stats` | engine size counters |
| `maintain` | purge tombstones, compact, build HNSW past the threshold (disk-first) |
| `checkpoint` | flush the journal into a fresh snapshot and clear it (leaves the database checkpointed) |
| `verify` | check content integrity (text UTF-8, vector↔fact consistency); exit 2 on damage |
| `scrub` | check the snapshot's byte-level container checksums; exit 2 on the first damaged section |
| `recover <DST>` | salvage a content-corrupt database into a clean `DST`; the source (`--db`) is left untouched |
| `export` | dump the currently-open facts as JSONL (one per line) to stdout |
| `import <FILE>` | load facts from a JSONL file (as written by `export`) |

Read-only commands (`recall`, `show`, `stats`, `export`) open the snapshot
**zero-copy over an mmap** (a shared lock, so several may run at once and
the whole file is not loaded) — falling back to a normal open if the
journal is un-checkpointed. `recall` uses that fast path only when no
embedder is configured, because embedding the query needs the read-write
handle. `scrub` also takes the shared mmap open, so it needs a checkpointed
database — run `checkpoint` (or `maintain`) first if the journal is dirty.

### Examples

```sh
# Remember, with a subject entity and a tag:
plugmem-cli remember "prefers tokio with pinned versions" --entity user --tag pref

# Recall — a ranked block ready to paste into a prompt:
plugmem-cli recall "which runtime"

# Bitemporal: correct a fact, then ask what was true earlier.
plugmem-cli remember "lives in Moscow" --entity user      # → fact 0
plugmem-cli revise 0 "lives in Berlin" --entity user
plugmem-cli recall "lives" --entity user --as-of 1700000000000   # → Moscow

# Machine-readable output for tooling / agents:
plugmem-cli --json recall "runtime" --tag pref
plugmem-cli --json stats

# Reclaim space held by forgotten facts:
plugmem-cli forget 3
plugmem-cli maintain

# Human-readable backup and restore:
plugmem-cli export > backup.jsonl
plugmem-cli --db other.plugmem import backup.jsonl

# Integrity & recovery (exit 2 on corruption — scriptable as a gate):
plugmem-cli verify                       # content consistency
plugmem-cli checkpoint && plugmem-cli scrub  # flush journal, then byte-level checksums
plugmem-cli recover agent.recovered.plugmem   # salvage into a clean copy
```

## Configuration

Optional `config.toml`, found by `--config PATH`, then `$PLUGMEM_CONFIG`,
then `$XDG_CONFIG_HOME/plugmem/config.toml` (all optional — the CLI works
with none). Precedence overall is **flag > environment > config file >
default**.

```toml
[engine]
dim = 768              # embedding size (0 = vectors off); other Config
                       # size fields: max_bytes, max_text, shards_* …

[embedder]             # default: none — lexical/tags/graph/time still work
kind = "ollama"        # ollama | openai | lmstudio | vllm | llamacpp | none
url = "http://localhost:11434/v1"
model = "nomic-embed-text"
api_key_env = "OPENAI_API_KEY"   # env var holding the bearer token (openai)

[maintenance]
snapshot_every_ops = 1024
snapshot_journal_bytes = 4194304
maintain_every_forgets = 100     # optional auto-purge
```

The embedder is what unlocks the **vector** recall source: with `kind =
"none"` (the default) `remember`/`recall` still answer from lexical, tag,
graph and temporal evidence, but no embeddings are computed. One
OpenAI-compatible client covers Ollama, OpenAI, LM Studio, vLLM and
llama.cpp-server. `$PLUGMEM_EMBEDDER` overrides `[embedder].kind`.

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
This is the same model as `sqlite3`, `git` and most file-backed tools:
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

One database file is a single-writer resource: `plugmem-cli` takes an
exclusive lock for the length of the (short-lived) command, so a second
`plugmem-cli` — or an MCP server holding the same file — is refused with
exit `1` rather than corrupting it. See the
[host concurrency model](https://docs.rs/plugmem-host/latest).

## License

MIT.
