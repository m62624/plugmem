# plugmem-cli

`plugmem-cli` is the command-line surface over the plugmem
[temporal-memory engine](../plugmem-core) — a thin shell around
[`plugmem-host`](../plugmem-host) that lets an agent (or you) keep a memory
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
| the same, but embedded in a Rust program | [`plugmem-host`](../plugmem-host) (`std`) |
| the engine alone, your own storage, `no_std` (browser/wasm) | [`plugmem-core`](../plugmem-core) |
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
| `maintain` | purge tombstones, compact, build HNSW past the threshold |

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
```

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
handle** case — embed [`plugmem-host`](../plugmem-host)'s `Database` in
your process (open once, call many verbs, all in RAM), or run the MCP
server, which keeps the memory resident. The CLI deliberately does not: it
trades a per-command load for a stateless, scriptable tool.

## Concurrency

One database file is a single-writer resource: `plugmem-cli` takes an
exclusive lock for the length of the (short-lived) command, so a second
`plugmem-cli` — or an MCP server holding the same file — is refused with
exit `1` rather than corrupting it. See the
[host concurrency model](../plugmem-host#concurrency-model).

## License

MIT.
