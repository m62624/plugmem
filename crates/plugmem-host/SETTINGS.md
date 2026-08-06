# plugmem settings

This is the canonical configuration reference for `plugmem-host` and every
wrapper that uses it: `plugmem-cli`, `plugmem-mcp`, `plugmem-napi` and
`plugmem-py`.

## Config-file discovery

The config file itself is resolved in this order:

1. An explicit `--config PATH` (CLI/MCP) or `config` option (Node, Python).
2. `$PLUGMEM_CONFIG`.
3. The platform config directory from `directories::ProjectDirs`:

   - Linux: `$XDG_CONFIG_HOME/plugmem/config.toml`, otherwise
     `~/.config/plugmem/config.toml`.
   - macOS: `~/Library/Application Support/plugmem/config.toml`.
   - Windows: `%APPDATA%\plugmem\config\config.toml`.

4. Built-in defaults when no config file exists.

An explicit path that does not exist is an error. A discovered default config
file is optional; its absence means that all defaults apply.

## Database-path precedence

The database path is resolved separately from the config-file path:

1. An explicit path (`--db` for CLI/MCP, or the path passed to `open` in Node
   and Python).
2. `$PLUGMEM_DB`.
3. `[database].path` from this config file.
4. The platform data path.

The default platform data path is:

- Linux: `$XDG_DATA_HOME/plugmem/memory.plugmem`, otherwise
  `~/.local/share/plugmem/memory.plugmem`.
- macOS: `~/Library/Application Support/plugmem/memory.plugmem`.
- Windows: `%LOCALAPPDATA%\plugmem\data\memory.plugmem`.

The database is a snapshot plus adjacent journal and lock files. The host
uses mmap/overlay opens, so a large database is not loaded into RAM in full;
it still requires enough free disk for snapshots and maintenance temporary
files. Use an explicit path when the database belongs on a particular disk or
project.

## Example

```toml
[database]
# Optional. An explicit --db or open path and PLUGMEM_DB override this.
path = "/path/to/memory.plugmem"

[engine]
dim = 768              # 0 disables vectors
max_bytes = 2147483648
max_text = 4096
max_blob = 65536

# Optional. Every key here has a tuned default — reach for this section when a
# specific memory answers badly, not before.
[recall]
w_vec = 2.0            # trust meaning over keywords in this memory
half_life_days = 30    # and treat anything older than a month as stale

[index]
flat_to_hnsw = 50000   # this memory is small; stay exact for longer

[embedder]
# none | ollama | openai | lmstudio | vllm | llamacpp
kind = "ollama"
url = "http://localhost:11434/v1"
model = "nomic-embed-text"
api_key_env = "OPENAI_API_KEY"

[maintenance]
snapshot_every_ops = 1024
snapshot_journal_bytes = 4194304
maintain_every_forgets = 100

# CLI only: facts per `import` batch.
batch_size = 128

[server]
# MCP only: defaults to half of available cores, at least one.
workers = 4
```

## Sections

### `[database]`

| Key | Default | Meaning |
|---|---:|---|
| `path` | platform data path | Persistent snapshot path. It is overridden by an explicit path and `$PLUGMEM_DB`. |

### `[workspace]`

**Omit this section unless you need it.** Without it there is one database,
addressed by path, and nothing below applies — that is the default and the
common case. A workspace is for one process serving many independent memories
(a database per chat, per tenant), where each request says which one it means.

| Key | Default | Meaning |
|---|---:|---|
| `dir` | unset | Directory of named databases. Setting it is what turns a workspace on. |
| `max_open` | `16` | Databases kept open at once; the least recently used is closed to make room. Must be between 1 and 240 — one open database costs several file descriptors, so an unbounded value would exhaust them somewhere else in the program. |
| `idle_timeout_ms` | `60000` | Close a database unused this long. `0` never closes. |

The layout under `dir` is fixed:

```text
<dir>/registry.plugmem      the registry — an ordinary plugmem database
<dir>/db/<name>.plugmem     the databases themselves
```

A name is `[a-z0-9][a-z0-9_-]*`, at most 64 bytes. It is not a path and cannot
become one: separators, dots and leading dashes are not names, so a name can
only ever resolve to one named database directly inside `<dir>/db`.

`idle_timeout_ms` is about **reachability, not memory**. An open database holds
an exclusive file lock, so a long-running server that never let go would make
its databases permanently unreachable from the CLI. The timeout is what returns
them.

### `[engine]`

The size-bearing fields — what a database is *built* with. Changing one on an
existing file is refused with a typed error rather than applied, because the
bytes on disk were laid out against it.

| Key | Default | Meaning |
|---|---:|---|
| `dim` | `0` | Embedding dimension; zero disables vector storage. |
| `max_bytes` | `2147483648` | Ceiling for **each** byte pool, not their sum — see below. |
| `max_text` | `4096` | Maximum fact text length in bytes. |
| `max_blob` | `65536` | Maximum single blob length in bytes. |

`max_bytes` applies to every pool separately (arena pages, the text and
metadata blob heaps, the tag and posting chunk pools, the vector pool), so a
database's total goes several times past it; the pool that binds first is
normally the fact texts. The default is not a capacity judgement — it is the
figure that keeps every pool addressable where `usize` is 32 bits, so a file
written anywhere opens anywhere. Raise it if you need to, and the only thing
you give up is that: a 32-bit host then refuses the file with a typed error
rather than reading it wrongly.

There is no shard-count setting. How many shards each arena gets is derived
from how much the database holds, and `maintain` moves it as that changes —
a thousand facts on a layout meant for a million cost fourteen megabytes
instead of one. `plugmem-cli stats` reports the current layout.

### `[recall]`

What comes back for a query, and in what order. **Unlike `[engine]`, these are
free to change on an existing database** — reopening with different weights is
how you change the ranking, and the next checkpoint records them in the file.

Every one of them is optional and the defaults are the tuned ones; reach for
this section when a specific memory answers badly, not before.

| Key | Default | Meaning |
|---|---:|---|
| `w_bm25` | `1.0` | Weight of the lexical source in the fused score. `0` switches it off. |
| `w_vec` | `1.0` | Weight of the vector source. `0` switches it off; it already costs nothing when `dim = 0`. |
| `w_graph` | `1.0` | Weight of the entity-graph source. `0` switches off relational expansion. |
| `w_time` | `1.0` | Weight of the temporal source (the `recorded_at` window). |
| `w_recency` | `0.25` | How much a fact's age discounts it, on top of the four sources. |
| `half_life_days` | `180` | Age at which the recency discount has halved. Larger keeps old facts competitive. |
| `rrf_k` | `60` | Reciprocal-rank-fusion constant. Larger flattens the gap between rank 1 and rank 10. |
| `bm25_k1` | `1.2` | BM25 term-frequency saturation: higher lets a repeated word keep counting. |
| `bm25_b` | `0.75` | BM25 length normalisation: `0` ignores fact length, `1` penalises long facts fully. |
| `graph_depth` | `2` | Default hops the graph source may follow from an anchor; a recall's own `graph_depth` overrides it, the way `ef` overrides `hnsw_ef_search`. Uncapped: what a walk costs is held by its entity and edge caps, not by the hop count. |
| `graph_decay` | `0.5` | How much each extra hop discounts a fact reached through the graph. |
| `hnsw_ef_search` | `64` | Default vector-search beam width. A recall's own `ef` overrides it, and it does nothing while the index is still flat. |
| `similar_cos` | `0.85` | Cosine above which `remember` reports an existing fact as possibly conflicting. |
| `similar_jaccard` | `0.5` | Token overlap that does the same for a memory with no vectors. |

The weights are relative, not probabilities: they scale each source's
contribution before fusion, so doubling all four changes nothing. Setting one
to `0` is the way to turn a source off entirely — useful for asking "what does
lexical search alone think", and cheaper than it sounds, since a source with no
weight is not consulted.

`similar_cos`/`similar_jaccard` are the only two that affect *writing*: they set
how alike an existing fact must be before `remember` reports it as a possible
conflict. The engine never revises on its own — the report is for you to judge.

### `[index]`

How the vector index is built. Also free to change on an existing database,
though `flat_to_hnsw` only takes effect at the next `maintain`.

| Key | Default | Meaning |
|---|---:|---|
| `hnsw_ef_construction` | `200` | Beam width while building the vector graph: higher builds a better index, slower. Must be at least `hnsw_m` (16). |
| `flat_to_hnsw` | `24000` | Vector count at which maintenance stops scanning flat and builds the HNSW graph. |

Below `flat_to_hnsw` the engine scans vectors linearly, which is exact and, at
that size, faster than a graph. The setting is where you decide that trade for
your data rather than taking the default's word for it.

The graph *degrees* (`hnsw_m`, `hnsw_m0`) are deliberately not here: they shape
the stored graph, so changing them on an existing database is refused with a
config mismatch, exactly like `[engine]`.

### `[embedder]`

The default is `kind = "none"`; lexical, tag, graph and temporal retrieval
still work without an embedder. `$PLUGMEM_EMBEDDER` overrides
`[embedder].kind`.

| Key | Default | Meaning |
|---|---|---|
| `kind` | `none` | `none`, `ollama`, `openai`, `lmstudio`, `vllm` or `llamacpp`. |
| `url` | unset | OpenAI-compatible `/v1/embeddings` endpoint. Required for an active embedder. |
| `model` | unset | Embedding model name. Required for an active embedder. |
| `api_key_env` | unset | Environment variable containing the bearer token. |

An active embedder also requires `[engine].dim > 0`. All supported providers
use the same OpenAI-compatible HTTP shape.

### `[maintenance]`

| Key | Default | Meaning |
|---|---:|---|
| `snapshot_every_ops` | `1024` | Snapshot after this many mutations. |
| `snapshot_journal_bytes` | `4194304` | Snapshot when the journal reaches this size. |
| `maintain_every_forgets` | off | Run policy maintenance after this many forgets. |
| `fsync` | `each_op` | When journal appends reach the disk. `each_op`: every acknowledged write survives a power cut. `on_snapshot`: faster, but an OS crash may lose the journal tail written since the last snapshot. |
| `batch_size` | `128` | CLI-only `import` batch size; `--batch` overrides it. |

One maintenance trigger has no key and is always on: a database that outgrows
(or falls far below) its shard layout re-shards itself on the next write. It
has to be automatic — the triggers above are opt-in, so otherwise a growing
database would keep the layout it was created with until somebody ran
`maintain` by hand. It is also self-limiting: the thresholds are a doubling up
and a fourfold drop, so it fires a handful of times over a database's life.

One consequence worth expecting: a database written by a version that used the
old fixed layout is stale the moment it opens, so its **first write re-shards
it**, at a cost proportional to its size. That happens once and leaves a
permanently smaller file.

### `[server]`

| Key | Default | Meaning |
|---|---:|---|
| `workers` | half of available cores | MCP worker threads; `--workers` overrides it. |

## A key nobody recognises

An unknown section or key does not stop anything — refusing one would mean an
older binary could not read a config written for a newer one, which is a worse
failure than a typo. It is **reported**, though, because the alternative is
silence: a misspelled `w_vec` changes no behaviour and says nothing, and you are
left believing you tuned something.

```console
$ plugmem-cli stats
plugmem: unknown config section [engin] — did you mean `engine`?
plugmem: unknown setting [recall].w_vector — did you mean `w_vec`?
facts       0
...
```

The CLI and the MCP server write these to stderr — stdout carries results, and
for MCP it carries the protocol itself. The Node addon returns them instead,
from `configWarnings()`, because a native addon has no business writing to its
host application's stderr:

```javascript
const db = await Plugmem.open("agent.plugmem", { config: "./plugmem.toml" });
for (const warning of db.configWarnings()) console.warn(warning);
```

A suggestion is only offered when it is close enough to be the obvious intent —
pointing at the wrong line is worse than pointing at none.

## Surface-specific overrides

| Surface | Explicit database path | Explicit config path | Extra override |
|---|---|---|---|
| CLI | `--db PATH` | `--config PATH` | `--batch`, `--json` |
| MCP | `--db PATH` | `--config PATH` | `--workers`, `--read-only` |
| Node | `Plugmem.open(path)` | `OpenOptions.config` | `OpenOptions.dim`, `readOnly` |
| Python | `Plugmem.open(path)` | `config=` | `dim=`, `read_only=` |
| Host | `Database::open(path, config)` | `Settings::load(path)` | programmatic builder options |

Use the runtime help surfaces when the full reference is not available:

```console
$ plugmem-cli help settings
$ plugmem-cli --json help settings
```

MCP exposes `plugmem_settings_help` with `format: "json"` or `"human"`; Node
exposes `settingsHelp()` and Python `settings_help()`.
