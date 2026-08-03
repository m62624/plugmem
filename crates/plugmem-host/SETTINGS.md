# plugmem settings

This is the canonical configuration reference for `plugmem-host` and every
wrapper that uses it: `plugmem-cli`, `plugmem-mcp` and `plugmem-napi`.

## Config-file discovery

The config file itself is resolved in this order:

1. An explicit `--config PATH` (CLI/MCP) or `config` option (NAPI).
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

1. An explicit path (`--db` for CLI/MCP, or the NAPI constructor path).
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
# Optional. Explicit --db / constructor path and PLUGMEM_DB override this.
path = "/path/to/memory.plugmem"

[engine]
dim = 768              # 0 disables vectors
max_bytes = 2147483648
max_text = 4096
max_blob = 65536

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

### `[engine]`

These are the size-bearing fields accepted from TOML. BM25, fusion, graph and
HNSW tuning fields remain programmatic `plugmem-core::Config` settings for now.

| Key | Default | Meaning |
|---|---:|---|
| `dim` | `0` | Embedding dimension; zero disables vector storage. |
| `max_bytes` | `2147483648` | Total byte-pool ceiling. |
| `max_text` | `4096` | Maximum fact text length in bytes. |
| `max_blob` | `65536` | Maximum single blob length in bytes. |

There is no shard-count setting. How many shards each arena gets is derived
from how much the database holds, and `maintain` moves it as that changes —
a thousand facts on a layout meant for a million cost fourteen megabytes
instead of one. `plugmem-cli stats` reports the current layout.

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
| `batch_size` | `128` | CLI-only `import` batch size; `--batch` overrides it. |

### `[server]`

| Key | Default | Meaning |
|---|---:|---|
| `workers` | half of available cores | MCP worker threads; `--workers` overrides it. |

## Surface-specific overrides

| Surface | Explicit database path | Explicit config path | Extra override |
|---|---|---|---|
| CLI | `--db PATH` | `--config PATH` | `--batch`, `--json` |
| MCP | `--db PATH` | `--config PATH` | `--workers`, `--read-only` |
| NAPI | constructor `path` | `OpenOptions.config` | `OpenOptions.dim`, `readOnly` |
| Host | `Database::open(path, config)` | `Settings::load(path)` | programmatic builder options |

Use the runtime help surfaces when the full reference is not available:

```console
$ plugmem-cli help settings
$ plugmem-cli --json help settings
```

MCP exposes `plugmem_settings_help` with `format: "json"` or `"human"`, and
NAPI exposes `settingsHelp()`.
