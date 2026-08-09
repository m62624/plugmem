# Local guide: `plugmem-mcp`

## Role and transport

`plugmem-mcp` is a sidecar MCP server. It reads newline-delimited JSON-RPC 2.0 messages from stdin and writes responses to stdout. It listens on no port, has no MCP SDK dependency, and uses `serde_json` with a hand-written protocol loop.

The layers are intentionally separate:

- `rpc.rs` owns JSON-RPC envelopes, stdio reading, dispatch, and worker coordination;
- `tools.rs` owns tool schemas, argument extraction, execution, and formatting;
- `messages.rs` owns all model/user-facing strings;
- `main.rs` opens settings/database and resolves worker count.

## Modes and concurrency

Writer mode exposes the full memory tool set. Read-only mode opens a shared checkpoint generation and exposes only read verbs plus `plugmem_generation` and `plugmem_refresh`; write verbs must be refused as tool-level errors.

Tag discovery is a bounded read tool and belongs in both modes. Global tag
removal is a bulk write tool and belongs only in writer mode; it delegates the
history-preserving revisions to host/core.

The default worker count is half of available parallelism, at least one, and can be overridden by `--workers` or `[server].workers`. Keep embedding HTTP work outside the database lock where the current tool path expects it. Do not add tokio or a network listener without redesigning the sidecar contract.

Database/config resolution follows `--db`, `PLUGMEM_DB`, and `./plugmem.db`, plus the shared host config loader. Startup failures go to stderr and use a non-zero exit code so the spawning process can detect failure.

## Protocol behavior

Tool definitions and dispatch must stay in sync. Missing params are JSON-RPC errors; unknown tools and host failures are tool-level errors with `isError`. Human and JSON result formats are selected through the common `format` argument.

There is deliberately no import tool: a remote/sandboxed server cannot assume access to a caller's local file. Bulk loading belongs to the CLI or a process with explicit disk access.

## Tests

```bash
cargo test -p plugmem-mcp
cargo run -p plugmem-mcp -- --help
```

Maintain coverage for `initialize`, `tools/list`, `tools/call`, missing/unknown methods, writer/read-only tool sets, malformed arguments, and typed host errors. Protocol output must remain one JSON object per line with no diagnostic noise on stdout.
