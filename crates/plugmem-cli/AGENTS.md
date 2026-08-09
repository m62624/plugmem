# Local guide: `plugmem-cli`

## Role

`plugmem-cli` is the human and shell interface over `plugmem-host`. It is a thin command parser/executor, not a second storage implementation. `src/cli.rs` owns clap definitions; `src/lib.rs` owns execution and rendering; `src/main.rs` is the binary entry point; `src/config.rs` handles CLI-facing configuration selection.

## Command surface

Global options include `--db`, `--config`, and `--json`. The default database path is `./plugmem.db`, overridden by `PLUGMEM_DB`; config resolution follows `--config`, `PLUGMEM_CONFIG`, and the XDG default.

The commands cover:

- `remember`, `recall`, `revise`, `forget`, tag listing/removal, and `link`;
- `show`, `stats`, `export`, and `import`;
- `maintain`, `checkpoint`, `verify`, `scrub`, and `recover`;
- `repl`, including an explicit read-only mode.

Recall options include tags, entities, validity time, recorded-time range, result count, and closed-fact inclusion. `import` streams JSONL in batches; ids and recorded timestamps are not preserved because facts are re-remembered.

Human output and `--json` must describe the same underlying result and use stable ids. Exit codes are part of the CLI contract: preserve them when changing error handling.

## Configuration and safety

The CLI delegates database lifecycle, locking, checkpointing, and recovery to `plugmem-host`. Do not open the same file through a second ad-hoc path. `recover` must require a destination different from the source; `scrub`/read-only workflows require the database state expected by host.

Keep long help text and user-facing wording in `cli.rs`; keep execution errors typed and testable. Avoid printing secrets from embedder configuration.

## Tests

```bash
cargo test -p plugmem-cli
cargo run -p plugmem-cli -- --help
cargo run -p plugmem-cli -- --json stats --db /tmp/example.plugmem
```

Use `tests/cli.rs` and `tests/fixtures/` for argument, JSONL, exit-code, and configuration behavior. If a command changes, update help text and fixture coverage together.
