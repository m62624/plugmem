# 07 — Wrappers: CLI, MCP, napi, skill, delivery

Every wrapper is thin: parse input → the core → render output. Memory logic in a
wrapper is forbidden (a review rule). The same capabilities on every surface; the only
differences are the transport and how the bytes are fetched.

## Shared: config and database discovery

Priority: flag/parameter > env > config file > default.

- Database: `--db PATH` | `PLUGMEM_DB` | `./plugmem.plugmem` if it exists |
  `$XDG_DATA_HOME/plugmem/default.plugmem` (created). With a workspace configured,
  `--db`/`PLUGMEM_DB` may also be a bare memory *name* — see `10-workspace.md`.
- Config file: `$XDG_CONFIG_HOME/plugmem/config.toml` — `[engine]` (the Config fields
  from `05-api.md`), `[embedder]` (`kind`, `url`, `model`, `api_key_env`),
  `[maintenance]` (`auto_after_ops`, `journal_snapshot_bytes`).
- The default embedder is `none` (the system must work out of the box with no
  services); turning it on is one config section or `PLUGMEM_EMBEDDER`.
- Locking: FileStorage holds an exclusive lock (see `03-snapshot.md`) — one database,
  one writer. When the database is busy (e.g. an MCP server holds it), the CLI prints
  "database is locked by another process" and exits 1.

## CLI (`plugmem-cli`)

Commands (all read verbs take `--json`; human output is the default):

| Command | What |
|---|---|
| `remember "text" [--entity E] [--tag T]… [--link REL:ENTITY]… [--meta KEY=VALUE]… [--valid-from TS]` | remember; prints id + similar hints. `--meta` is repeatable, last-wins per key |
| `recall [QUERY] [--tag]… [--entity]… [--as-of TS] [--range A B] [-k N] [--closed]` | recall; human output is the `rendered` block |
| `revise ID "text" […]` | revise (same flags as remember) |
| `forget ID` | forget |
| `link SRC REL DST` | upsert a typed edge |
| `show ID` | a fact's full card (both time axes, state, metadata) |
| `stats` | sizes, counters, identity |
| `maintain` / `checkpoint` | explicit maintenance / flush the journal to a fresh snapshot |
| `verify` / `scrub` | content integrity / byte-level container integrity |
| `recover DST` | salvage a content-corrupt database into a clean copy at DST |
| `export` / `import FILE [--batch N]` | dump/load facts as JSONL (backup, portability) |
| `repl [--read-only]` | keep the engine open, one command per line, at host speed |

Exit codes: 0 ok; 1 input error / not found; 2 database damage. `now` comes from the
system clock **here** (the only place time enters). Cold start is a critical path (a
process per command): no work before the command is parsed.

## MCP (`plugmem-mcp`)

stdio JSON-RPC (newline-delimited, one message per line). The tools mirror the core:
`plugmem_remember`, `plugmem_recall`, `plugmem_revise`, `plugmem_forget`,
`plugmem_link`, `plugmem_show`, `plugmem_stats`, `plugmem_export`, `plugmem_maintain`,
`plugmem_checkpoint`, `plugmem_verify`, plus `plugmem_version` and `plugmem_about`. A
read-only server adds `plugmem_generation` / `plugmem_refresh` (cross-process freshness)
and refuses the write verbs. (`scrub`, `recover` and `import` are CLI-only.) Each tool
takes `format` ("json" default | "human"). `plugmem_remember`'s description says
outright: "if similar contains a contradiction, decide: plugmem_revise or keep both".

The server owns one database (path from argument/env as in the CLI), the embedder from
the same config. With `--workspace DIR` it instead serves a directory of named memories,
and every tool that touches one gains a `db` argument — absent otherwise, so the
single-database default is unchanged. See `10-workspace.md`; that mode is opt-in. `maintain` runs on the `[maintenance]` policy between requests (no
background thread — a check after each call). `SKILL.md` is embedded via `include_str!`
and shipped as a release artifact; `plugmem_version`/`plugmem_about` tell the agent to
load the version-matched skill.

## Node addon (`plugmem-napi`)

A **native napi addon, not wasm**: the host is compiled as-is (real mmap/MVCC/locks, no
RAM bloat and no 4 GiB ceiling). The crate is `crate-type = ["cdylib", "rlib"]`,
`publish = false` (it ships to npm as the meta package `plugmem` plus six platform
packages). The `Plugmem` class mirrors the host `Database` 1:1; inputs/outputs are
`napi(object)` mapped to hand-written TypeScript interfaces; the heavy verbs
(`maintain`/`checkpoint`) are async on libuv. Node opens a file directly (real file
I/O), so there is no JS storage bridge. The test suite is `node --test` smoke plus
parity with native (the same scenario → the same rendered block).

## SKILL.md

Version-matched to the engine. It covers when to recall (start of a task / when the
past is mentioned; an empty block means "nothing relevant"), when to remember (durable
facts, one fact = one statement, entity/tag conventions), the contradiction loop
(remember → read similar → revise / keep both / forget), temporality (valid_from,
as_of, range), and worked examples per surface. A `<!-- skill-version: X.Y.Z -->`
marker must equal the workspace version at release; a CLI/MCP consumer of the skill
prints one version-check line and stops on a mismatch. A fenced section (the transport
ceremony) is stripped from the in-process npm distribution, where skill and engine ship
together.

## Delivery

cargo-dist builds the two binaries (`plugmem-cli`, `plugmem-mcp`) for
linux/windows/macOS × x64/arm64 on every tagged release, with shell/PowerShell/`.msi`
installers, a Homebrew tap (`m62624/homebrew-plugmem`) and `cargo binstall` support.
Libraries publish to crates.io (`dist = false`). The npm packages publish from CI over
OIDC trusted publishing. `SKILL.md` is a version-pinned artifact of each release. The CI
mechanics and required checks are in `08-performance.md`.

## Test plan

- CLI: integration tests (the binary + a temp database): each subcommand happy path +
  errors; `--json` schemas pinned by golden files.
- MCP: scenario JSON-RPC sessions: tools/list, each tool, bad inputs → correct JSON-RPC
  errors.
- napi: `node --test` smoke (open→remember→recall→checkpoint→reopen); rendered parity
  with native on a shared scenario.
