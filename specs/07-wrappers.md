# 07 — Wrappers: CLI, MCP, napi, py, skill, delivery

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
`plugmem_link`, `plugmem_show`, `plugmem_stats`, `plugmem_export` (`{ facts, edges }` —
an edge belongs to no single fact, so facts alone lose the graph), `plugmem_maintain`,
`plugmem_checkpoint`, `plugmem_verify`, plus `plugmem_version` and `plugmem_about`. A
read-only server adds `plugmem_generation` / `plugmem_refresh` (cross-process freshness)
and refuses the write verbs. (`scrub` and `recover` are deliberately absent: the server
is judged by what an agent should be handed, not by what host has — see the wrapper
parity rule in `AGENTS.md`. `import` is a CLI feature, not a host verb.) Each tool
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
packages). The `Plugmem` class wraps the host `Database` verb for verb — every method is
the identically-named host verb, and it is the *whole* of `Database`: `exportEdges`,
`scrub` and `recover` are there alongside the rest, per the wrapper-parity rule in
`AGENTS.md`. The one thing host has and this does not is `import`, which is not a host
verb at all — JSONL is a format the CLI defines. Inputs/outputs are
`napi(object)` mapped to hand-written TypeScript interfaces; the heavy verbs
(`maintain`/`checkpoint`) are async on libuv. Node opens a file directly (real file
I/O), so there is no JS storage bridge. A `Workspace` class mirrors the host type the
same way, and its `open(name)` returns the same `Plugmem` class — so a named memory has
every verb (see `10-workspace.md`). The test suite is `node --test` smoke plus parity
with native (the same scenario → the same rendered block), and a **type-check gate**:
the generated `index.d.ts` is compiled with a consumer-shaped type test under `strict`
plus `isolatedModules`, because a surface can be valid Rust and unusable TypeScript.

## Python extension (`plugmem-py`)

A **CPython extension module (PyO3)**, published to PyPI as `plugmem`. The crate
is `crate-type = ["cdylib", "rlib"]`, `publish = false`; the compiled module is
`plugmem._plugmem`, nested inside a pure-Python package so `py.typed`, the
generated stubs and the `export_pages` generator ship beside it.

**Its reference is `plugmem-napi`'s surface, not host's.** This is the one place
the wrapper-parity rule needs stating twice, because napi narrowed host on
purpose — one `export_page` and no `export_each`, one `maintain(mode)` and no
`maintain_with_options`, `scrub(budget)` and no `scrub_with_budget` — and a
third surface derived from host would disagree with the second. Verified rather
than asserted: `tests/test_parity.py` reads napi's generated `index.d.ts` and
its `error.rs`, maps camelCase to snake_case, and fails in both directions.
`Plugmem` 23 verbs, `Scrub` 3, `Workspace` 11, plus the module functions.

**No async layer, by the same reasoning that produced one in napi.** Node has a
single thread that runs JavaScript, so the blocking verbs had to move to a libuv
worker. The Python equivalent of that thread is the GIL, and the equivalent move
is to release it: every verb is synchronous with its body inside
`Python::detach`. That is what makes `asyncio.to_thread(db.recall, ...)` and a
`ThreadPoolExecutor` correct without a runtime to bridge.

Locks appear where napi needs none, and they guard the **handle slot**, not the
engine: host's `Database` and `Workspace` synchronize themselves, but several
Python threads share one wrapper object, so `close()` emptying the `Option`
while another thread is inside a verb would race on that `Option`. `RwLock` on
`Plugmem` (reads overlap; `refresh`/`close` are exclusive), `Mutex` on `Scrub`.

Results are hand-mapped, with the host type destructured exhaustively — a new
field in the engine is a compile error here rather than a value silently not
carried. Numbers stay exact, unlike napi's `f64` rendering: an open fact's
`valid_to` is `u64::MAX`, which no `f64` represents. Exceptions are a hierarchy
under `PlugmemError`, each carrying the `code` string napi puts on a thrown
`Error`.

`python/plugmem/_plugmem.pyi` is generated by `cargo run --bin stub_gen` and
committed, gated on `git diff` and on `mypy --strict` — the same treatment
`index.d.ts` gets. PyO3's own `experimental-inspect` plus
`maturin generate-stubs` was measured and produces an empty stub, hence
`pyo3-stub-gen`.

Delivery is 12 wheels plus an sdist: six platforms mirroring `napi-build` and
cargo-dist, each built twice — an `abi3-py310` wheel covering every CPython
3.10+ with a GIL, and a version-specific `cp314t` wheel because free-threading
(PEP 703) changed the object layout and has no stable ABI until `abi3t` in 3.15.
No musl, for the reason it is absent everywhere else in this repository.
Publishing is PyPI trusted publishing with no token at all — unlike npm and
crates.io, PyPI supports a *pending* publisher, so there is no manual first
release.

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
  with native on a shared scenario; `npm run typecheck` compiles the generated
  declarations and asserts the inferred types.
- py: pytest against a built wheel. Surface parity against napi's `index.d.ts`
  and `error.rs`, failing in both directions. Proof the GIL is released, stated
  as counters rather than clocks so no runner speed can make it flaky: an
  instrumented embedder records the peak number of threads inside it at once
  and the test requires more than one, six threads recalling one handle must
  overlap, and an asyncio heartbeat must tick while twelve `to_thread` recalls
  are in flight. `mypy --strict` over the generated stub and the package.
