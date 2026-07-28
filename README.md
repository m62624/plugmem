# plugmem

Temporal memory for LLM agents, as an embeddable library. plugmem stores
short **facts** — with a subject entity, tags, an optional embedding and
two time axes — and answers a query with a ranked, token-budgeted block
ready to paste into a prompt. It runs in-process, keeps a whole database
in one snapshot file plus an append-only journal, and needs no server and
no disk of its own — the same engine runs on native, in WebAssembly, and
in the browser.

It is **not** a vector database. A vector is one of four recall sources —
lexical (BM25), vector, entity graph and time — fused with reciprocal-rank
fusion; tags act as filters. What plugmem is actually about:

- **bitemporal facts** — `revise`/`forget`, "what was true *then*" vs now,
  revision chains kept intact, physical erasure on `maintain`;
- **an entity graph** — typed edges between entities, relational knowledge,
  not just nearest-neighbor vectors;
- **conflict surfacing on `remember`** — a new fact comes back with the
  live facts it may duplicate or contradict; the engine never merges on
  its own, the caller decides;
- **a compact rendered block** — the result is selected greedily under a
  token budget, ready for the prompt;
- **embeddable everywhere** — single-threaded `no_std + alloc` core,
  zero-allocation recall after warm-up, one file, no server; built and
  tested on `wasm32v1-none` and a real 32-bit wasm runtime in CI.

**Where it fits — and where it doesn't.** plugmem is for the memory of a
*local agent* or an *embedded system*: one process that remembers and recalls
on the order of thousands to hundreds of thousands of facts, in a single file,
with nothing to run. It is deliberately **not** a large-scale vector database —
if you need approximate-nearest-neighbour search over tens of millions of
embeddings, sharded across a cluster or behind a managed API, reach for a
dedicated one ([Qdrant](https://qdrant.tech), [Milvus](https://milvus.io),
[Weaviate](https://weaviate.io), [Pinecone](https://www.pinecone.io),
[pgvector](https://github.com/pgvector/pgvector)). plugmem trades that scale
for zero infrastructure, recall that fuses keyword + meaning + relationships +
time (not vectors alone), and correctness over time — the shape an agent's
memory actually needs.

## Which crate do I need?

**If you write Rust and just want a working memory, use
[`plugmem-host`](crates/plugmem-host) — it has everything.** The other crates
are for narrower needs.

| You want | Use | Why |
|---|---|---|
| **A memory in a Rust program** — the common case | **[`plugmem-host`](crates/plugmem-host)** (`std`) | Everything included: files, locking, read-only mmap, HTTP embedders, integrity, cross-process concurrency. One dependency — it re-exports the engine. |
| A memory in Rust with **no `std`** or **your own storage** (browser, wasm host, custom file layer) | [`plugmem-core`](crates/plugmem-core) (`no_std`) | The engine only. You bring the `Storage` trait, the clock, file I/O and embedding — so you also manage when the file opens and how memory loads. |
| Just the **flat byte-pool containers**, to build your own index/store | [`plugmem-arena`](crates/plugmem-arena) (`no_std`) | The storage substrate, engine-agnostic. |
| A memory from a **terminal or shell script** | [`plugmem-cli`](crates/plugmem-cli) (`plugmem`) | One file, no server; `plugmem repl` keeps the engine open for host speed. |
| A memory for an **LLM agent** or a **non-Rust program** | [`plugmem-mcp`](crates/plugmem-mcp) | Long-lived stdio JSON-RPC; language-independent. **Don't** front your own Rust with it — embed `host` instead. |
| A memory in **JavaScript / the browser** | [`plugmem-wasm`](crates/plugmem-wasm) | The engine compiled to WebAssembly. |

Rust programs use the library (`host`, or `core` for specialists) — embedded
in-process, like linking SQLite. Other languages and agents come in through
`mcp` (or `wasm` for JS) — not the CLI, which is the human/scripting door.

| Crate | What it is | State |
|---|---|---|
| [`plugmem-arena`](crates/plugmem-arena) | flat byte-pool storage structures (`no_std`, wasm-first) | done: tested, measured |
| [`plugmem-core`](crates/plugmem-core) | the engine: data model, indexes, recall, snapshots (`no_std`) | done |
| [`plugmem-host`](crates/plugmem-host) | OS glue: files, locking, mmap read-only, embedder clients (`std`) | done |
| [`plugmem-cli`](crates/plugmem-cli) | command-line surface, one-shot + interactive `repl` | done |
| [`plugmem-wasm`](crates/plugmem-wasm) | wasm bindings for non-Rust hosts | in progress |
| [`plugmem-mcp`](crates/plugmem-mcp) | MCP server (stdio JSON-RPC) for agents | done |
| `plugmem-testgen` | deterministic corpus generator for tests and benches | done |

## What recall does

Recall is not a vector lookup — it fuses four sources with
[reciprocal-rank fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) and a
recency boost (tags filter; they are not a source):

| Source | Algorithm | What it finds |
|---|---|---|
| **Lexical** | [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (Robertson idf) over a Unicode ([UAX #29](https://unicode.org/reports/tr29/)) tokenizer | exact terms / keyword overlap |
| **Semantic** | int8-quantized cosine — a flat two-phase scan below a threshold, an [HNSW](https://arxiv.org/abs/1603.09320) graph above | meaning / nearest neighbours |
| **Graph** | entity graph with typed edges, breadth-first from query anchors | relational knowledge |
| **Temporal** | range scans over a `recorded_at`-ordered index; bitemporal validity | "what was true *then*", time windows |

## Install

Two binaries — the `plugmem-cli` CLI (crate `plugmem-cli`) and the `plugmem-mcp`
server (crate `plugmem-mcp`) — built for **Linux, Windows and macOS (x64 &
arm64)** on every tagged release.

**Pick the one method that's convenient — you don't need more than one.** They all
install the *same* binaries. Quick guide:

| If you… | Use |
|---|---|
| are on macOS / Linux with Homebrew | **Homebrew** |
| don't want a Rust toolchain | **installer script**, or the Windows **`.msi`** |
| want managed install/uninstall on Windows | **`.msi`** |
| have `cargo` and want a cross-platform install | **`cargo binstall`** |
| want to compile it yourself | **from source** |

### Homebrew (macOS / Linux)

From the [`m62624/homebrew-plugmem`](https://github.com/m62624/homebrew-plugmem)
tap; `brew upgrade` / `brew uninstall` then manage it like any formula:

```console
$ brew install m62624/plugmem/plugmem-cli     # the `plugmem-cli` CLI
$ brew install m62624/plugmem/plugmem-mcp     # the `plugmem-mcp` server
```

### Installer scripts (no Rust toolchain)

Each binary has its own script on the
[Releases page](https://github.com/m62624/plugmem/releases); `latest` always
points at the newest tag.

```console
# Linux / macOS  (POSIX sh)
$ curl --proto '=https' --tlsv1.2 -LsSf https://github.com/m62624/plugmem/releases/latest/download/plugmem-cli-installer.sh | sh
$ curl --proto '=https' --tlsv1.2 -LsSf https://github.com/m62624/plugmem/releases/latest/download/plugmem-mcp-installer.sh | sh
```

### Windows `.msi`

Download `plugmem-cli-*.msi` or `plugmem-mcp-*.msi` from the
[Releases page](https://github.com/m62624/plugmem/releases). Double-click to
install; it registers the app in **"Add or remove programs"**, so upgrades and
uninstalls go through the normal Windows UI.

### Windows PowerShell script (alternative to `.msi`)

If you prefer a script over a GUI installer:

```powershell
> powershell -ExecutionPolicy Bypass -c "irm https://github.com/m62624/plugmem/releases/latest/download/plugmem-cli-installer.ps1 | iex"
> powershell -ExecutionPolicy Bypass -c "irm https://github.com/m62624/plugmem/releases/latest/download/plugmem-mcp-installer.ps1 | iex"
```

### `cargo binstall`

[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) downloads the
prebuilt binary instead of compiling. It reads the release's cargo-dist
manifest, so it just works on every OS/arch above — no extra config:

```console
$ cargo binstall plugmem-cli     # the `plugmem-cli` binary
$ cargo binstall plugmem-mcp     # the `plugmem-mcp` binary
```

### From source

Needs a Rust toolchain; compiles locally and works on any platform Rust targets.
Both crates are published to crates.io, so you can build straight from there:

```console
$ cargo install plugmem-cli     # the `plugmem-cli` binary
$ cargo install plugmem-mcp     # the `plugmem-mcp` binary
```

…or from a local checkout of this repo:

```console
$ cargo install --path crates/plugmem-cli
$ cargo install --path crates/plugmem-mcp
```

### Uninstall

**Installed with `cargo binstall` / `cargo install`** (either resolves to cargo's
own install tracking, so plain `cargo uninstall` works):

```console
$ cargo uninstall plugmem-cli      # removes the `plugmem-cli` binary
$ cargo uninstall plugmem-mcp
```

**Installed with Homebrew:** `brew uninstall plugmem-cli plugmem-mcp`.

**Installed from a Windows `.msi`:** uninstall from **"Add or remove programs"**
(or Settings → Apps), exactly like any other Windows app.

**Installed with the shell/PowerShell scripts:** cargo-dist does not ship an
uninstaller, so remove the binaries and their install receipts by hand. By default
the binaries land in `~/.cargo/bin` (note: `cargo uninstall` won't touch these —
cargo didn't track them), and a receipt is written per app.

```console
# Linux / macOS
$ rm -f  ~/.cargo/bin/plugmem-cli ~/.cargo/bin/plugmem-mcp
$ rm -rf ~/.config/plugmem-cli ~/.config/plugmem-mcp     # install receipts
```

```powershell
# Windows (PowerShell)
> Remove-Item "$env:USERPROFILE\.cargo\bin\plugmem-cli.exe","$env:USERPROFILE\.cargo\bin\plugmem-mcp.exe" -ErrorAction SilentlyContinue
> Remove-Item "$env:LOCALAPPDATA\plugmem-cli","$env:LOCALAPPDATA\plugmem-mcp" -Recurse -ErrorAction SilentlyContinue
```

If you pointed the installer somewhere else (`PLUGMEM_CLI_INSTALL_DIR` /
`PLUGMEM_MCP_INSTALL_DIR`, or `CARGO_DIST_FORCE_INSTALL_DIR`), delete from that
directory instead. The installer may also have added the bin dir to your `PATH` —
prune that line from your shell profile if nothing else uses it.

## Principles

- **Fast is a number.** Every performance claim has a benchmark in-repo;
  cross-runtime results (native / wasmtime / wasmer) reproduce with one
  command.
- **The memory image is the file format.** State lives in flat byte pools;
  a snapshot is a `memcpy`, loading is validation plus adoption, and the
  same file opens byte-identically on native, wasm32 and wasm64.
- **Embedded first.** Single-threaded `no_std + alloc` core; OS concerns
  stay in the thin `plugmem-host` layer.
- **Untrusted input never panics.** Arbitrary bytes can produce any typed
  `Error`, never a panic or UB; after a load every stored id is
  range-checked.

## License

MIT.
