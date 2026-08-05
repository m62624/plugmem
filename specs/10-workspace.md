# 10 — Workspace: many memories in one directory

**A workspace is optional and off by default.** With no `--workspace`, no
`$PLUGMEM_WORKSPACE` and no `[workspace].dir`, there is one database addressed
by a path and nothing in this document applies. That is the ordinary
configuration; this one is for a process that serves many independent memories
— a database per conversation, per tenant, per project.

Everything here lives in `plugmem-host` and the wrappers. **The core does not
know workspaces exist**: one `Memory` is one database, `no_std`, and the file
format is unchanged. A change that needs the core is a change that is not this.

## Why it is affordable at all

Before the shard layout became a function of the data (`01-arena.md`), a
database holding a thousand facts cost 14.13 MB of pages whatever it held, so a
bot with a memory per conversation was not a design — it was 3.6 GB for two
hundred chats. At 0.43 MB and ~2 ms to open, the same two hundred are ~90 MB and
a directory scan is cheap enough to be a repair tool.

The `fifty_small_memories_fit_in_a_budget_a_bot_can_afford` test in
`plugmem-host/tests/host.rs` is the gate on that property.

## Layout

```text
<root>/registry.plugmem      the registry — an ordinary plugmem database
<root>/db/<name>.plugmem     the memories themselves
```

The registry sits in the root while the memories sit one level below, so **no
name can collide with the registry**. That is layout, not a reserved word.

## Names

A name is ASCII `[a-z0-9][a-z0-9_-]*`, at most `MAX_DB_NAME` (64) bytes, and not
a Windows device name.

The point is not filtering. **A name cannot represent a path**, so `..`, `/`,
`\`, a drive letter and an absolute path are not rejected late — they are
unconstructible, and resolution is a join with nothing to get wrong. A property
test states exactly that: for an arbitrary string, either `DbName::parse`
refuses it or the resulting path is one component directly inside `<root>/db`.

Three rules follow from portability rather than from security, and all three are
enforced on every platform so a workspace directory can be copied between
machines:

- **lowercase only** — Windows and macOS filesystems are case-insensitive, so
  `Work` and `work` would address one database with two names;
- **no dots** — a name is the whole stem, and the sidecar files
  (`.lock`, `.journal`, `.snap.N`) are distinguished by what follows the first
  dot;
- **no Windows device names** (`con`, `prn`, `aux`, `nul`, `com1`–`com9`,
  `lpt1`–`lpt9`) — Windows resolves those as devices in every directory *and
  with any extension appended*, so `con.plugmem` opens the console. The alphabet
  already excludes `CONIN$` and the superscript `COM¹` forms.

## The directory is the truth; the registry is an index

`WorkspaceLayout::list` reads the filesystem, never the registry.

Each memory **describes itself**, in a fact on the reserved entity
`plugmem workspace self` inside it. The registry is therefore derivable, and
`reindex` derives it. The consequences are the design:

- delete the registry and *search* stops working. Nothing else does;
- copy, move or delete a memory file and its description travels inside it;
- the registry may be wrong — it is a cache — and `verify` reports how, without
  repairing anything.

A registry that were the source of truth would instead have four ways to
disagree with the disk, and each would lose data rather than lose search.

`reindex` cannot read a memory another process holds open (one database, one
writer), so it names those in its report rather than skipping them silently: the
rebuilt registry is knowingly incomplete. The normal path — `describe` keeping
the registry current — has no such limit.

The reserved self-description entity is a *reservation*: a caller that gives one
of its own facts that exact subject will be taken for a self-description, and
`verify` reports the ambiguity rather than guessing.

## What goes where in a registry record

Metadata is stored opaquely by the engine and **is not searchable**, so it holds
only what must be read back, never what must be found.

| field | where | why |
|---|---|---|
| description | the fact's text | this is what search matches |
| tags | tags | this is what filters |
| owner | an `owned-by` edge, and metadata | the edge answers "everything Ann owns"; the metadata reads it back exactly |
| name | metadata, and the record's subject entity | only ever returned — and the entity makes lookup by name an anchor, not a scan |

`find` passes its query as both text and a graph anchor, so a person's name
finds what they own even though an owner is an edge. Nothing special-cases
owners: the lexical and graph sources do what they already do, fused.

A record is revised, never duplicated, so the history of what a memory used to
be for is kept. The revision has a new fact id — which is why a memory's
identity is its name and never an id.

## The handle pool

`Workspace` keeps at most `max_open` databases open, evicting the least recently
used, and closes anything idle for `idle_timeout_ms`.

The idle timeout is a **liveness** setting, not a memory setting. An open writer
holds the file's exclusive lock, so a long-running server that never let go
would make its memories permanently unreachable from the CLI. `max_open` has a
derived ceiling (`MAX_OPEN_CEILING`): it can arrive from a config file, and a
process that runs out of file descriptors fails at some unrelated `open`
elsewhere, which is the worst place to learn about it.

Two consequences are documented rather than hidden:

- the pool lock is held across an open, so two callers cannot race to open the
  same file and have the loser told it is busy by its own process. An open never
  waits on the file lock, so the cost is a short queue;
- a handle handed out **outlives its pool entry**. `Database` is an `Arc`;
  eviction drops one clone and the lock goes when the last does. A caller that
  parks a handle for hours keeps that memory locked for hours.

## Surfaces

### MCP: three startup shapes, three schemas

| started with | `db` in the tool schema |
|---|---|
| `--db FILE` | absent entirely — today's behaviour, unchanged |
| `--workspace DIR --db NAME` | present, optional, defaulted |
| `--workspace DIR` | present, required, listed first |

There is **no verb to switch memory**. With a worker pool that would be shared
mutable state: worker A switches to X while B switches to Y, A reads and gets Y
— a race that reproduces once in a hundred calls and writes to the wrong
person's memory.

That the argument *disappears* in the single-database case is deliberate. In MCP
the model fills tool arguments, so a `db` field is a decision the model makes on
every call, while the process that spawned the server usually knew the answer
for certain. Where the answer is known the question is not asked.

`version`, `about` and `settings_help` never take `db`.

Creation follows the verb: a write to an unused name creates the memory — the
reason a new conversation needs no registration step — while a read of an
unknown name is refused, because such a read is a typo far more often than a new
memory, and an empty answer would hide it. `--no-create` turns even the write
case off.

`--read-only` has no workspace form. A read-only handle pins one immutable
generation, and a pool of pinned snapshots that silently age is worse than not
offering it.

**Permissions are not this project's responsibility**, and the README says so.
`db` comes from the caller, whose own harness sees the call before the server
does. `--allow` is a convenience for a single-tenant process, not a boundary.

### CLI

`--db` takes a name or a path. The rule needs no flag: a name has no separator
and no dot, so `work` is a name and `work.plugmem`, `./work` and `/srv/work` are
paths. With no workspace configured every value is a path, unchanged.

`workspace list/find/describe/archive/reindex/verify/use` are administrative;
everyday use needs none of them. `verify` exits 1 on disagreement so it can gate
a script.

`use` keeps **no state on disk**. It prints the line to `eval`, so the selection
lives in the shell that ran it and one terminal cannot silently redirect
another. It prints a path rather than a name, because that line may be evaluated
in a shell with no workspace configured, where a bare name would mean a file in
the current directory. The line is shell-specific (`export` for `sh`,
`$env:` for PowerShell); `--json` is the portable form.

### napi

`Workspace` is the class mirror of the host type, and `open(name)` returns the
same `Plugmem` class a path-opened memory does — a named memory therefore has
every verb, with no second implementation.

`reindex()` and `verify()` are promises (they read every memory in the
directory). A pool limit arriving from JS is range-checked like one arriving
from a config file, and an idle timeout is checked for finiteness before it is
cast, because a JS number is an `f64`.

The generated `index.d.ts` is type-checked as a gate, under `strict` plus
`isolatedModules` — the settings Vite, esbuild and swc imply. That is what
caught `maintain`'s parameter being an ambient `const enum`, which is valid Rust
and unusable TypeScript; it is declared as the string union the runtime already
accepted.

## Deliberately absent

Each is closed by measurement or by a structural fact, not by taste.

- **No merging results across memories.** Measured: asking only the right memory
  answers 98–99 %; asking all and fusing gives 92–94 %, and 62–66 % when the
  question crosses topics. A control split that preserves word statistics scores
  97–99 %, so the fusion machinery loses nothing — what it loses is the idf
  disagreement between corpora. Three fusion strategies differed by 1–3 %, so
  the algorithm is not the variable. Routing beats merging.
- **No composite ids** (`work/f3`). Nothing merges, so `f3` stays unambiguous.
- **No edges between memories.** `EntityId` is numbered per database; this is
  impossible without a global id space.
- **No shared corpus statistics.** That is the memories giving up independence.
- **No "I am a registry" flag in the format.** A wrapper makes a file a
  registry; the format does not know.
