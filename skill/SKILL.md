---
name: plugmem
description: >-
  Persistent long-term memory for an agent, embedded in one local file — no
  server, no cloud. Use it to REMEMBER durable facts about the user, the
  project or past decisions (one fact = one statement, with optional entity,
  tags and validity time), and to RECALL them later as a compact ranked block
  ready for the prompt: hybrid retrieval fuses BM25 text search, optional
  embedding vectors, an entity graph and time. Link entities with typed
  relationships, and unlink relationships when they stop being true without
  destroying their historical as-of answers. When a new fact contradicts an old
  one, the engine surfaces the conflict and YOU decide: revise (it changed),
  keep both (compatible) or forget (it was wrong). Supports "what was true
  then" (as-of) queries and episodic time ranges. Reach for it at the start of
  a task (recall context) and whenever you learn something worth keeping across
  sessions.
---

# plugmem — long-term memory for agents

plugmem is an embedded memory engine (the SQLite model: a library plus one
snapshot file and a journal — no server). You talk to it through these main
verbs:

- **remember** — store one durable fact: short text, optional subject entity,
  tags, optional embedding vector, optional `valid_from`.
- **recall** — ask for context: text and/or vector and/or tags/entities/time;
  you get a ranked, token-budgeted block (`rendered`) ready for the prompt.
- **revise** — close an old fact and chain its successor (history survives:
  "lived in Moscow (2023 → 2025)" stays answerable via `as_of`).
- **forget** — tombstone a fact immediately; `maintain` purges it physically.
- **link** — create or update a typed relationship between two entities.
- **unlink** — close a typed relationship for current recall while preserving
  its historical `as_of` interval.

## When to remember

Store a fact the moment you learn something that should outlive this session —
who the user is, a stated preference, a decision and its reason, a project
constraint, a durable relationship between entities. Skip the ephemera (the
current file, a transient value, anything re-derivable next turn).

- **One fact = one statement.** "Ada uses Postgres and prefers dark mode" is
  two facts, not one — split them so each can be recalled, revised or forgotten
  on its own.
- **Name the subject with `entity`** when the fact is *about* someone/something
  (`--entity ada`); reuse the same spelling every time so the entity graph
  links up. Entities are created lazily on first mention.
- **Tag for filtering, not phrasing** (`--tag prefs`, `--tag decision`). Tags
  are filters in recall (a fact must carry *all* requested tags); they are not
  a ranking source, so don't stuff the query into a tag.
- **Link related entities** with a typed edge (`--link works_at:acme`, or the
  standalone `link ada works_at acme`) — the graph source expands recall from
  an anchor entity to its neighbours. When the relationship stops being true,
  use `unlink ada works_at acme`; current recall stops using the edge, while
  `recall --as-of <then>` can still see the historical relationship.
- **Metadata is an opaque pointer, not content** (`--meta uri=s3://…/doc.pdf
  --meta mime=application/pdf`). The engine stores and returns it verbatim and
  never searches it — use it for a URI to the real payload in another store, or
  side attributes; keep the searchable statement in the text.

## When to recall

Recall at the **start of a task** (pull the standing context before you act)
and **whenever the past is referenced** ("as we discussed", "the usual setup",
a name you've seen before). The result is already ranked and token-budgeted —
paste the block, don't re-rank it yourself.

- The sources **compose**: a free-text `query` drives lexical + vector, `--tag`
  and `--entity` filter/anchor, `--as-of`/`--range` bound time. You can recall
  by tags/entities/time **with no query at all**.
- An **empty block means "nothing relevant"** — just continue; it is not an
  error and not a reason to retry with a looser query unless you have one.
- Each recalled line is `- [fN] text …`; **`N` is the fact id** you pass to
  `show N`, `revise N …` or `forget N`. In JSON the same id is the `"id"` field.

## The contradiction loop — you decide, never the engine

`remember` returns the new id **plus any similar or potentially-conflicting
live facts**. The engine never merges, revises or deletes on its own — it
surfaces the tension and hands you the choice:

- **revise** (`revise <id> "<new text>"`) — the fact *changed*. The old fact is
  closed and chained to the successor, so history survives: a later
  `recall --as-of <then>` still sees the old value.
- **keep both** — the facts are compatible truths (two preferences, two
  projects). Do nothing; both stay live.
- **forget** (`forget <id>`) — the old fact was simply *wrong*. It is
  tombstoned immediately and physically purged at the next `maintain`.

When in doubt, prefer `revise` over `forget`: it corrects the record without
destroying the "what was true then" answer.

## Temporality — two independent time axes

plugmem is bitemporal. Keep the axes straight:

- **`valid_from` / `valid_to` — the truth axis** ("*since* last month", "true
  *until* the reorg"). Set `--valid-from <unix-ms>` on remember/revise when a
  fact became true at a time other than now.
- **`recorded_at` — the knowledge axis** (when *you learned* it). Always
  stamped at write time; you never set it.

Query them separately:

- **`recall --as-of <unix-ms>`** — "what was true *then*": the validity instant.
  A revised fact answers with its old value before the revision, the new value
  after.
- **`recall --range <from> <to>`** — episodic "what did I record in this
  window", over the `recorded_at` axis.
- **`recall --closed`** — include closed revision chains (with their intervals),
  not just the currently-live facts.

<!-- wasm-strip:begin -->

## Run it — verify the engine first (CLI / MCP)

This whole section is CUT from the in-process npm (napi) distribution of this
skill: that package ships the skill and the engine from the same release and
calls the engine directly in the Node process, so there is no separate binary
to find and no version skew to check. For the CLI and MCP surfaces — where the
skill and the engine are installed independently — the ceremony below is
MANDATORY.

### Step 0a — pick your transport

- **Shell available →** the **`plugmem-cli`** binary (CLI). One database file
  per memory, addressed with `--db <file>`. The verbs and their flags:

  | Verb | Shape |
  |------|-------|
  | `remember` | `remember "<text>" [--entity E] [--tag T]… [--link REL:ENTITY]… [--meta K=V]… [--valid-from MS]` |
  | `recall` | `recall ["<query>"] [--tag T]… [--entity E]… [--as-of MS] [--range FROM TO] [-k N] [--closed]` |
  | `revise` | `revise <id> "<text>" [same flags as remember]` |
  | `forget` | `forget <id>` |
  | `link` | `link <src> <rel> <dst>` |
  | `unlink` | `unlink <src> <rel> <dst>` |
  | `show` / `stats` | `show <id>` · `stats` |
  | upkeep | `maintain [--mode M]` · `checkpoint` · `verify` · `scrub` · `recover <dst>` |
  | bulk | `export` (JSONL to stdout) · `import <file> [--batch N]` |
  | session | `repl [--read-only]` — keep the engine open, one command per line |

  Add `--json` to any read verb for machine output; `plugmem-cli --version`
  reports the engine version (used in Step 0c).

  **Which `maintain`.** Plain `maintain` is `--mode auto`: it does only what is
  pending and is a no-op when nothing is. That is the one to run routinely —
  after a batch of `forget`s, or before handing the file to another process.
  Reach for `--mode full` only when you want the file *small*: it rebuilds
  every index and repacks the edge arenas, which is work proportional to the
  whole database. The narrow modes (`compact`, `reindex-text`,
  `optimize-vectors`) exist for when you know exactly which structure you want
  rebuilt. **No mode ever deletes a fact revision or an edge version** — the
  heavier ones buy bytes and index freshness, never less history. The MCP tool
  takes the same values as an optional `mode` argument.

  **What `verify` is for.** Opening a database checks that nothing in the file
  can make a read unsafe — it does not check that the graph agrees with itself.
  `verify` is the pass that does: stored text, metadata, the vector mapping,
  and both directions of every edge. It costs a full sweep, so run it when you
  have a reason (a file from elsewhere, a crash, a suspicion), not before every
  session. An open that succeeded is not a clean bill of health; `verify` is.

- **No shell, but your tools include `plugmem_recall` →** MCP. The server
  exposes, as tools: `plugmem_remember`, `plugmem_recall`, `plugmem_revise`,
  `plugmem_forget`, `plugmem_link`, `plugmem_unlink`, `plugmem_show`, `plugmem_stats`,
  `plugmem_export`, `plugmem_maintain`, `plugmem_checkpoint`, `plugmem_verify`,
  plus `plugmem_version` and `plugmem_about`. A read-only server adds
  `plugmem_generation` / `plugmem_refresh` and refuses the write verbs.
  (`scrub`, `recover` and `import` are CLI-only — there is no MCP tool for
  them.) Each tool takes `format:"json"` (default) or `"human"`.

- **Neither →** plugmem is not installed here. Say so; do not fabricate
  memories. Install: https://github.com/m62624/plugmem

### Configuration — ask for it only when needed

All three external surfaces share the same config model. Request detailed
settings help only when configuration is relevant:

- CLI: `plugmem-cli help settings` (or `plugmem-cli --json help settings`).
- MCP: call `plugmem_settings_help` with `format:"json"` or `format:"human"`.
- NAPI: call the exported `settingsHelp()` function.

The config file itself resolves as `--config`/constructor `config`, then
`$PLUGMEM_CONFIG`, then the platform config directory. Database-path precedence
is an explicit `--db`/constructor path, then `$PLUGMEM_DB`, then
`[database].path`, then the platform data directory.

The common shape is:

```toml
[database]
path = "/path/to/memory.plugmem" # optional

[engine]
dim = 0                         # 0 keeps vectors disabled

[embedder]
kind = "none"                  # or ollama/openai/lmstudio/vllm/llamacpp

[maintenance]
snapshot_every_ops = 1024
snapshot_journal_bytes = 4194304
```

Do not invent provider URLs, model names or paths. Ask for settings help when
the user needs to configure them, and otherwise rely on the platform default.

### Step 0b — smoke-test

Before you trust the engine, prove one `remember` → `recall` round-trip against
a **throwaway** database (never the user's real memory file):

```console
$ plugmem-cli --db /tmp/plugmem-probe.plugmem remember "smoke test marker z%%q"
$ plugmem-cli --db /tmp/plugmem-probe.plugmem recall "smoke test marker"
```

Over MCP, call `plugmem_remember` with a unique marker string, then
`plugmem_recall` for it. Either way, confirm the recalled block contains the
marker you just stored; if it does not, stop and report the engine as
unverified rather than writing real memories through it.

### Step 0c — version check (MANDATORY; on ANY mismatch, STOP)

This skill targets the engine version in the marker below. Read the engine's
version (`plugmem-cli --version` / `plugmem_version`) and print one explicit
comparison line:

```
plugmem version check: skill <marker> vs engine <reported> → OK | MISMATCH
```

<!-- skill-version: 0.3.0 -->

If they differ in ANY way: **stop**, warn the user that skill and engine
describe different functionality, and proceed only on their explicit
confirmation — tagging every result "unverified — version mismatch".

<!-- wasm-strip:end -->
