---
name: plugmem
description: >-
  Persistent long-term memory for an agent, embedded in one local database — no
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

plugmem is an embedded memory engine: a library plus a file-backed local
database (snapshot, journal and lock), with no server. Its main verbs are:

- **remember** — store one durable fact: short text, optional subject entity,
  tags, optional embedding vector, optional `valid_from`.
- **guarded remember** — check the normal bounded similarity detector and store
  only if it is clear, with no race between the check and possible write.
  **Give it an entity**, or it has nothing to compare against (below).
- **recall** — ask for context: text and/or vector and/or tags/entities/time;
  you get a ranked, token-budgeted block (`rendered`) ready for the prompt.
- **revise** — close an old fact and chain its successor (history survives:
  "lived in Moscow (2023 → 2025)" stays answerable via `as_of`).
- **forget** — tombstone a fact immediately; `maintain` purges it physically.
  Forgetting several ids at once (CLI `forget <id>…`, MCP `ids:[...]`, or the
  host/napi/Python `forget_many`) runs under one write instead of N — reach for
  it whenever you already have the whole list of ids to drop, rather than
  looping one `forget` call per id.
- **link** — create or update a typed relationship between two entities.
- **unlink** — close a typed relationship for current recall while preserving
  its historical `as_of` interval.
- **list tags** — discover current tags and counts in bounded lexical pages.
- **remove tag** — remove one tag from every current fact by revising those
  facts; current classification changes, facts and historical tags survive.

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

Ordinary `remember` stores the new fact and returns its id **plus any similar or
potentially-conflicting live facts**. Use guarded remember (`remember
--guarded` in the CLI, `guarded:true` in MCP) when finding a candidate must
leave the memory unchanged. Its check and conditional write are one operation,
so concurrent callers cannot both pass a read-only preflight. A blocked result
has no new fact id.

Never use `recall` as a duplicate/conflict check. Recall ranks the best context
available and may return a weak nearest vector; its fused score is neither
cosine similarity nor the detector threshold.

**The detector is scoped to the fact's entity, and a fact with no entity is
compared against nothing.** It looks at that entity's most recent live facts and
at no others, so a guarded write carrying no entity has no candidates and is
always stored. It does not error and it does not warn — it silently behaves like
an ordinary `remember`, which is the one thing a caller reaching for the guard
did not want. Six identical guarded writes with no entity produce six facts; the
same six with `entity` produce one and five blocked. So: name an entity whenever
avoiding a duplicate is the reason you chose the guarded verb, and give related
facts the *same* entity spelling, since two spellings are two candidate sets.

A stored result reports `checked`: `false` means no comparison happened, so the
fact was stored exactly as an ordinary remember would have stored it. Never read
"stored" as "checked and clear" without it.

`similar` hints carry the id, the score and the reason — not the text. Read a
hit's wording with `get <id>` when you want to tell someone what it collided
with.

The engine never merges, revises or deletes on its own — it surfaces the
tension and hands you the choice:

- **revise** (`revise <id> "<new text>"`) — the fact *changed*. The old fact is
  closed and chained to the successor, so history survives: a later
  `recall --as-of <then>` still sees the old value.
- **keep both** — the facts are compatible truths (two preferences, two
  projects). After ordinary remember, do nothing; after a guarded block, repeat
  with ordinary remember to make that choice explicit.
- **forget** (`forget <id>`) — the old fact was simply *wrong*. It is
  tombstoned immediately and physically purged at the next `maintain`.

When in doubt, prefer `revise` over `forget`: it corrects the record without
destroying the "what was true then" answer.

## Discovering and removing tags

Do not guess which tags exist or inspect the general term count: text tokens,
entities, relations and tags share that interner. Use the bounded tag catalogue:

- CLI: `tags [--prefix P] [--cursor C] [--limit N]`, `remove-tag TAG`.
- MCP: `plugmem_tags`, `plugmem_remove_tag`.
- Node: `await db.listTags(options)`, `await db.removeTag(tag)`.
- Python: `db.list_tags(...)`, `db.remove_tag(tag)`; use
  `asyncio.to_thread` from an event loop.
- Rust: `list_tags(TagQuery { .. })`, `remove_tag(now, tag)`.

Pages are sorted by exact, case-sensitive UTF-8 name; default size is 64 and
the hard maximum is 256. Pass the opaque cursor unchanged. If writes make it
stale, restart from the first page. Global removal may touch many facts: it
closes each current version and creates an otherwise-identical successor
without the tag. Never use it when you meant to change only one fact.

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

This is the behaviour that makes plugmem different from a store that would just
overwrite the fact, so it is worth seeing once:

```
remember "kim lives in Moscow" --entity kim     # → fact 0
# ... time passes; note the instant, then correct the record ...
revise 0 "kim lives in Berlin" --entity kim     # → fact 1

recall --entity kim
- [f1] kim: kim lives in Berlin (2026-08; active)

recall --entity kim --as-of <the instant between them>
- [f0] kim: kim lives in Moscow (2026-08 → 2026-08; closed)

recall --entity kim --closed
- [f0] kim: kim lives in Moscow (2026-08 → 2026-08; closed)
- [f1] kim: kim lives in Berlin (2026-08; active)
```

`revise` **closed** the first fact's interval instead of deleting it, which is
why the middle query has something to answer with. `forget` would have destroyed
that answer — which is exactly when to prefer one over the other.

One trap worth knowing: `--as-of` moves **both** clocks. A fact answers only if
it was valid at that instant *and* had already been recorded by then. So an
`--as-of` earlier than a fact's `recorded_at` sees nothing — the memory
genuinely knew nothing then, and answering with today's knowledge would be the
wrong answer to "what did I hold".

## Sizing recalled context

- **`--token-budget <n>`** (default 512) caps the block. Lower it when you are
  `-k` caps facts; this caps rendered text.
- **`--ef <n>`** widens the vector search beam (default: the configured
  `hnsw_ef_search`). Higher is more accurate and slower. It does nothing until
  the database is past the flat-search threshold, so reach for it only on a
  large memory whose vector answers look thin.

## The entity graph — and it moves with time too

Facts are joined by **typed edges between entities**: `ann —hires→ bob`. That is
not decoration. When a recall anchors on an entity, the graph source walks its
edges and pulls in the neighbours' facts, so asking about `ann` answers with
what is known about `bob` as well:

```
remember "ann hired bob in march" --entity ann      # → fact 0
link ann hires bob --provenance 0
remember "bob owns the billing service" --entity bob

recall --entity ann
- [f0] ann: ann hired bob in march (2026-08; active)
- [f1] bob: bob owns the billing service (2026-08; active)
- links: ann —hires→ bob
```

Fact 1 never mentions ann. It came back because the edge led there — that is the
whole point of linking, and why reusing the same entity spelling matters.

**Edges are temporal in the same way facts are.** `unlink` closes a relationship
rather than deleting it, so current recall stops using it while `--as-of` still
walks the graph *as it stood then*:

```
unlink ann hires bob

recall --entity ann
- [f0] ann: ann hired bob in march (2026-08; active)

recall --entity ann --as-of <an instant before the unlink>
- [f0] ann: ann hired bob in march (2026-08; active)
- [f1] bob: bob owns the billing service (2026-08; active)
- links: ann —hires→ bob
```

So the two axes are not a fact-only feature: "who reported to whom last spring"
is a question this answers, and it is the reason to `unlink` rather than to
forget the fact that stated the relationship.

How far the walk goes is `--graph-depth N` on the recall (`graph_depth` over
MCP, `graphDepth` in Node, `graph_depth` in Python), defaulting to
`[recall].graph_depth`. Reach for it when *this* question wants a different net than the
memory's usual one: `--graph-depth 3` for "everything around ann",
`--graph-depth 0` for "ann's own facts and nothing her neighbours know".

## Saying *why* an edge exists

`link --provenance <fact_id>` records the fact an edge follows from, and graph
recall returns it. Use it whenever the relationship was *stated* somewhere:

```
remember "ann hired bob in march" --entity ann   # → fact 7
link ann hires bob --provenance 7
```

A later reader (you, next week) can then answer "why does the memory think ann
hires bob" instead of trusting a bare edge. An edge without it is fine — it just
has no citation.

<!-- wasm-strip:begin -->

## Run it — verify the engine first (CLI / MCP)

This whole section is CUT from the in-process library distributions of this
skill — the npm (napi) package and the PyPI (PyO3) one: each ships the skill
and the engine from the same release and calls the engine in your own process,
so there is no separate binary to find and no version skew to check. For the CLI and MCP surfaces — where the
skill and the engine are installed independently — the ceremony below is
MANDATORY.

### Step 0a — pick your transport

- **Shell available →** the **`plugmem-cli`** binary (CLI). One local database
  per memory, addressed with `--db <path>`. The verbs and their flags:

  | Verb | Shape |
  |------|-------|
  | `remember` | `remember "<text>" [--guarded] [--entity E] [--tag T]… [--link REL:ENTITY]… [--meta K=V]… [--valid-from MS] [--vector F32,…]` |
  | `recall` | `recall ["<query>"] [--tag T]… [--entity E]… [--as-of MS] [--range FROM TO] [-k N] [--closed] [--token-budget N] [--ef N] [--graph-depth N] [--vector F32,…]` |
  | `revise` | `revise <id> "<text>" [same flags as remember]` |
  | `forget` | `forget <id>…` (one or more ids, batched under one write) |
  | `tags` | `tags [--prefix P] [--cursor C] [--limit N]` |
  | `remove-tag` | `remove-tag <tag>` |
  | `link` | `link <src> <rel> <dst> [--provenance FACT_ID]` |
  | `unlink` | `unlink <src> <rel> <dst>` |
  | `show` / `stats` | `show <id>` · `stats` |
  | upkeep | `maintain [--mode M]` · `maintain --reembed [--batch-size N]` · `checkpoint` · `verify` · `scrub` · `recover <dst>` |
  | bulk | `export` (JSONL: facts then edges, streamed to stdout) · `import <file> [--batch N]` |

  Add `--json` to any read verb for machine output; `plugmem-cli --version`
  reports the engine version (used in Step 0c).

  `--vector` is for embeddings you already have: given one, nothing is sent to
  the configured embedder and its length must equal `dim`. You will rarely need
  it — omit it and the engine embeds the text itself.

  **Never run `repl`.** It is an interactive session: it reads commands from
  stdin until end-of-input, so an agent that starts it **hangs** — the command
  never returns and the turn is stuck. It exists for a person at a terminal,
  where keeping the engine open between commands is worth it. You are not that
  person: run one verb per invocation. Every one of them is available as a
  one-shot command, so nothing is lost by avoiding it.

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

  **Changing the embedding model.** Never assume a new model is compatible
  because its dimension matches. plugmem records a readable vector-space id
  with the file and ordinary writes refuse a mismatch. After changing the
  configured model, run `maintain --reembed` exactly once; MCP uses
  `plugmem_maintain {"mode":"reembed"}`. It recomputes every retained fact in
  bounded batches and publishes atomically. Plain/automatic `maintain` never
  calls the model. A legacy vector database whose old id was not recorded also
  needs this explicit transition.

  **What `verify` is for.** Opening a database checks that nothing in the file
  can make a read unsafe — it does not check that the graph agrees with itself.
  `verify` is the pass that does: stored text, metadata, the vector mapping,
  and both directions of every edge. It costs a full sweep, so run it when you
  have a reason (a file from elsewhere, a crash, a suspicion), not before every
  session. An open that succeeded is not a clean bill of health; `verify` is.

  **`verify` or `scrub`?** They ask different questions and neither replaces the
  other. `verify` asks whether the *content* agrees with itself. `scrub` asks
  whether the *bytes* are the ones that were written — it recomputes the stored
  checksums, which is what catches a flipped bit that the structure accepts.
  Suspect a bad answer, run `verify`; suspect the disk or a file that travelled,
  run `scrub`. If `scrub` reports damage, `recover <dst>` salvages what survives
  into a **new** file and leaves the original untouched as evidence — the swap
  is the user's decision, not yours.

- **No shell, but your tools include `plugmem_recall` →** MCP. The server
  exposes, as tools: `plugmem_remember`, `plugmem_recall`, `plugmem_revise`,
  `plugmem_forget`, `plugmem_tags`, `plugmem_remove_tag`, `plugmem_link`,
  `plugmem_unlink`, `plugmem_show`, `plugmem_stats`,
  `plugmem_export`, `plugmem_maintain`, `plugmem_checkpoint`, `plugmem_verify`,
  plus `plugmem_version` and `plugmem_about`. A read-only server keeps
  `plugmem_tags`, adds
  `plugmem_generation` / `plugmem_refresh` and refuses the write verbs.
  (`scrub`, `recover` and `import` have no MCP tool. The first two are an
  operator's job on the server's own disk; the third needs a file the server can
  see. Reach for `plugmem-cli` — or tell the user to.) Each tool takes
  `format:"json"` (default) or `"human"`.

  `plugmem_export` returns `{ facts, edges }`. Both halves matter: an edge is a
  statement *between* two entities and belongs to no single fact, so the facts
  alone are not a copy of the memory.

  Arguments that narrow an answer are checked, not guessed: `range` must be
  exactly `[from, to]` and `as_of` / `valid_from` must each be a whole,
  non-negative unix-millisecond number. A malformed one is a tool error rather
  than an answer computed without it. `plugmem_remember`, `plugmem_revise` and
  `plugmem_recall` also take an optional `vector` — a precomputed embedding
  that replaces the configured embedder for that call. `plugmem_remember`
  additionally accepts `guarded:true`: a similarity hit returns `blocked`
  without allocating an id or changing the memory.

- **Neither →** plugmem is not installed here. Say so; do not fabricate
  memories. Install: https://github.com/m62624/plugmem

### Configuration — ask for it only when needed

Every external surface shares the same config model. Request detailed
settings help only when configuration is relevant:

- CLI: `plugmem-cli help settings` (or `plugmem-cli --json help settings`).
- MCP: call `plugmem_settings_help` with `format:"json"` or `format:"human"`.
- Node: call the exported `settingsHelp()`; Python: `settings_help()`.

The config file itself resolves as `--config` (or the `config` option passed to
`open`), then `$PLUGMEM_CONFIG`, then the platform config directory.
Database-path precedence is an explicit `--db` or `open` path, then
`$PLUGMEM_DB`, then `[database].path`, then the platform data directory.

The common shape is below. It is **not** the whole catalogue — there are around
forty keys, and the settings-help call above prints every one of them with its
default, read out of the running build rather than out of this document. Two
files carry the same thing for a person to read:
https://github.com/m62624/plugmem/blob/main/config.example.toml (every key,
commented out, ready to copy) and
https://github.com/m62624/plugmem/blob/main/crates/plugmem-host/SETTINGS.md
(what each one is for).

```toml
[database]
path = "/path/to/memory.plugmem" # optional

[engine]
dim = 0                         # 0 keeps vectors disabled

[recall]                        # optional: every key has a tuned default
w_vec = 1.0                     # per-source weights, 0 turns a source off
half_life_days = 180            # how fast an old fact loses ground

[index]                         # optional
flat_to_hnsw = 24000            # vectors before the graph index is built

[embedder]
# Optional: omit the section for lexical/tags/graph/time only.
# The one client is OpenAiCompatEmbedder; url is the complete embeddings
# endpoint (no path is appended), and model selects the server-side model.
enabled = false                # keep settings but do not create/use the embedder
# To enable it, set [engine].dim > 0, change this to true (or omit it), and add:
# url = "http://localhost:11434/v1/embeddings"
# model = "nomic-embed-text"
# space_id = "nomic-embed-text@v1" # optional; defaults to model, never probed
# api_key_env = "OPENAI_API_KEY" # env var containing the bearer token
# on_error = "degrade"           # keep working when the provider is unreachable
# timeout_ms = 10000             # one request, end to end; 0 waits forever
# retry_after_ms = 0             # a suspended embedder: 0 = only on request,
#                                # unset = 1s doubling to retry_max_ms (60s)

[maintenance]
snapshot_every_ops = 1024
snapshot_journal_bytes = 4194304
```

`space_id` is the persisted semantic identity of the vectors. It defaults to
`model`; set it to an exact revision or digest when the provider model is an
alias. Plugmem trusts the value and never probes the provider to discover it.
Changing it for an existing vector database requires an explicit reembed;
routine and automatic maintenance never calls the model.

### When the provider is simply unreachable

A different failure from the one below, and the more common one: the endpoint
refuses the connection, times out, or answers with something else. By default
that fails the verb — `remember` and a text `recall` both embed, so a stopped
Ollama takes every write and every meaning-based read with it.

`[embedder].on_error = "degrade"` carries on without the vector instead: the
fact is stored, the query is answered from the lexical, tag, graph and time
sources, and the embedder is suspended so the next call does not pay the same
failure again. It retries by itself (1s, doubling to a minute, reset by the
first success), and `reembed` fills in the missing vectors from the stored text
once the provider is back. Nothing needs reopening.

For a consumer, three things are worth knowing:

- `embedder_state()` / `embedderState()` answers `absent`, `active` or
  `suspended` — the honest way to tell a person "your memory is running
  without meaning-based ranking right now". The MCP server puts the same word
  in `plugmem_stats`.
- `suspend_embedder()` / `resume_embedder()` are the manual switches, for when
  the caller already knows the provider is gone.
- `vectors < facts` in `stats` is the durable trace of a degraded stretch, and
  the reason to run a reembed on the next start with a working provider.

These four keys have environment overrides, which is the form that fits an
outage — nobody wants to edit a config file to get through one:
`$PLUGMEM_EMBEDDER_ON_ERROR`, `$PLUGMEM_EMBEDDER_TIMEOUT_MS`,
`$PLUGMEM_EMBEDDER_RETRY_AFTER_MS`, `$PLUGMEM_EMBEDDER_RETRY_MAX_MS` (and
`$PLUGMEM_EMBEDDER_ENABLED`, which predates them). `url`, `model`, `space_id`
and `api_key_env` are file-only. The environment wins over the file, the file
over the defaults.

### What a mismatched vector space actually does

Worth knowing exactly, because the shape of the failure decides how a consumer
should handle it, and guessing produces either needless panic or silent data
rot. Changing `model`, `space_id` or `dim` on a database that already holds
vectors does **not** stop it opening, and loses nothing:

| still works | fails, loudly |
|---|---|
| open (writer and read-only), `stats`, `get`, `tags_of`, `list_tags` | a **recall with text** |
| entity/graph recall, export/scan, `forget`, `link`, `unlink` | a **remember with text** |
| `verify`, `maintain`, `checkpoint`, `reembed` | |

So the content is intact and recovery is always possible — which is the reason
`open` is deliberately not the place this fails. What it costs is exactly the
two verbs a text-driven memory is built on, and a consumer only finds out at
the first lookup after the change. Detect it by making the cheapest text recall
and watching for the mismatch error, not by remembering what you configured
last time: a bookkeeping file drifts the moment somebody edits the config,
restores a backup, or copies a database between machines — and it drifts
towards "everything is fine".

The repair is `reembed`, and it is idempotent: an interrupted one leaves the
facts it finished with their new vectors, so running it again completes the
job. It rebuilds one database, so a workspace needs a pass over every memory in
it — half rebuilt is a workspace answering from two vector spaces at once, and
nothing reports that on its own.

Two more measured details, both easy to assume wrongly:

- `reembed` on an EMPTY database still makes one embedder request, whose input
  is the empty string. A provider that rejects empty input (OpenAI answers 400)
  fails a rebuild that had nothing to rebuild.
- switching the embedder **on** over a database built without one breaks
  nothing and warns about nothing: the old facts simply have no vectors, so
  meaning-based recall answers from a fraction of the memory. Compare
  `stats().vectors` against `stats().facts` to notice it.

`[engine]` is what a database is built with — changing one of those on an
existing memory is refused. `[recall]` and `[index]` are the opposite: reopening
with different weights is how ranking changes, so they are safe to tune on a
live memory.

**Do not tune `[recall]` on your own initiative.** The defaults are tuned, and a
weight you moved because one answer looked wrong will quietly change every
answer afterwards. Change it when the user asks for it, or when you can say
which source is misbehaving and why — then say what you changed.

Do not invent provider URLs, model names or paths. Ask for settings help when
the user needs to configure them, and otherwise rely on the platform default.

An unknown key is reported, not applied: the CLI and the MCP server print it to
stderr, Node returns it from `configWarnings()` and Python from
`config_warnings()`. If you see one, the setting did nothing — fix the spelling
rather than assuming it took effect.

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

<!-- skill-version: 0.12.0 -->

If they differ in ANY way: **stop**, warn the user that skill and engine
describe different functionality, and proceed only on their explicit
confirmation — tagging every result "unverified — version mismatch".

## Several memories (only if you see a `db` argument)

**Normally there is one memory and you never think about this.** No `db`
argument on the tools, no choice to make, nothing in this section applies. That
is the ordinary setup and the one you should assume.

**The signal is the schema, not this text.** If `plugmem_remember` and the rest
carry a `db` argument, this server holds several memories and you have to say
which one each call is for. If they do not, skip the rest of this section.

Two shapes, and the schema tells you which:

- **`db` is optional and shows a default.** One memory is this session's own —
  almost always the right one. Omit `db`. Name another only when the knowledge
  plainly belongs elsewhere (shared team knowledge going to a common memory).
- **`db` is required.** Nothing is implied; every call must name a memory.

### Picking a name

- **You were told the name** (the harness passed it, the user said it) → use it.
- **You were not** → call `plugmem_workspace_find` with a description of what
  you are looking for ("the chat about releases", "Ann's notes"). It returns
  names. A person's name works too — owners are searchable even though they are
  not in the description text. `plugmem_workspace_list` shows everything when
  there are few enough to read.
- **Never invent a name to guess with.** Writing to a name nobody has used
  *creates* a memory, so a guess does not fail loudly — it silently starts an
  empty one, and the knowledge you meant to file is now somewhere nobody looks.

### Which memory something belongs in

- Knowledge about this conversation → this conversation's memory.
- Knowledge true for everyone → the shared memory (usually `common`), if the
  workspace has one. Do not copy it into every conversation.
- **Do not spread one fact across memories.** They are independent; there is no
  search that spans them, so a fact filed in the wrong one is lost, not merely
  misplaced.

Names are lowercase letters, digits, `-` and `_`. A name is never a path.

On the CLI the same thing is `--db <name>` instead of `--db <file>`, plus a
`plugmem-cli workspace` command group (`list`, `find`, `describe`, `archive`,
`verify`, `reindex`). Ask for `plugmem-cli workspace --help` if you need it.

<!-- wasm-strip:end -->
