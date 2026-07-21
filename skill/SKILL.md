---
name: plugmem
description: >-
  Persistent long-term memory for an agent, embedded in one local file — no
  server, no cloud. Use it to REMEMBER durable facts about the user, the
  project or past decisions (one fact = one statement, with optional entity,
  tags and validity time), and to RECALL them later as a compact ranked block
  ready for the prompt: hybrid retrieval fuses BM25 text search, optional
  embedding vectors, an entity graph and time. When a new fact contradicts an
  old one, the engine surfaces the conflict and YOU decide: revise (it
  changed), keep both (compatible) or forget (it was wrong). Supports "what
  was true then" (as-of) queries and episodic time ranges. Reach for it at the
  start of a task (recall context) and whenever you learn something worth
  keeping across sessions.
---

# plugmem — long-term memory for agents

> **STATUS: skeleton.** The structure and the contract markers below are
> final; the instructional text is a stub to be written out before the first
> release (stage 5, specs/06). Do not treat the prose as complete guidance yet.

plugmem is an embedded memory engine (the SQLite model: a library plus one
snapshot file and a journal — no server). You talk to it through four verbs:

- **remember** — store one durable fact: short text, optional subject entity,
  tags, optional embedding vector, optional `valid_from`.
- **recall** — ask for context: text and/or vector and/or tags/entities/time;
  you get a ranked, token-budgeted block (`rendered`) ready for the prompt.
- **revise** — close an old fact and chain its successor (history survives:
  "lived in Moscow (2023 → 2025)" stays answerable via `as_of`).
- **forget** — tombstone a fact immediately; `maintain` purges it physically.

## When to remember (stub)

Durable facts, preferences, decisions, identity — not ephemera. One fact =
one statement. Entity and tag naming conventions live here (to be written).

## When to recall (stub)

At the start of a task and whenever the past is mentioned. An empty
`rendered` block means "nothing relevant" — just continue.

## The contradiction loop (stub)

`remember` returns `similar` hints. On a conflict decide explicitly:
`revise` (the fact changed), keep both (compatible truths), `forget` (it was
an error). The engine NEVER merges or deletes on its own.

## Temporality (stub)

`valid_from` for "since last month", `as_of` for "what was true then",
`range` for episodic "what happened in March". Worked examples per surface
(CLI commands and MCP calls) land here with stage 5.

<!-- wasm-strip:begin -->

## Run it — verify the engine first (CLI / MCP)

This whole section is CUT from the npm/wasm distribution of this skill (the
wasm package always ships skill and engine from the same release, and has a
single transport). For CLI and MCP the ceremony below is MANDATORY.

### Step 0a — pick your transport (stub)

- **Shell available →** the `plugmem` binary (CLI). Command table: specs/06
  (to be inlined here at stage 5).
- **No shell, but your tools include `plugmem_recall` →** MCP. The server
  also exposes `plugmem_version`, `plugmem_about` and `plugmem_skill`.
- **Neither →** plugmem is not installed here. Say so; do not fabricate
  memories. Install: https://github.com/m62624/plugmem

### Step 0b — smoke-test (stub)

Run one `remember` + `recall` round-trip against a throwaway database and
confirm the recalled block contains the remembered text.

### Step 0c — version check (MANDATORY; on ANY mismatch, STOP)

This skill targets the engine version in the marker below. Read the engine's
version (`plugmem-cli --version` / `plugmem_version`) and print one explicit
comparison line:

```
plugmem version check: skill <marker> vs engine <reported> → OK | MISMATCH
```

<!-- skill-version: 0.1.0 -->

If they differ in ANY way: **stop**, warn the user that skill and engine
describe different functionality, and proceed only on their explicit
confirmation — tagging every result "unverified — version mismatch".

<!-- wasm-strip:end -->
