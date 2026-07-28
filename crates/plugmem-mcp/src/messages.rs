//! Every model- and user-facing string the server emits, gathered in one place
//! so the wording can be read and tuned in isolation from the protocol plumbing.
//! `crate::rpc` and `crate::tools` pull their text from here — nothing
//! user-visible is written inline next to the JSON.

/// The MCP protocol version this server speaks (reported at `initialize`).
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// `serverInfo.name` reported at `initialize`.
pub const SERVER_NAME: &str = "plugmem";

/// The `--help` blurb for the binary itself (not a model-facing tool).
pub const ABOUT_CLI: &str = "plugmem MCP server: exposes a plugmem memory to AI agents over stdio \
JSON-RPC. The host spawns this process and talks to it on stdin/stdout; it is not a daemon and \
listens on no port. One process serves one memory file.";

/// Shared by every tool description: the `format` argument.
pub const ARG_FORMAT: &str = "Output format. \"json\" (default) returns compact machine JSON; \
\"human\" returns the same data pretty-printed for eyeballing.";

/// `plugmem_stats` tool description.
pub const STATS_TOOL: &str = "Return engine size counters for the memory: facts, entities, terms, edges, vectors, and the \
database uuid. A cheap health/size probe.";

/// `plugmem_version` tool description — the MCP analog of `plugmem-cli --version`.
pub const VERSION_TOOL: &str = "Return the running plugmem version (e.g. \"plugmem 0.1.0\"). Call \
this once up front and compare it to the version your skill targets; if they differ, warn the user \
about the version skew before relying on the other tools.";

/// Shared by the `plugmem_about` tool's *description* and its *returned text*,
/// so the two never drift. Deliberately version-free (`plugmem_version` owns the
/// number) and harness-agnostic (no product names).
pub const ABOUT_TOOL: &str = "plugmem is a temporal-memory engine for LLM agents: it remembers \
facts over time, then recalls the relevant ones for a prompt by fusing lexical (BM25), semantic \
(vector), graph and temporal evidence. You are calling it over MCP, so you are an AI agent: \
`plugmem_remember` stores a fact, `plugmem_recall` returns a ranked, prompt-ready block, and the \
`[fN]` id on each recalled line is how you address a fact in `plugmem_revise`/`plugmem_forget`. \
For the full verb loop and worked examples, load the matching plugmem skill or see the project: \
https://github.com/m62624/plugmem";

// ── Writer tools ──────────────────────────────────────────────────────────

/// `plugmem_remember` tool description.
pub const REMEMBER_TOOL: &str = "Store a fact in the memory. Returns the new fact's id plus any \
similar or potentially-conflicting live facts (the engine never revises on its own — you judge: \
`plugmem_revise`, keep both, or `plugmem_forget`). Time is the server's wall clock.";

/// `plugmem_recall` tool description.
pub const RECALL_TOOL: &str = "Recall the most relevant facts for a query as a ranked, \
token-budgeted block, fusing lexical (BM25), semantic (vector), graph and temporal evidence. \
`format:\"human\"` returns the prompt-ready block; `\"json\"` (default) returns the structured \
facts + edges. Each fact carries its id — the `[fN]` you pass to revise/forget/show.";

/// `plugmem_revise` tool description.
pub const REVISE_TOOL: &str = "Revise fact `id`: close the old fact and record its successor with \
the new text/flags. Use this to correct a fact while keeping the bitemporal history (a later \
`plugmem_recall --as-of` still sees the old value). Same arguments as remember, plus `id`.";

/// `plugmem_forget` tool description.
pub const FORGET_TOOL: &str = "Tombstone fact `id` (physically purged at the next \
`plugmem_maintain`). Returns whether the fact was live. Idempotent: forgetting an already-gone \
fact is not an error.";

/// `plugmem_link` tool description.
pub const LINK_TOOL: &str = "Upsert a typed edge `src -rel-> dst` between two entities (created \
lazily). Edges feed the graph recall source. Time is the server's wall clock.";

/// `plugmem_show` tool description.
pub const SHOW_TOOL: &str = "Return one fact's full card by id — text, both time axes \
(recorded_at, valid_from/valid_to) and state. A missing id is a tool error.";

/// `plugmem_export` tool description.
pub const EXPORT_TOOL: &str = "Dump every currently-open fact as a JSON array (text, entity, tags, \
recorded_at, valid_from). The counterpart of the CLI's JSONL export; useful for backup or \
inspection.";

/// `plugmem_maintain` tool description.
pub const MAINTAIN_TOOL: &str = "Purge tombstoned facts, compact storage, and build the vector \
index past its threshold. Returns a report (purged count, bytes before/after). Time is the \
server's wall clock.";

/// `plugmem_checkpoint` tool description.
pub const CHECKPOINT_TOOL: &str = "Flush the journal into a fresh snapshot and clear it, leaving \
the database checkpointed (read-ready for other processes). Returns ok.";

/// `plugmem_verify` tool description.
pub const VERIFY_TOOL: &str = "Check content integrity (fact text is valid UTF-8, the vector↔fact \
mapping is consistent). A tool error on damage; ok otherwise.";

// ── Argument descriptions ─────────────────────────────────────────────────

pub const ARG_TEXT: &str = "The fact text to store (required). One clear statement.";
pub const ARG_ENTITY: &str = "Subject entity name the fact is about (created lazily on first \
mention). Optional.";
pub const ARG_TAGS: &str = "Verbatim tag strings (≤ 32). In recall they filter (a fact must carry \
all of them); they are not a ranking source.";
pub const ARG_LINKS: &str = "Typed edges from this fact's subject entity, as an array of \
{ \"rel\": \"...\", \"entity\": \"...\" } (≤ 16). Optional.";
pub const ARG_VALID_FROM: &str =
    "Validity start, unix milliseconds (the truth axis). Defaults to now.";
pub const ARG_QUERY: &str = "Free-text query for the lexical/semantic sources. Optional — a recall \
can filter by tags/entities/time alone.";
pub const ARG_ENTITIES: &str = "Entity anchors for the graph source (relational expansion).";
pub const ARG_AS_OF: &str =
    "Validity instant, unix milliseconds — recall what was true *then*. Defaults to now.";
pub const ARG_RANGE: &str = "`recorded_at` window as [from, to) in unix milliseconds (the knowledge \
axis) for the temporal source.";
pub const ARG_K: &str = "Max facts to return (0 or omitted = 8; hard ceiling 64).";
pub const ARG_CLOSED: &str =
    "Include closed revisions too (whole chains, marked by intervals). Default false.";
pub const ARG_ID: &str = "The fact id (as printed by remember, or the `[fN]` in a recall block).";
pub const ARG_SRC: &str = "Source entity name (created lazily).";
pub const ARG_REL: &str = "Relation term, verbatim (e.g. \"works_at\").";
pub const ARG_DST: &str = "Destination entity name (created lazily).";

// ── Read-only mode: freshness meta ────────────────────────────────────────

/// `plugmem_generation` tool description (read-only mode only).
pub const GENERATION_TOOL: &str = "Return the snapshot generation this read-only session is pinned \
to (a number a writer's checkpoint bumps). A read-only session is frozen at its generation until \
you `plugmem_refresh`.";

/// `plugmem_refresh` tool description (read-only mode only).
pub const REFRESH_TOOL: &str = "Advance this read-only session to the writer's latest published \
checkpoint, if any. Returns whether it moved and the current generation. Cheap: a 24-byte manifest \
read, re-mapping only when the generation grew.";

/// Prefix of the tool-error text when a write verb is called in read-only mode.
pub const READ_ONLY_REFUSAL: &str = "read-only server: this tool is not available (start the server \
without --read-only to write)";
