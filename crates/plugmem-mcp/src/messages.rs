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
listens on no port. One process serves one local database.";

/// Shared by every tool description: the `format` argument.
pub const ARG_FORMAT: &str = "Output format. \"json\" (default) returns compact machine JSON; \
\"human\" returns the same data pretty-printed for eyeballing.";

/// `plugmem_stats` tool description.
pub const STATS_TOOL: &str = "Return engine size counters for the memory: facts, entities, terms, current edges, edge \
history versions, vectors, HNSW coverage, ids, and the database uuid. A cheap health/size probe.";

/// `plugmem_version` tool description — the MCP analog of `plugmem-cli --version`.
pub const VERSION_TOOL: &str = "Return the running plugmem version (e.g. \"plugmem 0.1.0\"). Call \
this once up front and compare it to the version your skill targets; if they differ, warn the user \
about the version skew before relying on the other tools.";

/// Shared by the `plugmem_about` tool's *description* and its *returned text*,
/// so the two never drift. Deliberately version-free (`plugmem_version` owns the
/// number) and harness-agnostic (no product names).
pub const ABOUT_TOOL: &str = "plugmem is a bitemporal memory and retrieval engine for local-first \
applications and agents: it remembers facts over time, then recalls relevant ones by fusing \
lexical (BM25), semantic (vector), graph and temporal evidence. You are calling it over MCP, so \
you may be an AI agent: `plugmem_remember` stores a fact, `plugmem_recall` returns ranked facts \
and edges plus an optional bounded rendered block, and the \
`[fN]` id on each recalled line is how you address a fact in `plugmem_revise`/`plugmem_forget`. \
For the full verb loop and worked examples, load the matching plugmem skill or see the project: \
https://github.com/m62624/plugmem";

/// `plugmem_settings_help` tool description.
pub const SETTINGS_HELP_TOOL: &str = "Return the complete config.toml settings catalogue, including \
config-file precedence, the platform default config path, every supported key, its default and \
which surface consumes it. Use this when configuration details are needed; ordinary about/help \
responses stay short.";

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

/// `plugmem_unlink` tool description.
pub const UNLINK_TOOL: &str = "Close the current typed edge `src -rel-> dst` between two \
entities. Historical `as_of` recall can still traverse the edge while it was active. Idempotent.";

/// `plugmem_show` tool description.
pub const SHOW_TOOL: &str = "Return one fact's full card by id — text, both time axes \
(recorded_at, valid_from/valid_to) and state. A missing id is a tool error.";

/// `plugmem_export` tool description.
pub const EXPORT_TOOL: &str = "Dump the whole memory as `{ facts, edges }`: every currently-open \
fact (id, text, entity, tags, metadata, recorded_at, valid_from) and every current edge (src, rel, \
dst, provenance). Both halves matter — an edge is a statement between two entities and belongs to \
no single fact, so facts alone lose the graph. The counterpart of the CLI's JSONL export; useful \
for backup or inspection.";

/// `plugmem_maintain` tool description.
pub const MAINTAIN_TOOL: &str = "Run policy-driven maintenance: no-op when nothing is pending, \
purge tombstoned facts, compact storage, reindex text when needed and advance the vector index. \
Returns a report with purge, byte and maintenance-action counters. Time is the server's wall clock.";

/// The `mode` argument of `plugmem_maintain`.
pub const ARG_MAINTAIN_MODE: &str = "How much work to do. `auto` (the default) does only what is \
pending and is cheap enough to run often. `compact` purges tombstones and compacts storage; \
`reindex-text` rebuilds the text index from the stored text; `optimize-vectors` builds or advances \
the vector graph; `full` rebuilds everything and repacks the edge arenas, which is O(database) \
work. No mode ever drops a fact revision or an edge version.";

/// `plugmem_checkpoint` tool description.
pub const CHECKPOINT_TOOL: &str = "Flush the journal into a fresh snapshot and clear it, leaving \
the database checkpointed (read-ready for other processes). Returns ok.";

/// `plugmem_verify` tool description.
pub const VERIFY_TOOL: &str = "Check the integrity an open defers: fact text is valid UTF-8, \
metadata decodes, the vector↔fact mapping is consistent, and the edge graph agrees with itself. \
A tool error on damage; ok otherwise.";

// ── Argument descriptions ─────────────────────────────────────────────────

pub const ARG_TEXT: &str = "The fact text to store (required). One clear statement.";
pub const ARG_ENTITY: &str = "Subject entity name the fact is about (created lazily on first \
mention). Optional.";
pub const ARG_TAGS: &str = "Verbatim tag strings (≤ 32). In recall they filter (a fact must carry \
all of them); they are not a ranking source.";
pub const ARG_LINKS: &str = "Typed edges from this fact's subject entity, as an array of \
{ \"rel\": \"...\", \"entity\": \"...\" } (≤ 16). Optional.";
pub const ARG_METADATA: &str = "Opaque metadata as a flat object of string values \
(e.g. { \"uri\": \"s3://…\", \"mime\": \"application/pdf\" }). The engine never \
interprets it — a place for a pointer to the real payload in another store, or \
side attributes. Optional.";
/// The `vector` argument of `plugmem_remember` / `plugmem_revise` /
/// `plugmem_recall`.
pub const ARG_VECTOR: &str = "A precomputed embedding as an array of numbers. \
Its length must equal the configured `dim`. Given, it REPLACES the embedder: \
nothing is sent to the provider. Use it when the vector already exists, or when \
the model is not an OpenAI-shaped HTTP endpoint. Omit it and a configured \
embedder produces one from the text.";

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
pub const ARG_TOKEN_BUDGET: &str = "Token budget of the rendered block (default 512). The block \
is what goes into a prompt, so this decides how much of your context window recall may spend.";
pub const ARG_EF: &str = "HNSW beam width for the vector source (default: the configured \
`hnsw_ef_search`). Higher is more accurate and slower. Ignored while the engine is still in the \
flat regime.";
pub const ARG_GRAPH_DEPTH: &str = "How many edges the graph source may follow from an \
anchor entity (default: the configured `graph_depth`). `0` asks for the anchors' own \
facts and no neighbours. Widen it when the question is \"what is known around this\", narrow it \
when you want one entity's own facts and not its neighbourhood.";
pub const ARG_ID: &str = "The fact id (as printed by remember, or the `[fN]` in a recall block).";
pub const ARG_SRC: &str = "Source entity name (created lazily).";
pub const ARG_REL: &str = "Relation term, verbatim (e.g. \"works_at\").";
pub const ARG_DST: &str = "Destination entity name (created lazily).";
pub const ARG_PROVENANCE: &str = "The fact this edge follows from. Recorded on the edge and \
returned by graph recall, so a later reader can answer \"why is this edge here\".";

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

// ── Workspace mode: many databases, one server ────────────────────────────

/// The `db` argument, present only when the server actually serves more than
/// one database (see the `tools` module for the three startup modes).
pub const ARG_DB: &str = "Which memory to use, by name (letters, digits, '-' and '_'). Not a path. \
Ask plugmem_workspace_find when you do not know the name — do not guess one, and do not invent a \
name for knowledge that belongs in an existing memory.";

/// The `db` argument when the server has a default, so it is optional.
pub const ARG_DB_OPTIONAL: &str = "Which memory to use, by name. Optional: omitted, it means this \
server's default memory, which is almost always the right one. Name another only when the \
knowledge plainly belongs elsewhere.";

/// `plugmem_workspace_list` tool description.
pub const WORKSPACE_LIST_TOOL: &str = "List every memory in this workspace with its description, \
tags and owner. Small enough to read in full at a few hundred memories; past that, search with \
plugmem_workspace_find.";

/// `plugmem_workspace_find` tool description.
pub const WORKSPACE_FIND_TOOL: &str = "Find memories by what they are for, in your own words \
(\"the chat about releases\", \"Ann's notes\"), and get their names back. This is how you pick a \
`db` when you do not already know its name. Owners are searchable too, even though they are not \
in the text.";

/// The `query` argument of `plugmem_workspace_find`.
pub const ARG_WORKSPACE_QUERY: &str = "What the memory is for, in your own words. A person's name \
also works — it finds what they own.";

/// Tool-error text when a workspace server with no default is called without a
/// `db` argument.
pub const WORKSPACE_DB_REQUIRED: &str = "this server holds several memories, so every call must \
say which one: pass `db`. Use plugmem_workspace_find to look one up by what it is for, or \
plugmem_workspace_list to see them all.";

/// Startup refusal: `--read-only` and `--workspace` together.
pub const WORKSPACE_READ_ONLY: &str = "--read-only has no workspace form: a read-only handle pins \
one immutable snapshot generation, and a pool of pinned snapshots that silently age is worse than \
not offering it. Serve one memory read-only (--db FILE --read-only), or serve the workspace \
read-write.";
