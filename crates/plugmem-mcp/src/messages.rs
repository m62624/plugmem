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
