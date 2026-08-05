//! The command-line surface: the [`Cli`] parser and its [`Command`]s
//! (clap derive). Kept apart from the execution logic so the long help
//! text lives in one place.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use plugmem_host::MaintenanceMode;

/// `plugmem` — a temporal memory for LLM agents in a single file.
#[derive(Parser)]
#[command(
    name = "plugmem-cli",
    version,
    about = "Temporal memory for LLM agents — remember, recall, revise, forget over one file.",
    long_about = LONG_ABOUT,
    after_help = AFTER_HELP,
    disable_help_subcommand = true,
)]
pub(crate) struct Cli {
    /// Database file (default: the platform data path, or $PLUGMEM_DB). With a
    /// workspace configured this also takes a bare memory *name* — `work` is a
    /// name, `./work.plugmem` is a path.
    #[arg(long, global = true, value_name = "PATH|NAME")]
    pub(crate) db: Option<String>,

    /// Directory of named memories (default: $PLUGMEM_WORKSPACE, else
    /// [workspace].dir). Unset means one database addressed by path, which is
    /// the default and needs nothing configured.
    #[arg(long, global = true, value_name = "DIR")]
    pub(crate) workspace: Option<PathBuf>,

    /// Config file (default: $PLUGMEM_CONFIG, else
    /// $XDG_CONFIG_HOME/plugmem/config.toml). Sections: [database],
    /// [engine], [embedder], [maintenance].
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) config: Option<PathBuf>,

    /// Machine-readable JSON output instead of the human report.
    #[arg(long, global = true)]
    pub(crate) json: bool,

    #[command(subcommand)]
    pub(crate) command: Command,
}

/// The `--help` long description.
const LONG_ABOUT: &str = "\
A local memory an agent talks to in four verbs — remember / recall / revise / forget — \
plus link / unlink / show / stats / maintain / checkpoint / export / import, integrity: verify / \
scrub / recover, and an interactive `repl` (one open handle, host speed). Recall fuses lexical \
(BM25), vector, entity-graph and temporal evidence into one \
ranked, token-budgeted block. One database is a single snapshot file plus a journal; point \
--db at it (default: the platform data path, or $PLUGMEM_DB). Human output by default, --json for \
tooling. Exit code: 0 ok, 1 not found / database locked, 2 usage / runtime error / \
corruption.";

/// The footer, aimed at an agent that reached the binary without the skill.
const AFTER_HELP: &str = "\
FOR AI AGENTS: you'll get markedly better results with the matching `plugmem` skill loaded \
— it carries the workflow, the remember/recall loop and the examples this binary expects. \
Check that its version matches `plugmem-cli --version`; the skill is attached to every \
release at https://github.com/m62624/plugmem/releases";

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Show detailed help for a topic without opening a database.
    Help {
        #[command(subcommand)]
        topic: HelpTopic,
    },
    /// Store a new fact; prints its id and any similar/conflicting facts.
    Remember {
        /// The fact text.
        text: String,
        /// Subject entity (created on first mention).
        #[arg(long)]
        entity: Option<String>,
        /// A tag; repeatable.
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        /// A typed edge `REL:ENTITY` from the subject; repeatable.
        #[arg(long = "link", value_name = "REL:ENTITY")]
        links: Vec<String>,
        /// A metadata entry `KEY=VALUE` (opaque to the engine — a URI, mime,
        /// external key); repeatable, last value wins per key.
        #[arg(long = "meta", value_name = "KEY=VALUE")]
        meta: Vec<String>,
        /// Validity start (unix millis); defaults to now.
        #[arg(long = "valid-from", value_name = "TS")]
        valid_from: Option<u64>,
        /// A precomputed embedding, comma-separated (`0.1,-0.2,…`). Its length
        /// must equal the configured `dim`. Given, it **replaces** the
        /// embedder: nothing is sent to the provider. This is the route for
        /// vectors you already have, or for a model that is not an
        /// OpenAI-shaped HTTP endpoint. Large ones come from a file:
        /// `--vector "$(cat vec.txt)"`.
        #[arg(long, value_name = "F32,…", value_delimiter = ',', num_args = 1..)]
        vector: Vec<f32>,
    },
    /// Retrieve a ranked, token-budgeted block; sources compose. Each line is
    /// `- [fN] text …`, where `N` is the fact's id — pass it to `forget`,
    /// `revise`, or `show` (e.g. `[f3]` → `forget 3`). `--json` carries the
    /// same id as a plain `"id"` field.
    Recall {
        /// Free-text query for the lexical/vector sources.
        query: Option<String>,
        /// Require this tag; repeatable (a fact must carry all).
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        /// Entity anchor for the graph source; repeatable.
        #[arg(long = "entity", value_name = "E")]
        entities: Vec<String>,
        /// "What was true then": validity instant (unix millis).
        #[arg(long = "as-of", value_name = "TS")]
        as_of: Option<u64>,
        /// recorded_at window `FROM TO` (unix millis) for the temporal source.
        #[arg(long, num_args = 2, value_names = ["FROM", "TO"])]
        range: Option<Vec<u64>>,
        /// Max facts (0 = engine default 8, ceiling 64).
        #[arg(short = 'k', long, default_value_t = 0)]
        k: usize,
        /// Include closed revisions (whole chains).
        #[arg(long)]
        closed: bool,
        /// Token budget of the rendered block (default 512). The block is the
        /// part that goes into a prompt, so this is the knob that decides how
        /// much of the context window recall is allowed to spend.
        #[arg(long = "token-budget", value_name = "N")]
        token_budget: Option<usize>,
        /// HNSW beam width for the vector source (default: `hnsw_ef_search`
        /// from the config). Higher is more accurate and slower. Ignored while
        /// the engine is still in the flat regime, below `flat_to_hnsw`.
        #[arg(long, value_name = "N")]
        ef: Option<usize>,
        /// How many edges the graph source may follow from an anchor entity
        /// (default: `graph_depth` from the config). `0` asks for the anchors'
        /// own facts and no neighbours. Use it when this
        /// particular question wants a wider or narrower net than the memory's
        /// usual one.
        #[arg(long = "graph-depth", value_name = "N")]
        graph_depth: Option<u32>,
        /// A precomputed query embedding, comma-separated. Given, it
        /// **replaces** the embedder for this query — nothing is sent to the
        /// provider — and its length must equal the configured `dim`.
        #[arg(long, value_name = "F32,…", value_delimiter = ',', num_args = 1..)]
        vector: Vec<f32>,
    },
    /// Supersede a fact: close the old one, record the successor.
    Revise {
        /// The fact id to revise.
        id: u32,
        /// The new fact text.
        text: String,
        #[arg(long)]
        entity: Option<String>,
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        #[arg(long = "link", value_name = "REL:ENTITY")]
        links: Vec<String>,
        #[arg(long = "meta", value_name = "KEY=VALUE")]
        meta: Vec<String>,
        #[arg(long = "valid-from", value_name = "TS")]
        valid_from: Option<u64>,
        /// A precomputed embedding, comma-separated (`0.1,-0.2,…`). Its length
        /// must equal the configured `dim`. Given, it **replaces** the
        /// embedder: nothing is sent to the provider. This is the route for
        /// vectors you already have, or for a model that is not an
        /// OpenAI-shaped HTTP endpoint. Large ones come from a file:
        /// `--vector "$(cat vec.txt)"`.
        #[arg(long, value_name = "F32,…", value_delimiter = ',', num_args = 1..)]
        vector: Vec<f32>,
    },
    /// Tombstone a fact (physically purged at the next `maintain`).
    Forget {
        /// The fact id to forget — the `N` from a `recall` line `[fN]`, a
        /// `show`, or a `remember`/`revise` confirmation.
        id: u32,
    },
    /// Upsert a typed edge between two entities.
    Link {
        /// Source entity.
        src: String,
        /// Relation.
        rel: String,
        /// Destination entity.
        dst: String,
        /// The fact this edge follows from. Recorded on the edge and returned
        /// by graph recall, so a caller can answer "why is this edge here".
        #[arg(long, value_name = "FACT_ID")]
        provenance: Option<u32>,
    },
    /// Close the current typed edge between two entities.
    Unlink {
        /// Source entity.
        src: String,
        /// Relation.
        rel: String,
        /// Destination entity.
        dst: String,
    },
    /// Print one fact's full card (text, both time axes, state).
    Show {
        /// The fact id.
        id: u32,
    },
    /// Print engine size counters and identity.
    Stats,
    /// Run a maintenance pass now (no-op, compact, reindex or optimize).
    Maintain {
        /// How much work to do. `auto` (the default) does only what is
        /// pending: purge tombstones, refresh a stale text index, and advance
        /// the vector graph within a bounded budget. `full` rebuilds
        /// everything and repacks the edge arenas — offline-grade work, and
        /// the only mode that reclaims edge-history page slack.
        #[arg(long, value_enum, default_value_t = MaintainMode::Auto)]
        mode: MaintainMode,
    },
    /// Flush the journal into a fresh snapshot now and clear it. Leaves the
    /// database checkpointed, so the read-only path (`scrub`, and any
    /// shared-lock open) can proceed without a dirty-journal `NeedsCheckpoint`.
    Checkpoint,
    /// Check the integrity an open defers: text UTF-8, metadata, vector↔fact
    /// consistency, and that the edge graph agrees with itself — the on-demand
    /// equivalent of SQLite's `integrity_check`. Exit 2 on damage.
    Verify,
    /// Scrub the snapshot's byte-level container integrity (per-section and
    /// whole-file checksums), a slice at a time. Requires a checkpointed
    /// database (run `maintain` first if the journal is dirty). Exit 2 on
    /// the first damaged section.
    Scrub,
    /// Salvage a content-corrupt database into a fresh file: drop the facts
    /// that fail `verify`, compact the survivors, and write a clean copy to
    /// DST — the source (`--db`) is left untouched. Disk-first (bounded RAM).
    Recover {
        /// Destination file for the recovered image (must differ from `--db`).
        dst: PathBuf,
    },
    /// Dump the current memory as JSONL to stdout: every open fact, then every
    /// open edge, one per line, each tagged with `kind`.
    ///
    /// Streamed on both halves, so a large memory never has to fit in RAM.
    /// Facts carry text, entity, tags, metadata, `recorded_at` and
    /// `valid_from`; edges carry `src`/`rel`/`dst` and the `provenance` fact,
    /// referenced by its id in *this* database so an import can translate it.
    ///
    /// **Closed revisions and vectors are not in the format.** History does
    /// not survive a round trip, and vectors are recomputed on import if an
    /// embedder is configured. This is a portable knowledge dump, not a
    /// backup — to back a database up, copy its files.
    Export,
    /// Load a JSONL file written by `export`, re-remembering each fact and
    /// re-linking each edge.
    ///
    /// Ids and `recorded_at` are not preserved (a fresh database assigns its
    /// own); text, entity, tags, metadata, `valid_from`, edges and their
    /// provenance are. A line without a `kind` is read as a fact, so files
    /// written before edges were in the format still load.
    Import {
        /// The JSONL file to read.
        file: PathBuf,
        /// Facts per batch write: one embedder round-trip and one journal
        /// fsync per batch, instead of per fact. The file is streamed in
        /// batches of this size (bounded memory / bounded HTTP body). Default
        /// 128; also settable via `[maintenance].batch_size`. Flag > config.
        #[arg(long, value_name = "N")]
        batch: Option<usize>,
    },
    /// Interactive session for a person at a terminal: open the database once
    /// and run commands from stdin (one per line, same grammar as the
    /// subcommands), keeping the engine in memory for native (host) speed
    /// instead of reloading per command. NOT for a script or an agent — it
    /// reads until end-of-input, so a caller that cannot type into it waits
    /// forever; run one verb per invocation instead. Type
    /// `help` for the verb list, `exit`/`quit` (or EOF) to leave; the session
    /// checkpoints on exit. `scrub`/`recover` stay one-shot.
    Repl {
        /// Observe another process's database read-only (a shared, zero-copy
        /// mmap) instead of opening read-write. Only the read verbs run
        /// (recall/show/stats/export/verify), plus two cross-process freshness
        /// verbs: `refresh` (advance to the writer's latest checkpoint) and
        /// `generation` (the pinned snapshot number). Those two exist ONLY for
        /// this mode — a normal writer repl and every one-shot command already
        /// see the newest data (read-your-writes, or a fresh open), so they are
        /// neither needed nor offered there. Requires a checkpointed database
        /// and does not write (no checkpoint on exit).
        #[arg(long)]
        read_only: bool,
    },
    /// Manage a directory of named memories. Only useful once `[workspace].dir`
    /// (or `--workspace`) points somewhere; without one there is a single
    /// database and nothing here applies.
    Workspace {
        #[command(subcommand)]
        command: WorkspaceCommand,
    },
}

/// The `workspace` subcommands.
#[derive(Subcommand)]
pub(crate) enum WorkspaceCommand {
    /// List every memory in the workspace, with its description when it has one.
    List,
    /// Find memories by what they are for — or by who owns them.
    Find {
        /// What the memory is for, in your own words. A person's name works too.
        query: String,
        /// Max results (default 8).
        #[arg(long, value_name = "N")]
        k: Option<usize>,
    },
    /// Say what a memory is for. Written into the memory itself and into the
    /// registry, so the registry can always be rebuilt from the memories.
    /// Creates the memory if it does not exist.
    Describe {
        /// The memory's name.
        name: String,
        /// What it is for.
        text: String,
        /// Tags to filter by (repeatable).
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        /// Who it belongs to.
        #[arg(long)]
        owner: Option<String>,
    },
    /// Label a memory archived. It stays where it is and stays openable.
    Archive {
        /// The memory's name.
        name: String,
    },
    /// Rebuild the registry from the memories' own descriptions.
    Reindex,
    /// Check the registry against the directory. Reports; never repairs.
    Verify,
    /// Print the shell line that points this terminal at a memory:
    /// `eval "$(plugmem-cli workspace use work)"`. It sets $PLUGMEM_DB in the
    /// shell you run it in — deliberately not a file on disk, so one window
    /// cannot silently redirect another.
    Use {
        /// The memory's name.
        name: String,
    },
}

/// The `maintain --mode` values, mirroring [`MaintenanceMode`] one to one.
///
/// A separate enum so the command line owns its own spelling and help text;
/// the engine's variants are not a CLI contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum MaintainMode {
    /// Only pending work: purge tombstones, refresh a stale text index, and
    /// advance the vector graph within a bounded budget. Cheap and safe to
    /// run often; a no-op when nothing is pending.
    Auto,
    /// Physically purge tombstoned facts and compact storage and indexes.
    Compact,
    /// Rebuild the text index by re-reading and re-tokenizing every fact.
    ReindexText,
    /// Build or advance the vector graph without compacting anything else.
    OptimizeVectors,
    /// Rebuild every rebuildable structure, fully optimize vectors, and
    /// repack the edge arenas. O(database) work; no history is ever dropped.
    Full,
}

impl From<MaintainMode> for MaintenanceMode {
    fn from(mode: MaintainMode) -> Self {
        match mode {
            MaintainMode::Auto => Self::Auto,
            MaintainMode::Compact => Self::Compact,
            MaintainMode::ReindexText => Self::ReindexText,
            MaintainMode::OptimizeVectors => Self::OptimizeVectors,
            MaintainMode::Full => Self::Full,
        }
    }
}

/// Detailed help topics that are intentionally separate from ordinary `--help`.
#[derive(Subcommand)]
pub(crate) enum HelpTopic {
    /// Explain config.toml discovery and every supported setting.
    Settings,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn settings_help_is_an_explicit_topic() {
        let cli = Cli::try_parse_from(["plugmem-cli", "help", "settings"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Help {
                topic: HelpTopic::Settings
            }
        ));
    }
}
