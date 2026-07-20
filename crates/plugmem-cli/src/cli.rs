//! The command-line surface: the [`Cli`] parser and its [`Command`]s
//! (clap derive). Kept apart from the execution logic so the long help
//! text lives in one place.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// `plugmem` — a temporal memory for LLM agents in a single file.
#[derive(Parser)]
#[command(
    name = "plugmem-cli",
    version,
    about = "Temporal memory for LLM agents — remember, recall, revise, forget over one file.",
    long_about = LONG_ABOUT,
    after_help = AFTER_HELP,
)]
pub(crate) struct Cli {
    /// Database file (default: ./plugmem.db, or $PLUGMEM_DB).
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) db: Option<PathBuf>,

    /// Config file (default: $PLUGMEM_CONFIG, else
    /// $XDG_CONFIG_HOME/plugmem/config.toml). Sections: [engine],
    /// [embedder], [maintenance].
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
plus link / show / stats / maintain / export / import. Recall fuses lexical (BM25), \
vector, entity-graph and temporal evidence into one ranked, token-budgeted block. One \
database is a single snapshot file plus a journal; point --db at it (default ./plugmem.db, \
or $PLUGMEM_DB). Human output by default, --json for tooling. Exit code: 0 ok, 1 not found \
/ database locked, 2 usage or runtime error.";

/// The footer, aimed at an agent that reached the binary without the skill.
const AFTER_HELP: &str = "\
FOR AI AGENTS: you'll get markedly better results with the matching `plugmem` skill loaded \
— it carries the workflow, the remember/recall loop and the examples this binary expects. \
Check that its version matches `plugmem-cli --version`; the skill is attached to every \
release at https://github.com/m62624/plugmem/releases";

#[derive(Subcommand)]
pub(crate) enum Command {
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
        /// Validity start (unix millis); defaults to now.
        #[arg(long = "valid-from", value_name = "TS")]
        valid_from: Option<u64>,
    },
    /// Retrieve a ranked, token-budgeted block; sources compose.
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
        #[arg(long = "valid-from", value_name = "TS")]
        valid_from: Option<u64>,
    },
    /// Tombstone a fact (physically purged at the next `maintain`).
    Forget {
        /// The fact id to forget.
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
    },
    /// Print one fact's full card (text, both time axes, state).
    Show {
        /// The fact id.
        id: u32,
    },
    /// Print engine size counters and identity.
    Stats,
    /// Run a maintenance pass now (purge tombstones, compact, build HNSW).
    Maintain,
    /// Dump the currently-open facts as JSONL (one fact per line) to stdout.
    Export,
    /// Load facts from a JSONL file (as written by `export`), re-remembering
    /// each. Ids and `recorded_at` are not preserved; text/entity/tags/
    /// valid_from are.
    Import {
        /// The JSONL file to read.
        file: PathBuf,
    },
}
