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
    disable_help_subcommand = true,
)]
pub(crate) struct Cli {
    /// Database file (default: the platform data path, or $PLUGMEM_DB).
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) db: Option<PathBuf>,

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
plus link / show / stats / maintain / checkpoint / export / import, integrity: verify / \
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
    },
    /// Print one fact's full card (text, both time axes, state).
    Show {
        /// The fact id.
        id: u32,
    },
    /// Print engine size counters and identity.
    Stats,
    /// Run a maintenance pass now (no-op, compact, reindex or optimize).
    Maintain,
    /// Flush the journal into a fresh snapshot now and clear it. Leaves the
    /// database checkpointed, so the read-only path (`scrub`, and any
    /// shared-lock open) can proceed without a dirty-journal `NeedsCheckpoint`.
    Checkpoint,
    /// Check content integrity (text UTF-8, vector↔fact consistency) — the
    /// on-demand equivalent of SQLite's `integrity_check`. Exit 2 on damage.
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
    /// Dump the currently-open facts as JSONL (one fact per line) to stdout.
    Export,
    /// Load facts from a JSONL file (as written by `export`), re-remembering
    /// each. Ids and `recorded_at` are not preserved; text/entity/tags/
    /// valid_from are.
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
    /// Interactive session: open the database once and run commands from stdin
    /// (one per line, same grammar as the subcommands), keeping the engine in
    /// memory for native (host) speed instead of reloading per command. Type
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
