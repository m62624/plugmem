//! `plugmem-mcp` — a Model Context Protocol server exposing a plugmem memory to
//! AI agents (and any program that can spawn a process and speak JSON over a
//! pipe: Python `subprocess`, Node `child_process`, an agent runner).
//!
//! Transport: stdio, newline-delimited JSON-RPC 2.0 (one message per line).
//! Hand-rolled with `serde_json` (no MCP SDK, no async runtime). It is a
//! **sidecar process, not a daemon**: the host spawns it and talks on
//! stdin/stdout; it listens on no port and serves one memory file for its
//! lifetime. Cross-language / many-reader use is many processes over the file
//! MVCC, not one network server.
//!
//! In a Rust program you do not need this — embed [`plugmem_host`] directly
//! (the engine in your process, like linking SQLite). MCP is the door for
//! agents and other languages.
//!
//! Layout, so protocol, behavior and wording are each editable in isolation:
//! `rpc` owns the JSON-RPC envelope and the stdio loop, `tools` the tool
//! definitions and their execution, and `messages` every model-facing string.

mod messages;
mod rpc;
mod tools;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

/// Environment variable naming the database file (below the `--db` flag).
const ENV_DB: &str = "PLUGMEM_DB";
/// Default database file when neither flag nor env is given.
const DEFAULT_DB: &str = "plugmem.db";

/// Command-line arguments. The host wires these once in its MCP config.
#[derive(Parser)]
#[command(name = "plugmem-mcp", version, about = messages::ABOUT_CLI)]
struct Args {
    /// Memory file to serve (else $PLUGMEM_DB, else ./plugmem.db).
    #[arg(long)]
    db: Option<PathBuf>,
    /// config.toml path (else $PLUGMEM_CONFIG, else the XDG default).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Observe-only: open a shared snapshot of another process's writer. Serves
    /// the read verbs plus `plugmem_generation`/`plugmem_refresh`; write verbs
    /// are refused. Requires a checkpointed database.
    #[arg(long)]
    read_only: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let path = args
        .db
        .or_else(|| std::env::var_os(ENV_DB).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DB));

    // Resolve the engine settings from config.toml + environment, then open the
    // single writer handle for the server's lifetime. A failure here (a bad
    // config, or the file already locked by another writer) is fatal: report to
    // stderr and exit, so the spawning host sees the server did not start.
    let settings = match plugmem_host::Settings::load(args.config.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("plugmem-mcp: {e}");
            return ExitCode::from(2);
        }
    };

    // Read-only: open a shared snapshot of another process's writer and keep the
    // embedder to embed recall queries (the read-only handle has none). Default:
    // open the single writer handle, consuming the settings (embedder included).
    let server = if args.read_only {
        let embedder = settings.embedder;
        match plugmem_host::Database::open_readonly(&path, settings.config) {
            Ok(db) => rpc::Server::Reader(Box::new(tools::ReaderState { db, embedder })),
            Err(e) => {
                eprintln!("plugmem-mcp: {}: {e}", path.display());
                return ExitCode::from(2);
            }
        }
    } else {
        match settings.open(&path) {
            Ok(db) => rpc::Server::Writer(db),
            Err(e) => {
                eprintln!("plugmem-mcp: {}: {e}", path.display());
                return ExitCode::from(2);
            }
        }
    };

    rpc::serve(server);
    ExitCode::SUCCESS
}
