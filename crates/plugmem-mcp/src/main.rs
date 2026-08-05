//! `plugmem-mcp` — a Model Context Protocol server exposing a plugmem memory to
//! AI agents (and any program that can spawn a process and speak JSON over a
//! pipe: Python `subprocess`, Node `child_process`, an agent runner).
//!
//! Transport: stdio, newline-delimited JSON-RPC 2.0 (one message per line).
//! Hand-rolled with `serde_json` (no MCP SDK, no async runtime). It is a
//! **sidecar process, not a daemon**: the host spawns it and talks on
//! stdin/stdout; it listens on no port and serves one local database for its
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
use std::sync::Arc;

use clap::Parser;

/// Environment variable naming the database file (below the `--db` flag).
const ENV_DB: &str = "PLUGMEM_DB";
/// Environment variable naming the workspace directory (below `--workspace`).
const ENV_WORKSPACE: &str = "PLUGMEM_WORKSPACE";
/// How often the janitor closes databases nobody is using.
const SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
/// MCP-owned config keys. The test below compares this inventory with the
/// shared host help catalogue whenever the parser gains a new key.
const MCP_SETTING_KEYS: &[(&str, &str)] = &[("server", "workers")];

/// Command-line arguments. The host wires these once in its MCP config.
#[derive(Parser)]
#[command(name = "plugmem-mcp", version, about = messages::ABOUT_CLI)]
struct Args {
    /// Memory file to serve (else $PLUGMEM_DB, else the per-user data path).
    /// With --workspace this is a memory *name* instead, and becomes the
    /// default for calls that do not say which memory they mean.
    #[arg(long)]
    db: Option<String>,

    /// Serve a directory of named memories instead of one local database (else
    /// $PLUGMEM_WORKSPACE, else `[workspace].dir`). Without it there is one
    /// memory and the tools have no `db` argument at all — the default, and
    /// the right choice for one process per conversation.
    #[arg(long)]
    workspace: Option<PathBuf>,

    /// Restrict a workspace server to these memory names (repeatable). Absent,
    /// any name in the workspace is servable. This is a convenience for a
    /// single-tenant process, not an access-control boundary: `db` comes from
    /// the caller, whose own harness sees the call before this server does.
    #[arg(long = "allow", value_name = "NAME")]
    allow: Vec<String>,

    /// Refuse to create a memory that does not exist yet. By default a write
    /// to an unused name creates it, which is what lets a new conversation get
    /// a memory without a registration step; reads never create.
    #[arg(long)]
    no_create: bool,
    /// config.toml path (else $PLUGMEM_CONFIG, else the platform default).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Observe-only: open a shared snapshot of another process's writer. Serves
    /// the read verbs plus `plugmem_generation`/`plugmem_refresh`; write verbs
    /// are refused. Requires a checkpointed database.
    #[arg(long)]
    read_only: bool,
    /// Worker threads for concurrent requests (else `[server].workers`, else
    /// half the available cores, at least 1).
    #[arg(long)]
    workers: Option<usize>,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("plugmem-mcp: {message}");
            ExitCode::from(2)
        }
    }
}

/// Resolve settings, pick the mode, serve. Every failure before serving is
/// fatal and reported to stderr, so the host that spawned the server sees it
/// did not start.
fn run(args: Args) -> Result<(), String> {
    let Args {
        db: cli_db,
        workspace,
        allow,
        no_create,
        config,
        read_only,
        workers,
    } = args;
    let env_db = std::env::var_os(ENV_DB).map(|s| s.to_string_lossy().into_owned());
    let use_config_or_default_db = cli_db.is_none() && env_db.is_none();

    // Read config.toml once: the shared loader builds the engine settings; the
    // server reads its own `[server].workers` from the same table.
    let table = plugmem_host::read_config(config.as_deref()).map_err(|e| e.to_string())?;
    let workers = workers.unwrap_or_else(|| resolve_workers(table.as_ref()));
    let settings = plugmem_host::Settings::from_table(table.as_ref()).map_err(|e| e.to_string())?;
    // Anything in config.toml nobody claimed. Stderr is the only place it can
    // go: stdout carries the JSON-RPC framing, and a stray line there would
    // break the protocol rather than inform anyone.
    for warning in &settings.warnings {
        eprintln!("plugmem-mcp: {warning}");
    }

    // Workspace mode is opt-in and stays opt-in: with no flag, no environment
    // variable and no `[workspace].dir`, everything below behaves exactly as it
    // did before workspaces existed — one local database, and no `db` argument anywhere.
    let root = workspace
        .or_else(|| std::env::var_os(ENV_WORKSPACE).map(PathBuf::from))
        .or_else(|| settings.workspace.dir.clone());
    if let Some(root) = root {
        return serve_workspace(
            settings,
            &root,
            cli_db.or(env_db),
            &allow,
            WorkspaceMode {
                read_only,
                create: !no_create,
            },
            workers,
        );
    }

    let path = cli_db
        .map(PathBuf::from)
        .or(env_db.map(PathBuf::from))
        .or_else(|| settings.database_path.clone())
        .or_else(plugmem_host::default_database_path)
        .unwrap_or_else(|| PathBuf::from("plugmem.db"));

    if use_config_or_default_db
        && let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return Err(format!(
            "cannot create database directory {}: {e}",
            parent.display()
        ));
    }

    // Read-only: open a shared snapshot of another process's writer and keep the
    // embedder to embed recall queries (the read-only handle has none). Default:
    // open the single writer handle, consuming the settings (embedder included).
    let shared = if read_only {
        let embedder = settings.embedder;
        let db = plugmem_host::Database::open_readonly(&path, settings.config)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        rpc::Shared::Reader(Arc::new(tools::ReaderShared::new(db, embedder)))
    } else {
        let db = settings
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        rpc::Shared::Writer(db)
    };

    rpc::serve(shared, workers);
    Ok(())
}

/// The two workspace-relevant flags, so the serving path takes what it uses.
struct WorkspaceMode {
    read_only: bool,
    create: bool,
}

/// Serve a directory of named memories.
///
/// `--read-only` has no workspace form: a read-only handle pins one immutable
/// snapshot generation, and "a pool of pinned snapshots that silently age" is a
/// worse thing to offer than nothing. Refusing is the honest answer.
fn serve_workspace(
    settings: plugmem_host::Settings,
    root: &std::path::Path,
    default: Option<String>,
    allow: &[String],
    mode: WorkspaceMode,
    workers: usize,
) -> Result<(), String> {
    if mode.read_only {
        return Err(messages::WORKSPACE_READ_ONLY.to_string());
    }
    let default = default.map(|s| parse_name(&s, "--db")).transpose()?;
    let allowed = allow
        .iter()
        .map(|s| parse_name(s, "--allow"))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(name) = &default
        && !allowed.is_empty()
        && !allowed.contains(name)
    {
        return Err(format!(
            "--db {name} is not in the --allow set, so the default memory could never be served"
        ));
    }

    let workspace = settings
        .open_workspace(root)
        .map_err(|e| format!("{}: {e}", root.display()))?;
    let shared = Arc::new(tools::WorkspaceShared::new(
        workspace,
        default,
        allowed,
        mode.create,
    ));

    // The janitor is what makes the idle timeout real. An open memory holds an
    // exclusive file lock, so a server that never let go would keep every
    // memory it has ever touched unreachable from anything else on the machine.
    // A detached thread is right here: it owns nothing the shutdown path needs,
    // and the process exiting is a perfectly good way to stop sweeping.
    let janitor = Arc::clone(&shared);
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(SWEEP_INTERVAL);
            janitor.workspace().close_idle(now_ms());
        }
    });

    rpc::serve(rpc::Shared::Workspace(shared), workers);
    Ok(())
}

/// Parses a memory name from the command line, naming the flag it came from.
fn parse_name(s: &str, flag: &str) -> Result<plugmem_host::DbName, String> {
    plugmem_host::DbName::parse(s).map_err(|e| format!("{flag}: {e}"))
}

/// Wall-clock now in unix milliseconds.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The worker count: `[server].workers` if a positive integer, else half the
/// available parallelism (at least 1) — leaving cores for the agent, the OS and
/// a local embedder rather than monopolizing the machine.
fn resolve_workers(table: Option<&toml::Table>) -> usize {
    table
        .and_then(|t| t.get(MCP_SETTING_KEYS[0].0))
        .and_then(toml::Value::as_table)
        .and_then(|s| s.get(MCP_SETTING_KEYS[0].1))
        .and_then(toml::Value::as_integer)
        .filter(|n| *n > 0)
        .map(|n| n as usize)
        .unwrap_or_else(default_workers)
}

/// Half the available cores, at least 1.
fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_from_config_else_default() {
        let table: toml::Table = "[server]\nworkers = 3\n".parse().unwrap();
        assert_eq!(resolve_workers(Some(&table)), 3);

        // No `[server]` / no table / non-positive → the default (>= 1).
        let empty: toml::Table = "[engine]\ndim = 8\n".parse().unwrap();
        assert!(resolve_workers(Some(&empty)) >= 1);
        assert!(resolve_workers(None) >= 1);
        let zero: toml::Table = "[server]\nworkers = 0\n".parse().unwrap();
        assert!(resolve_workers(Some(&zero)) >= 1);

        assert!(default_workers() >= 1);
    }

    #[test]
    fn every_mcp_setting_is_documented() {
        let docs = plugmem_host::settings_help().docs();
        let documented: Vec<_> = docs
            .iter()
            .filter(|doc| doc.scope == plugmem_host::SettingScope::Mcp)
            .map(|doc| (doc.section, doc.key))
            .collect();
        assert_eq!(documented.as_slice(), MCP_SETTING_KEYS);
    }
}
