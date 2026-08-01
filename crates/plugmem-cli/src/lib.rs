//! `plugmem` — the command-line surface over the
//! [temporal-memory engine](plugmem_core), a thin wrapper around
//! [`plugmem_host::Database`]. Parse the arguments, call one
//! engine verb, render the result — human text by default, `--json` for
//! tooling and agents. No memory logic lives here; that is the engine's.
//!
//! Exit codes: `0` success; `1` a soft miss (the target fact does not
//! exist, or the database is locked by another process); `2` a usage or
//! runtime error. This makes the binary scriptable as a gate.
//!
//! The logic is in this library (not `main.rs`) so it is unit-testable:
//! [`run`] wires argv and the database, and `execute` runs one command
//! against an open [`Database`] into any writer.

mod cli;
mod config;

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use plugmem_host::{
    Database, ExportedFact, FactId, HostError, LinkInput, ReadOnlyDatabase, RecallQuery,
    RecallResult, RememberInput, RememberOutcome, Settings, Stats, VALID_TO_OPEN,
};
use serde_json::json;

use crate::cli::{Cli, Command, HelpTopic};
use crate::config::read_batch_size;

/// Environment variable naming the database file (below the `--db` flag).
pub(crate) const ENV_DB: &str = "PLUGMEM_DB";
/// Last-resort relative database name if the platform data directory is unavailable.
pub(crate) const DEFAULT_DB: &str = "plugmem.db";

/// A failure before or during a command: a runtime engine/host error, or a
/// usage error (a malformed argument the parser could not catch).
#[derive(Debug)]
pub(crate) enum CliError {
    Host(HostError),
    Usage(String),
}

impl From<HostError> for CliError {
    fn from(e: HostError) -> Self {
        CliError::Host(e)
    }
}

/// Wall-clock now in unix milliseconds (the engine keeps no clock).
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Parses argv and runs one command, mapping the result to a process exit
/// code. The binary's `main` is a one-liner over this; the wiring itself is
/// [`run_parsed`], which is unit-testable (only `Cli::parse` is not).
pub fn run() -> ExitCode {
    let stdout = io::stdout();
    ExitCode::from(run_parsed(Cli::parse(), &mut stdout.lock()))
}

/// The testable core of [`run`]: resolve settings and the database path,
/// open the right handle, run the command into `out`, return the exit code
/// (`0` ok, `1` soft miss / locked, `2` error). Errors go to stderr.
fn run_parsed(cli: Cli, out: &mut impl Write) -> u8 {
    if let Command::Help { topic } = &cli.command {
        return execute_help(topic, cli.json, out);
    }

    // Read config.toml once: the shared loader builds engine/embedder/
    // maintenance settings; the CLI reads its own `[maintenance].batch_size`
    // from the same table (used by `import` below).
    let table = match plugmem_host::read_config(cli.config.as_deref()) {
        Ok(t) => t,
        Err(e) => return report_err(&e.into()),
    };
    let cfg_batch_size = read_batch_size(table.as_ref());
    let mut settings = match Settings::from_table(table.as_ref()) {
        Ok(s) => s,
        Err(e) => return report_err(&e.into()),
    };
    let path = resolve_db_path(cli.db.as_deref(), settings.database_path.as_deref());

    // `recover` is a standalone salvage on file paths — it opens the source
    // itself (under an exclusive lock) and writes a fresh destination, so it
    // runs before the normal open. `scrub` is a byte-level container check over
    // a read-only (shared-lock) open, which requires a checkpointed database.
    match &cli.command {
        Command::Recover { dst } => return do_recover(&path, dst, &settings, cli.json, out),
        Command::Scrub => return do_scrub(&path, &settings, cli.json, out),
        // The interactive session opens one handle and reads commands from
        // stdin, so it is dispatched before the per-command open below. The
        // read-only variant observes another process's writer over a shared
        // mmap; the default variant opens the single writer handle.
        Command::Repl { read_only: true } => {
            return run_repl_ro(&path, settings, cli.json, io::stdin().lock(), out);
        }
        Command::Repl { read_only: false } => {
            return run_repl(&path, settings, cli.json, io::stdin().lock(), out);
        }
        _ => {}
    }

    // Read-only commands open the snapshot zero-copy (mmap, shared lock) and
    // coexist with a live writer process (Variant 2 MVCC) — they never take the
    // writer lock. `verify` is a pure content check, so it belongs here too.
    // `recall` embeds its text query *before* the open (mirroring the host's
    // "embed outside the lock" rule) so it can search by vector on the read-only
    // path, which carries no embedder. A dirty (un-checkpointed) journal forbids
    // a read-only open, so those fall through to the read-write path.
    let readonly_ok = matches!(
        &cli.command,
        Command::Show { .. }
            | Command::Stats
            | Command::Export
            | Command::Verify
            | Command::Recall { .. }
    );
    if readonly_ok {
        let recall_vector = match embed_recall_query(&mut settings, &cli.command) {
            Ok(v) => v,
            Err(e) => return report_err(&e),
        };
        match Database::open_readonly(&path, settings.config.clone()) {
            Ok(ro) => {
                return execute_ro(&ro, &cli.command, recall_vector.as_deref(), cli.json, out);
            }
            Err(HostError::Locked { path }) => return report_locked(&path),
            // Any other failure — a missing snapshot (fresh db), a dirty
            // journal (NeedsCheckpoint), or a corrupt image — is handled by
            // the read-write path: it creates/checkpoints, or surfaces the
            // same corruption as a typed error.
            Err(_) => {}
        }
    }

    // `cfg_batch_size` was read from the config table above (before `open`
    // consumes `settings`); the `--batch` flag still wins over it.
    let db = match settings.open(&path) {
        Ok(db) => db,
        Err(HostError::Locked { path }) => return report_locked(&path),
        Err(e) => return report_err(&CliError::Host(e)),
    };
    // Import is dispatched here, not in `execute`: its batch size comes from the
    // `--batch` flag or `[maintenance].batch_size` (flag > config > default).
    if let Command::Import { file, batch } = &cli.command {
        let batch_size = batch
            .or(cfg_batch_size.map(|n| n as usize))
            .unwrap_or(DEFAULT_IMPORT_BATCH)
            .max(1);
        return match do_import(&db, now_ms(), file, batch_size, out) {
            Ok(n) => {
                if cli.json {
                    writeln!(out, "{}", json!({ "imported": n })).ok();
                } else {
                    writeln!(out, "imported {n} facts").ok();
                }
                0
            }
            Err(e) => {
                let _ = out.flush();
                report_err(&e)
            }
        };
    }
    match execute(&db, &cli.command, cli.json, now_ms(), out) {
        Ok(code) => code,
        Err(e) => {
            let _ = out.flush();
            report_err(&e)
        }
    }
}

/// Default facts-per-batch for `import` when neither `--batch` nor
/// `[maintenance].batch_size` is set — safe for provider batch limits.
const DEFAULT_IMPORT_BATCH: usize = 128;

/// Prints an error to stderr and returns its exit code (`2`).
fn report_err(e: &CliError) -> u8 {
    match e {
        CliError::Usage(msg) => eprintln!("plugmem: {msg}"),
        CliError::Host(err) => eprintln!("plugmem: {err}"),
    }
    2
}

/// Prints the locked message and returns its exit code (`1`).
fn report_locked(path: &std::path::Path) -> u8 {
    eprintln!(
        "plugmem: database is locked by another process: {}",
        path.display()
    );
    1
}

/// Database path precedence: `--db` flag > `$PLUGMEM_DB` >
/// `[database].path` > the platform default.
fn resolve_db_path(
    flag: Option<&std::path::Path>,
    config_path: Option<&std::path::Path>,
) -> PathBuf {
    flag.map(PathBuf::from)
        .or_else(|| std::env::var_os(ENV_DB).map(PathBuf::from))
        .or_else(|| config_path.map(PathBuf::from))
        .or_else(plugmem_host::default_database_path)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DB))
}

/// Render the opt-in detailed help topics without reading a config file or
/// opening a database.
fn execute_help(topic: &HelpTopic, json_output: bool, out: &mut impl Write) -> u8 {
    match topic {
        HelpTopic::Settings => {
            if json_output {
                let help = plugmem_host::settings_help();
                let settings: Vec<_> = help
                    .docs()
                    .iter()
                    .map(|doc| {
                        json!({
                            "section": doc.section,
                            "key": doc.key,
                            "type": doc.value_type,
                            "default": doc.default,
                            "description": doc.description,
                            "scope": doc.scope.as_str(),
                        })
                    })
                    .collect();
                let value = json!({
                    "topic": "settings",
                    "config_path_precedence": help.config_path_precedence(),
                    "default_config_path": plugmem_host::default_config_path()
                        .map(|path| path.display().to_string()),
                    "settings": settings,
                });
                writeln!(out, "{value}").ok();
            } else {
                write!(out, "{}", plugmem_host::settings_help().render_human()).ok();
            }
            0
        }
    }
}

/// Runs a read-only command over a zero-copy [`ReadOnlyDatabase`] (mmap,
/// shared lock). Only the commands `run_parsed` routes here appear.
fn execute_ro(
    ro: &ReadOnlyDatabase,
    cmd: &Command,
    recall_vector: Option<&[f32]>,
    json: bool,
    out: &mut impl Write,
) -> u8 {
    match cmd {
        Command::Recall { .. } => {
            match with_recall_query(cmd, now_ms(), recall_vector, |q| ro.recall(q)) {
                Ok(res) => {
                    render_recall(&res, json, out);
                    0
                }
                Err(e) => report_err(&CliError::Host(e)),
            }
        }
        Command::Show { id } => render_show(ro.get(FactId(*id)), *id, json, out),
        Command::Stats => {
            render_stats(&ro.stats(), json, out);
            0
        }
        Command::Export => {
            ro.export_each(|f| write_export_line(out, &f));
            0
        }
        // A clean image returns Ok; corruption is a typed error mapped to exit 2.
        Command::Verify => match ro.verify() {
            Ok(()) => {
                if json {
                    writeln!(out, "{}", json!({ "ok": true })).ok();
                } else {
                    writeln!(out, "integrity ok").ok();
                }
                0
            }
            Err(e) => report_err(&CliError::Host(e)),
        },
        _ => unreachable!("execute_ro only receives read-only commands"),
    }
}

/// Runs one command against an open database, writing the result to `out`.
/// Returns the process exit code (`0` ok, `1` soft miss). Split from
/// [`run`] so tests drive it directly against a temp database.
fn execute(
    db: &Database,
    cmd: &Command,
    json: bool,
    now: u64,
    out: &mut impl Write,
) -> Result<u8, CliError> {
    match cmd {
        Command::Remember {
            text,
            entity,
            tags,
            links,
            meta,
            valid_from,
        } => {
            let outcome = do_remember(db, now, text, entity, tags, links, meta, *valid_from, None)?;
            render_remember(&outcome, json, out);
            Ok(0)
        }
        Command::Revise {
            id,
            text,
            entity,
            tags,
            links,
            meta,
            valid_from,
        } => {
            let outcome = do_remember(
                db,
                now,
                text,
                entity,
                tags,
                links,
                meta,
                *valid_from,
                Some(FactId(*id)),
            )?;
            render_remember(&outcome, json, out);
            Ok(0)
        }
        Command::Recall { .. } => {
            let res = with_recall_query(cmd, now, None, |q| db.recall(q))?;
            render_recall(&res, json, out);
            Ok(0)
        }
        Command::Forget { id } => {
            let fresh = db.forget(now, FactId(*id))?;
            if json {
                writeln!(out, "{}", json!({ "id": id, "forgotten": fresh })).ok();
            } else if fresh {
                writeln!(out, "forgot fact {id}").ok();
            } else {
                writeln!(out, "fact {id} was already gone").ok();
            }
            Ok(0)
        }
        Command::Link { src, rel, dst } => {
            db.link(LinkInput {
                now,
                src,
                rel,
                dst,
                provenance: None,
            })?;
            if json {
                writeln!(out, "{}", json!({ "src": src, "rel": rel, "dst": dst })).ok();
            } else {
                writeln!(out, "linked {src} -{rel}-> {dst}").ok();
            }
            Ok(0)
        }
        Command::Show { id } => Ok(render_show(db.get(FactId(*id)), *id, json, out)),
        Command::Stats => {
            render_stats(&db.stats(), json, out);
            Ok(0)
        }
        Command::Export => {
            db.export_each(|f| write_export_line(out, &f));
            Ok(0)
        }
        Command::Maintain => {
            let report = db.maintain(now)?;
            if json {
                writeln!(
                    out,
                    "{}",
                    json!({
                        "purged": report.purged,
                        "bytes_before": report.bytes_before,
                        "bytes_after": report.bytes_after,
                    })
                )
                .ok();
            } else {
                writeln!(
                    out,
                    "maintained: purged {}, {} -> {} bytes",
                    report.purged, report.bytes_before, report.bytes_after
                )
                .ok();
            }
            Ok(0)
        }
        Command::Checkpoint => {
            db.checkpoint(now)?;
            if json {
                writeln!(out, "{}", json!({ "ok": true })).ok();
            } else {
                writeln!(out, "checkpointed: journal flushed to snapshot").ok();
            }
            Ok(0)
        }
        Command::Verify => {
            // A clean image returns Ok; corruption is a typed error the caller
            // maps to exit 2.
            db.verify()?;
            if json {
                writeln!(out, "{}", json!({ "ok": true })).ok();
            } else {
                writeln!(out, "integrity ok").ok();
            }
            Ok(0)
        }
        // Handled in `run_parsed` (Import needs `settings` for its batch size).
        Command::Scrub
        | Command::Recover { .. }
        | Command::Repl { .. }
        | Command::Import { .. }
        | Command::Help { .. } => {
            unreachable!("this command is dispatched before execute")
        }
    }
}

/// Salvages `src` into a fresh `dst`: `Database::recover` opens
/// the source under an exclusive lock, drops the content-corrupt facts, and
/// writes a clean disk-first copy. The source is left untouched.
fn do_recover(src: &Path, dst: &Path, settings: &Settings, json: bool, out: &mut impl Write) -> u8 {
    match Database::recover(src, dst, settings.config.clone(), now_ms()) {
        Ok(r) => {
            if json {
                writeln!(
                    out,
                    "{}",
                    json!({
                        "kept": r.kept,
                        "dropped_text": r.dropped_text,
                        "dropped_vector": r.dropped_vector,
                        "dropped_metadata": r.dropped_metadata,
                        "dst": dst.display().to_string(),
                    })
                )
                .ok();
            } else {
                writeln!(
                    out,
                    "recovered to {}: kept {}, dropped {} text + {} vector + {} metadata",
                    dst.display(),
                    r.kept,
                    r.dropped_text,
                    r.dropped_vector,
                    r.dropped_metadata
                )
                .ok();
            }
            0
        }
        Err(HostError::Locked { path }) => report_locked(&path),
        Err(e) => report_err(&CliError::Host(e)),
    }
}

/// Runs a byte-level container scrub over a read-only (shared-lock) open. A
/// clean image exits 0; the first damaged section is a typed error (exit 2). A
/// dirty journal forbids the read-only open — the reported `NeedsCheckpoint`
/// tells the caller to run `maintain` first.
fn do_scrub(path: &Path, settings: &Settings, json: bool, out: &mut impl Write) -> u8 {
    let ro = match Database::open_readonly(path, settings.config.clone()) {
        Ok(ro) => ro,
        Err(HostError::Locked { path }) => return report_locked(&path),
        Err(e) => return report_err(&CliError::Host(e)),
    };
    let scrub = match ro.scrub() {
        Ok(s) => s,
        Err(e) => return report_err(&CliError::Host(e)),
    };
    let mut done = 0u64;
    let mut total = 0u64;
    for step in scrub {
        match step {
            Ok(p) => {
                done = p.done_bytes;
                total = p.total_bytes;
            }
            Err(e) => return report_err(&CliError::Host(e)),
        }
    }
    if json {
        writeln!(out, "{}", json!({ "ok": true, "bytes": done })).ok();
    } else {
        writeln!(out, "scrub ok: {done}/{total} bytes verified").ok();
    }
    0
}

/// Parses a single REPL line: the subcommand grammar, with no leading binary
/// name (the line is `recall tokio`, not `plugmem recall tokio`).
#[derive(Parser)]
#[command(
    no_binary_name = true,
    name = "plugmem",
    disable_help_subcommand = true
)]
struct ReplLine {
    #[command(subcommand)]
    command: Command,
}

/// Splits a REPL line into tokens, honoring single/double quotes so
/// `remember "two words"` is one argument. No escape handling — a quote runs to
/// its match or the end of the line.
fn split_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut has = false;
    for c in line.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else {
                    cur.push(c);
                }
            }
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                has = true;
            }
            None if c.is_whitespace() => {
                if has {
                    tokens.push(std::mem::take(&mut cur));
                    has = false;
                }
            }
            None => {
                cur.push(c);
                has = true;
            }
        }
    }
    if has {
        tokens.push(cur);
    }
    tokens
}

/// Runs the interactive session over one open writer handle: read a line, parse
/// it as a subcommand, run it against the in-memory engine, repeat. The engine
/// stays resident, so each command is host-speed (no per-command reload). The
/// session checkpoints on exit, leaving a read-ready file. Prompts and the
/// banner go to stderr so stdout carries only command output.
fn run_repl(
    path: &Path,
    settings: Settings,
    json: bool,
    input: impl BufRead,
    out: &mut impl Write,
) -> u8 {
    let db = match settings.open(path) {
        Ok(db) => db,
        Err(HostError::Locked { path }) => return report_locked(&path),
        Err(e) => return report_err(&CliError::Host(e)),
    };
    eprintln!("plugmem repl — one open handle, host speed. `help` for verbs, `exit` to quit.");
    eprint!("plugmem> ");
    for line in input.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            eprint!("plugmem> ");
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        } else if line == "help" {
            writeln!(
                out,
                "verbs: remember recall revise forget link show stats maintain checkpoint \
                 verify export import  (scrub/recover stay one-shot)  exit"
            )
            .ok();
        } else {
            run_repl_line(&db, line, json, out);
        }
        eprint!("plugmem> ");
    }
    eprintln!();
    // Leave the database checkpointed (read-ready) for the next opener.
    match db.checkpoint(now_ms()) {
        Ok(()) => 0,
        Err(e) => report_err(&CliError::Host(e)),
    }
}

/// Parses and runs one non-meta REPL line, reporting errors to `out` without
/// ending the session.
fn run_repl_line(db: &Database, line: &str, json: bool, out: &mut impl Write) {
    let cmd = match ReplLine::try_parse_from(split_line(line)) {
        Ok(r) => r.command,
        // clap's message (usage / unknown command / `--help`) — print, continue.
        Err(e) => {
            let _ = writeln!(out, "{e}");
            return;
        }
    };
    match &cmd {
        Command::Repl { .. } => {
            let _ = writeln!(out, "already in a repl session");
        }
        Command::Scrub | Command::Recover { .. } => {
            let _ = writeln!(
                out,
                "scrub/recover are one-shot commands; run them outside the repl"
            );
        }
        _ => {
            if let Err(e) = execute(db, &cmd, json, now_ms(), out) {
                let _ = match &e {
                    CliError::Usage(m) => writeln!(out, "plugmem: {m}"),
                    CliError::Host(h) => writeln!(out, "plugmem: {h}"),
                };
            }
        }
    }
}

/// Runs the interactive session **read-only** over one open
/// [`ReadOnlyDatabase`] (a shared, zero-copy mmap): it observes another
/// process's writer at the generation it opened on. Only the read verbs run;
/// writes and one-shot commands are refused. Two extra meta-verbs make the
/// cross-process freshness observable by hand — `generation` prints the pinned
/// snapshot number, and `refresh` advances to the writer's latest published
/// checkpoint (see [`ReadOnlyDatabase::refresh`](plugmem_host::ReadOnlyDatabase::refresh)).
///
/// These two verbs exist **only** in this mode. A normal (writer) `repl` and
/// any one-shot command already see the freshest data — read-your-writes over
/// the overlay, or a fresh open per command — so there is nothing to refresh
/// there. This session never writes: it does not checkpoint on exit.
fn run_repl_ro(
    path: &Path,
    mut settings: Settings,
    json: bool,
    input: impl BufRead,
    out: &mut impl Write,
) -> u8 {
    let mut ro = match Database::open_readonly(path, settings.config.clone()) {
        Ok(ro) => ro,
        Err(HostError::Locked { path }) => return report_locked(&path),
        // A dirty (un-checkpointed) journal, a fresh database with no published
        // generation, or a corrupt image — surfaced as a typed error.
        Err(e) => return report_err(&CliError::Host(e)),
    };
    eprintln!(
        "plugmem repl --read-only — observing generation {} of another process's writer. \
         `help` for verbs, `refresh`/`generation` for cross-process freshness, `exit` to quit.",
        ro.generation()
    );
    eprint!("plugmem(ro)> ");
    for line in input.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            eprint!("plugmem(ro)> ");
            continue;
        }
        match line {
            "exit" | "quit" => break,
            "help" => {
                writeln!(
                    out,
                    "read verbs: recall show stats export verify  \
                     freshness: generation refresh  exit  \
                     (writes and scrub/recover are refused in a read-only session)"
                )
                .ok();
            }
            // Freshness meta-verbs — only meaningful for a read-only observer of
            // another process's writer (a writer repl sees its own writes at once).
            "generation" => {
                let g = ro.generation();
                if json {
                    writeln!(out, "{}", json!({ "generation": g })).ok();
                } else {
                    writeln!(out, "generation {g}").ok();
                }
            }
            "refresh" => match ro.refresh() {
                Ok(advanced) => {
                    let g = ro.generation();
                    if json {
                        writeln!(out, "{}", json!({ "advanced": advanced, "generation": g })).ok();
                    } else if advanced {
                        writeln!(out, "refreshed → generation {g}").ok();
                    } else {
                        writeln!(out, "already current → generation {g}").ok();
                    }
                }
                Err(e) => {
                    writeln!(out, "plugmem: {e}").ok();
                }
            },
            _ => run_repl_ro_line(&ro, &mut settings, line, json, out),
        }
        eprint!("plugmem(ro)> ");
    }
    eprintln!();
    // Read-only: nothing to checkpoint, the writer owns the file.
    0
}

/// Parses and runs one non-meta line of a read-only repl, refusing anything but
/// the read verbs (writes/one-shot are not available without the writer lock).
fn run_repl_ro_line(
    ro: &ReadOnlyDatabase,
    settings: &mut Settings,
    line: &str,
    json: bool,
    out: &mut impl Write,
) {
    let cmd = match ReplLine::try_parse_from(split_line(line)) {
        Ok(r) => r.command,
        Err(e) => {
            let _ = writeln!(out, "{e}");
            return;
        }
    };
    let readable = matches!(
        &cmd,
        Command::Show { .. }
            | Command::Stats
            | Command::Export
            | Command::Verify
            | Command::Recall { .. }
    );
    if !readable {
        let _ = writeln!(
            out,
            "read-only session: only recall/show/stats/export/verify run \
             (plus refresh/generation); writes and one-shot commands need a writer handle"
        );
        return;
    }
    // Embed a text recall query up front, exactly like the one-shot read-only
    // path — the read-only handle carries no embedder of its own.
    let recall_vector = match embed_recall_query(settings, &cmd) {
        Ok(v) => v,
        Err(e) => {
            let _ = match &e {
                CliError::Usage(m) => writeln!(out, "plugmem: {m}"),
                CliError::Host(h) => writeln!(out, "plugmem: {h}"),
            };
            return;
        }
    };
    let _ = execute_ro(ro, &cmd, recall_vector.as_deref(), json, out);
}

/// Embeds a `recall` command's text query into a vector using the configured
/// embedder, so the read-only path (which carries no embedder) can still search
/// by meaning while a writer process holds the database. Returns `None` when the
/// command is not `recall`, carries no query text, or no embedder is configured
/// — recall then falls back to lexical/structural sources. Mirrors the host's
/// "embed before the lock" rule; the embed happens before the open
/// so a locked database only costs the embed on the rare read-write fallback.
fn embed_recall_query(
    settings: &mut Settings,
    cmd: &Command,
) -> Result<Option<Vec<f32>>, CliError> {
    let Command::Recall {
        query: Some(text), ..
    } = cmd
    else {
        return Ok(None);
    };
    let Some(embedder) = settings.embedder.as_mut() else {
        return Ok(None);
    };
    let mut vectors = embedder.embed(&[text.as_str()]).map_err(CliError::Host)?;
    Ok(vectors.pop())
}

/// Builds the [`RecallQuery`] for a `recall` command and passes it to `f`.
/// A closure (not a return) because the query borrows temporary tag/entity
/// slices that must outlive the call. Used by both the read-write and
/// read-only paths.
fn with_recall_query<R>(
    cmd: &Command,
    now: u64,
    override_vector: Option<&[f32]>,
    f: impl FnOnce(RecallQuery<'_>) -> R,
) -> R {
    let Command::Recall {
        query,
        tags,
        entities,
        as_of,
        range,
        k,
        closed,
    } = cmd
    else {
        unreachable!("with_recall_query called on a non-recall command");
    };
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let ent_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
    let range_pair = range.as_ref().map(|v| (v[0], v[1]));
    // `override_vector` is set only on the read-only path, where the CLI has
    // already embedded the text query; the read-write path leaves it `None` and
    // the host embeds inside `recall`.
    let q = RecallQuery {
        now,
        text: query.as_deref(),
        vector: override_vector,
        tags: &tag_refs,
        entities: &ent_refs,
        as_of: *as_of,
        range: range_pair,
        k: *k,
        token_budget: None,
        include_closed: *closed,
        ef: None,
    };
    f(q)
}

/// Renders a recall result — the engine's block (human) or facts + block
/// (JSON).
fn render_recall(res: &RecallResult, json: bool, out: &mut impl Write) {
    if json {
        let facts: Vec<_> = res
            .facts
            .iter()
            .map(|f| {
                json!({
                    "id": f.id.0,
                    "score": f.score,
                    "sources": f.sources,
                    "recorded_at": f.recorded_at,
                    "valid_from": f.valid_from,
                    "valid_to": open_or(f.valid_to),
                })
            })
            .collect();
        writeln!(
            out,
            "{}",
            json!({ "facts": facts, "rendered": res.rendered, "truncated": res.truncated })
        )
        .ok();
    } else if res.rendered.is_empty() {
        writeln!(out, "(nothing recalled)").ok();
    } else {
        writeln!(out, "{}", res.rendered).ok();
    }
}

/// Renders one fact's card. Returns the exit code (`0` found, `1` missing).
fn render_show(
    fact: Option<plugmem_host::FactSnapshot>,
    id: u32,
    json: bool,
    out: &mut impl Write,
) -> u8 {
    let Some(fact) = fact else {
        if json {
            writeln!(out, "{}", json!({ "id": id, "found": false })).ok();
        } else {
            writeln!(out, "fact {id} not found").ok();
        }
        return 1;
    };
    let r = &fact.record;
    if json {
        writeln!(
            out,
            "{}",
            json!({
                "id": r.id.0,
                "text": fact.text,
                "recorded_at": r.recorded_at,
                "valid_from": r.valid_from,
                "valid_to": open_or(r.valid_to),
                "closed": r.is_closed(),
                "tombstone": r.is_tombstone(),
                "revises": (r.revises != FactId::NONE).then_some(r.revises.0),
                "metadata": fact.metadata,
            })
        )
        .ok();
    } else {
        writeln!(out, "fact {}", r.id.0).ok();
        writeln!(out, "  text        {}", fact.text).ok();
        writeln!(out, "  recorded_at {}", r.recorded_at).ok();
        write!(out, "  valid       [{}, ", r.valid_from).ok();
        match r.valid_to {
            VALID_TO_OPEN => writeln!(out, "open)").ok(),
            to => writeln!(out, "{to})").ok(),
        };
        if r.revises != FactId::NONE {
            writeln!(out, "  revises     fact {}", r.revises.0).ok();
        }
        if !fact.metadata.is_empty() {
            let rendered = fact
                .metadata
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "  metadata    {rendered}").ok();
        }
        if r.is_tombstone() {
            writeln!(out, "  state       tombstoned").ok();
        }
    }
    0
}

/// Renders engine size counters.
fn render_stats(s: &Stats, json: bool, out: &mut impl Write) {
    if json {
        writeln!(
            out,
            "{}",
            json!({
                "facts": s.facts,
                "entities": s.entities,
                "terms": s.terms,
                "edges": s.edges,
                "vectors": s.vectors,
                "next_fact": s.next_fact,
                "next_entity": s.next_entity,
                "pool_bytes": s.pool_bytes,
            })
        )
        .ok();
    } else {
        writeln!(out, "facts       {}", s.facts).ok();
        writeln!(out, "entities    {}", s.entities).ok();
        writeln!(out, "terms       {}", s.terms).ok();
        writeln!(out, "edges       {}", s.edges).ok();
        writeln!(out, "vectors     {}", s.vectors).ok();
        writeln!(out, "next_fact   {}", s.next_fact).ok();
        writeln!(out, "pool_bytes  {}", s.pool_bytes).ok();
    }
}

/// Writes one exported fact as a JSONL line. The unit of the streaming export
/// — the same shape with or without `--json` (JSONL is already machine-readable).
fn write_export_line(out: &mut impl Write, f: &ExportedFact) {
    writeln!(
        out,
        "{}",
        json!({
            "text": f.text,
            "entity": f.entity,
            "tags": f.tags,
            "metadata": f.metadata,
            "recorded_at": f.recorded_at,
            "valid_from": f.valid_from,
        })
    )
    .ok();
}

/// Renders a whole slice of exported facts as JSONL (test helper — the runtime
/// path streams via [`write_export_line`]).
#[cfg(test)]
fn render_export(facts: &[ExportedFact], _json: bool, out: &mut impl Write) {
    for f in facts {
        write_export_line(out, f);
    }
}

/// Loads facts from a JSONL file (as written by `export`) in **streamed
/// batches** of `batch_size`: the file is read line-by-line (memory bounded to
/// a batch, not the whole file), and each full batch is one
/// [`remember_many`](Database::remember_many) — one embedder round-trip and one
/// journal fsync, instead of per fact. Returns the count imported. A malformed
/// line is a usage error naming its 1-based number.
fn do_import(
    db: &Database,
    now: u64,
    file: &std::path::Path,
    batch_size: usize,
    _out: &mut impl Write,
) -> Result<usize, CliError> {
    let f = std::fs::File::open(file)
        .map_err(|e| CliError::Usage(format!("reading {}: {e}", file.display())))?;
    let reader = io::BufReader::new(f);
    let mut count = 0usize;
    let mut batch: Vec<ParsedFact> = Vec::with_capacity(batch_size);
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| CliError::Usage(format!("line {}: {e}", i + 1)))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        batch.push(parse_import_line(line, i + 1)?);
        if batch.len() >= batch_size {
            count += flush_import_batch(db, now, &batch)?;
            batch.clear();
        }
    }
    count += flush_import_batch(db, now, &batch)?;
    Ok(count)
}

/// One parsed JSONL fact, owned so a whole batch can be buffered before its
/// `remember_many`.
struct ParsedFact {
    text: String,
    entity: Option<String>,
    tags: Vec<String>,
    metadata: Vec<(String, String)>,
    valid_from: Option<u64>,
}

/// Parses one JSONL line into an owned fact. Bad JSON, or a missing/non-string
/// `text`, is a usage error naming the 1-based line.
fn parse_import_line(line: &str, lineno: usize) -> Result<ParsedFact, CliError> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| CliError::Usage(format!("line {lineno}: {e}")))?;
    let text = v["text"]
        .as_str()
        .ok_or_else(|| CliError::Usage(format!("line {lineno}: missing string \"text\"")))?
        .to_string();
    let entity = v["entity"].as_str().map(String::from);
    let tags = v["tags"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    // Metadata: an object of string values. Keys are sorted (via `BTreeMap`) so
    // the imported pairs are canonical; non-string values are skipped.
    let metadata = v["metadata"]
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default();
    let valid_from = v["valid_from"].as_u64();
    Ok(ParsedFact {
        text,
        entity,
        tags,
        metadata,
        valid_from,
    })
}

/// Writes one batch of parsed facts via `remember_many` (one embed round-trip,
/// one fsync). Returns how many were written; an empty batch is a no-op.
fn flush_import_batch(db: &Database, now: u64, batch: &[ParsedFact]) -> Result<usize, CliError> {
    if batch.is_empty() {
        return Ok(0);
    }
    // Per-fact `&[&str]` tag slices and `&[(&str,&str)]` metadata pairs must
    // outlive the `remember_many` call.
    let tag_refs: Vec<Vec<&str>> = batch
        .iter()
        .map(|p| p.tags.iter().map(String::as_str).collect())
        .collect();
    let meta_refs: Vec<Vec<(&str, &str)>> = batch
        .iter()
        .map(|p| {
            p.metadata
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect()
        })
        .collect();
    let inputs: Vec<RememberInput> = batch
        .iter()
        .zip(&tag_refs)
        .zip(&meta_refs)
        .map(|((p, tags), meta)| RememberInput {
            entity: p.entity.as_deref(),
            tags,
            metadata: (!meta.is_empty()).then_some(meta.as_slice()),
            valid_from: p.valid_from,
            ..RememberInput::text(now, &p.text)
        })
        .collect();
    db.remember_many(inputs)?;
    Ok(batch.len())
}

/// Shared `remember`/`revise` body: build the input and dispatch.
#[allow(clippy::too_many_arguments)]
fn do_remember(
    db: &Database,
    now: u64,
    text: &str,
    entity: &Option<String>,
    tags: &[String],
    links: &[String],
    meta: &[String],
    valid_from: Option<u64>,
    revise: Option<FactId>,
) -> Result<RememberOutcome, CliError> {
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let link_pairs = parse_links(links)?;
    let link_refs: Vec<(&str, &str)> = link_pairs
        .iter()
        .map(|(r, e)| (r.as_str(), e.as_str()))
        .collect();
    // A `BTreeMap` dedups keys (last `--meta` for a key wins) and sorts them;
    // the engine re-canonicalizes regardless, but this keeps the borrowed pairs
    // clean and dup-free.
    let meta_map = parse_meta(meta)?;
    let meta_refs: Vec<(&str, &str)> = meta_map
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let input = RememberInput {
        entity: entity.as_deref(),
        tags: &tag_refs,
        links: &link_refs,
        metadata: (!meta_refs.is_empty()).then_some(meta_refs.as_slice()),
        valid_from,
        ..RememberInput::text(now, text)
    };
    match revise {
        Some(target) => Ok(db.revise(target, input)?),
        None => Ok(db.remember(input)?),
    }
}

/// Parses `--meta KEY=VALUE` strings into a sorted, deduped map (last value per
/// key wins).
fn parse_meta(meta: &[String]) -> Result<BTreeMap<String, String>, CliError> {
    let mut map = BTreeMap::new();
    for s in meta {
        let (k, v) = s
            .split_once('=')
            .filter(|(k, _)| !k.is_empty())
            .ok_or_else(|| CliError::Usage(format!("bad --meta `{s}` — expected KEY=VALUE")))?;
        map.insert(k.to_string(), v.to_string());
    }
    Ok(map)
}

/// Parses `--link REL:ENTITY` strings into `(rel, entity)` pairs.
fn parse_links(links: &[String]) -> Result<Vec<(String, String)>, CliError> {
    links
        .iter()
        .map(|s| {
            s.split_once(':')
                .filter(|(r, e)| !r.is_empty() && !e.is_empty())
                .map(|(r, e)| (r.to_string(), e.to_string()))
                .ok_or_else(|| CliError::Usage(format!("bad --link `{s}` — expected REL:ENTITY")))
        })
        .collect()
}

/// Renders a `remember`/`revise` outcome (shared shape).
fn render_remember(outcome: &RememberOutcome, json: bool, out: &mut impl Write) {
    if json {
        let similar: Vec<_> = outcome
            .similar
            .iter()
            .map(|s| json!({ "id": s.id.0, "score": s.score, "reason": format!("{:?}", s.reason) }))
            .collect();
        writeln!(
            out,
            "{}",
            json!({
                "id": outcome.id.0,
                "entity": outcome.entity.map(|e| e.0),
                "similar": similar,
            })
        )
        .ok();
    } else {
        writeln!(out, "remembered fact {}", outcome.id.0).ok();
        for s in &outcome.similar {
            writeln!(
                out,
                "  ~ similar to fact {} ({:?}, {:.2})",
                s.id.0, s.reason, s.score
            )
            .ok();
        }
    }
}

/// `VALID_TO_OPEN` → JSON `null`, a real bound → the number.
fn open_or(valid_to: u64) -> Option<u64> {
    (valid_to != VALID_TO_OPEN).then_some(valid_to)
}

#[cfg(test)]
mod tests {
    use plugmem_host::Config;

    use super::*;

    /// A stub embedder returning a fixed vector per input — no network.
    struct StubEmbedder;
    impl plugmem_host::Embedder for StubEmbedder {
        fn dim(&self) -> usize {
            3
        }
        fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
            Ok(texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
        }
    }

    fn recall_cmd(query: Option<&str>) -> Command {
        Command::Recall {
            query: query.map(str::to_owned),
            tags: vec![],
            entities: vec![],
            as_of: None,
            range: None,
            k: 0,
            closed: false,
        }
    }

    fn settings_with(embedder: Option<Box<dyn plugmem_host::Embedder>>) -> Settings {
        Settings {
            database_path: None,
            config: Config::default(),
            embedder,
            snapshot_every_ops: None,
            snapshot_journal_bytes: None,
            maintain_every_forgets: None,
        }
    }

    #[test]
    fn embed_recall_query_embeds_recall_text_only_when_an_embedder_is_set() {
        // recall text + embedder → a vector.
        let mut with = settings_with(Some(Box::new(StubEmbedder)));
        assert_eq!(
            embed_recall_query(&mut with, &recall_cmd(Some("tokio"))).unwrap(),
            Some(vec![0.1, 0.2, 0.3])
        );

        // no embedder → None (recall falls back to lexical/structural sources).
        let mut without = settings_with(None);
        assert_eq!(
            embed_recall_query(&mut without, &recall_cmd(Some("tokio"))).unwrap(),
            None
        );

        // recall with no query text → None (nothing to embed).
        let mut with_empty = settings_with(Some(Box::new(StubEmbedder)));
        assert_eq!(
            embed_recall_query(&mut with_empty, &recall_cmd(None)).unwrap(),
            None
        );

        // a non-recall command → None even with an embedder configured.
        let mut with_stats = settings_with(Some(Box::new(StubEmbedder)));
        assert_eq!(
            embed_recall_query(&mut with_stats, &Command::Stats).unwrap(),
            None
        );
    }

    #[test]
    fn split_line_honors_quotes_and_whitespace() {
        assert_eq!(split_line("remember hello"), ["remember", "hello"]);
        assert_eq!(
            split_line(r#"remember "two words" --tag x"#),
            ["remember", "two words", "--tag", "x"]
        );
        assert_eq!(split_line("  recall   'a b'  "), ["recall", "a b"]);
        assert_eq!(split_line(""), Vec::<String>::new());
        // An empty quoted string is a real (empty) argument.
        assert_eq!(split_line(r#"remember """#), ["remember", ""]);
    }

    #[test]
    fn repl_runs_over_one_handle_and_checkpoints_on_exit() {
        let (db, tmp) = TempDb::open();
        let path = tmp.0.join("m.plugmem");
        drop(db); // release the writer lock so run_repl can open it

        let settings = settings_with(None);
        // Multi-word text is quoted, same grammar as the one-shot CLI.
        let script = b"remember \"hello tokio world\"\nrecall tokio\nrevise 0 \"goodbye tokio\"\nbadcmd\nexit\n";
        let mut out = Vec::new();
        let code = run_repl(&path, settings, false, &script[..], &mut out);
        let text = String::from_utf8(out).unwrap();

        assert_eq!(code, 0);
        assert!(text.contains("remembered fact 0"), "{text}");
        assert!(text.contains("tokio"), "{text}");
        // A bad line is reported but does not end the session (revise ran after).
        assert!(text.contains("unrecognized subcommand"), "{text}");

        // Checkpointed on exit → a fresh read-only open sees the data with a
        // clean journal. The revise chain leaves two facts: the closed original
        // and its active successor.
        let ro = Database::open_readonly(&path, Config::default()).unwrap();
        assert_eq!(ro.stats().facts, 2, "original + successor after the revise");
    }

    #[test]
    fn read_only_repl_observes_a_writer_reports_freshness_and_refuses_writes() {
        let (db, tmp) = TempDb::open();
        let path = tmp.0.join("m.plugmem");
        // Seed and publish generation 1, then keep the writer open and live —
        // the read-only repl observes it cross-process (Variant 2 MVCC).
        let mut sink = Vec::new();
        execute(
            &db,
            &remember("seed fact tokio", None, &[]),
            false,
            1_000,
            &mut sink,
        )
        .unwrap();
        db.checkpoint(1_001).unwrap();

        let settings = settings_with(None);
        // A read verb, both freshness verbs, and a write (must be refused).
        let script = b"generation\nstats\nrefresh\nremember \"nope\"\nexit\n";
        let mut out = Vec::new();
        let code = run_repl_ro(&path, settings, false, &script[..], &mut out);
        let text = String::from_utf8(out).unwrap();

        assert_eq!(code, 0);
        assert!(text.contains("generation 1"), "generation verb: {text}");
        assert!(text.contains("fact"), "stats ran: {text}");
        // The writer published nothing after the reader opened, so refresh is a
        // no-op that stays on generation 1.
        assert!(
            text.contains("already current → generation 1"),
            "refresh no-op: {text}"
        );
        // A write verb is refused without ending the session (exit still ran).
        assert!(text.contains("read-only session"), "write refused: {text}");

        // The read-only session never wrote: the writer is still on generation 1
        // with its single seeded fact, untouched by the repl.
        assert_eq!(db.stats().facts, 1);
    }

    #[test]
    fn read_only_repl_refresh_advances_after_the_writer_checkpoints() {
        let (db, tmp) = TempDb::open();
        let path = tmp.0.join("m.plugmem");
        let mut sink = Vec::new();
        execute(&db, &remember("first", None, &[]), false, 1_000, &mut sink).unwrap();
        db.checkpoint(1_001).unwrap();

        // A reader hook that publishes a *new* generation the first time the repl
        // pulls a line, so the subsequent `refresh` deterministically advances —
        // exercising the "refreshed" branch without a background thread.
        struct HookOnFirstRead<'a> {
            script: std::io::Cursor<&'a [u8]>,
            db: &'a Database,
            fired: bool,
        }
        impl std::io::Read for HookOnFirstRead<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if !self.fired {
                    self.fired = true;
                    // Publish generation 2 before the first command is read, so
                    // the reader (opened on gen 1) sees something newer.
                    let mut s = Vec::new();
                    execute(
                        self.db,
                        &remember("second", None, &[]),
                        false,
                        2_000,
                        &mut s,
                    )
                    .unwrap();
                    self.db.checkpoint(2_001).unwrap();
                }
                self.script.read(buf)
            }
        }
        let reader = std::io::BufReader::new(HookOnFirstRead {
            script: std::io::Cursor::new(b"refresh\nstats\nexit\n" as &[u8]),
            db: &db,
            fired: false,
        });

        let mut out = Vec::new();
        let code = run_repl_ro(&path, settings_with(None), false, reader, &mut out);
        let text = String::from_utf8(out).unwrap();

        assert_eq!(code, 0);
        // Opened on gen 1, the writer published gen 2, refresh advanced onto it.
        assert!(text.contains("refreshed → generation 2"), "advance: {text}");
        // And the advanced reader now sees the writer's second fact.
        assert!(text.contains("fact"), "stats after refresh: {text}");
        assert_eq!(db.stats().facts, 2);
    }

    #[test]
    fn read_only_repl_freshness_verbs_emit_json() {
        let (db, tmp) = TempDb::open();
        let path = tmp.0.join("m.plugmem");
        let mut sink = Vec::new();
        execute(&db, &remember("j", None, &[]), false, 1_000, &mut sink).unwrap();
        db.checkpoint(1_001).unwrap();

        let script = b"generation\nrefresh\nexit\n";
        let mut out = Vec::new();
        let code = run_repl_ro(&path, settings_with(None), true, &script[..], &mut out);
        let text = String::from_utf8(out).unwrap();

        assert_eq!(code, 0);
        assert!(
            text.contains(r#""generation":1"#),
            "generation json: {text}"
        );
        assert!(text.contains(r#""advanced":false"#), "refresh json: {text}");
    }

    /// A throwaway database on a unique temp path; removed on drop.
    struct TempDb(PathBuf);
    impl TempDb {
        fn open() -> (Database, Self) {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-cli-{}-{}",
                std::process::id(),
                now_ms_unique()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("m.plugmem");
            let (db, _) = Database::open(&path, Config::default()).unwrap();
            (db, TempDb(dir))
        }
    }
    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A strictly-increasing counter so temp dirs never collide within a run
    /// (the wall clock alone can repeat at millisecond resolution).
    fn now_ms_unique() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        format!("{}-{}", now_ms(), N.fetch_add(1, Ordering::Relaxed))
    }

    fn run_cmd(db: &Database, cmd: &Command, json: bool, now: u64) -> (u8, String) {
        let mut buf = Vec::new();
        let code = execute(db, cmd, json, now, &mut buf).expect("execute");
        (code, String::from_utf8(buf).unwrap())
    }

    fn remember(text: &str, entity: Option<&str>, tags: &[&str]) -> Command {
        Command::Remember {
            text: text.into(),
            entity: entity.map(Into::into),
            tags: tags.iter().map(|t| (*t).into()).collect(),
            links: Vec::new(),
            meta: Vec::new(),
            valid_from: None,
        }
    }

    fn remember_with_meta(text: &str, meta: &[&str]) -> Command {
        Command::Remember {
            text: text.into(),
            entity: None,
            tags: Vec::new(),
            links: Vec::new(),
            meta: meta.iter().map(|m| (*m).into()).collect(),
            valid_from: None,
        }
    }

    #[test]
    fn meta_flag_renders_sorted_in_show_and_export_and_rejects_bad_input() {
        let (db, _t) = TempDb::open();
        // Keys given out of order; last value for a repeated key wins.
        let cmd = remember_with_meta("a scan", &["uri=s3://b/x", "page=2", "page=3"]);
        assert_eq!(run_cmd(&db, &cmd, false, 1_000).0, 0);

        // show (human): sorted `key=value`, last-write-wins on `page`.
        let (_, human) = run_cmd(&db, &Command::Show { id: 0 }, false, 2_000);
        assert!(
            human.contains("metadata    page=3, uri=s3://b/x"),
            "{human}"
        );
        // show (json): a metadata object.
        let (_, jshow) = run_cmd(&db, &Command::Show { id: 0 }, true, 2_000);
        let v: serde_json::Value = serde_json::from_str(&jshow).unwrap();
        assert_eq!(v["metadata"]["page"], "3");
        assert_eq!(v["metadata"]["uri"], "s3://b/x");

        // export: the JSONL line carries the same object.
        let (_, exp) = run_cmd(&db, &Command::Export, false, 2_000);
        let line: serde_json::Value = serde_json::from_str(exp.lines().next().unwrap()).unwrap();
        assert_eq!(line["metadata"]["uri"], "s3://b/x");

        // A `--meta` without `=` is a usage error.
        assert!(matches!(
            parse_meta(&["noequals".to_string()]),
            Err(CliError::Usage(_))
        ));
        assert!(parse_meta(&["=noKey".to_string()]).is_err());
    }

    #[test]
    fn remember_then_recall_human_and_json() {
        let (db, _t) = TempDb::open();
        let (code, out) = run_cmd(
            &db,
            &remember("prefers tokio", Some("user"), &["pref"]),
            false,
            1_000,
        );
        assert_eq!(code, 0);
        assert!(out.starts_with("remembered fact 0"), "{out}");

        // human recall
        let recall = Command::Recall {
            query: Some("tokio".into()),
            tags: Vec::new(),
            entities: Vec::new(),
            as_of: None,
            range: None,
            k: 0,
            closed: false,
        };
        let (code, out) = run_cmd(&db, &recall, false, 2_000);
        assert_eq!(code, 0);
        assert!(out.contains("tokio"), "{out}");

        // json recall
        let (code, out) = run_cmd(&db, &recall, true, 2_000);
        assert_eq!(code, 0);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert!(!v["facts"].as_array().unwrap().is_empty(), "{out}");
    }

    #[test]
    fn recall_empty_is_ok_with_a_note() {
        let (db, _t) = TempDb::open();
        let recall = Command::Recall {
            query: Some("nothing here".into()),
            tags: Vec::new(),
            entities: Vec::new(),
            as_of: None,
            range: None,
            k: 0,
            closed: false,
        };
        let (code, out) = run_cmd(&db, &recall, false, 1_000);
        assert_eq!(code, 0);
        assert!(out.contains("nothing recalled"), "{out}");
    }

    #[test]
    fn revise_closes_the_predecessor_and_conflict_is_surfaced() {
        let (db, _t) = TempDb::open();
        run_cmd(
            &db,
            &remember("lives in Moscow", Some("user"), &[]),
            false,
            1_000,
        );
        // a near-duplicate surfaces a similar hint
        let (_, out) = run_cmd(
            &db,
            &remember("lives in Moscow now", Some("user"), &[]),
            false,
            1_500,
        );
        assert!(out.contains("similar to fact"), "{out}");

        let revise = Command::Revise {
            id: 0,
            text: "lives in Berlin".into(),
            entity: Some("user".into()),
            tags: Vec::new(),
            links: Vec::new(),
            meta: Vec::new(),
            valid_from: None,
        };
        let (code, out) = run_cmd(&db, &revise, false, 2_000);
        assert_eq!(code, 0);
        assert!(out.starts_with("remembered fact"), "{out}");
    }

    #[test]
    fn show_found_and_missing() {
        let (db, _t) = TempDb::open();
        run_cmd(&db, &remember("a note", None, &[]), false, 1_000);

        let (code, out) = run_cmd(&db, &Command::Show { id: 0 }, false, 2_000);
        assert_eq!(code, 0);
        assert!(
            out.contains("a note") && out.contains("recorded_at 1000"),
            "{out}"
        );

        let (code, out) = run_cmd(&db, &Command::Show { id: 999 }, false, 2_000);
        assert_eq!(code, 1, "missing id is a soft miss");
        assert!(out.contains("not found"), "{out}");

        // json card
        let (_, out) = run_cmd(&db, &Command::Show { id: 0 }, true, 2_000);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["text"], "a note");
        assert_eq!(v["valid_to"], serde_json::Value::Null); // open interval
    }

    #[test]
    fn forget_then_maintain_purges() {
        let (db, _t) = TempDb::open();
        run_cmd(&db, &remember("temp", None, &[]), false, 1_000);

        let (code, out) = run_cmd(&db, &Command::Forget { id: 0 }, false, 2_000);
        assert_eq!(code, 0);
        assert!(out.contains("forgot fact 0"), "{out}");
        // second forget is idempotent
        let (_, out) = run_cmd(&db, &Command::Forget { id: 0 }, false, 2_100);
        assert!(out.contains("already gone"), "{out}");

        let (code, out) = run_cmd(&db, &Command::Maintain, false, 3_000);
        assert_eq!(code, 0);
        assert!(out.contains("purged 1"), "{out}");
    }

    #[test]
    fn link_and_stats_and_json() {
        let (db, _t) = TempDb::open();
        run_cmd(
            &db,
            &remember("uses tokio", Some("plugmem"), &[]),
            false,
            1_000,
        );
        let link = Command::Link {
            src: "plugmem".into(),
            rel: "depends_on".into(),
            dst: "tokio".into(),
        };
        let (code, out) = run_cmd(&db, &link, false, 2_000);
        assert_eq!(code, 0);
        assert!(out.contains("plugmem -depends_on-> tokio"), "{out}");

        let (code, out) = run_cmd(&db, &Command::Stats, true, 3_000);
        assert_eq!(code, 0);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["facts"], 1);
        assert!(v["edges"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn bad_link_is_a_usage_error() {
        let (db, _t) = TempDb::open();
        let cmd = Command::Remember {
            text: "x".into(),
            entity: Some("user".into()),
            tags: Vec::new(),
            links: vec!["not-a-pair".into()],
            meta: Vec::new(),
            valid_from: None,
        };
        let mut buf = Vec::new();
        let err = execute(&db, &cmd, false, 1_000, &mut buf).unwrap_err();
        assert!(matches!(err, CliError::Usage(_)));
    }

    #[test]
    fn as_of_time_travel_via_recall() {
        let (db, _t) = TempDb::open();
        run_cmd(
            &db,
            &remember("lives in Moscow", Some("user"), &[]),
            false,
            1_000,
        );
        let revise = Command::Revise {
            id: 0,
            text: "lives in Berlin".into(),
            entity: Some("user".into()),
            tags: Vec::new(),
            links: Vec::new(),
            meta: Vec::new(),
            valid_from: None,
        };
        run_cmd(&db, &revise, false, 2_000);

        let as_of = Command::Recall {
            query: Some("lives".into()),
            tags: Vec::new(),
            entities: vec!["user".into()],
            as_of: Some(1_500),
            range: None,
            k: 0,
            closed: false,
        };
        let (_, out) = run_cmd(&db, &as_of, false, 3_000);
        assert!(out.contains("Moscow"), "as-of 1500 → Moscow: {out}");
    }

    #[test]
    fn every_command_has_a_json_shape() {
        let (db, _t) = TempDb::open();
        // remember --json: id + similar array
        let (_, out) = run_cmd(
            &db,
            &remember("uses tokio", Some("plugmem"), &["pref"]),
            true,
            1_000,
        );
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["id"], 0);
        assert!(v["similar"].is_array());

        // revise --json
        let revise = Command::Revise {
            id: 0,
            text: "uses tokio now".into(),
            entity: Some("plugmem".into()),
            tags: Vec::new(),
            links: Vec::new(),
            meta: Vec::new(),
            valid_from: None,
        };
        let (_, out) = run_cmd(&db, &revise, true, 1_500);
        assert!(serde_json::from_str::<serde_json::Value>(out.trim()).is_ok());

        // link --json
        let link = Command::Link {
            src: "plugmem".into(),
            rel: "depends_on".into(),
            dst: "tokio".into(),
        };
        let (_, out) = run_cmd(&db, &link, true, 2_000);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["rel"], "depends_on");

        // forget --json then maintain --json
        let (_, out) = run_cmd(&db, &Command::Forget { id: 1 }, true, 2_500);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["forgotten"], true);
        let (_, out) = run_cmd(&db, &Command::Maintain, true, 3_000);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert!(v["purged"].as_u64().unwrap() >= 1);

        // show --json of a missing id
        let (code, out) = run_cmd(&db, &Command::Show { id: 999 }, true, 3_500);
        assert_eq!(code, 1);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["found"], false);

        // recall --json with a range window (covers the range/closed paths)
        let recall = Command::Recall {
            query: None,
            tags: Vec::new(),
            entities: vec!["plugmem".into()],
            as_of: None,
            range: Some(vec![0, 10_000]),
            k: 4,
            closed: true,
        };
        let (_, out) = run_cmd(&db, &recall, true, 4_000);
        assert!(serde_json::from_str::<serde_json::Value>(out.trim()).is_ok());
    }

    #[test]
    fn stats_human_lists_the_counters() {
        let (db, _t) = TempDb::open();
        run_cmd(&db, &remember("a", None, &[]), false, 1_000);
        let (code, out) = run_cmd(&db, &Command::Stats, false, 2_000);
        assert_eq!(code, 0);
        assert!(out.contains("facts") && out.contains("pool_bytes"), "{out}");
    }

    #[test]
    fn show_json_of_a_revised_predecessor_is_closed() {
        let (db, _t) = TempDb::open();
        run_cmd(&db, &remember("v1", Some("e"), &[]), false, 1_000);
        let revise = Command::Revise {
            id: 0,
            text: "v2".into(),
            entity: Some("e".into()),
            tags: Vec::new(),
            links: Vec::new(),
            meta: Vec::new(),
            valid_from: None,
        };
        run_cmd(&db, &revise, false, 2_000);
        // the successor records `revises`; its card names the predecessor
        let (_, out) = run_cmd(&db, &Command::Show { id: 1 }, false, 3_000);
        assert!(out.contains("revises     fact 0"), "{out}");
    }

    #[test]
    fn resolve_db_path_prefers_the_flag() {
        let p = std::path::Path::new("/tmp/explicit.plugmem");
        assert_eq!(resolve_db_path(Some(p), None), PathBuf::from(p));
        let configured = std::path::Path::new("/tmp/configured.plugmem");
        assert_eq!(
            resolve_db_path(None, Some(configured)),
            PathBuf::from(configured)
        );
        // With no flag/config it falls back to $PLUGMEM_DB or the platform default — we
        // only assert the code path runs and yields some path.
        let _ = resolve_db_path(None, None);
    }

    #[test]
    fn settings_help_runs_without_opening_a_database() {
        let cli = Cli::try_parse_from(["plugmem-cli", "help", "settings"]).unwrap();
        let mut output = Vec::new();
        assert_eq!(run_parsed(cli, &mut output), 0);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("plugmem settings"));
        assert!(output.contains("[database]"));
        assert!(output.contains("path (path string"));
    }

    #[test]
    fn run_parsed_opens_runs_and_reports() {
        let dir = std::env::temp_dir().join(format!(
            "plugmem-run-{}-{}",
            std::process::id(),
            now_ms_unique()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.plugmem");
        let cli = Cli {
            db: Some(path.clone()),
            config: None,
            json: false,
            command: Command::Stats,
        };
        let mut buf = Vec::new();
        let code = run_parsed(cli, &mut buf);
        assert_eq!(code, 0);
        assert!(String::from_utf8(buf).unwrap().contains("facts"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_parsed_on_a_locked_database_returns_one() {
        let (_held, dir) = {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-lock-{}-{}",
                std::process::id(),
                now_ms_unique()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("m.plugmem");
            (Database::open(&path, Config::default()).unwrap(), dir)
        };
        let cli = Cli {
            db: Some(dir.join("m.plugmem")),
            config: None,
            json: false,
            command: Command::Stats,
        };
        let mut buf = Vec::new();
        assert_eq!(run_parsed(cli, &mut buf), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_parsed_propagates_a_usage_error_as_two() {
        let dir = std::env::temp_dir().join(format!(
            "plugmem-usage-{}-{}",
            std::process::id(),
            now_ms_unique()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cli = Cli {
            db: Some(dir.join("m.plugmem")),
            config: None,
            json: false,
            command: Command::Remember {
                text: "x".into(),
                entity: None,
                tags: Vec::new(),
                links: vec!["bad".into()],
                meta: Vec::new(),
                valid_from: None,
            },
        };
        let mut buf = Vec::new();
        assert_eq!(run_parsed(cli, &mut buf), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A scratch directory (no db) for config/checkpoint tests; removed on drop.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-cli-{tag}-{}-{}",
                std::process::id(),
                now_ms_unique()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn export_import_roundtrip_preserves_open_facts() {
        // A deliberately nested scenario: entities, multi-tag facts, a
        // revision (closes its predecessor), a forget (tombstone), and an
        // explicit valid_from — export must dump exactly the open facts, and
        // import must reconstruct that set faithfully.
        let (a, _ta) = TempDb::open();
        run_cmd(
            &a,
            &Command::Remember {
                text: "prefers tokio".into(),
                entity: Some("user".into()),
                tags: vec!["pref".into(), "lang".into()],
                links: Vec::new(),
                meta: vec!["uri=s3://b/x".into(), "src=chat".into()],
                valid_from: Some(500),
            },
            false,
            1_000,
        );
        run_cmd(
            &a,
            &remember("lives in Moscow", Some("user"), &[]),
            false,
            1_100,
        ); // id 1
        run_cmd(
            &a,
            &Command::Revise {
                id: 1,
                text: "lives in Berlin".into(),
                entity: Some("user".into()),
                tags: vec!["geo".into()],
                links: Vec::new(),
                meta: Vec::new(),
                valid_from: None,
            },
            false,
            1_200,
        ); // id 2 open, id 1 closed
        run_cmd(&a, &remember("junk", None, &[]), false, 1_300); // id 3
        run_cmd(&a, &Command::Forget { id: 3 }, false, 1_400); // tombstone id 3
        run_cmd(
            &a,
            &remember("uses rust", Some("plugmem"), &["lang"]),
            false,
            1_500,
        ); // id 4

        // Export A into a JSONL file.
        let mut dump = Vec::new();
        render_export(&a.export(), false, &mut dump);
        let scratch = Scratch::new("roundtrip");
        let file = scratch.0.join("dump.jsonl");
        std::fs::write(&file, &dump).unwrap();

        // Import into a fresh B.
        let (b, _tb) = TempDb::open();
        let n = do_import(&b, 9_000, &file, 128, &mut Vec::new()).unwrap();

        // Both sides, compared as sets keyed by the preserved fields.
        let key = |f: &ExportedFact| {
            let mut tags = f.tags.clone();
            tags.sort();
            (f.text.clone(), f.entity.clone(), tags, f.valid_from)
        };
        let mut ak: Vec<_> = a.export().iter().map(key).collect();
        let mut bk: Vec<_> = b.export().iter().map(key).collect();
        ak.sort();
        bk.sort();
        assert_eq!(n, ak.len());
        assert_eq!(
            ak, bk,
            "roundtrip must preserve text/entity/tags/valid_from"
        );

        // Spot-checks: the open facts survive with their metadata; the closed
        // revision and the tombstone do not.
        let b_open = b.export();
        assert!(b_open.iter().any(|f| f.text == "prefers tokio"
            && f.valid_from == 500
            && f.entity.as_deref() == Some("user")
            && f.tags == vec!["pref".to_string(), "lang".to_string()]
            && f.metadata.get("uri").map(String::as_str) == Some("s3://b/x")
            && f.metadata.get("src").map(String::as_str) == Some("chat")));
        assert!(b_open.iter().any(|f| f.text == "lives in Berlin"));
        assert!(b_open.iter().any(|f| f.text == "uses rust"));
        assert!(!b_open.iter().any(|f| f.text.contains("Moscow")));
        assert!(!b_open.iter().any(|f| f.text == "junk"));
    }

    #[test]
    fn export_command_emits_jsonl_regardless_of_json_flag() {
        let (db, _t) = TempDb::open();
        run_cmd(&db, &remember("a fact", Some("e"), &["t"]), false, 1_000);
        for json in [false, true] {
            let (code, out) = run_cmd(&db, &Command::Export, json, 2_000);
            assert_eq!(code, 0);
            let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
            assert_eq!(v["text"], "a fact");
            assert_eq!(v["entity"], "e");
            assert_eq!(v["tags"][0], "t");
        }
    }

    #[test]
    fn import_command_counts_and_rejects_bad_lines() {
        let (db, _t) = TempDb::open();
        let scratch = Scratch::new("import");
        let good = scratch.0.join("in.jsonl");
        std::fs::write(
            &good,
            "{\"text\":\"from jsonl\",\"entity\":\"user\",\"tags\":[\"x\"],\"valid_from\":42}\n\n{\"text\":\"second\"}\n",
        )
        .unwrap();
        // A tiny batch size exercises the streaming/chunking path (two batches).
        let n = do_import(&db, 9_000, &good, 1, &mut Vec::new()).unwrap();
        assert_eq!(n, 2, "both facts imported, blank line skipped");

        let bad = scratch.0.join("bad.jsonl");
        std::fs::write(&bad, "not json at all\n").unwrap();
        let err = do_import(&db, 9_000, &bad, 128, &mut Vec::new()).unwrap_err();
        assert!(matches!(err, CliError::Usage(_)));
    }

    #[test]
    fn import_batch_size_does_not_change_the_result() {
        // The chunk size is a performance knob only: importing the same file with
        // batch 1 and batch 100 yields the identical fact set.
        let scratch = Scratch::new("import-batch");
        let file = scratch.0.join("facts.jsonl");
        let mut jsonl = String::new();
        for i in 0..5 {
            jsonl.push_str(&format!("{{\"text\":\"fact number {i}\"}}\n"));
        }
        std::fs::write(&file, &jsonl).unwrap();

        let (a, _ta) = TempDb::open();
        let (b, _tb) = TempDb::open();
        let na = do_import(&a, 9_000, &file, 1, &mut Vec::new()).unwrap();
        let nb = do_import(&b, 9_000, &file, 100, &mut Vec::new()).unwrap();

        assert_eq!(na, 5);
        assert_eq!(nb, 5);
        let texts = |db: &Database| {
            let mut t: Vec<_> = db.export().into_iter().map(|f| f.text).collect();
            t.sort();
            t
        };
        assert_eq!(texts(&a), texts(&b), "batch size must not change the facts");
    }

    #[test]
    fn config_table_feeds_settings_and_the_cli_batch_size() {
        // The CLI reads config.toml once (host `read_config`), builds the
        // shared `Settings`, and pulls its own `batch_size` from the same
        // table — the exact flow of `run_parsed`.
        let scratch = Scratch::new("settings");
        let cfgfile = scratch.0.join("config.toml");
        std::fs::write(
            &cfgfile,
            "[engine]\ndim = 512\n[embedder]\nkind = \"none\"\n\
             [maintenance]\nsnapshot_every_ops = 64\nbatch_size = 200\n",
        )
        .unwrap();
        let table = plugmem_host::read_config(Some(&cfgfile)).unwrap();
        let s = Settings::from_table(table.as_ref()).unwrap();
        assert_eq!(s.config.dim, 512);
        assert!(s.embedder.is_none());
        assert_eq!(s.snapshot_every_ops, Some(64));
        assert_eq!(read_batch_size(table.as_ref()), Some(200));

        // An explicit --config that does not exist is a usage error.
        assert!(plugmem_host::read_config(Some(&scratch.0.join("nope.toml"))).is_err());
    }

    #[test]
    fn checkpoint_command_flushes_the_journal_and_enables_the_readonly_path() {
        let scratch = Scratch::new("checkpoint-cmd");
        let path = scratch.0.join("m.plugmem");

        // A remember through the read-write path leaves a dirty journal.
        let remember = Cli {
            db: Some(path.clone()),
            config: None,
            json: false,
            command: Command::Remember {
                text: "hello tokio".into(),
                entity: None,
                tags: Vec::new(),
                links: Vec::new(),
                meta: Vec::new(),
                valid_from: None,
            },
        };
        assert_eq!(run_parsed(remember, &mut Vec::new()), 0);

        // The new command: human shape.
        let checkpoint = |json| Cli {
            db: Some(path.clone()),
            config: None,
            json,
            command: Command::Checkpoint,
        };
        let mut buf = Vec::new();
        assert_eq!(run_parsed(checkpoint(false), &mut buf), 0);
        assert!(String::from_utf8(buf).unwrap().contains("checkpointed"));

        // json shape.
        let mut buf = Vec::new();
        assert_eq!(run_parsed(checkpoint(true), &mut buf), 0);
        let v: serde_json::Value =
            serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
        assert_eq!(v["ok"], true);

        // The journal is now clean, so scrub (a shared-lock, read-only open)
        // succeeds — it would fail `NeedsCheckpoint` on a dirty journal.
        let scrub = Cli {
            db: Some(path),
            config: None,
            json: false,
            command: Command::Scrub,
        };
        let mut buf = Vec::new();
        assert_eq!(run_parsed(scrub, &mut buf), 0);
        assert!(String::from_utf8(buf).unwrap().contains("scrub ok"));
    }

    #[test]
    fn run_parsed_uses_the_readonly_path_after_a_checkpoint() {
        let scratch = Scratch::new("ro-route");
        let path = scratch.0.join("m.plugmem");
        {
            let (db, _) = Database::open(&path, Config::default()).unwrap();
            db.remember(RememberInput::text(1_000, "hello tokio"))
                .unwrap();
            db.checkpoint(2_000).unwrap(); // empty journal → open_readonly succeeds
        }
        // stats routes through open_readonly (mmap, shared)
        let cli = Cli {
            db: Some(path.clone()),
            config: None,
            json: false,
            command: Command::Stats,
        };
        let mut buf = Vec::new();
        assert_eq!(run_parsed(cli, &mut buf), 0);
        assert!(String::from_utf8(buf).unwrap().contains("facts"));

        // recall with no embedder also uses the read-only path
        let cli = Cli {
            db: Some(path),
            config: None,
            json: false,
            command: Command::Recall {
                query: Some("tokio".into()),
                tags: Vec::new(),
                entities: Vec::new(),
                as_of: None,
                range: None,
                k: 0,
                closed: false,
            },
        };
        let mut buf = Vec::new();
        assert_eq!(run_parsed(cli, &mut buf), 0);
        assert!(String::from_utf8(buf).unwrap().contains("tokio"));
    }
}
