//! `plugmem` — the command-line surface over the
//! [temporal-memory engine](plugmem_core), a thin wrapper around
//! [`plugmem_host::Database`] (specs/06). Parse the arguments, call one
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

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use plugmem_host::{
    Database, ExportedFact, FactId, HostError, LinkInput, ReadOnlyDatabase, RecallQuery,
    RecallResult, RememberInput, RememberOutcome, Stats, VALID_TO_OPEN,
};
use serde_json::json;

use crate::cli::{Cli, Command};
use crate::config::{Settings, load_settings};

/// Environment variable naming the database file (below the `--db` flag).
pub(crate) const ENV_DB: &str = "PLUGMEM_DB";
/// Environment variable naming the config file (below the `--config` flag).
pub(crate) const ENV_CONFIG: &str = "PLUGMEM_CONFIG";
/// Environment variable selecting the embedder kind (above the config file).
pub(crate) const ENV_EMBEDDER: &str = "PLUGMEM_EMBEDDER";
/// Default database file when neither flag nor env is given.
pub(crate) const DEFAULT_DB: &str = "plugmem.db";
/// The app's config subdirectory and file, under `$XDG_CONFIG_HOME`.
pub(crate) const CONFIG_DIR: &str = "plugmem";
pub(crate) const CONFIG_FILE: &str = "config.toml";

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
    let path = resolve_db_path(cli.db.as_deref());
    let mut settings = match load_settings(&cli) {
        Ok(s) => s,
        Err(e) => return report_err(&e),
    };

    // `recover` is a standalone salvage on file paths — it opens the source
    // itself (under an exclusive lock) and writes a fresh destination, so it
    // runs before the normal open. `scrub` is a byte-level container check over
    // a read-only (shared-lock) open, which requires a checkpointed database.
    match &cli.command {
        Command::Recover { dst } => return do_recover(&path, dst, &settings, cli.json, out),
        Command::Scrub => return do_scrub(&path, &settings, cli.json, out),
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

    let db = match settings.open(&path) {
        Ok(db) => db,
        Err(HostError::Locked { path }) => return report_locked(&path),
        Err(e) => return report_err(&CliError::Host(e)),
    };
    match execute(&db, &cli.command, cli.json, now_ms(), out) {
        Ok(code) => code,
        Err(e) => {
            let _ = out.flush();
            report_err(&e)
        }
    }
}

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

/// Database path precedence (specs/06): `--db` flag > `$PLUGMEM_DB` >
/// `./plugmem.db`.
fn resolve_db_path(flag: Option<&std::path::Path>) -> PathBuf {
    flag.map(PathBuf::from)
        .or_else(|| std::env::var_os(ENV_DB).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DB))
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
            render_export(&ro.export(), json, out);
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
            valid_from,
        } => {
            let outcome = do_remember(db, now, text, entity, tags, links, *valid_from, None)?;
            render_remember(&outcome, json, out);
            Ok(0)
        }
        Command::Revise {
            id,
            text,
            entity,
            tags,
            links,
            valid_from,
        } => {
            let outcome = do_remember(
                db,
                now,
                text,
                entity,
                tags,
                links,
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
            render_export(&db.export(), json, out);
            Ok(0)
        }
        Command::Import { file } => {
            let n = do_import(db, now, file, out)?;
            if json {
                writeln!(out, "{}", json!({ "imported": n })).ok();
            } else {
                writeln!(out, "imported {n} facts").ok();
            }
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
        // Handled in `run_parsed` before the read-write open.
        Command::Scrub | Command::Recover { .. } => {
            unreachable!("scrub/recover are dispatched before execute")
        }
    }
}

/// Salvages `src` into a fresh `dst` (specs/16 §9): `Database::recover` opens
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
                        "dst": dst.display().to_string(),
                    })
                )
                .ok();
            } else {
                writeln!(
                    out,
                    "recovered to {}: kept {}, dropped {} text + {} vector",
                    dst.display(),
                    r.kept,
                    r.dropped_text,
                    r.dropped_vector
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

/// Embeds a `recall` command's text query into a vector using the configured
/// embedder, so the read-only path (which carries no embedder) can still search
/// by meaning while a writer process holds the database. Returns `None` when the
/// command is not `recall`, carries no query text, or no embedder is configured
/// — recall then falls back to lexical/structural sources. Mirrors the host's
/// "embed before the lock" rule (specs/13 §1); the embed happens before the open
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
    // the host embeds inside `recall` (specs/13 §1).
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

/// Renders exported facts as JSONL (one object per line) — the same shape
/// with or without `--json` (JSONL is already machine-readable).
fn render_export(facts: &[ExportedFact], _json: bool, out: &mut impl Write) {
    for f in facts {
        writeln!(
            out,
            "{}",
            json!({
                "text": f.text,
                "entity": f.entity,
                "tags": f.tags,
                "recorded_at": f.recorded_at,
                "valid_from": f.valid_from,
            })
        )
        .ok();
    }
}

/// Loads facts from a JSONL file (as written by `export`), re-`remember`ing
/// each. Returns the count imported. A malformed line is a usage error.
fn do_import(
    db: &Database,
    now: u64,
    file: &std::path::Path,
    _out: &mut impl Write,
) -> Result<usize, CliError> {
    let text = std::fs::read_to_string(file)
        .map_err(|e| CliError::Usage(format!("reading {}: {e}", file.display())))?;
    let mut count = 0usize;
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| CliError::Usage(format!("line {}: {e}", i + 1)))?;
        let fact_text = v["text"]
            .as_str()
            .ok_or_else(|| CliError::Usage(format!("line {}: missing string \"text\"", i + 1)))?;
        let entity = v["entity"].as_str();
        let tags: Vec<String> = v["tags"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        let valid_from = v["valid_from"].as_u64();
        db.remember(RememberInput {
            entity,
            tags: &tag_refs,
            valid_from,
            ..RememberInput::text(now, fact_text)
        })?;
        count += 1;
    }
    Ok(count)
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
    valid_from: Option<u64>,
    revise: Option<FactId>,
) -> Result<RememberOutcome, CliError> {
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let link_pairs = parse_links(links)?;
    let link_refs: Vec<(&str, &str)> = link_pairs
        .iter()
        .map(|(r, e)| (r.as_str(), e.as_str()))
        .collect();
    let input = RememberInput {
        entity: entity.as_deref(),
        tags: &tag_refs,
        links: &link_refs,
        valid_from,
        ..RememberInput::text(now, text)
    };
    match revise {
        Some(target) => Ok(db.revise(target, input)?),
        None => Ok(db.remember(input)?),
    }
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

    fn settings_with(embedder: Option<Box<dyn plugmem_host::Embedder>>) -> crate::config::Settings {
        crate::config::Settings {
            config: Config::default(),
            embedder,
            maintenance: crate::config::Maintenance::default(),
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
            valid_from: None,
        }
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
        assert_eq!(resolve_db_path(Some(p)), PathBuf::from(p));
        // With no flag it falls back (to $PLUGMEM_DB or the default) — we
        // only assert the code path runs and yields some path.
        let _ = resolve_db_path(None);
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
        let n = do_import(&b, 9_000, &file, &mut Vec::new()).unwrap();

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
            && f.tags == vec!["pref".to_string(), "lang".to_string()]));
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
        let (code, out) = run_cmd(&db, &Command::Import { file: good }, false, 9_000);
        assert_eq!(code, 0);
        assert!(out.contains("imported 2 facts"), "{out}");

        let bad = scratch.0.join("bad.jsonl");
        std::fs::write(&bad, "not json at all\n").unwrap();
        let mut buf = Vec::new();
        let err = execute(&db, &Command::Import { file: bad }, false, 9_000, &mut buf).unwrap_err();
        assert!(matches!(err, CliError::Usage(_)));
    }

    #[test]
    fn load_settings_reads_the_config_file() {
        let scratch = Scratch::new("settings");
        let cfgfile = scratch.0.join("config.toml");
        std::fs::write(
            &cfgfile,
            "[engine]\ndim = 512\n[embedder]\nkind = \"none\"\n[maintenance]\nsnapshot_every_ops = 64\n",
        )
        .unwrap();
        let cli = Cli {
            db: None,
            config: Some(cfgfile),
            json: false,
            command: Command::Stats,
        };
        let s = load_settings(&cli).unwrap();
        assert_eq!(s.config.dim, 512);
        assert!(s.embedder.is_none());
        assert_eq!(s.maintenance.snapshot_every_ops, Some(64));
        // an explicit --config that does not exist is a usage error
        let cli = Cli {
            db: None,
            config: Some(scratch.0.join("nope.toml")),
            json: false,
            command: Command::Stats,
        };
        assert!(load_settings(&cli).is_err());
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
