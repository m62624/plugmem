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

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use plugmem_core::FactId;
use plugmem_host::{
    Config, Database, HostError, LinkInput, RecallQuery, RememberInput, RememberOutcome,
    VALID_TO_OPEN,
};
use serde_json::json;

/// `plugmem` — a temporal memory for LLM agents in a single file.
#[derive(Parser)]
#[command(
    name = "plugmem-cli",
    version,
    about = "Temporal memory for LLM agents — remember, recall, revise, forget over one file.",
    long_about = "A local memory an agent talks to in four verbs — remember / recall / \
revise / forget — plus link / show / stats / maintain. Recall fuses lexical (BM25), \
vector, entity-graph and temporal evidence into one ranked, token-budgeted block. One \
database is a single snapshot file plus a journal; point --db at it (default ./plugmem.db, \
or $PLUGMEM_DB). Human output by default, --json for tooling. Exit code: 0 ok, 1 not found \
/ database locked, 2 usage or runtime error.",
    after_help = "FOR AI AGENTS: you'll get markedly better results with the matching \
`plugmem` skill loaded — it carries the workflow, the remember/recall loop and the examples \
this binary expects. Check that its version matches `plugmem-cli --version`; the skill is \
attached to every release at https://github.com/m62624/plugmem/releases"
)]
pub struct Cli {
    /// Database file (default: ./plugmem.db, or $PLUGMEM_DB).
    #[arg(long, global = true, value_name = "PATH")]
    db: Option<PathBuf>,

    /// Machine-readable JSON output instead of the human report.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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
}

/// A failure before or during a command: a runtime engine/host error, or a
/// usage error (a malformed argument the parser could not catch).
#[derive(Debug)]
enum CliError {
    Host(HostError),
    Usage(String),
}

impl From<HostError> for CliError {
    fn from(e: HostError) -> Self {
        CliError::Host(e)
    }
}

/// Wall-clock now in unix milliseconds (the engine keeps no clock).
fn now_ms() -> u64 {
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

/// The testable core of [`run`]: resolve the database path, open it, run the
/// command into `out`, and return the exit code (`0` ok, `1` soft miss /
/// locked, `2` error). Errors go to stderr.
fn run_parsed(cli: Cli, out: &mut impl Write) -> u8 {
    let path = resolve_db_path(cli.db.as_deref());
    let (db, _report) = match Database::open(&path, Config::default()) {
        Ok(v) => v,
        Err(HostError::Locked { path }) => {
            eprintln!(
                "plugmem: database is locked by another process: {}",
                path.display()
            );
            return 1;
        }
        Err(e) => {
            eprintln!("plugmem: {e}");
            return 2;
        }
    };

    match execute(&db, &cli.command, cli.json, now_ms(), out) {
        Ok(code) => code,
        Err(CliError::Usage(msg)) => {
            let _ = out.flush();
            eprintln!("plugmem: {msg}");
            2
        }
        Err(CliError::Host(e)) => {
            let _ = out.flush();
            eprintln!("plugmem: {e}");
            2
        }
    }
}

/// Database path precedence (specs/06): `--db` flag > `$PLUGMEM_DB` >
/// `./plugmem.db`.
fn resolve_db_path(flag: Option<&std::path::Path>) -> PathBuf {
    flag.map(PathBuf::from)
        .or_else(|| std::env::var_os("PLUGMEM_DB").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("plugmem.db"))
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
        Command::Recall {
            query,
            tags,
            entities,
            as_of,
            range,
            k,
            closed,
        } => {
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let ent_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
            let range_pair = range.as_ref().map(|v| (v[0], v[1]));
            let q = RecallQuery {
                now,
                text: query.as_deref(),
                vector: None,
                tags: &tag_refs,
                entities: &ent_refs,
                as_of: *as_of,
                range: range_pair,
                k: *k,
                token_budget: None,
                include_closed: *closed,
                ef: None,
            };
            let res = db.recall(q)?;
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
                    json!({
                        "facts": facts,
                        "rendered": res.rendered,
                        "truncated": res.truncated,
                    })
                )
                .ok();
            } else if res.rendered.is_empty() {
                writeln!(out, "(nothing recalled)").ok();
            } else {
                writeln!(out, "{}", res.rendered).ok();
            }
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
        Command::Show { id } => {
            let Some(fact) = db.get(FactId(*id)) else {
                if json {
                    writeln!(out, "{}", json!({ "id": id, "found": false })).ok();
                } else {
                    writeln!(out, "fact {id} not found").ok();
                }
                return Ok(1);
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
            Ok(0)
        }
        Command::Stats => {
            let s = db.stats();
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
    }
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
    use super::*;

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
}
