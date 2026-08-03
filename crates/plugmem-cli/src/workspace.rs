//! The `workspace` command group: a directory of named memories.
//!
//! **Nothing here is on by default.** With no `--workspace`, no
//! `$PLUGMEM_WORKSPACE` and no `[workspace].dir`, every command in this file
//! says so and exits — there is one database, addressed by path, and that is
//! the ordinary case.
//!
//! These commands are the administrative half of a workspace: what is in it,
//! what each memory is for, and whether the registry still agrees with the
//! directory. The everyday half is `--db <name>`, which needs none of them.

use std::io::Write;
use std::path::PathBuf;

use plugmem_host::{
    DbEntry, DbName, Description, Settings, Workspace, WorkspaceIssue, WorkspaceLayout,
};
use serde_json::json;

use crate::cli::WorkspaceCommand;
use crate::{CliError, ENV_DB, now_ms};

/// Environment variable naming the workspace directory (below `--workspace`).
pub(crate) const ENV_WORKSPACE: &str = "PLUGMEM_WORKSPACE";

/// What to say when a workspace command is run without a workspace.
const NO_WORKSPACE: &str = "no workspace configured: pass --workspace DIR, set $PLUGMEM_WORKSPACE, \
or add [workspace].dir to config.toml. Without one there is a single database and --db takes a \
path, which is the default and needs none of this.";

/// The workspace root: `--workspace` > `$PLUGMEM_WORKSPACE` > `[workspace].dir`.
///
/// The same shape as the database-path precedence, so a person who knows one
/// knows the other.
pub(crate) fn resolve_root(flag: Option<&std::path::Path>, settings: &Settings) -> Option<PathBuf> {
    flag.map(PathBuf::from)
        .or_else(|| std::env::var_os(ENV_WORKSPACE).map(PathBuf::from))
        .or_else(|| settings.workspace.dir.clone())
}

/// Resolves what `--db` (or `$PLUGMEM_DB`) meant.
///
/// A bare name is a memory in the workspace; anything else is a path, exactly
/// as before. The two cannot be confused: a name has no separator and no dot,
/// so `work` is a name and `work.plugmem`, `./work` and `/srv/work` are paths.
/// With no workspace configured, everything is a path — which is what keeps
/// the single-database default byte-for-byte unchanged.
pub(crate) fn resolve_target(value: &str, root: Option<&PathBuf>) -> PathBuf {
    match (root, DbName::parse(value)) {
        (Some(root), Ok(name)) => WorkspaceLayout::new(root).path_of(&name),
        _ => PathBuf::from(value),
    }
}

/// Creates the workspace's database directory when `path` is a memory inside it.
///
/// A first write to a new name has to just work — that is the whole point of
/// naming a memory rather than registering one — and the directory may not
/// exist yet. Deliberately scoped to the workspace: `--db some/new/dir/x` is
/// still a plain path, and the CLI does not invent directories for those.
pub(crate) fn ensure_dir(path: &std::path::Path, root: Option<&PathBuf>) -> Result<(), CliError> {
    let Some(root) = root else { return Ok(()) };
    let dir = WorkspaceLayout::new(root).db_dir();
    if !path.starts_with(&dir) {
        return Ok(());
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| CliError::Usage(format!("cannot create {}: {e}", dir.display())))
}

/// Runs one `workspace` subcommand. Returns the process exit code.
pub(crate) fn execute(
    command: &WorkspaceCommand,
    root: Option<PathBuf>,
    settings: Settings,
    json_output: bool,
    out: &mut impl Write,
) -> Result<u8, CliError> {
    let Some(root) = root else {
        return Err(CliError::Usage(NO_WORKSPACE.into()));
    };

    // `use` prints a line for the shell and touches nothing else, so it is
    // answered before the workspace is opened — no lock, no registry, no
    // directory created just to be told a name.
    if let WorkspaceCommand::Use { name } = command {
        return use_line(&root, name, json_output, out);
    }

    let ws = settings.open_workspace(&root).map_err(usage)?;
    match command {
        WorkspaceCommand::List => list(&ws, json_output, out),
        WorkspaceCommand::Find { query, k } => find(&ws, query, k.unwrap_or(8), json_output, out),
        WorkspaceCommand::Describe {
            name,
            text,
            tags,
            owner,
        } => describe(&ws, name, text, tags, owner.as_deref(), json_output, out),
        WorkspaceCommand::Archive { name } => archive(&ws, name, json_output, out),
        WorkspaceCommand::Reindex => reindex(&ws, json_output, out),
        WorkspaceCommand::Verify => verify(&ws, json_output, out),
        WorkspaceCommand::Use { .. } => unreachable!("answered before the workspace is opened"),
    }
}

/// `workspace list` — every memory on disk, described or not.
///
/// Reads the directory and joins the registry onto it, rather than listing the
/// registry: a memory that exists but was never described still has to appear,
/// or the list would be a list of *descriptions* pretending to be a list of
/// memories.
fn list(ws: &Workspace, json_output: bool, out: &mut impl Write) -> Result<u8, CliError> {
    let names = ws.layout().list().map_err(usage)?;
    let described = ws.entries().map_err(usage)?;

    if json_output {
        let rows: Vec<_> = names
            .iter()
            .map(|name| {
                let entry = described.iter().find(|e| &e.name == name);
                json!({
                    "db": name.as_str(),
                    "description": entry.map(|e| e.description.as_str()),
                    "tags": entry.map(|e| e.tags.clone()).unwrap_or_default(),
                    "owner": entry.and_then(|e| e.owner.clone()),
                    "archived": entry.is_some_and(DbEntry::is_archived),
                })
            })
            .collect();
        writeln!(out, "{}", json!(rows)).ok();
        return Ok(0);
    }

    if names.is_empty() {
        writeln!(out, "no memories in {}", ws.layout().db_dir().display()).ok();
        return Ok(0);
    }
    let width = names.iter().map(|n| n.as_str().len()).max().unwrap_or(0);
    for name in &names {
        match described.iter().find(|e| &e.name == name) {
            Some(entry) => {
                let archived = if entry.is_archived() {
                    " [archived]"
                } else {
                    ""
                };
                writeln!(
                    out,
                    "{:width$}  {}{archived}",
                    name.as_str(),
                    entry.description
                )
                .ok();
            }
            None => {
                writeln!(out, "{:width$}  (no description)", name.as_str()).ok();
            }
        }
    }
    Ok(0)
}

/// `workspace find` — which memory is the one about…
fn find(
    ws: &Workspace,
    query: &str,
    k: usize,
    json_output: bool,
    out: &mut impl Write,
) -> Result<u8, CliError> {
    let hits = ws.find(query, k, now_ms()).map_err(usage)?;
    if json_output {
        writeln!(out, "{}", json!(entries_json(&hits))).ok();
    } else if hits.is_empty() {
        writeln!(out, "nothing matches {query:?}").ok();
    } else {
        for entry in &hits {
            writeln!(out, "{}  {}", entry.name, entry.description).ok();
        }
    }
    // A search that found nothing is a successful search, not a failure.
    Ok(0)
}

/// `workspace describe`.
fn describe(
    ws: &Workspace,
    name: &str,
    text: &str,
    tags: &[String],
    owner: Option<&str>,
    json_output: bool,
    out: &mut impl Write,
) -> Result<u8, CliError> {
    let name = DbName::parse(name).map_err(usage)?;
    let tags: Vec<&str> = tags.iter().map(String::as_str).collect();
    ws.describe(
        &name,
        now_ms(),
        Description {
            text,
            tags: &tags,
            owner,
        },
    )
    .map_err(usage)?;
    if json_output {
        writeln!(out, "{}", json!({ "db": name.as_str(), "described": true })).ok();
    } else {
        writeln!(out, "described {name}").ok();
    }
    Ok(0)
}

/// `workspace archive`.
fn archive(
    ws: &Workspace,
    name: &str,
    json_output: bool,
    out: &mut impl Write,
) -> Result<u8, CliError> {
    let name = DbName::parse(name).map_err(usage)?;
    let changed = ws.archive(&name, now_ms()).map_err(usage)?;
    if json_output {
        writeln!(
            out,
            "{}",
            json!({ "db": name.as_str(), "archived": changed })
        )
        .ok();
    } else if changed {
        writeln!(out, "archived {name}").ok();
    } else {
        writeln!(out, "{name} was already archived").ok();
    }
    Ok(0)
}

/// `workspace reindex`.
fn reindex(ws: &Workspace, json_output: bool, out: &mut impl Write) -> Result<u8, CliError> {
    let report = ws.reindex(now_ms()).map_err(usage)?;
    let names = |v: &[DbName]| -> Vec<String> { v.iter().map(DbName::to_string).collect() };
    if json_output {
        writeln!(
            out,
            "{}",
            json!({
                "indexed": names(&report.indexed),
                "undescribed": names(&report.undescribed),
                "busy": names(&report.busy),
            })
        )
        .ok();
        return Ok(0);
    }
    writeln!(out, "indexed {} memories", report.indexed.len()).ok();
    if !report.undescribed.is_empty() {
        writeln!(
            out,
            "{} have no description yet: {}",
            report.undescribed.len(),
            names(&report.undescribed).join(", ")
        )
        .ok();
    }
    // A busy memory is the honest limit of rebuilding a live workspace: one
    // file has one writer, so this pass could not read it and the registry is
    // knowingly incomplete. Saying so beats a silent gap.
    if !report.busy.is_empty() {
        writeln!(
            out,
            "{} in use by another process, not reindexed: {}",
            report.busy.len(),
            names(&report.busy).join(", ")
        )
        .ok();
    }
    Ok(0)
}

/// `workspace verify` — exit `0` when the registry agrees with the directory,
/// `1` when it does not. A scriptable gate, like `verify` on one database.
fn verify(ws: &Workspace, json_output: bool, out: &mut impl Write) -> Result<u8, CliError> {
    let issues = ws.verify(now_ms()).map_err(usage)?;
    if json_output {
        let rows: Vec<_> = issues.iter().map(issue_json).collect();
        writeln!(
            out,
            "{}",
            json!({ "ok": issues.is_empty(), "issues": rows })
        )
        .ok();
    } else if issues.is_empty() {
        writeln!(out, "workspace ok").ok();
    } else {
        for issue in &issues {
            writeln!(out, "{}", issue_text(issue)).ok();
        }
    }
    Ok(u8::from(!issues.is_empty()))
}

/// `workspace use` — the line to `eval` in a shell.
fn use_line(
    root: &std::path::Path,
    name: &str,
    json_output: bool,
    out: &mut impl Write,
) -> Result<u8, CliError> {
    let name = DbName::parse(name).map_err(usage)?;
    let layout = WorkspaceLayout::new(root);
    if !layout.exists(&name) {
        return Err(CliError::Usage(format!(
            "no memory named {name} in {}",
            layout.db_dir().display()
        )));
    }
    // A path, not the name: this line is going into some other shell, which may
    // not have the workspace configured, and there `PLUGMEM_DB=work` would
    // quietly mean a file called `work` in the current directory.
    let path = layout.path_of(&name);
    if json_output {
        writeln!(
            out,
            "{}",
            json!({ "db": name.as_str(), "path": path, "env": ENV_DB })
        )
        .ok();
    } else {
        writeln!(out, "{}", export_line(&path.to_string_lossy())).ok();
    }
    Ok(0)
}

/// The line that sets the environment variable, in the shell this platform's
/// user is overwhelmingly likely to be running.
///
/// A path is not shell syntax, so printing one for a shell is the one place in
/// this crate that is genuinely platform-specific — and it has to be, because
/// `export VAR='…'` is meaningless in PowerShell and `$env:VAR` is meaningless
/// in `sh`. `--json` stays the portable form for anything scripting this.
fn export_line(path: &str) -> String {
    if cfg!(windows) {
        // PowerShell: single quotes are literal, and a literal quote doubles.
        format!("$env:{ENV_DB} = '{}'", path.replace('\'', "''"))
    } else {
        format!("export {ENV_DB}={}", sh_quote(path))
    }
}

/// Single-quotes a value for `sh`, so a path with a space or a quote in it
/// survives the `eval` this line is written for.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn entries_json(entries: &[DbEntry]) -> Vec<serde_json::Value> {
    entries
        .iter()
        .map(|e| {
            json!({
                "db": e.name.as_str(),
                "description": e.description,
                "tags": e.tags,
                "owner": e.owner,
                "archived": e.is_archived(),
            })
        })
        .collect()
}

fn issue_json(issue: &WorkspaceIssue) -> serde_json::Value {
    match issue {
        WorkspaceIssue::Missing { name } => json!({ "db": name.as_str(), "issue": "missing" }),
        WorkspaceIssue::Undescribed { name } => {
            json!({ "db": name.as_str(), "issue": "undescribed" })
        }
        WorkspaceIssue::Stale { name } => json!({ "db": name.as_str(), "issue": "stale" }),
        WorkspaceIssue::Unreadable { name, why } => {
            json!({ "db": name.as_str(), "issue": "unreadable", "why": why })
        }
        WorkspaceIssue::AmbiguousSelf { name, facts } => {
            json!({ "db": name.as_str(), "issue": "ambiguous-self", "facts": facts })
        }
        // `WorkspaceIssue` is non_exhaustive: a kind added later is reported as
        // itself rather than dropped from the output.
        other => json!({ "issue": format!("{other:?}") }),
    }
}

fn issue_text(issue: &WorkspaceIssue) -> String {
    match issue {
        WorkspaceIssue::Missing { name } => {
            format!("{name}: described in the registry, but not on disk")
        }
        WorkspaceIssue::Undescribed { name } => {
            format!("{name}: no description — it works, it just cannot be found by one")
        }
        WorkspaceIssue::Stale { name } => {
            format!(
                "{name}: the registry disagrees with the memory; `workspace reindex` settles it"
            )
        }
        WorkspaceIssue::Unreadable { name, why } => format!("{name}: could not be read: {why}"),
        WorkspaceIssue::AmbiguousSelf { name, facts } => {
            format!("{name}: {facts} facts claim the reserved self-description anchor")
        }
        other => format!("{other:?}"),
    }
}

/// A workspace failure as a usage error: these are all "what you asked for
/// cannot be done", not engine faults.
fn usage(e: impl std::fmt::Display) -> CliError {
    CliError::Usage(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    /// A unique temp directory; removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-cli-ws-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Runs `plugmem-cli --workspace <root> ...` and returns (exit code, stdout).
    fn run(root: &std::path::Path, args: &[&str]) -> (u8, String) {
        let mut argv = vec!["plugmem-cli", "--workspace"];
        let root = root.to_string_lossy().into_owned();
        argv.push(&root);
        argv.extend_from_slice(args);
        let cli = Cli::try_parse_from(argv).expect("parse");
        let mut buf = Vec::new();
        let code = crate::run_parsed(cli, &mut buf);
        (code, String::from_utf8(buf).unwrap())
    }

    #[test]
    fn a_workspace_command_without_a_workspace_says_what_to_do() {
        // Never a panic and never a silent default: a workspace verb with no
        // workspace names all three ways to configure one.
        let cli = Cli::try_parse_from(["plugmem-cli", "workspace", "list"]).unwrap();
        let Command::Workspace { command } = &cli.command else {
            panic!("expected the workspace group");
        };
        let settings = Settings::from_table(None).unwrap();
        let e = execute(command, None, settings, false, &mut Vec::new()).unwrap_err();
        let CliError::Usage(message) = e else {
            panic!("expected a usage error");
        };
        assert!(message.contains("--workspace"));
        assert!(message.contains("PLUGMEM_WORKSPACE"));
        assert!(message.contains("[workspace].dir"));
    }

    #[test]
    fn list_shows_every_memory_including_the_ones_nobody_described() {
        let tmp = TempDir::new("list");
        assert!(
            run(&tmp.0, &["workspace", "list"])
                .1
                .contains("no memories in")
        );

        // Two memories, one described. Both must be listed: this is a list of
        // memories, not of descriptions.
        run(&tmp.0, &["--db", "chat-42", "remember", "a fact"]);
        run(&tmp.0, &["--db", "scratch", "remember", "another"]);
        run(
            &tmp.0,
            &["workspace", "describe", "chat-42", "release planning"],
        );

        let (code, out) = run(&tmp.0, &["workspace", "list"]);
        assert_eq!(code, 0);
        assert!(out.contains("chat-42"), "{out}");
        assert!(out.contains("release planning"), "{out}");
        assert!(out.contains("scratch"), "{out}");
        assert!(out.contains("(no description)"), "{out}");

        let (_, json) = run(&tmp.0, &["--json", "workspace", "list"]);
        let rows: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(rows[0]["db"], "chat-42");
        assert_eq!(rows[0]["description"], "release planning");
        assert_eq!(rows[1]["db"], "scratch");
        assert!(rows[1]["description"].is_null());
    }

    #[test]
    fn a_bare_name_after_db_reaches_the_memory_in_the_workspace() {
        let tmp = TempDir::new("byname");
        let (code, _) = run(&tmp.0, &["--db", "work", "remember", "prefers tokio"]);
        assert_eq!(code, 0);
        assert!(tmp.0.join("db/work.plugmem.journal").exists());

        let (code, out) = run(&tmp.0, &["--db", "work", "recall", "tokio"]);
        assert_eq!(code, 0);
        assert!(out.contains("prefers tokio"), "{out}");

        // A different memory does not see it.
        let (_, out) = run(&tmp.0, &["--db", "other", "recall", "tokio"]);
        assert!(!out.contains("prefers tokio"), "{out}");
    }

    #[test]
    fn describe_archive_and_find_work_through_the_command_line() {
        let tmp = TempDir::new("describe");
        let (code, out) = run(
            &tmp.0,
            &[
                "workspace",
                "describe",
                "chat-42",
                "release planning and performance",
                "--tag",
                "kind:chat",
                "--owner",
                "ann",
            ],
        );
        assert_eq!(code, 0);
        assert!(out.contains("described chat-42"), "{out}");

        let (code, out) = run(&tmp.0, &["workspace", "find", "release planning"]);
        assert_eq!(code, 0);
        assert!(out.contains("chat-42"), "{out}");

        // The owner is an edge, not text, and is still findable.
        let (_, out) = run(&tmp.0, &["workspace", "find", "ann"]);
        assert!(out.contains("chat-42"), "{out}");

        // A search that matches nothing succeeded at searching.
        let (code, out) = run(&tmp.0, &["workspace", "find", "quantum knitting"]);
        assert_eq!(code, 0);
        assert!(out.contains("nothing matches"), "{out}");

        let (code, out) = run(&tmp.0, &["workspace", "archive", "chat-42"]);
        assert_eq!(code, 0);
        assert!(out.contains("archived chat-42"), "{out}");
        let (code, out) = run(&tmp.0, &["workspace", "archive", "chat-42"]);
        assert_eq!(code, 0);
        assert!(out.contains("already archived"), "{out}");

        let (_, out) = run(&tmp.0, &["--json", "workspace", "archive", "chat-42"]);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&out).unwrap()["archived"],
            false
        );
    }

    #[test]
    fn verify_is_a_gate_and_reindex_settles_it() {
        let tmp = TempDir::new("verify");
        run(&tmp.0, &["workspace", "describe", "chat-42", "planning"]);
        let (code, out) = run(&tmp.0, &["workspace", "verify"]);
        assert_eq!(code, 0);
        assert!(out.contains("workspace ok"), "{out}");

        // A memory nobody described is a reportable disagreement, and the exit
        // code says so — this is meant to be usable in a script.
        run(&tmp.0, &["--db", "scratch", "remember", "x"]);
        let (code, out) = run(&tmp.0, &["workspace", "verify"]);
        assert_eq!(code, 1);
        assert!(out.contains("scratch"), "{out}");
        assert!(out.contains("no description"), "{out}");

        let (code, out) = run(&tmp.0, &["--json", "workspace", "verify"]);
        assert_eq!(code, 1);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["ok"], false);
        assert_eq!(parsed["issues"][0]["issue"], "undescribed");

        let (code, out) = run(&tmp.0, &["workspace", "reindex"]);
        assert_eq!(code, 0);
        assert!(out.contains("indexed 1 memories"), "{out}");
        assert!(out.contains("have no description yet"), "{out}");

        let (_, out) = run(&tmp.0, &["--json", "workspace", "reindex"]);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["indexed"][0], "chat-42");
        assert_eq!(parsed["undescribed"][0], "scratch");
        assert_eq!(parsed["busy"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn use_prints_a_line_for_the_shell_and_writes_nothing() {
        let tmp = TempDir::new("use");
        run(&tmp.0, &["--db", "work", "remember", "a fact"]);

        let (code, out) = run(&tmp.0, &["workspace", "use", "work"]);
        assert_eq!(code, 0);
        // A path, not the name: this line is going into a shell that may not
        // have the workspace configured, where a name would silently mean a
        // file in the current directory.
        let expected_prefix = if cfg!(windows) {
            "$env:PLUGMEM_DB = '"
        } else {
            "export PLUGMEM_DB='"
        };
        assert!(out.starts_with(expected_prefix), "{out}");
        // The path is compared as a path, not as a string with a separator
        // baked into it — that separator is the OS's to choose.
        let expected =
            WorkspaceLayout::new(&tmp.0).path_of(&plugmem_host::DbName::parse("work").unwrap());
        assert!(out.contains(&*expected.to_string_lossy()), "{out}");

        // No state file anywhere — the shell holds the state, so one window
        // cannot redirect another.
        let entries: Vec<_> = std::fs::read_dir(&tmp.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, ["db"], "{entries:?}");

        let (_, out) = run(&tmp.0, &["--json", "workspace", "use", "work"]);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["db"], "work");
        assert_eq!(parsed["env"], "PLUGMEM_DB");

        // A name nobody has used is a typo, caught here rather than later.
        let (code, _) = run(&tmp.0, &["workspace", "use", "nope"]);
        assert_eq!(code, 2);
    }

    #[test]
    fn a_name_outside_the_alphabet_is_refused_by_every_verb_that_takes_one() {
        let tmp = TempDir::new("badname");
        for args in [
            vec!["workspace", "describe", "../etc", "x"],
            vec!["workspace", "archive", "../etc"],
            vec!["workspace", "use", "../etc"],
        ] {
            let (code, _) = run(&tmp.0, &args);
            assert_eq!(code, 2, "{args:?}");
        }
    }

    #[test]
    fn every_issue_class_renders_in_both_output_modes() {
        // `verify` is only useful if a person can read what it found, and the
        // rare classes are exactly the ones nobody sees until they matter.
        let name = plugmem_host::DbName::parse("chat-42").unwrap();
        let cases = [
            (
                WorkspaceIssue::Missing { name: name.clone() },
                "missing",
                "not on disk",
            ),
            (
                WorkspaceIssue::Undescribed { name: name.clone() },
                "undescribed",
                "no description",
            ),
            (
                WorkspaceIssue::Stale { name: name.clone() },
                "stale",
                "reindex",
            ),
            (
                WorkspaceIssue::Unreadable {
                    name: name.clone(),
                    why: "held elsewhere".into(),
                },
                "unreadable",
                "held elsewhere",
            ),
            (
                WorkspaceIssue::AmbiguousSelf {
                    name: name.clone(),
                    facts: 2,
                },
                "ambiguous-self",
                "reserved self-description",
            ),
        ];
        for (issue, tag, phrase) in cases {
            let rendered = issue_json(&issue);
            assert_eq!(rendered["issue"], tag, "{issue:?}");
            assert_eq!(rendered["db"], "chat-42", "{issue:?}");
            let text = issue_text(&issue);
            assert!(text.contains("chat-42"), "{text}");
            assert!(text.contains(phrase), "{text}");
        }
    }

    #[test]
    fn a_described_memory_renders_its_whole_record() {
        let entries = [DbEntry {
            name: plugmem_host::DbName::parse("chat-42").unwrap(),
            description: "release planning".into(),
            tags: vec!["kind:chat".into(), plugmem_host::ARCHIVED_TAG.into()],
            owner: Some("ann".into()),
        }];
        let rendered = entries_json(&entries);
        assert_eq!(rendered[0]["db"], "chat-42");
        assert_eq!(rendered[0]["description"], "release planning");
        assert_eq!(rendered[0]["owner"], "ann");
        assert_eq!(rendered[0]["archived"], true);
        assert_eq!(rendered[0]["tags"][0], "kind:chat");
    }

    #[test]
    fn find_and_describe_answer_in_json_too() {
        let tmp = TempDir::new("json");
        let (code, out) = run(
            &tmp.0,
            &["--json", "workspace", "describe", "chat-42", "planning"],
        );
        assert_eq!(code, 0);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&out).unwrap()["described"],
            true
        );

        let (code, out) = run(&tmp.0, &["--json", "workspace", "find", "planning"]);
        assert_eq!(code, 0);
        let hits: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(hits[0]["db"], "chat-42");

        // An archived memory is marked as such in the listing.
        run(&tmp.0, &["workspace", "archive", "chat-42"]);
        let (_, out) = run(&tmp.0, &["workspace", "list"]);
        assert!(out.contains("[archived]"), "{out}");
    }

    #[test]
    fn reindex_reports_a_memory_another_process_is_holding() {
        let tmp = TempDir::new("reindex-busy");
        run(&tmp.0, &["workspace", "describe", "chat-42", "planning"]);
        run(&tmp.0, &["workspace", "describe", "notes", "other things"]);

        // A live writer elsewhere: this pass genuinely cannot read that memory,
        // and the rebuilt registry is knowingly incomplete. It has to say so.
        let held =
            WorkspaceLayout::new(&tmp.0).path_of(&plugmem_host::DbName::parse("chat-42").unwrap());
        let _outsider =
            plugmem_host::Database::open(held, plugmem_host::Config::default()).unwrap();

        let (code, out) = run(&tmp.0, &["workspace", "reindex"]);
        assert_eq!(code, 0);
        assert!(out.contains("in use by another process"), "{out}");
        assert!(out.contains("chat-42"), "{out}");
    }

    #[test]
    fn a_path_outside_the_workspace_creates_no_directory() {
        // `ensure_dir` is scoped: the CLI invents `<root>/db` for a named
        // memory, and nothing at all for a plain path.
        let tmp = TempDir::new("ensure");
        let elsewhere = tmp.0.join("elsewhere/m.plugmem");
        ensure_dir(&elsewhere, Some(&tmp.0)).unwrap();
        assert!(!tmp.0.join("elsewhere").exists());
        assert!(!tmp.0.join("db").exists());

        let inside =
            WorkspaceLayout::new(&tmp.0).path_of(&plugmem_host::DbName::parse("x").unwrap());
        ensure_dir(&inside, Some(&tmp.0)).unwrap();
        assert!(tmp.0.join("db").is_dir());

        // No workspace at all: nothing to create, and no error.
        ensure_dir(&elsewhere, None).unwrap();
    }

    #[test]
    fn a_path_with_a_quote_survives_the_shell_it_is_written_for() {
        // The one platform-specific line in the crate: a path is not shell
        // syntax, and the two shells quote differently.
        assert_eq!(sh_quote("/srv/bot"), "'/srv/bot'");
        assert_eq!(sh_quote("/srv/it's"), r"'/srv/it'\''s'");
        assert_eq!(sh_quote("/srv/a b"), "'/srv/a b'");

        let line = export_line("/srv/it's");
        if cfg!(windows) {
            assert_eq!(line, "$env:PLUGMEM_DB = '/srv/it''s'");
        } else {
            assert_eq!(line, r"export PLUGMEM_DB='/srv/it'\''s'");
        }
    }
}
