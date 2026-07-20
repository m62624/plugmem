//! Black-box tests of the `plugmem` binary: argument parsing, exit codes,
//! the human/JSON split, and lock behaviour. The command logic itself is
//! unit-tested in the library; here we exercise the actual executable.

use std::path::PathBuf;
use std::process::{Command, Output};

use plugmem_host::{Config, Database};

/// A unique temp directory per test; removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let unique = format!(
            "plugmem-cli-it-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn db(&self) -> PathBuf {
        self.0.join("m.plugmem")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs the binary with `--db <path>` prepended to `args`.
fn plugmem(db: &std::path::Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(db)
        .args(args)
        .output()
        .expect("run plugmem")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A checked-in fixture file under `tests/fixtures/`.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn help_and_version_succeed() {
    for flag in ["--help", "--version"] {
        let out = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
            .arg(flag)
            .output()
            .unwrap();
        assert!(out.status.success(), "{flag} should exit 0");
    }
}

#[test]
fn no_command_is_a_usage_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn remember_recall_roundtrip() {
    let tmp = TempDir::new("roundtrip");
    let r = plugmem(
        &tmp.db(),
        &[
            "remember",
            "prefers tokio",
            "--entity",
            "user",
            "--tag",
            "pref",
        ],
    );
    assert_eq!(r.status.code(), Some(0));
    assert!(
        stdout(&r).starts_with("remembered fact 0"),
        "{}",
        stdout(&r)
    );

    let r = plugmem(&tmp.db(), &["recall", "tokio"]);
    assert_eq!(r.status.code(), Some(0));
    assert!(stdout(&r).contains("tokio"), "{}", stdout(&r));
}

#[test]
fn show_missing_is_exit_one() {
    let tmp = TempDir::new("missing");
    plugmem(&tmp.db(), &["remember", "a note"]);
    let r = plugmem(&tmp.db(), &["show", "999"]);
    assert_eq!(r.status.code(), Some(1));
    assert!(stdout(&r).contains("not found"));
}

#[test]
fn json_output_is_parseable() {
    let tmp = TempDir::new("json");
    plugmem(
        &tmp.db(),
        &["remember", "uses tokio", "--entity", "plugmem"],
    );
    let r = plugmem(&tmp.db(), &["--json", "stats"]);
    assert_eq!(r.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_str(stdout(&r).trim()).unwrap();
    assert_eq!(v["facts"], 1);
}

#[test]
fn env_var_selects_the_database() {
    let tmp = TempDir::new("env");
    // Seed via PLUGMEM_DB (no --db flag), then read it back the same way.
    let seed = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .env("PLUGMEM_DB", tmp.db())
        .args(["remember", "from the env path"])
        .output()
        .unwrap();
    assert_eq!(seed.status.code(), Some(0));
    let read = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .env("PLUGMEM_DB", tmp.db())
        .args(["show", "0"])
        .output()
        .unwrap();
    assert!(
        stdout(&read).contains("from the env path"),
        "{}",
        stdout(&read)
    );
}

#[test]
fn a_locked_database_exits_one() {
    let tmp = TempDir::new("locked");
    // Hold the exclusive lock in-process, then the CLI must refuse.
    let (_held, _report) = Database::open(tmp.db(), Config::default()).unwrap();
    let r = plugmem(&tmp.db(), &["stats"]);
    assert_eq!(r.status.code(), Some(1), "locked database → exit 1");
    let msg = String::from_utf8_lossy(&r.stderr);
    assert!(msg.contains("locked"), "stderr = {msg}");
}

#[test]
fn config_file_is_accepted_and_missing_one_is_a_usage_error() {
    let tmp = TempDir::new("config");
    let with_config = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
            .arg("--db")
            .arg(tmp.db())
            .arg("--config")
            .arg(fixture("config.toml"))
            .args(args)
            .output()
            .unwrap()
    };
    assert_eq!(
        with_config(&["remember", "hello from config", "--entity", "user"])
            .status
            .code(),
        Some(0)
    );
    let r = with_config(&["recall", "config"]);
    assert_eq!(r.status.code(), Some(0));
    assert!(stdout(&r).contains("config"), "{}", stdout(&r));

    // An explicit --config that does not exist is a usage error (exit 2).
    let bad = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(tmp.db())
        .args(["--config", "/no/such/plugmem-config.toml", "stats"])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
}

#[test]
fn import_from_a_fixture_jsonl() {
    let tmp = TempDir::new("import-fix");
    let out = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(tmp.db())
        .arg("import")
        .arg(fixture("facts.jsonl"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert!(stdout(&out).contains("imported 3 facts"), "{}", stdout(&out));

    // The imported facts are recallable, entity and tags preserved.
    let r = plugmem(&tmp.db(), &["recall", "tokio"]);
    assert!(
        stdout(&r).contains("tokio") && stdout(&r).contains("#pref"),
        "{}",
        stdout(&r)
    );
}
