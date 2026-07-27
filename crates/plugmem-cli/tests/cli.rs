//! Black-box tests of the `plugmem` binary: argument parsing, exit codes,
//! the human/JSON split, and lock behaviour. The command logic itself is
//! unit-tested in the library; here we exercise the actual executable.

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
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

/// The current snapshot generation file for a database `base`: `base` is now a
/// tiny manifest (magic/ver/gen/checksum, little-endian) naming `base.snap.<gen>`.
fn snapshot_file(base: &std::path::Path) -> PathBuf {
    let m = std::fs::read(base).expect("manifest present");
    let generation = u64::from_le_bytes(m[8..16].try_into().unwrap());
    let mut p = base.to_path_buf().into_os_string();
    p.push(format!(".snap.{generation}"));
    PathBuf::from(p)
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
fn read_only_commands_coexist_with_a_live_writer() {
    let tmp = TempDir::new("coexist");
    // Seed and checkpoint via the CLI so a snapshot generation is published.
    assert!(
        plugmem(&tmp.db(), &["remember", "a fact about tokio"])
            .status
            .success()
    );
    assert!(plugmem(&tmp.db(), &["checkpoint"]).status.success());

    // Hold the writer lock in-process — a live writer that HAS published a
    // generation. Variant 2 (MVCC): read-only commands pin that generation and
    // run alongside the writer, unlike `a_locked_database_exits_one` where the
    // fresh writer published nothing to read.
    let (_writer, _r) = Database::open(tmp.db(), Config::default()).unwrap();

    for cmd in [
        ["stats"].as_slice(),
        &["show", "0"],
        &["export"],
        &["verify"],
        &["recall", "tokio"],
    ] {
        let r = plugmem(&tmp.db(), cmd);
        assert_eq!(
            r.status.code(),
            Some(0),
            "read command {cmd:?} must coexist with a live writer; stderr={}",
            String::from_utf8_lossy(&r.stderr)
        );
    }
    // The two newly read-only-routed commands produce their expected output.
    assert!(stdout(&plugmem(&tmp.db(), &["verify"])).contains("integrity ok"));
    assert!(stdout(&plugmem(&tmp.db(), &["recall", "tokio"])).contains("tokio"));

    // A second *writer* is still refused while the lock is held.
    assert_eq!(
        plugmem(&tmp.db(), &["remember", "second writer"])
            .status
            .code(),
        Some(1),
        "a second writer is still locked out",
    );
}

/// A minimal `/v1/embeddings` mock: one thread, `responses` sequential canned
/// replies with deterministic `dim`-length vectors (seeded by input length),
/// honoring request order. Returns the base URL. No network leaves localhost.
fn spawn_mock_embedder(dim: usize, responses: usize) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        for _ in 0..responses {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let mut read = 0usize;
            let body_start = loop {
                read += sock.read(&mut buf[read..]).unwrap();
                let head = String::from_utf8_lossy(&buf[..read]);
                if let Some(at) = head.find("\r\n\r\n") {
                    let len: usize = head
                        .lines()
                        .find_map(|l| {
                            l.strip_prefix("content-length: ")
                                .or_else(|| l.strip_prefix("Content-Length: "))
                        })
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap();
                    if read >= at + 4 + len {
                        break at + 4;
                    }
                }
            };
            let body: serde_json::Value = serde_json::from_slice(&buf[body_start..read]).unwrap();
            let inputs = body["input"].as_array().unwrap();
            let data: Vec<serde_json::Value> = inputs
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    let seed = text.as_str().unwrap().len() as f32;
                    let embedding: Vec<f32> = (0..dim).map(|j| (seed + j as f32).sin()).collect();
                    serde_json::json!({ "index": i, "embedding": embedding })
                })
                .collect();
            let payload = serde_json::json!({ "data": data }).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                payload.len()
            );
            sock.write_all(response.as_bytes()).unwrap();
        }
    });
    (format!("http://{addr}/v1"), handle)
}

#[test]
fn recall_with_an_embedder_coexists_with_a_live_writer() {
    let dim = 8;
    // Exactly two embed calls: the seeding `remember`, then the `recall` query.
    let (url, server) = spawn_mock_embedder(dim, 2);
    let tmp = TempDir::new("embed-coexist");

    // A config pointing the embedder at the mock server.
    let config = tmp.0.join("config.toml");
    std::fs::write(
        &config,
        format!("[engine]\ndim = {dim}\n\n[embedder]\nkind = \"openai\"\nurl = \"{url}\"\nmodel = \"mock\"\n"),
    )
    .unwrap();

    let plug = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
            .arg("--db")
            .arg(tmp.db())
            .arg("--config")
            .arg(&config)
            .args(args)
            .output()
            .unwrap()
    };

    // Seed a fact (embeds its text via the mock) and publish a generation.
    assert!(
        plug(&["remember", "a fact about tokio and работа"])
            .status
            .success()
    );
    assert!(plug(&["checkpoint"]).status.success());

    // Hold the writer lock in-process, matching the on-disk vector dimension.
    let mut wcfg = Config::default();
    wcfg.dim = dim;
    let (_writer, _r) = Database::open(tmp.db(), wcfg).unwrap();

    // recall embeds its query via the mock (before the open), then opens
    // read-only and pins the published generation — coexisting with the writer.
    // Without the read-only path this would have opened the writer handle and
    // hit `Locked` (exit 1).
    let r = plug(&["recall", "tokio"]);
    assert_eq!(
        r.status.code(),
        Some(0),
        "embedded recall must coexist with a live writer; stderr={}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(stdout(&r).contains("tokio"), "{}", stdout(&r));

    server.join().unwrap();
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
    assert!(
        stdout(&out).contains("imported 3 facts"),
        "{}",
        stdout(&out)
    );

    // The imported facts are recallable, entity and tags preserved.
    let r = plugmem(&tmp.db(), &["recall", "tokio"]);
    assert!(
        stdout(&r).contains("tokio") && stdout(&r).contains("#pref"),
        "{}",
        stdout(&r)
    );
}

#[test]
fn verify_reports_a_clean_database() {
    let tmp = TempDir::new("verify-ok");
    plugmem(&tmp.db(), &["remember", "a clean fact"]);
    let out = plugmem(&tmp.db(), &["verify"]);
    assert!(out.status.success(), "verify of a clean db exits 0");
    assert!(stdout(&out).contains("integrity ok"), "{}", stdout(&out));
}

#[test]
fn scrub_verifies_a_checkpointed_database() {
    let tmp = TempDir::new("scrub-ok");
    plugmem(&tmp.db(), &["remember", "a fact worth some bytes to scrub"]);
    // maintain checkpoints (empties the journal) so the read-only scrub opens.
    plugmem(&tmp.db(), &["maintain"]);
    let out = plugmem(&tmp.db(), &["scrub"]);
    assert!(
        out.status.success(),
        "scrub of a clean db exits 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout(&out).contains("scrub ok"), "{}", stdout(&out));
}

#[test]
fn scrub_detects_on_disk_corruption() {
    let tmp = TempDir::new("scrub-corrupt");
    plugmem(&tmp.db(), &["remember", "a corruptible tokio fact"]);
    plugmem(&tmp.db(), &["maintain"]);
    // Flip a byte inside a section body (the text pool) of the snapshot.
    let snap = snapshot_file(&tmp.db());
    let mut bytes = std::fs::read(&snap).unwrap();
    let at = bytes.windows(5).position(|w| w == b"tokio").unwrap();
    bytes[at] ^= 0xFF;
    std::fs::write(&snap, bytes).unwrap();
    let out = plugmem(&tmp.db(), &["scrub"]);
    assert_eq!(out.status.code(), Some(2), "corruption exits 2");
}

#[test]
fn recover_writes_a_clean_copy_and_preserves_the_source() {
    let tmp = TempDir::new("recover");
    plugmem(&tmp.db(), &["remember", "keep me"]);
    plugmem(&tmp.db(), &["maintain"]); // materialize a snapshot file to salvage
    let dst = tmp.0.join("recovered.plugmem");
    let out = plugmem(&tmp.db(), &["recover", dst.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "recover exits 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout(&out).contains("recovered to"), "{}", stdout(&out));
    assert!(dst.exists(), "the destination was written");
    assert!(tmp.db().exists(), "the source is preserved");
    // The recovered database is usable.
    assert!(plugmem(&dst, &["stats"]).status.success());
}

#[test]
fn recover_refuses_a_destination_equal_to_the_source() {
    let tmp = TempDir::new("recover-same");
    plugmem(&tmp.db(), &["remember", "a fact"]);
    let out = plugmem(&tmp.db(), &["recover", tmp.db().to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2), "dst == src is a usage error");
}
