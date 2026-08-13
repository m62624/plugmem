//! Black-box tests of the `plugmem` binary: argument parsing, exit codes,
//! the human/JSON split, and lock behaviour. The command logic itself is
//! unit-tested in the library; here we exercise the actual executable.

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
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
/// honoring request order. Returns the full embeddings endpoint URL. No network
/// leaves localhost.
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
    (format!("http://{addr}/v1/embeddings"), handle)
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
        format!("[engine]\ndim = {dim}\n\n[embedder]\nurl = \"{url}\"\nmodel = \"mock\"\n"),
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

/// A mock embedder that serves until the test ends and counts what it embedded.
///
/// [`spawn_mock_embedder`] answers a fixed number of requests and is joined to
/// prove they all arrived — which turns "one request too few" into a hang
/// rather than a failure. Where the *count* is the assertion, this is the shape
/// to use: the number is read directly, and a session that stopped embedding
/// fails on the spot with both numbers printed.
fn spawn_counting_embedder(dim: usize) -> (String, std::sync::Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let embedded = std::sync::Arc::new(AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&embedded);
    std::thread::spawn(move || {
        for sock in listener.incoming() {
            let Ok(mut sock) = sock else { break };
            let mut buf = vec![0u8; 65536];
            let mut read = 0usize;
            let body_start = loop {
                match sock.read(&mut buf[read..]) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => read += n,
                }
                let head = String::from_utf8_lossy(&buf[..read]);
                if let Some(at) = head.find("\r\n\r\n") {
                    let len: usize = head
                        .lines()
                        .find_map(|l| {
                            l.strip_prefix("content-length: ")
                                .or_else(|| l.strip_prefix("Content-Length: "))
                        })
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                    if read >= at + 4 + len {
                        break at + 4;
                    }
                }
            };
            let body: serde_json::Value = serde_json::from_slice(&buf[body_start..read]).unwrap();
            let inputs = body["input"].as_array().unwrap();
            counter.fetch_add(inputs.len(), Ordering::SeqCst);
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
            let _ = sock.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1/embeddings"), embedded)
}

/// An address nothing listens on: bound, read, and dropped.
fn dead_embedder_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}/v1/embeddings")
}

#[test]
fn every_recall_of_a_read_only_repl_session_embeds_its_query() {
    // The repl is the CLI's long-lived mode, and it runs many commands against
    // one `Settings`. A first version of the embedder gate *took* the provider
    // out of the settings for the first recall, so the second and third lines
    // of a session silently searched without a vector source at all — the
    // quietest possible regression, since a lexical answer still looks like an
    // answer.
    use std::io::Write as _;
    use std::process::Stdio;

    let dim = 8;
    let (url, embedded) = spawn_counting_embedder(dim);
    let tmp = TempDir::new("repl-embed");
    let config = tmp.0.join("config.toml");
    std::fs::write(
        &config,
        format!("[engine]\ndim = {dim}\n\n[embedder]\nurl = \"{url}\"\nmodel = \"mock\"\n"),
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

    // Seed one fact and publish a generation, so the repl can open read-only.
    let seeded = plug(&["remember", "a fact about tokio"]);
    assert!(seeded.status.success(), "{}", stderr(&seeded));
    assert!(plug(&["checkpoint"]).status.success());
    assert_eq!(
        embedded.load(Ordering::SeqCst),
        1,
        "the write embedded once"
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(tmp.db())
        .arg("--config")
        .arg(&config)
        .args(["repl", "--read-only"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"recall tokio\nrecall tokio\nrecall tokio\nexit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "repl exit: {:?}", out.status);

    // Four in total: the seeding write plus one per recall line. Three would
    // mean a session that embedded once and then quietly stopped.
    assert_eq!(
        embedded.load(Ordering::SeqCst),
        4,
        "every repl recall must embed its query"
    );
}

#[test]
fn an_unreachable_embedder_fails_a_recall_by_default_and_degrades_on_request() {
    // The CLI embeds a text `recall` itself, before opening read-only, so the
    // policy has to reach that call too — otherwise `on_error = "degrade"`
    // would be honoured everywhere except the surface a person types into.
    let dim = 8;
    let tmp = TempDir::new("embed-degrade");
    let url = dead_embedder_url();

    let write_config = |name: &str, extra: &str| {
        let path = tmp.0.join(name);
        std::fs::write(
            &path,
            format!(
                "[engine]\ndim = {dim}\n\n[embedder]\nurl = \"{url}\"\nmodel = \"mock\"\n{extra}\n"
            ),
        )
        .unwrap();
        path
    };
    let strict = write_config("strict.toml", "");
    let lenient = write_config("lenient.toml", "on_error = \"degrade\"");

    let plug = |config: &std::path::Path, args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
            .arg("--db")
            .arg(tmp.db())
            .arg("--config")
            .arg(config)
            .args(args)
            .output()
            .unwrap()
    };

    // Seeded through the lenient config: the fact is stored without a vector,
    // which is the state a degraded write leaves behind.
    let stored = plug(&lenient, &["remember", "a fact about tokio"]);
    assert_eq!(stored.status.code(), Some(0), "{}", stderr(&stored));
    assert!(plug(&lenient, &["checkpoint"]).status.success());

    // The default: the provider is unreachable, so the command says so.
    let strict_recall = plug(&strict, &["recall", "tokio"]);
    // 2, the CLI's code for a reported error, not 1 (a locked database).
    assert_eq!(strict_recall.status.code(), Some(2));
    assert!(
        stderr(&strict_recall).contains("embedder"),
        "{}",
        stderr(&strict_recall)
    );

    // Degrade: the same query answers from the lexical source instead.
    let lenient_recall = plug(&lenient, &["recall", "tokio"]);
    assert_eq!(
        lenient_recall.status.code(),
        Some(0),
        "{}",
        stderr(&lenient_recall)
    );
    assert!(
        stdout(&lenient_recall).contains("tokio"),
        "{}",
        stdout(&lenient_recall)
    );
}

#[test]
fn a_recall_that_falls_back_to_the_writer_embeds_its_query_once() {
    // The read-only open fails on a database with no published generation yet,
    // and the read-write path embeds inside `Database::recall`. Embedding
    // before the open therefore paid for the same query twice — invisible in
    // the answer, and a doubled bill on a metered provider.
    let dim = 8;
    let (url, embedded) = spawn_counting_embedder(dim);
    let tmp = TempDir::new("fallback-embed");
    let config = tmp.0.join("config.toml");
    std::fs::write(
        &config,
        format!("[engine]\ndim = {dim}\n\n[embedder]\nurl = \"{url}\"\nmodel = \"mock\"\n"),
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

    let stored = plug(&["remember", "a fact about tokio"]);
    assert_eq!(stored.status.code(), Some(0), "{}", stderr(&stored));
    assert_eq!(
        embedded.load(Ordering::SeqCst),
        1,
        "the write embedded once"
    );

    // No checkpoint: there is no snapshot to map, so this recall takes the
    // read-write fallback.
    let recalled = plug(&["recall", "tokio"]);
    assert_eq!(recalled.status.code(), Some(0), "{}", stderr(&recalled));
    assert_eq!(
        embedded.load(Ordering::SeqCst),
        2,
        "the fallback recall must embed once, not once per path"
    );

    // And the read-only path still embeds exactly once, now that a generation
    // exists to map.
    assert!(plug(&["checkpoint"]).status.success());
    let readonly = plug(&["recall", "tokio"]);
    assert_eq!(readonly.status.code(), Some(0), "{}", stderr(&readonly));
    assert_eq!(embedded.load(Ordering::SeqCst), 3);
}

#[test]
fn stats_names_the_embedder_state_the_vector_count_cannot_explain() {
    // `vectors` below `facts` looks the same whether the provider died
    // mid-write or was never configured, and a person reading `stats` has no
    // other way to tell. The MCP server answers this in `plugmem_stats`; the
    // CLI printing nothing would make the same question unanswerable at a
    // terminal.
    use std::io::Write as _;
    use std::process::Stdio;

    let dim = 8;
    let tmp = TempDir::new("stats-embedder");
    let url = dead_embedder_url();
    let config = tmp.0.join("config.toml");
    std::fs::write(
        &config,
        format!(
            "[engine]\ndim = {dim}\n\n[embedder]\nurl = \"{url}\"\nmodel = \"mock\"\n\
             on_error = \"degrade\"\n"
        ),
    )
    .unwrap();
    let bare = tmp.0.join("bare.toml");
    std::fs::write(&bare, format!("[engine]\ndim = {dim}\n")).unwrap();

    let plug = |config: &std::path::Path, args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
            .arg("--db")
            .arg(tmp.db())
            .arg("--config")
            .arg(config)
            .args(args)
            .output()
            .unwrap()
    };

    // No embedder in the config at all: "absent", and the writer path prints
    // it too (this database has no snapshot yet, so `stats` opens read-write).
    let absent = plug(&bare, &["stats"]);
    assert!(
        stdout(&absent).contains("embedder    absent"),
        "{}",
        stdout(&absent)
    );

    // Configured but never called yet: "active". Nothing has failed, and a
    // state read must not invent a failure to report.
    let stored = plug(&config, &["remember", "a fact about tokio"]);
    assert_eq!(stored.status.code(), Some(0), "{}", stderr(&stored));
    assert!(plug(&config, &["checkpoint"]).status.success());
    let json = plug(&config, &["--json", "stats"]);
    assert!(
        stdout(&json).contains("\"embedder\":\"active\""),
        "{}",
        stdout(&json)
    );

    // And after a failure inside one process: "suspended". It has to be one
    // process, because the suspension is per-process state — which is exactly
    // why a one-shot command can only ever report "active" here.
    let mut child = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(tmp.db())
        .arg("--config")
        .arg(&config)
        .args(["repl", "--read-only"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"recall tokio\nstats\nexit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "repl exit: {:?}", out.status);
    assert!(
        stdout(&out).contains("embedder    suspended"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn import_streams_the_file_in_batches_one_http_each() {
    let dim = 8;
    // 5 facts imported with --batch 2 → ceil(5/2) = 3 embedder round-trips
    // (batches of 2, 2, 1). The mock is told to serve exactly 3; `join` at the
    // end proves it neither hung waiting for a 4th nor was hit by an extra.
    let (url, server) = spawn_mock_embedder(dim, 3);
    let tmp = TempDir::new("import-chunk");

    let config = tmp.0.join("config.toml");
    std::fs::write(
        &config,
        format!("[engine]\ndim = {dim}\n\n[embedder]\nurl = \"{url}\"\nmodel = \"mock\"\n"),
    )
    .unwrap();

    let facts = tmp.0.join("facts.jsonl");
    let mut jsonl = String::new();
    for i in 0..5 {
        jsonl.push_str(&format!("{{\"text\":\"fact number {i}\"}}\n"));
    }
    std::fs::write(&facts, &jsonl).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(tmp.db())
        .arg("--config")
        .arg(&config)
        .arg("import")
        .arg(&facts)
        .arg("--batch")
        .arg("2")
        .output()
        .unwrap();

    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout(&out).contains("imported 5 facts"),
        "{}",
        stdout(&out)
    );
    // Exactly three round-trips were made.
    server.join().unwrap();
}

#[test]
fn repl_session_over_the_binary_runs_and_checkpoints_on_exit() {
    use std::io::Write as _;
    use std::process::Stdio;

    let tmp = TempDir::new("repl");
    let mut child = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--db")
        .arg(tmp.db())
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"remember \"user likes tokio\"\nrecall tokio\nstats\nexit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "repl exit: {:?}", out.status);
    let so = String::from_utf8_lossy(&out.stdout);
    assert!(so.contains("remembered fact 0"), "stdout={so}");
    assert!(so.contains("tokio"), "stdout={so}");

    // The session checkpointed on exit → a standalone read-only recall sees the
    // data with a clean journal (no dirty-journal fallback).
    let r = plugmem(&tmp.db(), &["recall", "tokio"]);
    assert!(
        r.status.success() && stdout(&r).contains("tokio"),
        "post-repl recall: {}",
        stdout(&r)
    );
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
fn embedder_enabled_env_overrides_a_configured_endpoint() {
    let tmp = TempDir::new("embedder-enabled-env");
    let config = tmp.0.join("config.toml");
    std::fs::write(
        &config,
        "[engine]\ndim = 8\n[embedder]\nenabled = true\nurl = \"http://127.0.0.1:1/v1/embeddings\"\nmodel = \"mock\"\n",
    )
    .unwrap();

    // false prevents the configured client from being created, so a write
    // succeeds without contacting the unreachable endpoint.
    let disabled = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .args(["--config", config.to_str().unwrap(), "--db"])
        .arg(tmp.db())
        .args(["remember", "env-disabled"])
        .env("PLUGMEM_EMBEDDER_ENABLED", "false")
        .output()
        .unwrap();
    assert!(disabled.status.success(), "{}", stdout(&disabled));

    // true overrides a file-level false/omission in the opposite direction;
    // the unreachable endpoint now surfaces as an embedder failure.
    let enabled = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .args(["--config", config.to_str().unwrap(), "--db"])
        .arg(tmp.0.join("enabled.plugmem"))
        .args(["remember", "env-enabled"])
        .env("PLUGMEM_EMBEDDER_ENABLED", "true")
        .output()
        .unwrap();
    assert!(!enabled.status.success());
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
    // checkpoint empties the journal so the read-only scrub opens.
    plugmem(&tmp.db(), &["checkpoint"]);
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
    plugmem(&tmp.db(), &["checkpoint"]);
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
    plugmem(&tmp.db(), &["checkpoint"]); // materialize a snapshot file to salvage
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

/// The three query/edge knobs that used to be reachable only from Rust.
///
/// Each was hardcoded to `None` in every wrapper — CLI, MCP and napi alike —
/// so the engine supported them and nobody could reach them. These tests are
/// the parity gate: they fail if a wrapper stops passing one through.
#[test]
fn the_token_budget_bounds_the_rendered_block() {
    let tmp = TempDir::new("token-budget");
    for i in 0..40 {
        plugmem(
            &tmp.db(),
            &[
                "remember",
                &format!("fact number {i} about the deployment pipeline and its many stages"),
            ],
        );
    }

    let generous = plugmem(&tmp.db(), &["recall", "deployment pipeline", "-k", "40"]);
    let tight = plugmem(
        &tmp.db(),
        &[
            "recall",
            "deployment pipeline",
            "-k",
            "40",
            "--token-budget",
            "40",
        ],
    );
    assert!(generous.status.success() && tight.status.success());

    let (big, small) = (stdout(&generous).len(), stdout(&tight).len());
    assert!(
        small < big,
        "a tight budget must shrink the block: {small} vs {big} bytes"
    );
}

#[test]
fn ef_is_accepted_and_does_not_change_a_lexical_answer() {
    // `ef` only steers the vector source, and this database has no vectors,
    // so the observable contract here is "accepted, and answers identically".
    let tmp = TempDir::new("ef");
    plugmem(&tmp.db(), &["remember", "the release ships on friday"]);

    let plain = plugmem(&tmp.db(), &["recall", "release", "--json"]);
    let with_ef = plugmem(&tmp.db(), &["recall", "release", "--ef", "64", "--json"]);
    assert!(with_ef.status.success(), "--ef must be accepted");
    assert_eq!(stdout(&plain), stdout(&with_ef));
}

#[test]
fn provenance_is_recorded_on_an_edge_and_comes_back_from_recall() {
    let tmp = TempDir::new("provenance");
    // The fact that justifies the edge.
    let out = plugmem(&tmp.db(), &["remember", "ann hired bob in march", "--json"]);
    let source: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("remember json");
    let fact_id = source["id"].as_u64().expect("fact id");

    let linked = plugmem(
        &tmp.db(),
        &[
            "link",
            "ann",
            "hires",
            "bob",
            "--provenance",
            &fact_id.to_string(),
        ],
    );
    assert!(linked.status.success(), "--provenance must be accepted");

    let recalled = plugmem(&tmp.db(), &["recall", "--entity", "ann", "--json"]);
    let res: serde_json::Value = serde_json::from_str(&stdout(&recalled)).expect("recall json");
    let edges = res["edges"].as_array().expect("recall JSON carries edges");
    assert!(!edges.is_empty(), "the graph source walked the edge");
    assert!(
        edges
            .iter()
            .any(|e| e["provenance"].as_u64() == Some(fact_id)),
        "the edge names the fact it follows from: {edges:?}"
    );
}

#[test]
fn an_edge_without_provenance_reports_none_rather_than_a_sentinel() {
    // The engine stores `FactId::NONE` for "no source fact". Leaking that
    // sentinel as a number would make callers compare against a magic value.
    let tmp = TempDir::new("no-provenance");
    plugmem(&tmp.db(), &["link", "team", "hires", "bob"]);
    let recalled = plugmem(&tmp.db(), &["recall", "--entity", "team", "--json"]);
    let res: serde_json::Value = serde_json::from_str(&stdout(&recalled)).expect("recall json");
    for edge in res["edges"].as_array().expect("edges") {
        assert!(
            edge["provenance"].is_null(),
            "an unsourced edge reports null, not a sentinel: {edge}"
        );
    }
}

#[test]
fn the_fsync_policy_is_settable_from_the_config_file() {
    // `FsyncPolicy` was public in the host and reachable from no wrapper at
    // all. It is a config setting rather than a flag because it changes what
    // survives a power cut, which is not a per-command decision.
    let tmp = TempDir::new("fsync");
    let cfg = tmp.0.join("config.toml");
    std::fs::write(&cfg, "[maintenance]\nfsync = \"on_snapshot\"\n").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--config")
        .arg(&cfg)
        .arg("--db")
        .arg(tmp.db())
        .args(["remember", "written under a relaxed fsync policy"])
        .output()
        .expect("run plugmem");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // And a misspelling is refused rather than silently ignored.
    std::fs::write(&cfg, "[maintenance]\nfsync = \"on-snapshot\"\n").unwrap();
    let bad = Command::new(env!("CARGO_BIN_EXE_plugmem-cli"))
        .arg("--config")
        .arg(&cfg)
        .arg("--db")
        .arg(tmp.db())
        .args(["stats"])
        .output()
        .expect("run plugmem");
    assert!(!bad.status.success(), "a bad fsync value must be refused");
}

#[test]
fn settings_help_lists_shared_settings() {
    let tmp = TempDir::new("settings-fsync");
    let out = plugmem(&tmp.db(), &["help", "settings"]);
    let text = stdout(&out);
    assert!(
        text.contains("fsync"),
        "the settings catalogue documents every key it parses"
    );
    assert!(
        text.contains("space_id"),
        "the CLI must expose the shared embedding-space setting"
    );
}

/// The JSONL round trip used to lose the graph: `export` wrote facts only, so
/// `import` rebuilt a memory with none of its edges — one of the four recall
/// sources, gone without a word. These hold the format honest.
#[test]
fn the_round_trip_carries_edges_and_their_provenance() {
    let src = TempDir::new("rt-src");
    let out = plugmem(&src.db(), &["remember", "ann hired bob in march", "--json"]);
    let fact: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("remember json");
    let fact_id = fact["id"].as_u64().expect("fact id");
    plugmem(&src.db(), &["remember", "bob reviews the notes"]);
    plugmem(
        &src.db(),
        &[
            "link",
            "ann",
            "hires",
            "bob",
            "--provenance",
            &fact_id.to_string(),
        ],
    );
    plugmem(&src.db(), &["link", "ann", "knows", "bob"]);

    let dump = plugmem(&src.db(), &["export"]);
    let text = stdout(&dump);
    let facts = text
        .lines()
        .filter(|l| l.contains("\"kind\":\"fact\""))
        .count();
    let edges = text
        .lines()
        .filter(|l| l.contains("\"kind\":\"edge\""))
        .count();
    assert_eq!((facts, edges), (2, 2), "both kinds are written:\n{text}");

    // Every fact line precedes every edge line: that ordering is what lets the
    // importer resolve a provenance in one forward pass.
    let kinds: Vec<&str> = text.lines().collect();
    let last_fact = kinds
        .iter()
        .rposition(|l| l.contains("\"kind\":\"fact\""))
        .unwrap();
    let first_edge = kinds
        .iter()
        .position(|l| l.contains("\"kind\":\"edge\""))
        .unwrap();
    assert!(last_fact < first_edge, "facts must come before edges");

    let dst = TempDir::new("rt-dst");
    let file = src.0.join("dump.jsonl");
    std::fs::write(&file, &text).unwrap();
    let imported = plugmem(&dst.db(), &["import", file.to_str().unwrap(), "--json"]);
    let report: serde_json::Value = serde_json::from_str(&stdout(&imported)).expect("import json");
    assert_eq!(report["imported"], 2);
    assert_eq!(report["edges"], 2, "edges are imported, not skipped");

    let recalled = plugmem(&dst.db(), &["recall", "--entity", "ann", "--json"]);
    let res: serde_json::Value = serde_json::from_str(&stdout(&recalled)).expect("recall json");
    let got = res["edges"].as_array().expect("edges");
    assert_eq!(got.len(), 2, "both edges came back");
    assert_eq!(
        got.iter().filter(|e| !e["provenance"].is_null()).count(),
        1,
        "exactly the sourced edge kept its provenance: {got:?}"
    );
}

#[test]
fn provenance_is_retargeted_to_the_id_the_new_database_assigns() {
    // The subtle half: ids do not survive an import, so a provenance copied
    // verbatim would point at an unrelated fact. It has to be translated.
    let src = TempDir::new("remap-src");
    let out = plugmem(&src.db(), &["remember", "ann hired bob", "--json"]);
    let old_id = serde_json::from_str::<serde_json::Value>(&stdout(&out)).unwrap()["id"]
        .as_u64()
        .unwrap();
    plugmem(
        &src.db(),
        &[
            "link",
            "ann",
            "hires",
            "bob",
            "--provenance",
            &old_id.to_string(),
        ],
    );
    let file = src.0.join("dump.jsonl");
    std::fs::write(&file, stdout(&plugmem(&src.db(), &["export"]))).unwrap();

    // Pre-fill the destination so the same fact lands on a different id.
    let dst = TempDir::new("remap-dst");
    for i in 0..3 {
        plugmem(&dst.db(), &["remember", &format!("filler {i}")]);
    }
    plugmem(&dst.db(), &["import", file.to_str().unwrap()]);

    let hit = plugmem(&dst.db(), &["recall", "ann hired", "--json"]);
    let new_id =
        serde_json::from_str::<serde_json::Value>(&stdout(&hit)).unwrap()["facts"][0]["id"]
            .as_u64()
            .unwrap();
    assert_ne!(new_id, old_id, "the destination must assign a different id");

    let recalled = plugmem(&dst.db(), &["recall", "--entity", "ann", "--json"]);
    let res: serde_json::Value = serde_json::from_str(&stdout(&recalled)).unwrap();
    assert!(
        res["edges"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["provenance"].as_u64() == Some(new_id)),
        "provenance follows the fact to its new id: {}",
        res["edges"]
    );
}

#[test]
fn a_file_written_before_edges_existed_still_imports() {
    // Backwards compatibility is the reason `kind` is optional on read: a line
    // without one is a fact, exactly as the old format meant it.
    let tmp = TempDir::new("legacy");
    let file = tmp.0.join("legacy.jsonl");
    std::fs::write(
        &file,
        "{\"text\":\"legacy fact\",\"entity\":\"old\",\"tags\":[\"x\"],\"valid_from\":42}\n",
    )
    .unwrap();

    let out = plugmem(&tmp.db(), &["import", file.to_str().unwrap(), "--json"]);
    let report: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("import json");
    assert_eq!(report["imported"], 1);
    assert_eq!(report["edges"], 0);

    let hit = plugmem(&tmp.db(), &["recall", "legacy", "--json"]);
    let res: serde_json::Value = serde_json::from_str(&stdout(&hit)).unwrap();
    assert_eq!(
        res["facts"][0]["valid_from"], 42,
        "the old fields still apply"
    );
}

#[test]
fn an_unknown_line_kind_is_refused_rather_than_skipped() {
    let tmp = TempDir::new("bad-kind");
    let file = tmp.0.join("bad.jsonl");
    std::fs::write(&file, "{\"kind\":\"vector\",\"text\":\"x\"}\n").unwrap();
    let out = plugmem(&tmp.db(), &["import", file.to_str().unwrap()]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an unknown kind is a usage error"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("line 1"), "the message names the line: {err}");
}
