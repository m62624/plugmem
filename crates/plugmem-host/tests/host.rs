//! Host-layer tests (specs/13 §6): file storage roundtrips, locking,
//! maintenance policy, auto-embedding against a local mock server, and
//! multi-threaded handles.

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::PathBuf;

use plugmem_host::{
    Config, Database, Embedder, FactId, FsyncPolicy, HostError, NullEmbedder, OpenAiCompatEmbedder,
    RecallQuery, RememberInput,
};

/// A unique temp directory per test; removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let unique = format!(
            "plugmem-host-{tag}-{}-{}",
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
        self.0.join("agent.plugmem")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cfg() -> Config {
    let mut cfg = Config::default();
    cfg.shards_facts = 8;
    cfg.shards_entities = 4;
    cfg.shards_edges = 4;
    cfg.shards_temporal = 4;
    cfg.shards_postings = 16;
    cfg
}

#[test]
fn open_remember_reopen_replays_the_journal() {
    let tmp = TempDir::new("roundtrip");
    let id = {
        let (db, report) = Database::open(tmp.db(), cfg()).unwrap();
        assert_eq!(report.replayed, 0);
        let out = db
            .remember(RememberInput {
                entity: Some("user"),
                tags: &["pref"],
                ..RememberInput::text(1_000, "prefers tokio")
            })
            .unwrap();
        db.link(plugmem_host::LinkInput {
            now: 2_000,
            src: "user",
            rel: "works_on",
            dst: "plugmem",
            provenance: Some(out.id),
        })
        .unwrap();
        out.id
    }; // drop releases the lock

    let (db, report) = Database::open(tmp.db(), cfg()).unwrap();
    assert_eq!(report.replayed, 2, "the journal replays on reopen");
    let fact = db.get(id).expect("the fact survived the reopen");
    assert_eq!(fact.text, "prefers tokio");
    let out = db.recall(RecallQuery::text(3_000, "tokio")).unwrap();
    assert!(out.rendered.contains("prefers tokio"));
}

#[test]
fn the_lock_refuses_a_second_owner() {
    let tmp = TempDir::new("lock");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    match Database::open(tmp.db(), cfg()) {
        Err(HostError::Locked { path }) => assert_eq!(path, tmp.db()),
        other => panic!("expected Locked, got {other:?}"),
    }
    drop(db);
    // Released on drop: the file opens again.
    Database::open(tmp.db(), cfg()).unwrap();
}

#[test]
fn checkpoint_and_snapshot_policy() {
    let tmp = TempDir::new("policy");
    let (db, _) = Database::builder(cfg())
        .snapshot_every_ops(5)
        .fsync(FsyncPolicy::OnSnapshot)
        .open(tmp.db())
        .unwrap();
    let journal = {
        let mut p = tmp.db().into_os_string();
        p.push(".journal");
        PathBuf::from(p)
    };
    for i in 0..5u64 {
        db.remember(RememberInput::text(i + 1, "some fact text here"))
            .unwrap();
    }
    // The fifth mutation crossed the threshold: snapshot written,
    // journal truncated.
    assert!(tmp.db().exists(), "the snapshot file must exist");
    assert_eq!(std::fs::metadata(&journal).unwrap().len(), 0);
    // And no tmp scrap survives.
    assert!(!tmp.0.join("agent.plugmem.tmp").exists());

    // An explicit checkpoint works too.
    db.remember(RememberInput::text(10, "one more")).unwrap();
    assert!(std::fs::metadata(&journal).unwrap().len() > 0);
    db.checkpoint(11).unwrap();
    assert_eq!(std::fs::metadata(&journal).unwrap().len(), 0);

    // Reopen from the snapshot alone.
    drop(db);
    let (db, report) = Database::open(tmp.db(), cfg()).unwrap();
    assert_eq!(report.replayed, 0, "everything came from the snapshot");
    assert_eq!(db.stats().facts, 6);
}

#[test]
fn torn_journal_tail_is_dropped_on_open() {
    let tmp = TempDir::new("torn");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        db.remember(RememberInput::text(1, "fact one")).unwrap();
    }
    // Simulate a crash mid-append: garbage that parses as a huge frame.
    let journal = {
        let mut p = tmp.db().into_os_string();
        p.push(".journal");
        PathBuf::from(p)
    };
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(&journal)
        .unwrap();
    f.write_all(&[0xFF, 0xFF, 0x00, 0x00, 0xAA]).unwrap();
    drop(f);

    let (db, report) = Database::open(tmp.db(), cfg()).unwrap();
    assert!(report.truncated_tail, "the torn tail must be reported");
    assert_eq!(report.replayed, 1);
    assert_eq!(db.stats().facts, 1);
}

#[test]
fn maintain_policy_fires_on_forgets() {
    let tmp = TempDir::new("maintain");
    let (db, _) = Database::builder(cfg())
        .maintain_every_forgets(3)
        .open(tmp.db())
        .unwrap();
    for i in 0..6u64 {
        db.remember(RememberInput::text(i + 1, "fact to forget later"))
            .unwrap();
    }
    for id in 0..3u32 {
        db.forget(100, FactId(id)).unwrap();
    }
    // The third forget crossed the threshold: the tombstones are gone
    // physically (maintain v2), ids stay burned.
    let stats = db.stats();
    assert_eq!(stats.facts, 3, "purged records are removed");
    assert_eq!(stats.next_fact, 6, "ids are never reused");
}

#[test]
fn concurrent_handles_share_one_file() {
    let tmp = TempDir::new("threads");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    let threads: Vec<_> = (0..4u64)
        .map(|t| {
            let db = db.clone();
            std::thread::spawn(move || {
                for i in 0..50u64 {
                    db.remember(RememberInput::text(
                        t * 1_000 + i + 1,
                        "a fact from a worker thread",
                    ))
                    .unwrap();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(db.stats().facts, 200, "every write landed exactly once");
    let out = db.recall(RecallQuery::text(10_000, "worker")).unwrap();
    assert!(!out.rendered.is_empty());
}

/// A minimal `/v1/embeddings` mock: one thread, canned deterministic
/// vectors derived from each input's length, honoring the request order.
fn spawn_mock_embedder(dim: usize, responses: usize) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        for _ in 0..responses {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let mut read = 0usize;
            // Read until the full body arrived (Content-Length framed).
            let body_start = loop {
                read += sock.read(&mut buf[read..]).unwrap();
                let head = String::from_utf8_lossy(&buf[..read]);
                if let Some(at) = head.find("\r\n\r\n") {
                    let len: usize = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length: "))
                        .or_else(|| {
                            head.lines()
                                .find_map(|l| l.strip_prefix("Content-Length: "))
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
fn auto_embedding_end_to_end() {
    let dim = 8;
    let (url, server) = spawn_mock_embedder(dim, 3);
    let tmp = TempDir::new("embed");
    let mut config = cfg();
    config.dim = dim;
    let (db, _) = Database::builder(config)
        .embedder(Box::new(OpenAiCompatEmbedder::new(&url, "mock-model", dim)))
        .open(tmp.db())
        .unwrap();

    // remember embeds the text; recall embeds the query; a same-length
    // query gets the identical mock vector, so the match is exact.
    let id = db
        .remember(RememberInput::text(1, "twelve chars"))
        .unwrap()
        .id;
    db.remember(RememberInput::text(2, "a very different length text"))
        .unwrap();
    let out = db
        .recall(RecallQuery {
            text: Some("also 12 char"), // same length -> same mock vector
            k: 1,
            ..RecallQuery::text(3, "")
        })
        .unwrap();
    assert_eq!(out.facts[0].id, id, "semantic recall through the mock");
    server.join().unwrap();
}

#[test]
fn embedder_transport_and_shape_errors_are_typed() {
    // A refused connection is a typed Embed error.
    let mut refused = OpenAiCompatEmbedder::new("http://127.0.0.1:1/v1", "m", 4);
    assert!(matches!(refused.embed(&["x"]), Err(HostError::Embed(_))));

    // A server sending the wrong dimension is a typed Embed error.
    let (url, server) = spawn_mock_embedder(4, 1);
    let mut wrong = OpenAiCompatEmbedder::new(&url, "m", 5);
    assert!(matches!(wrong.embed(&["abc"]), Err(HostError::Embed(_))));
    server.join().unwrap();

    // The dimension gate at open: embedder dim must equal Config::dim.
    let tmp = TempDir::new("dimgate");
    let mut config = cfg();
    config.dim = 4;
    let err = Database::builder(config)
        .embedder(Box::new(OpenAiCompatEmbedder::new("http://x/v1", "m", 8)))
        .open(tmp.db())
        .unwrap_err();
    assert!(matches!(err, HostError::Engine(_)));

    // NullEmbedder means "no vectors": remember works, nothing embeds.
    let tmp = TempDir::new("null");
    let (db, _) = Database::builder(cfg())
        .embedder(Box::new(NullEmbedder))
        .open(tmp.db())
        .unwrap();
    db.remember(RememberInput::text(1, "plain")).unwrap();
    assert_eq!(db.stats().vectors, 0);
}

/// A one-shot server answering any request with the given raw JSON body.
fn spawn_canned(payload: String) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 65536];
        let _ = sock.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
            payload.len()
        );
        sock.write_all(response.as_bytes()).unwrap();
    });
    (format!("http://{addr}/v1"), handle)
}

#[test]
fn revise_and_journal_bytes_policy_and_debug() {
    let tmp = TempDir::new("revise");
    // A 1-byte journal threshold: every mutation snapshots immediately.
    let (db, _) = Database::builder(cfg())
        .snapshot_journal_bytes(1)
        .snapshot_every_ops(0)
        .open(tmp.db())
        .unwrap();
    let old = db
        .remember(RememberInput {
            entity: Some("user"),
            ..RememberInput::text(1, "lives in moscow")
        })
        .unwrap()
        .id;
    let journal = {
        let mut p = tmp.db().into_os_string();
        p.push(".journal");
        PathBuf::from(p)
    };
    assert_eq!(
        std::fs::metadata(&journal).unwrap().len(),
        0,
        "the byte threshold snapshots after every op"
    );

    let new = db
        .revise(
            old,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(2, "lives in berlin")
            },
        )
        .unwrap()
        .id;
    let closed = db.get(old).expect("closed facts stay readable");
    assert!(closed.record.is_closed());
    assert_eq!(db.get(new).unwrap().record.revises, old);

    let shown = format!("{db:?}");
    assert!(shown.contains("facts"), "Debug prints a summary: {shown}");
}

#[test]
fn leftover_tmp_scrap_is_discarded_on_open() {
    let tmp = TempDir::new("scrap");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        db.remember(RememberInput::text(1, "keep me")).unwrap();
        db.checkpoint(2).unwrap();
    }
    // A crashed half-write leaves a tmp file; the snapshot is intact.
    let scrap = tmp.0.join("agent.plugmem.tmp");
    std::fs::write(&scrap, b"half-written garbage").unwrap();
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    assert!(!scrap.exists(), "the scrap must be removed");
    assert_eq!(db.stats().facts, 1, "the real snapshot loaded");
}

#[test]
fn embedder_edge_cases() {
    // NullEmbedder's contract, called directly.
    let mut null = NullEmbedder;
    assert_eq!(null.dim(), 0);
    assert_eq!(null.embed(&["a", "b"]).unwrap(), vec![Vec::<f32>::new(); 2]);

    // Empty input short-circuits without a network call.
    let mut e = OpenAiCompatEmbedder::new("http://127.0.0.1:1/v1", "m", 4);
    assert!(e.embed(&[]).unwrap().is_empty());

    // The api-key path sends and succeeds against the mock.
    let (url, server) = spawn_mock_embedder(4, 1);
    let mut keyed = OpenAiCompatEmbedder::new(&url, "m", 4).with_api_key("sk-test");
    assert_eq!(keyed.embed(&["abc"]).unwrap()[0].len(), 4);
    server.join().unwrap();

    // Malformed server answers are typed Embed errors, never panics.
    for payload in [
        r#"{"nodata": true}"#,                                         // no data array
        r#"{"data": []}"#,                                             // wrong count
        r#"{"data": [{"index": 7, "embedding": [1.0,2.0,3.0,4.0]}]}"#, // bad index
        r#"{"data": [{"embedding": [1.0,2.0,3.0,4.0]}]}"#,             // no index
        r#"{"data": [{"index": 0, "embedding": "nope"}]}"#,            // not an array
        r#"{"data": [{"index": 0, "embedding": [1.0, "x", 3.0, 4.0]}]}"#, // NaN-ish member
        "not json at all",                                             // body parse
    ] {
        let (url, server) = spawn_canned(payload.to_string());
        let mut e = OpenAiCompatEmbedder::new(&url, "m", 4);
        assert!(
            matches!(e.embed(&["abc"]), Err(HostError::Embed(_))),
            "payload {payload:?} must be a typed error"
        );
        server.join().unwrap();
    }
}

#[test]
fn file_storage_direct_and_io_errors() {
    use plugmem_core::Storage as _;
    use plugmem_host::{FileStorage, FsyncPolicy};

    // Direct storage use: accessors and the journal growth counter.
    let tmp = TempDir::new("storage");
    let mut fs = FileStorage::open(tmp.db(), FsyncPolicy::EachOp).unwrap();
    assert_eq!(fs.path(), tmp.db());
    assert_eq!(fs.journal_bytes(), 0);
    assert_eq!(fs.read_snapshot().unwrap(), None);
    fs.append_journal(b"0123456789").unwrap();
    assert_eq!(fs.journal_bytes(), 10);
    assert_eq!(fs.read_journal().unwrap(), b"0123456789");
    fs.write_snapshot(b"image-bytes").unwrap();
    assert_eq!(
        fs.read_snapshot().unwrap().as_deref(),
        Some(&b"image-bytes"[..])
    );
    fs.clear_journal().unwrap();
    assert_eq!(fs.journal_bytes(), 0);

    // A base inside a nonexistent directory is a typed Io error.
    let missing = tmp.0.join("no/such/dir/agent.plugmem");
    match FileStorage::open(&missing, FsyncPolicy::EachOp) {
        Err(HostError::Io { .. }) => {}
        other => panic!("expected Io, got {other:?}"),
    }
}
