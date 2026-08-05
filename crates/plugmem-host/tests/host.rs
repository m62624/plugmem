//! Host-layer tests: file storage roundtrips, locking,
//! maintenance policy, auto-embedding against a local mock server, and
//! multi-threaded handles.

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use plugmem_host::{
    Config, Database, Embedder, FactId, FsyncPolicy, HostError, NullEmbedder, OpenAiCompatEmbedder,
    ReadOnlyDatabase, RecallQuery, RememberInput, ShardLayout, UnlinkInput,
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
    Config::default()
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
        assert!(
            db.unlink(UnlinkInput {
                now: 2_500,
                src: "user",
                rel: "works_on",
                dst: "plugmem",
            })
            .unwrap()
        );
        out.id
    }; // drop releases the lock

    let (db, report) = Database::open(tmp.db(), cfg()).unwrap();
    assert_eq!(report.replayed, 3, "the journal replays on reopen");
    let fact = db.get(id).expect("the fact survived the reopen");
    assert_eq!(fact.text, "prefers tokio");
    let out = db.recall(RecallQuery::text(3_000, "tokio")).unwrap();
    assert!(out.rendered.contains("prefers tokio"));
    let current = db
        .recall(RecallQuery {
            entities: &["user"],
            ..RecallQuery::text(3_000, "")
        })
        .unwrap();
    assert!(current.edges.is_empty());
    let historical = db
        .recall(RecallQuery {
            entities: &["user"],
            as_of: Some(2_250),
            ..RecallQuery::text(3_000, "")
        })
        .unwrap();
    assert_eq!(historical.edges.len(), 1);
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
fn maintain_noop_does_not_rewrite_the_snapshot() {
    let tmp = TempDir::new("maintain-noop");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(1, "stable fact")).unwrap();
    db.checkpoint(2).unwrap();
    let before = snapshot_file(&tmp.db());
    let before_len = std::fs::metadata(&before).unwrap().len();

    let report = db.maintain(3).unwrap();
    assert!(report.no_op);
    assert_eq!(report.purged, 0);
    assert!(!report.structural_compacted);
    assert_eq!(snapshot_file(&tmp.db()), before);
    assert_eq!(std::fs::metadata(before).unwrap().len(), before_len);
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

/// An in-process embedder that counts calls and texts, so a test can prove a
/// batch of K texts costs **one** embed call (not K). Deterministic vectors
/// (slot 0 = text length) keep recall reproducible.
struct CountingEmbedder {
    dim: usize,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    texts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl CountingEmbedder {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            calls: Default::default(),
            texts: Default::default(),
        }
    }
}

impl Embedder for CountingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        use std::sync::atomic::Ordering;
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dim];
                v[0] = t.len() as f32;
                v
            })
            .collect())
    }
}

#[test]
fn remember_many_embeds_the_whole_batch_in_one_call() {
    use std::sync::atomic::Ordering;
    let dim = 8;
    let emb = CountingEmbedder::new(dim);
    let (calls, texts) = (emb.calls.clone(), emb.texts.clone());
    let tmp = TempDir::new("batch-embed");
    let mut config = cfg();
    config.dim = dim;
    let (db, _) = Database::builder(config)
        .embedder(Box::new(emb))
        .open(tmp.db())
        .unwrap();

    let outs = db
        .remember_many(vec![
            RememberInput::text(1, "first fact"),
            RememberInput::text(2, "second fact longer"),
            RememberInput::text(3, "third"),
        ])
        .unwrap();

    assert_eq!(outs.len(), 3);
    assert_eq!(db.stats().facts, 3);
    // The whole batch is ONE embed call over three texts — not three calls.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "one round-trip for the batch"
    );
    assert_eq!(texts.load(Ordering::SeqCst), 3, "all three texts in it");
}

#[test]
fn remember_many_matches_single_remembers() {
    let dim = 8;
    let mut config = cfg();
    config.dim = dim;

    let tmp = TempDir::new("batch-eq");
    let (db, _) = Database::builder(config.clone())
        .embedder(Box::new(CountingEmbedder::new(dim)))
        .open(tmp.db())
        .unwrap();
    db.remember_many(vec![
        RememberInput::text(1, "alpha runtime tokio"),
        RememberInput::text(2, "beta lives berlin"),
    ])
    .unwrap();

    // A twin database written one at a time.
    let tmp2 = TempDir::new("batch-eq2");
    let (db2, _) = Database::builder(config)
        .embedder(Box::new(CountingEmbedder::new(dim)))
        .open(tmp2.db())
        .unwrap();
    db2.remember(RememberInput::text(1, "alpha runtime tokio"))
        .unwrap();
    db2.remember(RememberInput::text(2, "beta lives berlin"))
        .unwrap();

    assert_eq!(db.stats().facts, db2.stats().facts);
    let q = RecallQuery {
        k: 5,
        ..RecallQuery::text(9, "runtime")
    };
    let batch_ids: Vec<_> = db.recall(q).unwrap().facts.iter().map(|f| f.id).collect();
    let single_ids: Vec<_> = db2.recall(q).unwrap().facts.iter().map(|f| f.id).collect();
    assert_eq!(batch_ids, single_ids, "batch and singles agree");
}

#[test]
fn remember_many_skips_inputs_that_already_have_a_vector() {
    use std::sync::atomic::Ordering;
    let dim = 4;
    let emb = CountingEmbedder::new(dim);
    let texts = emb.texts.clone();
    let tmp = TempDir::new("batch-skip");
    let mut config = cfg();
    config.dim = dim;
    let (db, _) = Database::builder(config)
        .embedder(Box::new(emb))
        .open(tmp.db())
        .unwrap();

    let v = vec![1.0f32; dim];
    let with_vec = RememberInput {
        vector: Some(&v),
        ..RememberInput::text(1, "has a vector already")
    };
    db.remember_many(vec![RememberInput::text(2, "needs embedding"), with_vec])
        .unwrap();

    // Only the vector-less input's text reached the embedder.
    assert_eq!(texts.load(Ordering::SeqCst), 1);
    assert_eq!(db.stats().facts, 2);
}

#[test]
fn remember_many_empty_is_a_noop() {
    let tmp = TempDir::new("batch-empty");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    assert!(db.remember_many(vec![]).unwrap().is_empty());
    assert_eq!(db.stats().facts, 0);
}

#[test]
fn remember_many_is_fail_fast_on_a_bad_input() {
    let dim = 4;
    let tmp = TempDir::new("batch-fail");
    let mut config = cfg();
    config.dim = dim;
    let (db, _) = Database::builder(config)
        .embedder(Box::new(CountingEmbedder::new(dim)))
        .open(tmp.db())
        .unwrap();

    // A wrong-dimension vector fails the engine; the good fact before it stays.
    let bad = vec![1.0f32; dim + 1];
    let bad_in = RememberInput {
        vector: Some(&bad),
        ..RememberInput::text(2, "bad vector")
    };
    let r = db.remember_many(vec![RememberInput::text(1, "good"), bad_in]);
    assert!(r.is_err(), "wrong-dimension vector → Err");
    assert_eq!(
        db.stats().facts,
        1,
        "fail-fast: the good fact before it stayed"
    );
}

#[test]
fn remember_many_is_durable_after_reopen() {
    // Batch mode defers the per-record fsync to one `sync_journal` at the end.
    // With no checkpoint, durability rests entirely on that final sync: a reopen
    // must replay all facts from the journal. (EachOp is the default policy.)
    let dim = 4;
    let mut config = cfg();
    config.dim = dim;
    let tmp = TempDir::new("batch-durable");
    {
        let (db, _) = Database::builder(config.clone())
            .embedder(Box::new(CountingEmbedder::new(dim)))
            .open(tmp.db())
            .unwrap();
        db.remember_many(vec![
            RememberInput::text(1, "durable one"),
            RememberInput::text(2, "durable two"),
            RememberInput::text(3, "durable three"),
        ])
        .unwrap();
        // No checkpoint here — only the batch's single sync_journal.
    } // drop releases the writer lock; the journal stays on disk.

    let (db2, _) = Database::builder(config)
        .embedder(Box::new(CountingEmbedder::new(dim)))
        .open(tmp.db())
        .unwrap();
    assert_eq!(
        db2.stats().facts,
        3,
        "the batch survived via one sync_journal"
    );
}

#[test]
fn export_each_streams_the_same_facts_as_export() {
    let tmp = TempDir::new("export-each");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(1, "alpha")).unwrap();
    db.remember(RememberInput::text(2, "beta")).unwrap();
    db.revise(FactId(1), RememberInput::text(3, "alpha prime"))
        .unwrap(); // closes id 1, opens a successor — export skips the closed one

    let mut streamed = Vec::new();
    db.export_each(|f| streamed.push(f.text.clone()));
    let mut collected: Vec<_> = db.export().into_iter().map(|f| f.text).collect();
    streamed.sort();
    collected.sort();
    assert_eq!(
        streamed, collected,
        "export_each visits exactly export()'s facts"
    );
}

#[test]
fn export_pages_are_bounded_and_advance_across_closed_and_tombstoned_ids() {
    let tmp = TempDir::new("export-page");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(1, "alpha")).unwrap();
    let beta = db.remember(RememberInput::text(2, "beta")).unwrap();
    db.revise(beta.id, RememberInput::text(3, "beta prime"))
        .unwrap();
    let gone = db.remember(RememberInput::text(4, "gone")).unwrap();
    db.forget(5, gone.id).unwrap();
    db.remember(RememberInput::text(6, "omega")).unwrap();

    let expected = db.export();
    let mut cursor = 0;
    let mut paged = Vec::new();
    loop {
        let page = db.export_page(cursor, NonZeroUsize::new(1).unwrap());
        assert!(page.facts.len() <= 1, "the requested bound is hard");
        paged.extend(page.facts);
        let Some(next) = page.next_cursor else { break };
        assert!(next > cursor, "a non-terminal page must make progress");
        cursor = next;
    }
    assert_eq!(paged, expected);

    let beyond_end = db.export_page(u32::MAX, NonZeroUsize::new(4).unwrap());
    assert!(beyond_end.facts.is_empty());
    assert_eq!(beyond_end.next_cursor, None);
}

#[test]
fn embedder_transport_and_shape_errors_are_typed() {
    // A refused connection is a typed Embed error.
    let refused = OpenAiCompatEmbedder::new("http://127.0.0.1:1/v1", "m", 4);
    assert!(matches!(refused.embed(&["x"]), Err(HostError::Embed(_))));

    // A server sending the wrong dimension is a typed Embed error.
    let (url, server) = spawn_mock_embedder(4, 1);
    let wrong = OpenAiCompatEmbedder::new(&url, "m", 5);
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
    // A crashed checkpoint leaves an orphan generation (and its staging tmp)
    // that the manifest never came to point at; the live snapshot is intact.
    let orphan = tmp.0.join("agent.plugmem.snap.999");
    let orphan_tmp = tmp.0.join("agent.plugmem.snap.999.tmp");
    std::fs::write(&orphan, b"half-written generation").unwrap();
    std::fs::write(&orphan_tmp, b"staging garbage").unwrap();
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    assert!(!orphan.exists(), "the orphan generation must be removed");
    assert!(!orphan_tmp.exists(), "the staging tmp must be removed");
    assert_eq!(db.stats().facts, 1, "the real snapshot loaded");
}

/// The number of `base.snap.<N>` generation files present (committed, not tmp).
fn generation_count(base: &std::path::Path) -> usize {
    let dir = base.parent().unwrap();
    let name = base.file_name().unwrap().to_str().unwrap();
    let prefix = format!("{name}.snap.");
    std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            let f = e.file_name();
            let f = f.to_string_lossy();
            f.strip_prefix(&prefix)
                .is_some_and(|rest| rest.parse::<u64>().is_ok())
        })
        .count()
}

#[test]
fn checkpoint_advances_the_generation_and_reclaims_the_old() {
    let tmp = TempDir::new("generations");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();

    db.remember(RememberInput::text(1, "first")).unwrap();
    db.checkpoint(10).unwrap();
    // The manifest is a small fixed record naming generation 1; exactly one
    // committed generation file exists.
    assert_eq!(
        std::fs::metadata(tmp.db()).unwrap().len(),
        24,
        "manifest size"
    );
    assert!(snapshot_file(&tmp.db()).exists(), "generation 1 exists");
    assert_eq!(generation_count(&tmp.db()), 1);

    db.remember(RememberInput::text(2, "second")).unwrap();
    db.checkpoint(20).unwrap();
    // The manifest now names generation 2; generation 1 has been reclaimed —
    // still exactly one generation on disk (no accumulation).
    let g2 = snapshot_file(&tmp.db());
    assert!(
        g2.to_string_lossy().ends_with(".snap.2"),
        "advanced to gen 2"
    );
    assert!(g2.exists());
    assert!(
        !tmp.0.join("agent.plugmem.snap.1").exists(),
        "the old generation is reclaimed"
    );
    assert_eq!(generation_count(&tmp.db()), 1);
}

#[test]
fn gc_reclaims_unpinned_generations_and_a_pin_keeps_one() {
    let tmp = TempDir::new("gc-pin");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(1, "one")).unwrap();
    db.checkpoint(10).unwrap();
    // Repeated checkpoints with no reader: each reclaims the previous, so
    // exactly one generation is ever on disk.
    for i in 0..5u64 {
        db.remember(RememberInput::text(20 + i, "more")).unwrap();
        db.checkpoint(30 + i).unwrap();
    }
    assert_eq!(
        generation_count(&tmp.db()),
        1,
        "unpinned generations are reclaimed"
    );

    // Pin the current generation with a reader, then checkpoint: the pinned
    // generation survives GC (its shared lock blocks reclaim), so two coexist.
    let pinned = snapshot_file(&tmp.db());
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(100, "after pin")).unwrap();
    db.checkpoint(110).unwrap();
    assert!(pinned.exists(), "the pinned generation survives GC");
    assert_eq!(generation_count(&tmp.db()), 2, "pinned + current");
    assert_eq!(ro.stats().facts, 6, "the reader sees its pinned snapshot");

    // Drop the reader; the next checkpoint reclaims the now-unpinned generation.
    drop(ro);
    db.remember(RememberInput::text(200, "after drop")).unwrap();
    db.checkpoint(210).unwrap();
    assert!(!pinned.exists(), "the unpinned generation is reclaimed");
    assert_eq!(generation_count(&tmp.db()), 1);
}

/// Open file descriptors belonging to `base`, via `/proc/self/fd` (Linux).
/// Restricting the count to this test's unique database matters: the Rust test
/// harness runs tests in parallel, and the workspace tests deliberately keep
/// a pool of unrelated databases open. Counting the whole process would turn
/// that legitimate activity into a false leak report.
///
/// On other platforms it returns 0, so the fd-growth assertion below is a
/// no-op there — the generation-count bound still holds everywhere, and
/// valgrind (Linux) supplies the authoritative heap/mapping verdict.
#[cfg(target_os = "linux")]
fn open_fd_count(base: &std::path::Path) -> usize {
    let prefix = base.to_string_lossy();
    std::fs::read_dir("/proc/self/fd")
        .map(|d| {
            d.flatten()
                .filter(|entry| {
                    std::fs::read_link(entry.path())
                        .map(|target| target.to_string_lossy().starts_with(&*prefix))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}
#[cfg(not(target_os = "linux"))]
fn open_fd_count(_base: &std::path::Path) -> usize {
    0
}

/// Leak-growth guard for the Variant 2 mmap/generation machinery.
///
/// Many rounds of {writer mutate + checkpoint} interleaved with an external
/// {reader open → recall → drop} must keep two independent resources bounded
/// *regardless of round count* — a leak makes either grow without bound:
///
/// - **Generations on disk stay bounded**: each checkpoint reclaims the
///   previous unpinned generation, so a leaked generation would accumulate on
///   disk round after round.
/// - **Open fds stay bounded**: every reader pins its generation with a `File`
///   and mmaps it; a leaked pin or mapping would grow the fd table by one per
///   reader-open, i.e. linearly in the round count.
///
/// Deterministic (no timing/watchdog) so it runs on every PR, and small enough
/// to run under valgrind — which supplies the authoritative heap verdict that a
/// constant mmap residency (not a growing leak) is what remains.
#[test]
fn readers_and_checkpoints_do_not_leak_across_rounds() {
    let tmp = TempDir::new("leak-growth");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(1, "seed fact")).unwrap();
    db.checkpoint(2).unwrap();

    // Enough rounds to make any linear leak obvious; overridable so the
    // valgrind job (≈100x slower) can run fewer — a leak still shows as growing
    // fds and as per-allocation loss under memcheck regardless of count.
    let rounds: u64 = std::env::var("PLUGMEM_LEAK_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let churn = |db: &Database| {
        for r in 0..rounds {
            let now = 100 + r;
            db.remember(RememberInput::text(now, "a churning fact about tokio"))
                .unwrap();
            db.checkpoint(now + 1).unwrap();
            // External reader pins the current generation, maps it, reads, and
            // drops — releasing the pin (fd) and unmapping.
            let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
            let _ = ro.recall(RecallQuery::text(now, "tokio")).unwrap();
        }
    };

    // Two identical passes: comparing the second delta against the first cancels
    // any one-time initialization (lazily-built caches, allocator arenas).
    let fds_start = open_fd_count(&tmp.db());
    churn(&db);
    let fds_after_first = open_fd_count(&tmp.db());
    churn(&db);
    let fds_after_second = open_fd_count(&tmp.db());

    // Generations never accumulate: readers are dropped each round, so GC
    // settles the disk to the single current generation.
    let gens = generation_count(&tmp.db());
    assert!(
        gens <= 2,
        "generations must not accumulate across {rounds} rounds: found {gens}"
    );

    // A per-reader pin/mapping leak would grow fds by `rounds` each pass; with no
    // leak the second identical churn adds ~0 over the first.
    let growth = fds_after_second.saturating_sub(fds_after_first);
    assert!(
        growth <= 2,
        "fd count grew by {growth} over a second identical {rounds}-round churn \
         (start={fds_start}, after1={fds_after_first}, after2={fds_after_second}) \
         — a per-reader pin or mapping leak"
    );
}

#[test]
fn a_corrupt_manifest_is_rejected_on_open() {
    let tmp = TempDir::new("bad-manifest");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        db.remember(RememberInput::text(1, "x")).unwrap();
        db.checkpoint(2).unwrap();
    }
    // Flip the manifest magic: it no longer validates, so an open refuses it
    // rather than trusting a garbage generation number.
    let mut m = std::fs::read(tmp.db()).unwrap();
    m[0] ^= 0xFF;
    std::fs::write(tmp.db(), &m).unwrap();
    match Database::open(tmp.db(), cfg()) {
        Err(HostError::Engine(plugmem_host::Error::Corrupt(_))) => {}
        other => panic!("expected a Corrupt manifest error, got {other:?}"),
    }
}

#[test]
fn embedder_edge_cases() {
    // NullEmbedder's contract, called directly.
    let null = NullEmbedder;
    assert_eq!(null.dim(), 0);
    assert_eq!(null.embed(&["a", "b"]).unwrap(), vec![Vec::<f32>::new(); 2]);

    // Empty input short-circuits without a network call.
    let e = OpenAiCompatEmbedder::new("http://127.0.0.1:1/v1", "m", 4);
    assert!(e.embed(&[]).unwrap().is_empty());

    // The api-key path sends and succeeds against the mock.
    let (url, server) = spawn_mock_embedder(4, 1);
    let keyed = OpenAiCompatEmbedder::new(&url, "m", 4).with_api_key("sk-test");
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
        let e = OpenAiCompatEmbedder::new(&url, "m", 4);
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

/// A workload touching several sources, then a checkpoint so the journal
/// is empty (the read-only precondition).
fn seed_checkpointed(db: &Database) {
    for i in 0..30u64 {
        db.remember(RememberInput {
            entity: Some(["user", "plugmem", "кот"][(i % 3) as usize]),
            tags: if i % 2 == 0 { &["pref"] } else { &[] },
            ..RememberInput::text(i + 1, "some fact about работа and tokio")
        })
        .unwrap();
    }
    db.link(plugmem_host::LinkInput {
        now: 100,
        src: "plugmem",
        rel: "depends_on",
        dst: "tokio",
        provenance: None,
    })
    .unwrap();
    db.checkpoint(200).unwrap();
}

#[test]
fn export_dumps_open_facts_with_names_and_tags() {
    let tmp = TempDir::new("export");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput {
        entity: Some("user"),
        tags: &["pref", "lang"],
        ..RememberInput::text(1_000, "prefers tokio")
    })
    .unwrap();
    // A closed revision must NOT appear (export is the current state).
    let old = db
        .remember(RememberInput::text(2_000, "lived in Moscow"))
        .unwrap();
    db.revise(old.id, RememberInput::text(3_000, "lives in Berlin"))
        .unwrap();
    // A tombstone must not appear either.
    let gone = db
        .remember(RememberInput::text(4_000, "temporary"))
        .unwrap();
    db.forget(5_000, gone.id).unwrap();

    let facts = db.export();
    // tokio (open, with subject + tags) and Berlin (open); Moscow is closed,
    // temporary is tombstoned → 2 facts.
    assert_eq!(facts.len(), 2, "{facts:?}");
    let tokio = facts.iter().find(|f| f.text == "prefers tokio").unwrap();
    assert_eq!(tokio.entity.as_deref(), Some("user"));
    assert_eq!(tokio.tags, vec!["pref".to_string(), "lang".to_string()]);
    assert_eq!(tokio.valid_from, 1_000);
    assert!(facts.iter().any(|f| f.text == "lives in Berlin"));
    assert!(!facts.iter().any(|f| f.text.contains("Moscow")));
}

#[test]
fn open_readonly_matches_read_write() {
    let tmp = TempDir::new("ro-match");
    let q = RecallQuery {
        entities: &["plugmem"],
        ..RecallQuery::text(1_000, "работа tokio")
    };
    // Build and checkpoint, capture the read-write answers, then release
    // the lock.
    let (facts, rendered, got1) = {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        seed_checkpointed(&db);
        let stats = db.stats();
        let rendered = db.recall(q).unwrap().rendered;
        let got1 = db.get(FactId(1)).unwrap();
        (stats.facts, rendered, got1)
    };

    // The read-only open borrows the mmap; its answers are identical.
    let ro: ReadOnlyDatabase = Database::open_readonly(tmp.db(), cfg()).unwrap();
    assert_eq!(ro.stats().facts, facts);
    assert_eq!(ro.recall(q).unwrap().rendered, rendered);
    assert_eq!(ro.get(FactId(1)), Some(got1));
    let page = ro.export_page(0, NonZeroUsize::new(1).unwrap());
    assert_eq!(page.facts.len(), 1);
    assert!(page.next_cursor.is_some());
    assert_eq!(ro.path(), tmp.db());
    // Debug is a summary, never the contents.
    assert!(format!("{ro:?}").contains("ReadOnlyDatabase"));
}

#[test]
fn open_readonly_refuses_a_dirty_journal() {
    let tmp = TempDir::new("ro-dirty");
    {
        let (db, _) = Database::builder(cfg())
            .snapshot_every_ops(0) // never auto-snapshot: keep the journal dirty
            .open(tmp.db())
            .unwrap();
        db.remember(RememberInput::text(1, "uncheckpointed"))
            .unwrap();
    }
    // A non-empty journal is a typed refusal with the base path.
    match Database::open_readonly(tmp.db(), cfg()) {
        Err(HostError::NeedsCheckpoint { path }) => assert_eq!(path, tmp.db()),
        other => panic!("expected NeedsCheckpoint, got {other:?}"),
    }
    // Checkpointing read-write once clears the way.
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        db.checkpoint(2).unwrap();
    }
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    assert_eq!(ro.stats().facts, 1);
}

#[test]
fn open_readonly_does_not_block_the_writer() {
    let tmp = TempDir::new("ro-nonblock");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        seed_checkpointed(&db);
    }
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    // A reader does not take the writer lock (it pins its generation instead),
    // so a writer opens alongside it and writes — no Locked.
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(300, "written while a reader is live"))
        .unwrap();
    // The reader is unaffected — still its pinned generation.
    assert_eq!(ro.stats().facts, 30);
}

#[test]
fn many_readers_share_one_snapshot() {
    let tmp = TempDir::new("ro-multi");
    let q = RecallQuery {
        entities: &["plugmem"],
        ..RecallQuery::text(1_000, "работа tokio")
    };
    let (facts, rendered) = {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        seed_checkpointed(&db);
        (db.stats().facts, db.recall(q).unwrap().rendered)
    };

    // Several read-only handles map the same generation at once and answer
    // identically.
    let readers: Vec<ReadOnlyDatabase> = (0..4)
        .map(|_| Database::open_readonly(tmp.db(), cfg()).unwrap())
        .collect();
    for ro in &readers {
        assert_eq!(ro.stats().facts, facts);
        assert_eq!(ro.recall(q).unwrap().rendered, rendered);
    }

    // A writer opens alongside the live readers (they hold no writer lock) and
    // writes; the readers still answer from their pinned generation.
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(300, "while readers are live"))
        .unwrap();
    for ro in &readers {
        assert_eq!(ro.stats().facts, facts, "readers are pinned");
    }
    drop(readers);
}

#[test]
fn a_reader_opens_alongside_a_live_writer() {
    let tmp = TempDir::new("ro-vs-rw");
    // Seed, then keep a read-write (exclusive) handle open.
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    seed_checkpointed(&db);
    // A read-only open succeeds against a live writer (cross-process MVCC): it
    // pins and maps the current published generation.
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    assert_eq!(ro.stats().facts, 30);
    // The writer keeps checkpointing while the reader is live; the reader stays
    // on its pinned generation (snapshot isolation), a fresh open sees more.
    db.remember(RememberInput::text(500, "after the reader opened"))
        .unwrap();
    db.checkpoint(600).unwrap();
    assert_eq!(ro.stats().facts, 30, "the open reader is pinned");
    let ro2 = Database::open_readonly(tmp.db(), cfg()).unwrap();
    assert_eq!(
        ro2.stats().facts,
        31,
        "a fresh reader sees the new checkpoint"
    );
}

#[test]
fn refresh_advances_a_reader_to_a_new_generation() {
    let tmp = TempDir::new("ro-refresh");
    // Seed and publish a first generation, then keep the writer open.
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    seed_checkpointed(&db);

    let mut ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    let gen0 = ro.generation();
    assert_eq!(ro.stats().facts, 30);

    // Nothing new published yet: refresh is a no-op, cheap and false, and the
    // handle stays exactly where it was.
    assert!(!ro.refresh().unwrap(), "no new generation → no advance");
    assert_eq!(ro.generation(), gen0);
    assert_eq!(ro.stats().facts, 30);

    // The writer publishes a newer generation.
    db.remember(RememberInput::text(500, "after the reader opened"))
        .unwrap();
    db.checkpoint(600).unwrap();

    // Before refresh the reader is still pinned to its point in time.
    assert_eq!(ro.stats().facts, 30, "pinned until refreshed");
    assert_eq!(ro.generation(), gen0);

    // refresh advances it: true, a strictly higher generation, and the new fact
    // is now visible.
    assert!(ro.refresh().unwrap(), "a newer generation → advance");
    assert!(ro.generation() > gen0, "generation is monotonic");
    assert_eq!(ro.stats().facts, 31, "the refreshed reader sees the write");

    // A second refresh with nothing newer is again a false no-op.
    let gen1 = ro.generation();
    assert!(!ro.refresh().unwrap());
    assert_eq!(ro.generation(), gen1);
    assert_eq!(ro.stats().facts, 31);
}

#[test]
fn journal_survives_repeated_clears() {
    // Regression guard for the Windows `clear_journal` bug: an append
    // handle cannot be truncated with `set_len` on Windows (FILE_WRITE_DATA
    // is masked off append handles), so clearing must go through a fresh
    // write handle and re-establish the append handle. This exercises many
    // clear/append cycles and asserts appends after a clear still land and
    // read back exactly — it fails on the pre-fix code on Windows.
    use plugmem_core::Storage as _;
    use plugmem_host::{FileStorage, FsyncPolicy};

    let tmp = TempDir::new("clears");
    let mut fs = FileStorage::open(tmp.db(), FsyncPolicy::EachOp).unwrap();
    for round in 0..8u8 {
        let record = [round; 16];
        fs.append_journal(&record).unwrap();
        fs.append_journal(&record).unwrap();
        assert_eq!(fs.journal_bytes(), 32, "two records landed this round");
        assert_eq!(fs.read_journal().unwrap(), [record, record].concat());

        fs.clear_journal().unwrap();
        assert_eq!(fs.journal_bytes(), 0, "the clear emptied the journal");
        assert!(fs.read_journal().unwrap().is_empty());

        // The re-established append handle must still write to the file.
        fs.append_journal(&[0xAB]).unwrap();
        assert_eq!(fs.journal_bytes(), 1, "appends resume after a clear");
        fs.clear_journal().unwrap();
    }
    assert_eq!(fs.journal_bytes(), 0);
}

// ---------------------------------------------------------------------------
// Overlay write path: the default `open` mmaps the snapshot and
// borrows it, re-mapping on snapshot. These tests prove the re-map is durable
// and canonical, and that opening residents far less RAM than an owned load.
// ---------------------------------------------------------------------------

/// The journal path next to a base database file.
fn journal_of(base: &std::path::Path) -> PathBuf {
    let mut p = base.to_path_buf().into_os_string();
    p.push(".journal");
    PathBuf::from(p)
}

/// The current snapshot generation file for a database `base`: read the
/// manifest (magic/ver/gen/checksum, little-endian) and build `base.snap.<gen>`.
/// `base` itself is the tiny manifest now, not the image.
fn snapshot_file(base: &std::path::Path) -> PathBuf {
    let m = std::fs::read(base).expect("manifest present");
    assert_eq!(m.len(), 24, "manifest is a fixed 24-byte record");
    let generation = u64::from_le_bytes(m[8..16].try_into().unwrap());
    let mut p = base.to_path_buf().into_os_string();
    p.push(format!(".snap.{generation}"));
    PathBuf::from(p)
}

#[test]
fn overlay_writes_survive_repeated_snapshots_and_reopen() {
    let tmp = TempDir::new("overlay-remap");
    {
        // Snapshot every 4 ops: the first crosses Owned -> Mapped, every later
        // one is a Mapped -> Mapped re-map. The engine must keep working across
        // all of them.
        let (db, _) = Database::builder(cfg())
            .snapshot_every_ops(4)
            .open(tmp.db())
            .unwrap();
        for i in 0..40u64 {
            db.remember(RememberInput {
                entity: Some("user"),
                ..RememberInput::text(i + 1, "a durable fact written across many snapshots")
            })
            .unwrap();
        }
        assert_eq!(
            db.stats().facts,
            40,
            "writes survive the re-maps while live"
        );
        let out = db.recall(RecallQuery::text(1_000, "durable")).unwrap();
        assert!(out.rendered.contains("durable"));
        let has_tmp = std::fs::read_dir(&tmp.0)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().ends_with(".tmp"));
        assert!(!has_tmp, "no staging tmp survives a snapshot");
    } // drop releases the lock

    // 40 ops at every-4 snapshots means the last op snapshotted: the journal is
    // empty and the reopen replays nothing.
    let (db, report) = Database::open(tmp.db(), cfg()).unwrap();
    assert_eq!(
        report.replayed, 0,
        "the last op checkpointed; nothing to replay"
    );
    assert_eq!(db.stats().facts, 40, "all writes survived reopen");
    let out = db.recall(RecallQuery::text(2_000, "durable")).unwrap();
    assert!(out.rendered.contains("durable"));
}

#[test]
fn a_re_mapped_snapshot_is_canonical_against_an_owned_replay() {
    use plugmem_core::Memory;

    let tmp = TempDir::new("overlay-canonical");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    // Write, checkpoint (Owned -> Mapped re-map), write more (incl. a revision),
    // checkpoint again (Mapped -> Mapped re-map): the final on-disk image is a
    // product of the re-map path.
    for i in 0..20u64 {
        db.remember(RememberInput {
            entity: Some(["user", "plugmem", "кот"][(i % 3) as usize]),
            tags: if i % 2 == 0 { &["pref"] } else { &[] },
            ..RememberInput::text(i + 1, "some fact about работа and tokio")
        })
        .unwrap();
    }
    db.checkpoint(100).unwrap();
    for i in 20..35u64 {
        db.remember(RememberInput::text(
            i + 1,
            "more facts after the first snapshot",
        ))
        .unwrap();
    }
    let old = db
        .remember(RememberInput::text(500, "lived in Moscow"))
        .unwrap();
    db.revise(old.id, RememberInput::text(600, "lives in Berlin"))
        .unwrap();
    const T: u64 = 1_000;
    db.checkpoint(T).unwrap();
    drop(db);

    // A checkpointed database has a full image and an empty journal.
    let file = std::fs::read(snapshot_file(&tmp.db())).unwrap();
    let journal = std::fs::read(journal_of(&tmp.db())).unwrap();
    assert!(journal.is_empty(), "checkpoint clears the journal");

    // The re-mapped snapshot dumps byte-identically to an owned replay of the
    // same image (canonical save->load->save), and the overlay open of that
    // image dumps the same bytes — tying the host's on-disk output to the core
    // canonicality guarantee.
    let (owned, _) = Memory::from_bytes(Some(&file), &journal, cfg()).unwrap();
    assert_eq!(
        owned.snapshot_bytes(T),
        file,
        "the re-mapped snapshot is canonical"
    );
    let (overlay, _) = Memory::from_bytes_overlay(&file, &journal, cfg()).unwrap();
    assert_eq!(
        overlay.snapshot_bytes(T),
        file,
        "overlay open matches the owned image byte-for-byte"
    );
}

#[test]
fn verify_passes_on_a_clean_database() {
    let tmp = TempDir::new("verify");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    seed_checkpointed(&db);
    db.verify().expect("a clean read-write database verifies");
    drop(db);
    // The read-only handle exposes the same on-demand integrity check.
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    ro.verify().expect("a clean read-only open verifies");
}

/// Process resident-set size in bytes (cross-OS: linux/macos/windows).
fn rss() -> usize {
    memory_stats::memory_stats()
        .expect("a resident-set-size reading")
        .physical_mem
}

#[test]
fn an_overlay_open_residents_far_less_than_the_image() {
    let tmp = TempDir::new("overlay-rss");
    // The default open is trust/sparse: it does not verify the whole-file xxh3,
    // and lazy validation skips the text/vector scans — so an open
    // faults in only the metadata and the large text pool stays non-resident.
    let c = cfg();

    // A sizable checkpointed database whose text pool dominates the image.
    {
        let (db, _) = Database::builder(c.clone())
            .snapshot_every_ops(0) // one checkpoint at the end, no mid-snapshots
            .fsync(FsyncPolicy::OnSnapshot)
            .open(tmp.db())
            .unwrap();
        // Long texts from a tiny template set: the blob heap (every fact stores
        // its full text) dominates the image, while the small shared vocabulary
        // keeps the posting lists compact. Lazy validation means an
        // open does not scan this text pool, so it stays non-resident.
        let templates = [
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do \
             eiusmod tempor incididunt ut labore et dolore magna aliqua",
            "ut enim ad minim veniam quis nostrud exercitation ullamco \
             laboris nisi ut aliquip ex ea commodo consequat duis aute",
            "excepteur sint occaecat cupidatat non proident sunt in culpa \
             qui officia deserunt mollit anim id est laborum sed perspiciatis",
            "at vero eos et accusamus et iusto odio dignissimos ducimus qui \
             blanditiis praesentium voluptatum deleniti atque corrupti quos",
        ];
        for i in 0..20_000u64 {
            // Repeat a template so the text is long (the blob heap dominates)
            // without adding distinct terms (the posting lists stay compact).
            let t = templates[(i % 4) as usize];
            let text = format!("{t} {t} {t} {t}");
            db.remember(RememberInput::text(i + 1, &text)).unwrap();
        }
        db.checkpoint(20_000_000).unwrap();
    }
    let file_len = std::fs::metadata(snapshot_file(&tmp.db())).unwrap().len() as usize;
    assert!(
        file_len > 8 * 1024 * 1024,
        "the test database must be large enough to measure ({file_len} bytes)"
    );

    // Overlay open (the default): mmap + borrow, validating only metadata.
    let before = rss();
    let (overlay, _) = Database::open(tmp.db(), c.clone()).unwrap();
    let overlay_rss = rss().saturating_sub(before);
    assert_eq!(overlay.stats().facts, 20_000, "the overlay opened the base");
    drop(overlay);

    // The overlay open residents well under half the image: the text pool (the
    // majority of the file) is never scanned, so it stays out of the resident
    // set — true sparse residency, not just no-copy. An owned
    // load, by contrast, copies every pool onto the heap; that copy is proven
    // absent at the allocation level in core's `zero_alloc` gate.
    assert!(
        overlay_rss < file_len / 2,
        "overlay open should resident far less than the image \
         (overlay {overlay_rss}, file {file_len})"
    );
}

// ---------------------------------------------------------------------------
// On-demand scrub: the default open trusts the file, so byte-level
// container integrity is a separate, resumable read-handle op. These prove the
// scrub verifies a clean image, catches a flipped section byte on disk, and
// holds a reader's lock for its whole life — independent of the handle it came
// from.
// ---------------------------------------------------------------------------

/// Flips one byte of the on-disk snapshot at the first occurrence of `needle`
/// (a substring of some fact's text, so the flip lands inside a section body,
/// which the structural parse accepts but the scrub must catch).
fn flip_byte_at(path: &std::path::Path, needle: &[u8]) {
    let mut bytes = std::fs::read(path).unwrap();
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("needle present in the snapshot");
    bytes[at] ^= 0xFF;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn scrub_verifies_a_clean_image_and_slices_by_budget() {
    let tmp = TempDir::new("scrub-clean");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        seed_checkpointed(&db);
    }
    let file_len = std::fs::metadata(snapshot_file(&tmp.db())).unwrap().len();

    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();

    // A tiny budget forces many slices; progress is monotonic and ends exactly
    // at the file length, every slice Ok.
    let mut steps = 0;
    let mut prev = 0u64;
    let mut last = None;
    for step in ro.scrub_with_budget(64).unwrap() {
        let p = step.expect("a clean image scrubs Ok");
        assert!(p.done_bytes >= prev, "progress went backwards");
        assert!(p.done_bytes <= file_len);
        assert_eq!(p.total_bytes, file_len);
        prev = p.done_bytes;
        last = Some(p);
        steps += 1;
    }
    assert!(
        steps > 1,
        "a tiny budget should take many slices, got {steps}"
    );
    assert_eq!(last.unwrap().done_bytes, file_len, "the scan reached EOF");

    // The default budget scrubs the same clean image to completion.
    assert!(ro.scrub().unwrap().all(|s| s.is_ok()));
}

#[test]
fn scrub_catches_a_flipped_section_byte_on_disk() {
    let tmp = TempDir::new("scrub-corrupt");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        seed_checkpointed(&db);
    }
    // Corrupt a byte inside the text pool while the lock is free.
    flip_byte_at(&snapshot_file(&tmp.db()), b"tokio");

    // The default open still succeeds (trust/sparse parse is structural)...
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    // ...and the scrub reports the container mismatch, then fuses.
    let mut cur = ro.scrub_with_budget(64).unwrap();
    let mut err = None;
    for step in cur.by_ref() {
        if let Err(e) = step {
            err = Some(e);
            break;
        }
    }
    match err {
        Some(HostError::Engine(plugmem_host::Error::Corrupt(msg))) => {
            assert_eq!(msg, "section checksum mismatch");
        }
        other => panic!("expected a Corrupt scrub error, got {other:?}"),
    }
    assert!(cur.next().is_none(), "the scrub is fused after an error");
}

#[test]
fn scrub_pins_its_generation_and_coexists_with_the_writer() {
    let tmp = TempDir::new("scrub-lock");
    {
        let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
        seed_checkpointed(&db);
    }
    let ro = Database::open_readonly(tmp.db(), cfg()).unwrap();
    let snap = snapshot_file(&tmp.db()); // the pinned generation file
    let mut scrub = ro.scrub().unwrap();
    // Debug is a summary, never the contents.
    assert!(format!("{scrub:?}").contains("Scrub"));
    // The scrub pins its own generation — it survives dropping the handle it
    // came from, and a writer opens and checkpoints alongside it (no Locked).
    drop(ro);
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();
    db.remember(RememberInput::text(500, "while a scrub runs"))
        .unwrap();
    db.checkpoint(600).unwrap();
    // The pinned generation survives the writer's GC while the scrub holds it.
    assert!(snap.exists(), "a live scrub pins its generation against GC");
    // The scrub still completes over its pinned image.
    assert!(scrub.all(|s| s.is_ok()));
}

// ---------------------------------------------------------------------------
// recover salvage (Tier 2): open a content-corrupt database,
// drop the bad records, maintain, and stream a clean image to a new file —
// leaving the source untouched. Structural corruption is not salvageable
// (Tier 0); the RAM guard refuses an image too large to rebuild.
// ---------------------------------------------------------------------------

#[test]
fn recover_drops_a_text_corrupt_fact_and_preserves_the_source() {
    let tmp = TempDir::new("recover-text");
    let src = tmp.db();
    let dst = tmp.0.join("recovered.plugmem");
    {
        let (db, _) = Database::open(&src, cfg()).unwrap();
        for i in 0..5u64 {
            db.remember(RememberInput::text(
                i + 1,
                &format!("clean fact number {i}"),
            ))
            .unwrap();
        }
        db.remember(RememberInput::text(100, "CORRUPTME marker fact"))
            .unwrap();
        db.checkpoint(200).unwrap();
    }
    // Turn the marker fact's text into invalid UTF-8, then snapshot the source
    // generation bytes so we can prove recover never rewrites them.
    let src_snap = snapshot_file(&src);
    flip_byte_at(&src_snap, b"CORRUPTME");
    let src_after_flip = std::fs::read(&src_snap).unwrap();

    let report = Database::recover(&src, &dst, cfg(), 300).unwrap();
    assert_eq!(report.dropped_text, 1);
    assert_eq!(report.dropped_vector, 0);
    assert_eq!(report.kept, 5, "the five clean facts survived");

    // The destination is a clean image: it opens, has the survivors, and passes
    // both integrity checks.
    {
        let ro = Database::open_readonly(&dst, cfg()).unwrap();
        assert_eq!(ro.stats().facts, 5);
        ro.verify().unwrap();
        assert!(ro.scrub().unwrap().all(|s| s.is_ok()));
        assert!(
            !ro.export().iter().any(|f| f.text.contains("marker")),
            "the corrupt fact is gone from the recovered image"
        );
    }
    // The source on disk is byte-for-byte what it was (evidence preserved).
    assert_eq!(
        std::fs::read(&src_snap).unwrap(),
        src_after_flip,
        "recover must leave the source untouched"
    );
}

#[test]
fn recover_drops_a_vector_corrupt_fact() {
    let tmp = TempDir::new("recover-vec");
    let src = tmp.db();
    let dst = tmp.0.join("recovered.plugmem");
    let mut c = cfg();
    c.dim = 8;
    {
        let (db, _) = Database::open(&src, c.clone()).unwrap();
        let v = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for i in 0..4u64 {
            db.remember(RememberInput {
                vector: Some(&v),
                ..RememberInput::text(i + 1, "a vector fact")
            })
            .unwrap();
        }
        db.checkpoint(200).unwrap();
    }
    // Break slot 0's owning-fact backpointer in the vector pool, so the fact
    // that owns slot 0 no longer has a slot that names it back (the fact<->slot
    // bijection verify() checks).
    const VEC_POOL_KIND: u16 = 37; // persist.rs `kind::VEC_POOL`
    let src_snap = snapshot_file(&src);
    let mut bytes = std::fs::read(&src_snap).unwrap();
    let start = {
        let snap = plugmem_core::snapshot::Snapshot::parse(&bytes).unwrap();
        let sec = snap.section(VEC_POOL_KIND).expect("a vector pool section");
        sec.as_ptr() as usize - bytes.as_ptr() as usize
    };
    bytes[start] ^= 0xFF; // the low byte of slot 0's owning fact id
    std::fs::write(&src_snap, bytes).unwrap();

    let report = Database::recover(&src, &dst, c.clone(), 300).unwrap();
    assert_eq!(report.dropped_vector, 1);
    assert_eq!(report.dropped_text, 0);
    assert_eq!(report.kept, 3, "the three intact vector facts survived");

    let ro = Database::open_readonly(&dst, c).unwrap();
    ro.verify().unwrap();
    assert!(ro.scrub().unwrap().all(|s| s.is_ok()));
}

#[test]
fn recover_refuses_structural_corruption() {
    let tmp = TempDir::new("recover-struct");
    let src = tmp.db();
    let dst = tmp.0.join("recovered.plugmem");
    {
        let (db, _) = Database::open(&src, cfg()).unwrap();
        seed_checkpointed(&db);
    }
    // Break the snapshot's magic so the image will not parse at all.
    let src_snap = snapshot_file(&src);
    let mut bytes = std::fs::read(&src_snap).unwrap();
    bytes[0] = b'X';
    std::fs::write(&src_snap, &bytes).unwrap();

    match Database::recover(&src, &dst, cfg(), 300) {
        Err(HostError::Engine(plugmem_host::Error::Corrupt(_))) => {}
        other => panic!("expected a structural Corrupt error, got {other:?}"),
    }
    assert!(
        !dst.exists(),
        "no destination is written when the source will not parse"
    );
}

#[test]
fn recover_refuses_a_destination_equal_to_the_source() {
    let tmp = TempDir::new("recover-same");
    let src = tmp.db();
    {
        let (db, _) = Database::open(&src, cfg()).unwrap();
        seed_checkpointed(&db);
    }
    match Database::recover(&src, &src, cfg(), 300) {
        Err(HostError::Engine(plugmem_host::Error::Invalid(_))) => {}
        other => panic!("expected Invalid (dst == src), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// FileScratch (milestone H): a temp-file staging area — sequential
// appends, a memory-mapped read-back on freeze, and cleanup on drop.
// ---------------------------------------------------------------------------

#[test]
fn file_scratch_streams_freezes_and_cleans_up() {
    use plugmem_host::{FileScratch, Scratch as _};

    let tmp = TempDir::new("scratch");
    let path = tmp.0.join("stage.tmp");

    // Build a payload with many small appends, then freeze and read it back
    // from the map — it must equal the concatenation of every write.
    let mut expect = Vec::new();
    {
        let mut s = FileScratch::create(&path).unwrap();
        assert!(s.is_empty());
        for i in 0..1_000u32 {
            s.write(&i.to_le_bytes()).unwrap();
            expect.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(s.len(), 4_000);
        let frozen = s.freeze().unwrap();
        assert_eq!(frozen, &expect[..], "freeze returns every written byte");
        // Random read into the map.
        assert_eq!(&frozen[8..12], &2u32.to_le_bytes());
        assert!(path.exists(), "the staging file exists while live");
    }
    // Dropped: the temp file is gone.
    assert!(!path.exists(), "the staging file is removed on drop");
}

#[test]
fn file_scratch_refuses_a_write_after_freeze() {
    use plugmem_host::{FileScratch, Scratch as _};

    let tmp = TempDir::new("scratch-frozen");
    let path = tmp.0.join("stage.tmp");
    let mut s = FileScratch::create(&path).unwrap();
    s.write(b"payload").unwrap();
    let _ = s.freeze().unwrap(); // the borrow ends at the statement
    assert!(
        matches!(s.write(b"more"), Err(HostError::Engine(_))),
        "a write after freeze is a typed error, not a silent corruption"
    );
}

#[test]
fn maintain_compacts_disk_first_and_the_engine_stays_live() {
    let tmp = TempDir::new("maintain-df");
    let mut c = cfg();
    c.dim = 8;
    let (db, _) = Database::builder(c.clone())
        .snapshot_every_ops(0) // no auto-snapshot noise
        .open(tmp.db())
        .unwrap();
    let v = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    for i in 0..40u64 {
        db.remember(RememberInput {
            vector: Some(&v),
            ..RememberInput::text(i + 1, "a fact worth some bytes to compact away")
        })
        .unwrap();
    }
    db.checkpoint(100).unwrap();

    // Forget half, then compact disk-first.
    for id in 0..20u32 {
        db.forget(200, FactId(id)).unwrap();
    }
    let report = db.maintain(300).unwrap();
    assert_eq!(report.purged, 20, "twenty tombstones purged");
    assert!(
        report.bytes_after < report.bytes_before,
        "the on-disk image shrank ({} -> {})",
        report.bytes_before,
        report.bytes_after
    );
    // No scratch temp files linger.
    assert!(!tmp.0.join("agent.plugmem.mtext.tmp").exists());
    assert!(!tmp.0.join("agent.plugmem.mvec.tmp").exists());

    // The engine keeps working: the survivors remain and it accepts writes.
    assert_eq!(db.stats().facts, 20);
    db.remember(RememberInput {
        vector: Some(&v),
        ..RememberInput::text(400, "after maintain")
    })
    .unwrap();
    assert_eq!(db.stats().facts, 21);

    // Reopen from the compacted file alone.
    drop(db);
    let (db2, _) = Database::open(tmp.db(), c).unwrap();
    assert_eq!(db2.stats().facts, 21);
    db2.verify().unwrap();
}

/// Metadata round-trips through the host: shuffled input keys come back sorted
/// from `get` and `export`, and re-remembering the exported map preserves it.
/// The cross-layer order/count parity (host `BTreeMap` == core pairs) is proven
/// in `plugmem-core/tests/metadata.rs`; here we confirm the host surface.
#[test]
fn metadata_round_trips_through_get_and_export_sorted() {
    use std::collections::BTreeMap;
    let tmp = TempDir::new("metadata");
    let (db, _) = Database::open(tmp.db(), cfg()).unwrap();

    // Input keys deliberately out of order.
    let id = db
        .remember(RememberInput {
            metadata: Some(&[
                ("uri", "s3://b/x"),
                ("mime", "application/pdf"),
                ("page", "3"),
            ]),
            ..RememberInput::text(100, "a scanned contract")
        })
        .unwrap()
        .id;

    let want: BTreeMap<String, String> = [
        ("mime", "application/pdf"),
        ("page", "3"),
        ("uri", "s3://b/x"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();

    // `get` decodes the sorted map.
    let snap = db.get(id).expect("fact exists");
    assert_eq!(snap.metadata, want);

    // A fact with no metadata reports an empty map, not an error.
    let bare = db
        .remember(RememberInput::text(200, "no metadata"))
        .unwrap()
        .id;
    assert!(db.get(bare).unwrap().metadata.is_empty());

    // `export` carries the same map, and re-remembering it into a fresh
    // database reproduces it exactly (import round-trip).
    let exported = db.export();
    let with_meta = exported
        .iter()
        .find(|f| !f.metadata.is_empty())
        .expect("one exported fact carries metadata");
    assert_eq!(with_meta.metadata, want);

    let tmp2 = TempDir::new("metadata-import");
    let (db2, _) = Database::open(tmp2.db(), cfg()).unwrap();
    let pairs: Vec<(&str, &str)> = with_meta
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let rid = db2
        .remember(RememberInput {
            metadata: Some(&pairs),
            ..RememberInput::text(300, &with_meta.text)
        })
        .unwrap()
        .id;
    assert_eq!(
        db2.get(rid).unwrap().metadata,
        want,
        "import preserves metadata"
    );
}

/// A growing database re-shards itself. Nothing else would: automatic
/// maintenance is opt-in, so without this trigger a database would keep the
/// layout it was created with until somebody ran `maintain` by hand, paying
/// for it in memory the whole time.
#[test]
fn a_growing_database_reshards_itself_without_being_asked() {
    let tmp = TempDir::new("reshard");
    // This test proves the automatic layout transition, not per-record
    // durability or the ordinary snapshot cadence. Keep those policies out of
    // the workload: on Windows, the default EachOp journal sync would turn
    // the ~175k records below into a many-minute CI durability benchmark.
    let (db, _) = Database::builder(cfg())
        .fsync(FsyncPolicy::OnSnapshot)
        .snapshot_every_ops(0)
        .snapshot_journal_bytes(0)
        .open(tmp.db())
        .unwrap();
    let floor = db.stats().shards;

    // Enough facts to push the fact arena one step past the floor. The rule
    // sizes each group by its payload, so this follows from the slot width and
    // the per-shard target rather than being a guessed number.
    const FACT_SLOT: usize = 48;
    const SHARD_TARGET: usize = 64 * 4096;
    // Past the floor is not enough — a rebuild waits for the growth margin.
    let n = floor.facts * ShardLayout::GROWTH_MARGIN * SHARD_TARGET / FACT_SLOT + 1;
    for i in 0..n {
        db.remember(RememberInput::text(i as u64 + 1, "a short fact"))
            .unwrap();
    }

    let grown = db.stats().shards;
    assert!(
        grown.facts > floor.facts,
        "layout stayed at {floor:?} after {n} facts"
    );
    assert_eq!(db.stats().facts, n, "re-sharding is a rearrangement");

    // It survives a reopen: the file records the layout, and the caller's
    // config — still on the floor — does not override it.
    drop(db);
    let (reopened, _) = Database::open(tmp.db(), cfg()).unwrap();
    assert_eq!(reopened.stats().shards, grown);
    assert_eq!(reopened.stats().facts, n);
    reopened.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Workspace: many small databases at once. This is what the derived shard
// layout bought — before it, a database holding a thousand facts cost 14 MB of
// pages, so a bot with a memory per conversation was not affordable at all. The
// gate belongs here rather than in a README because it is the property that
// makes the whole feature viable, and it is exactly the kind of thing a future
// change to shard sizing would silently undo.
// ---------------------------------------------------------------------------

#[test]
fn fifty_small_memories_fit_in_a_budget_a_bot_can_afford() {
    use plugmem_host::{DbName, IfMissing, Settings, WorkspaceLimits};

    const MEMORIES: usize = 50;
    const FACTS_EACH: u64 = 200;
    /// Engine pool bytes one memory of this size may hold.
    ///
    /// This, not the resident set, is the gate. `pool_bytes` counts what the
    /// engine allocated and is identical on every platform and profile, while
    /// RSS is the allocator's and the OS's answer to a different question — it
    /// varies by page accounting and malloc zone, which makes a tight budget on
    /// it a tripwire for the CI runner rather than for the engine.
    ///
    /// The regression it guards is not subtle: with the fixed shard counts this
    /// branch was built on top of, a memory this size cost ~14 MB of pages
    /// whatever it held. A hundredth of that is still generous for 200 facts.
    const POOL_BUDGET: usize = 256 * 1024;
    /// A resident-set sanity bound, deliberately loose. It exists to catch a
    /// pool that never evicts (fifty memories held open at once), not to count
    /// bytes — sixteen pooled handles at the old floor would be past 200 MB, so
    /// anything under this is unambiguously "the floor is gone".
    const RSS_BUDGET: usize = 96 * 1024 * 1024;

    let tmp = TempDir::new("workspace-rss");
    let settings = Settings::from_table(None).unwrap();
    let ws = settings.open_workspace(&tmp.0).unwrap();
    assert_eq!(
        ws.limits(),
        WorkspaceLimits::default(),
        "the budget below is sized against the default pool"
    );

    let before = rss();
    for i in 0..MEMORIES {
        let name = DbName::parse(&format!("chat-{i}")).unwrap();
        let db = ws.get(&name, 1_000, IfMissing::Create).unwrap();
        for f in 0..FACTS_EACH {
            db.remember(RememberInput::text(
                1_000 + f,
                &format!("memory {i} fact {f}: the sky was a shade of blue that day"),
            ))
            .unwrap();
        }
        db.checkpoint(2_000).unwrap();
    }
    let grew = rss().saturating_sub(before);

    assert_eq!(ws.layout().list().unwrap().len(), MEMORIES);
    // The pool is what bounds this: fifty memories were written, at most
    // `max_open` are held, and the rest were closed on the way through.
    assert!(
        ws.open_count() <= WorkspaceLimits::default().max_open,
        "pool held {} handles",
        ws.open_count()
    );
    assert!(
        grew < RSS_BUDGET,
        "{MEMORIES} memories of {FACTS_EACH} facts added {grew} resident bytes \
         (loose bound {RSS_BUDGET})"
    );

    // The deterministic half: every memory is intact, independent, and small.
    for i in 0..MEMORIES {
        let name = DbName::parse(&format!("chat-{i}")).unwrap();
        let db = ws.get(&name, 3_000, IfMissing::Fail).unwrap();
        let stats = db.stats();
        assert_eq!(stats.facts, FACTS_EACH as usize);
        assert!(
            stats.pool_bytes < POOL_BUDGET,
            "chat-{i} holds {} pool bytes for {FACTS_EACH} facts (budget {POOL_BUDGET})",
            stats.pool_bytes
        );
    }

    for i in [0, MEMORIES / 2, MEMORIES - 1] {
        let name = DbName::parse(&format!("chat-{i}")).unwrap();
        let db = ws.get(&name, 3_000, IfMissing::Fail).unwrap();
        let out = db
            .recall(plugmem_host::RecallQuery::text(3_000, "shade of blue"))
            .unwrap();
        assert!(out.rendered.contains(&format!("memory {i} fact")), "{i}");
    }
}
