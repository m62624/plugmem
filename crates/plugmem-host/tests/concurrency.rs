//! Special-purpose concurrency invariants (Variant 1): many readers run
//! against one engine at once — on a live [`Database`] (a writer churning the
//! file, incl. the checkpoint re-map) and on a shared [`ReadOnlyDatabase`] —
//! and **never observe a torn (half-applied) fact**.
//!
//! The guarantee is structural: readers take a shared `RwLock` guard, the
//! writer an exclusive one, so no reader ever runs while a mutation (or the
//! snapshot re-map) is in flight. These stress tests guard against a
//! regression that moves shared mutable state *outside* that lock — then a
//! reader could splice old and new bytes, which the self-consistency check
//! below would catch. High iteration + a start barrier make the race likely.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use plugmem_host::{Config, Database, FactId, ReadOnlyDatabase, RecallQuery, RememberInput};

/// A unique temp directory per test; removed on drop.
struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let unique = format!(
            "plugmem-conc-{tag}-{}-{}",
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

/// Every fact's text is a self-describing frame: `len8 marker DDDDDDDD split
/// DDDDDDDD`, the two 8-digit groups identical. The writer only ever stores a
/// whole, consistent frame; a torn read (old bytes spliced with new, or a fault
/// mid re-map) would break the frame or make the halves disagree.
fn fact_text(v: u64) -> String {
    format!("len8 marker {v:08} split {v:08}")
}

/// Asserts a fact text is a whole, self-consistent frame — the torn-read gate.
fn assert_consistent(text: &str) {
    let toks: Vec<&str> = text.split(' ').collect();
    assert_eq!(toks.len(), 5, "garbled/torn text: {text:?}");
    assert_eq!(toks[0], "len8", "torn frame: {text:?}");
    assert_eq!(toks[1], "marker", "torn frame: {text:?}");
    assert_eq!(toks[3], "split", "torn frame: {text:?}");
    assert_eq!(toks[2].len(), 8, "torn length: {text:?}");
    assert_eq!(toks[2], toks[4], "torn halves disagree: {text:?}");
}

/// A tiny per-thread PRNG (xorshift64*) — no external rng, deterministic seed.
fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn _assert_send_sync<T: Send + Sync>() {}

/// Runs `body` (which spawns and joins its own worker threads) under a wall
/// clock deadline. A hang inside — a lock-ordering deadlock, a writer starved
/// forever, a guard never released — trips the deadline instead of blocking the
/// whole test binary; a panic inside is re-raised. This turns "no deadlock"
/// from an implicit belief into a checked property.
fn run_with_deadline(secs: u64, label: &str, body: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        body();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) => worker.join().unwrap(),
        // The worker dropped `tx` without signalling — it panicked; re-raise it.
        Err(RecvTimeoutError::Disconnected) => worker.join().unwrap(),
        Err(RecvTimeoutError::Timeout) => {
            panic!("{label}: did not finish within {secs}s — possible deadlock/hang")
        }
    }
}

#[test]
fn database_and_readonly_are_send_sync() {
    // The whole point of the design: share one handle across reader threads.
    _assert_send_sync::<Database>();
    _assert_send_sync::<ReadOnlyDatabase>();
}

#[test]
#[cfg_attr(tarpaulin, ignore)] // timing stress: runs under `cargo test`, skipped under coverage
fn concurrent_readers_with_a_writer_never_see_a_torn_fact() {
    const WRITES: u64 = 3_000;
    const READERS: usize = 8;

    let tmp = TempDir::new("rw-torn");
    // Small snapshot cadence so the checkpoint re-map fires often under the
    // readers — the re-map (engine swap) is the sharpest torn-read risk.
    let (db, _) = Database::builder(Config::default())
        .snapshot_every_ops(128)
        .open(tmp.db())
        .unwrap();

    let db = Arc::new(db);
    let done = Arc::new(AtomicBool::new(false));
    let max_id = Arc::new(AtomicU64::new(0)); // count of committed facts
    let barrier = Arc::new(Barrier::new(READERS + 1));

    let mut handles = Vec::new();
    for r in 0..READERS {
        let db = Arc::clone(&db);
        let done = Arc::clone(&done);
        let max_id = Arc::clone(&max_id);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            let mut rng = xorshift(0xABCD_0000 ^ r as u64);
            let mut reads = 0u64;
            barrier.wait();
            while !done.load(Ordering::Acquire) {
                let n = max_id.load(Ordering::Acquire);
                if n > 0 {
                    // Read a random committed fact by id — the content check.
                    let id = (rng() % n) as u32;
                    if let Some(snap) = db.get(FactId(id)) {
                        assert_consistent(&snap.text);
                    }
                }
                // Exercise the recall read-path too; any returned fact must be
                // whole (its id resolved through the same shared guard).
                let out = db
                    .recall(RecallQuery {
                        k: 16,
                        ..RecallQuery::text(1 << 40, "marker split")
                    })
                    .unwrap();
                for f in &out.facts {
                    if let Some(snap) = db.get(f.id) {
                        assert_consistent(&snap.text);
                    }
                }
                let _ = db.stats();
                reads += 1;
            }
            reads
        }));
    }

    // The writer: churn the file (remember + periodic checkpoint) while the
    // readers hammer it.
    barrier.wait();
    for v in 0..WRITES {
        let now = 1_000 + v;
        db.remember(RememberInput::text(now, &fact_text(v)))
            .unwrap();
        max_id.store(v + 1, Ordering::Release);
        if v.is_multiple_of(256) {
            db.checkpoint(now).unwrap();
        }
    }
    done.store(true, Ordering::Release);

    let total_reads: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert!(total_reads > 0, "readers must have observed the writer");
    assert_eq!(
        db.stats().facts,
        WRITES as usize,
        "every write landed exactly once"
    );
}

#[test]
#[cfg_attr(tarpaulin, ignore)] // timing stress: runs under `cargo test`, skipped under coverage
fn readonly_serves_concurrent_readers_consistently() {
    const FACTS: u64 = 500;
    const READERS: usize = 8;
    const LOOPS: u64 = 400;

    let tmp = TempDir::new("ro-conc");
    // Write a consistent corpus, then checkpoint so the read-only path (which
    // needs an empty journal) can map the snapshot.
    {
        let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();
        for v in 0..FACTS {
            db.remember(RememberInput::text(1_000 + v, &fact_text(v)))
                .unwrap();
        }
        db.checkpoint(1_000 + FACTS).unwrap();
    } // drop releases the exclusive lock

    let ro = Arc::new(Database::open_readonly(tmp.db(), Config::default()).unwrap());
    let barrier = Arc::new(Barrier::new(READERS));

    let handles: Vec<_> = (0..READERS)
        .map(|r| {
            let ro = Arc::clone(&ro);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let mut rng = xorshift(0x5151_0000 ^ r as u64);
                barrier.wait();
                for _ in 0..LOOPS {
                    let id = (rng() % FACTS) as u32;
                    let snap = ro.get(FactId(id)).expect("every id in range exists");
                    assert_consistent(&snap.text);
                    // recall shares the mapped engine (&Memory) + a per-thread
                    // scratch — the concurrency this test exists to prove.
                    let out = ro
                        .recall(RecallQuery {
                            k: 8,
                            ..RecallQuery::text(1 << 40, "marker split")
                        })
                        .unwrap();
                    for f in &out.facts {
                        if let Some(snap) = ro.get(f.id) {
                            assert_consistent(&snap.text);
                        }
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(ro.stats().facts, FACTS as usize);
}

#[test]
#[cfg_attr(tarpaulin, ignore)] // timing stress: runs under `cargo test`, skipped under coverage
fn mixed_read_write_contention_progresses_without_deadlock() {
    // Many threads each interleave writes, checkpoints and reads on one shared
    // handle. Writes and checkpoints take the exclusive guard, reads the shared
    // one; a lock-ordering bug or a never-released guard would hang here. The
    // deadline turns that into a failure, not a stuck binary. No lost updates:
    // the final fact count must equal the writes that reported success.
    run_with_deadline(60, "mixed contention", || {
        const THREADS: usize = 6;
        const OPS: u64 = 400;

        let tmp = TempDir::new("mixed");
        let (db, _) = Database::builder(Config::default())
            .snapshot_every_ops(97) // frequent re-map under the readers
            .open(tmp.db())
            .unwrap();
        let db = Arc::new(db);
        let barrier = Arc::new(Barrier::new(THREADS));
        let writes = Arc::new(AtomicU64::new(0));
        let clock = Arc::new(AtomicU64::new(1));

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                let writes = Arc::clone(&writes);
                let clock = Arc::clone(&clock);
                std::thread::spawn(move || {
                    let mut rng = xorshift(0x9E37_0000 ^ t as u64);
                    barrier.wait();
                    for _ in 0..OPS {
                        match rng() % 8 {
                            0..=2 => {
                                let now = clock.fetch_add(1, Ordering::Relaxed);
                                let v = writes.fetch_add(1, Ordering::Relaxed);
                                db.remember(RememberInput::text(now, &fact_text(v)))
                                    .unwrap();
                            }
                            3 => {
                                let now = clock.fetch_add(1, Ordering::Relaxed);
                                db.checkpoint(now).unwrap();
                            }
                            _ => {
                                let out = db
                                    .recall(RecallQuery {
                                        k: 8,
                                        ..RecallQuery::text(1 << 40, "marker split")
                                    })
                                    .unwrap();
                                for f in &out.facts {
                                    if let Some(s) = db.get(f.id) {
                                        assert_consistent(&s.text);
                                    }
                                }
                                let _ = db.stats();
                            }
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Every remember that returned Ok is present — writers serialized
        // cleanly, none clobbered another (remember only appends here).
        assert_eq!(
            db.stats().facts as u64,
            writes.load(Ordering::Relaxed),
            "a counted write went missing — a lost update"
        );
    });
}

#[test]
#[cfg_attr(tarpaulin, ignore)] // timing stress: runs under `cargo test`, skipped under coverage
fn concurrent_writers_with_checkpoints_never_lose_a_write() {
    // Pure write contention: every thread appends its own disjoint block while
    // some also checkpoint. Writers serialize on the exclusive guard; the exact
    // final count proves none deadlocked and none was lost. Under a deadline so
    // a serialization bug fails fast instead of hanging.
    run_with_deadline(60, "writer serialization", || {
        const THREADS: u64 = 8;
        const PER: u64 = 200;

        let tmp = TempDir::new("writers");
        let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();
        let db = Arc::new(db);
        let barrier = Arc::new(Barrier::new(THREADS as usize));

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..PER {
                        // Disjoint timestamps per thread so nothing collides.
                        let now = 1 + t * PER + i;
                        db.remember(RememberInput::text(now, &fact_text(now)))
                            .unwrap();
                        if i.is_multiple_of(64) {
                            db.checkpoint(now).unwrap();
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            db.stats().facts as u64,
            THREADS * PER,
            "every write from every thread must land exactly once"
        );
    });
}
