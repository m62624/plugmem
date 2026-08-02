//! Criterion bench for the on-demand container scrub over an mmap'd snapshot
//! The core `scrub` bench sweeps the slice budget on a warm
//! buffer and finds throughput flat (xxh3/bandwidth-bound); this adds the
//! host path — a real file, mapped read-only — so the sweep here carries the
//! cold-page fault cost too. Informational, not a CI gate.

use std::path::PathBuf;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use plugmem_host::{Config, Database, RememberInput};

/// A unique temp directory holding one benchmark database; removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let unique = format!(
            "plugmem-bench-{tag}-{}-{}",
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
        self.0.join("bench.plugmem")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bench_scrub_mmap(c: &mut Criterion) {
    let tmp = TempDir::new("scrub");
    // A multi-megabyte checkpointed database: wide texts so the container is
    // large enough to sweep MiB-scale slice budgets.
    {
        let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();
        let big = "lorem ipsum dolor sit amet ".repeat(75); // ~2 KiB
        for i in 0..4_000u64 {
            db.remember(RememberInput::text(i + 1, &big)).unwrap();
        }
        db.checkpoint(4_001).unwrap();
    }
    let snapshot_prefix = format!("{}.snap.", tmp.db().file_name().unwrap().to_string_lossy());
    let snapshot = std::fs::read_dir(&tmp.0)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with(&snapshot_prefix))
        })
        .expect("checkpoint should create a snapshot generation");
    let total = std::fs::metadata(snapshot).unwrap().len();
    eprintln!("scrub_mmap: snapshot is {total} bytes");

    let mut g = c.benchmark_group("scrub_mmap");
    g.sample_size(20);
    g.throughput(Throughput::Bytes(total));
    for budget in [
        64usize << 10,
        256 << 10,
        1 << 20,
        4 << 20,
        16 << 20,
        64 << 20,
    ] {
        g.bench_function(format!("budget_{}KiB", budget >> 10), |b| {
            b.iter(|| {
                // A fresh read-only handle per iteration: each scrub maps the
                // file anew, so the cold-page fault cost is included.
                let ro = Database::open_readonly(tmp.db(), Config::default()).unwrap();
                let mut done = 0u64;
                for step in ro.scrub_with_budget(budget).unwrap() {
                    done = step.unwrap().done_bytes;
                }
                done
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_scrub_mmap);
criterion_main!(benches);
