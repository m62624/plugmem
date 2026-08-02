//! Maintenance modes and what they are for.
//!
//! Run with:
//!
//! ```text
//! cargo run -p plugmem-host --example maintain_modes
//! ```
//!
//! `maintain` is the explicit background-work boundary. Normal writes stay
//! append-friendly; maintenance is where the database can purge tombstoned
//! facts, compact pools, rebuild text indexes, and optimize vector indexes.

use plugmem_host::{
    Config, Database, FactId, FsyncPolicy, HostError, MaintenanceMode, MaintenanceOptions,
    RememberInput,
};

fn main() -> Result<(), HostError> {
    let dir = std::env::temp_dir().join(format!("plugmem-maintain-modes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("memory.plugmem");

    let (db, _) = Database::builder(Config::default())
        .fsync(FsyncPolicy::OnSnapshot)
        .snapshot_every_ops(0)
        .snapshot_journal_bytes(0)
        .open(&path)?;

    for i in 0..32u32 {
        db.remember(RememberInput {
            now: 1_700_000_000_000 + u64::from(i),
            text: &format!("maintain example fact {i}"),
            entity: Some("maintain-demo"),
            tags: &["maintenance"],
            links: &[],
            vector: None,
            valid_from: None,
            metadata: None,
        })?;
    }
    for id in [1, 3, 5, 7] {
        db.forget(1_700_000_100_000 + u64::from(id), FactId(id))?;
    }

    print_stats("before maintenance", &db);

    // Auto is the production default: do only the work that is currently
    // needed. With tombstones present it may compact; with no pending work it
    // is intentionally cheap.
    print_report(
        "auto",
        db.maintain_with_options(1_700_000_200_000, MaintenanceOptions::auto())?,
    );

    // Compact asks for structural compaction/purge. It is the right focused
    // mode when you mainly want to reclaim tombstoned facts and pool space.
    print_report(
        "compact",
        db.maintain_with_options(
            1_700_000_300_000,
            MaintenanceOptions {
                mode: MaintenanceMode::Compact,
                max_hnsw_inserts: None,
            },
        )?,
    );

    // ReindexText rebuilds lexical postings from live text. Use it after text
    // index format changes or when validating text-index recovery.
    print_report(
        "reindex-text",
        db.maintain_with_options(
            1_700_000_400_000,
            MaintenanceOptions {
                mode: MaintenanceMode::ReindexText,
                max_hnsw_inserts: None,
            },
        )?,
    );

    // OptimizeVectors advances/rebuilds vector graph work without forcing a
    // text/fact compaction. In this dim=0 example it is a documented no-op.
    print_report(
        "optimize-vectors",
        db.maintain_with_options(
            1_700_000_500_000,
            MaintenanceOptions {
                mode: MaintenanceMode::OptimizeVectors,
                max_hnsw_inserts: None,
            },
        )?,
    );

    // Full is the offline rebuild: use it for explicit maintenance windows and
    // benchmarks, not on every interactive write.
    print_report(
        "full",
        db.maintain_with_options(1_700_000_600_000, MaintenanceOptions::full())?,
    );

    print_stats("after maintenance", &db);
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

fn print_report(label: &str, report: plugmem_host::MaintainReport) {
    println!(
        "{label}: no_op={} purged={} structural={} bm25_reindexed={} hnsw_rebuilt={} hnsw_remapped={} hnsw_inserted={}",
        report.no_op,
        report.purged,
        report.structural_compacted,
        report.bm25_reindexed,
        report.hnsw_rebuilt,
        report.hnsw_remapped,
        report.hnsw_inserted
    );
}

fn print_stats(label: &str, db: &Database) {
    let stats = db.stats();
    println!(
        "{label}: facts={} tombstones={} edges={} edge_versions={} pool_bytes={}",
        stats.facts, stats.tombstones, stats.edges, stats.edge_versions, stats.pool_bytes
    );
}
