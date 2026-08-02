//! Edge-history churn benchmark.
//!
//! `bench_edges` grows history by *breadth*: one hub, one version per edge.
//! This one grows it by *depth*: a small set of relations relinked and
//! unlinked over and over, so a single `(src, rel, dst)` triple accumulates
//! hundreds of versions. That is the shape a long-lived database reaches —
//! "employer", "assignee", "status" change again and again — and it stresses
//! the historical `as_of` traversal rather than the current graph.
//!
//! At most one version of a triple is valid at any instant, so an `as_of`
//! query wants one version per triple and must not pay for the rest.
//!
//! Deterministic synthetic data, no embedding model, no network.
//!
//! ```text
//! cargo run --release -p plugmem-host --example bench_edge_churn -- 200 500
//! ```

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use plugmem_host::{
    Config, Database, FsyncPolicy, LinkInput, MaintenanceOptions, RecallQuery, RememberInput,
    Stats, UnlinkInput,
};

const DEFAULT_TRIPLES: usize = 200;
const DEFAULT_ROUNDS: usize = 500;
const RECALL_WARMUP: usize = 3;
const RECALL_SAMPLES: usize = 20;
const ANCHOR: &str = "hub";
const REL: &str = "assigned_to";
/// Timestamp of the first link; every later round advances by `ROUND_STEP`.
const EPOCH: u64 = 1_000_000;
/// Milliseconds between two consecutive churn rounds.
const ROUND_STEP: u64 = 1_000;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "plugmem-churn-bench-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is before the Unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn db(&self) -> PathBuf {
        self.path.join("churn.plugmem")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct Options {
    triples: usize,
    rounds: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run(parse_options()?)
}

fn parse_options() -> Result<Options, String> {
    let mut positional = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "usage: bench_edge_churn [TRIPLES] [ROUNDS]\n\n\
                     TRIPLES defaults to 200, ROUNDS to 500. The benchmark \
                     links and unlinks each triple ROUNDS times, retaining \
                     TRIPLES * ROUNDS edge versions, then measures `as_of` \
                     graph recall at both ends of that history."
                );
                std::process::exit(0);
            }
            value if value.starts_with('-') => {
                return Err(format!("unrecognized argument `{value}`"));
            }
            value => positional.push(
                value
                    .parse::<usize>()
                    .map_err(|_| format!("expected a positive integer, got `{value}`"))?,
            ),
        }
    }
    let triples = positional.first().copied().unwrap_or(DEFAULT_TRIPLES);
    let rounds = positional.get(1).copied().unwrap_or(DEFAULT_ROUNDS);
    if positional.len() > 2 {
        return Err("expected at most TRIPLES and ROUNDS".into());
    }
    if triples == 0 || rounds == 0 {
        return Err("triples and rounds must be greater than zero".into());
    }
    Ok(Options { triples, rounds })
}

fn run(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let versions = options.triples * options.rounds;
    let corpus = format!("churn-{}x{}", options.triples, options.rounds);
    let (db, _) = Database::builder(Config::default())
        .fsync(FsyncPolicy::OnSnapshot)
        .snapshot_every_ops(0)
        .snapshot_journal_bytes(0)
        .open(temp.db())?;

    println!(
        "# plugmem edge churn benchmark: triples={} rounds={} versions={versions}",
        options.triples, options.rounds
    );
    println!("# persistence: OnSnapshot, no auto-snapshot");
    println!("# metric format: #DB<TAB>corpus<TAB>runtime<TAB>phase<TAB>metric<TAB>value");

    db.remember(RememberInput {
        now: 1,
        text: "edge churn hub anchor",
        entity: Some(ANCHOR),
        tags: &[],
        links: &[],
        vector: None,
        valid_from: None,
        metadata: None,
    })?;
    let names: Vec<String> = (0..options.triples)
        .map(|i| format!("target-{i}"))
        .collect();
    for name in &names {
        db.remember(RememberInput {
            now: 2,
            text: "edge churn target",
            entity: Some(name),
            tags: &[],
            links: &[],
            vector: None,
            valid_from: None,
            metadata: None,
        })?;
    }

    // Churn: every round reopens and recloses every triple, so each triple
    // ends up with `rounds` non-overlapping versions.
    let churn_start = Instant::now();
    for round in 0..options.rounds {
        let opened = EPOCH + round as u64 * ROUND_STEP;
        for name in &names {
            db.link(LinkInput {
                now: opened,
                src: ANCHOR,
                rel: REL,
                dst: name,
                provenance: None,
            })?;
        }
        // Leave the last round open, so the current graph is not empty.
        if round + 1 < options.rounds {
            for name in &names {
                black_box(db.unlink(UnlinkInput {
                    now: opened + ROUND_STEP / 2,
                    src: ANCHOR,
                    rel: REL,
                    dst: name,
                })?);
            }
        }
    }
    let churn_elapsed = churn_start.elapsed();
    emit_ms(&corpus, "churn", "elapsed_ms", churn_elapsed);
    emit_f64(
        &corpus,
        "churn",
        "latency_us_per_version",
        churn_elapsed.as_secs_f64() * 1_000_000.0 / versions as f64,
    );
    emit_stats(&corpus, "after_churn", db.stats());

    let last_round = EPOCH + (options.rounds - 1) as u64 * ROUND_STEP;
    let now = last_round + ROUND_STEP;
    let anchors = [ANCHOR];
    measure_query(&corpus, "current_graph_recall", || {
        db.recall(graph_query(now, None, &anchors))
    })?;
    // Recent history: the versions valid one round before the end.
    measure_query(&corpus, "as_of_recent", || {
        db.recall(graph_query(now, Some(last_round - ROUND_STEP), &anchors))
    })?;
    // Ancient history: the versions valid in the very first round.
    measure_query(&corpus, "as_of_oldest", || {
        db.recall(graph_query(now, Some(EPOCH), &anchors))
    })?;
    // Between two rounds every triple is closed, so nothing is valid and the
    // traversal has to establish that. Early in the history only the first
    // round precedes the instant, so the scan is short.
    measure_query(&corpus, "as_of_gap_early", || {
        db.recall(graph_query(now, Some(EPOCH + ROUND_STEP / 2 + 1), &anchors))
    })?;
    // The same question late in the history is the worst case a time-ordered
    // index still has: every version ever opened precedes the instant, every
    // one of them is closed by it, and there is no valid edge to stop at — so
    // the backward walk reads them all.
    measure_query(&corpus, "as_of_gap_late", || {
        db.recall(graph_query(
            now,
            Some(last_round - ROUND_STEP + ROUND_STEP / 2 + 1),
            &anchors,
        ))
    })?;

    // Churn fragments the edge arenas: the incoming mirror is keyed by the
    // far endpoint, so interleaved relations keep landing mid-page and pages
    // split in half. `Full` rewrites them in key order.
    let maintain_start = Instant::now();
    let report = db.maintain_with_options(now + 1, MaintenanceOptions::full())?;
    let maintain_elapsed = maintain_start.elapsed();
    emit_ms(&corpus, "full_maintain", "elapsed_ms", maintain_elapsed);
    emit_f64(
        &corpus,
        "full_maintain",
        "latency_us_per_version",
        maintain_elapsed.as_secs_f64() * 1_000_000.0 / versions as f64,
    );
    emit_usize(
        &corpus,
        "full_maintain",
        "edges_compacted",
        usize::from(report.edges_compacted),
    );
    emit_stats(&corpus, "after_full_maintain", db.stats());
    measure_query(&corpus, "as_of_recent_after_maintain", || {
        db.recall(graph_query(now, Some(last_round - ROUND_STEP), &anchors))
    })?;

    Ok(())
}

fn graph_query<'a>(now: u64, as_of: Option<u64>, entities: &'a [&'a str]) -> RecallQuery<'a> {
    let mut query = RecallQuery::text(now, "");
    query.text = None;
    query.entities = entities;
    query.as_of = as_of;
    query.k = 64;
    query.token_budget = Some(4096);
    query
}

fn measure_query(
    corpus: &str,
    phase: &str,
    mut call: impl FnMut() -> Result<plugmem_host::RecallResult, plugmem_host::HostError>,
) -> Result<(), plugmem_host::HostError> {
    for _ in 0..RECALL_WARMUP {
        black_box(call()?.edges.len());
    }
    let mut samples = Vec::with_capacity(RECALL_SAMPLES);
    let mut last_edges = 0usize;
    for _ in 0..RECALL_SAMPLES {
        let started = Instant::now();
        let result = call()?;
        last_edges = result.edges.len();
        black_box(last_edges);
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    emit_us_duration(corpus, phase, "p50_us", samples[samples.len() / 2]);
    emit_us_duration(corpus, phase, "p95_us", samples[samples.len() * 95 / 100]);
    emit_usize(corpus, phase, "edges", last_edges);
    Ok(())
}

fn emit_stats(corpus: &str, phase: &str, stats: Stats) {
    emit_usize(corpus, phase, "edges", stats.edges);
    emit_usize(corpus, phase, "edge_versions", stats.edge_versions);
    emit_usize(corpus, phase, "pool_bytes", stats.pool_bytes);
}

fn emit_ms(corpus: &str, phase: &str, metric: &str, duration: Duration) {
    emit_f64(corpus, phase, metric, duration.as_secs_f64() * 1_000.0);
}

fn emit_us_duration(corpus: &str, phase: &str, metric: &str, duration: Duration) {
    emit_f64(corpus, phase, metric, duration.as_secs_f64() * 1_000_000.0);
}

fn emit_usize(corpus: &str, phase: &str, metric: &str, value: usize) {
    emit_f64(corpus, phase, metric, value as f64);
}

fn emit_f64(corpus: &str, phase: &str, metric: &str, value: f64) {
    println!("#DB\t{corpus}\tnative\t{phase}\t{metric}\t{value:.3}");
}
