//! Edge-lifecycle benchmark.
//!
//! This is a focused file-backed benchmark for graph relationship churn:
//! standalone `link`, standalone `unlink`, retained edge-history growth,
//! current graph recall versus historical `as_of` graph recall, and a full
//! maintenance pass after many closed edges.
//!
//! It uses deterministic synthetic entities and facts, no embedding model, and
//! no network access.
//!
//! ```text
//! cargo run --release -p plugmem-host --example bench_edges -- 100000 | tee edge-benchmark-100k.tsv
//! cargo run -p plugmem-bench-charts -- edge-benchmark-100k.tsv --force
//! ```

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use plugmem_host::{
    Config, Database, FsyncPolicy, LinkInput, MaintenanceOptions, RecallQuery, RememberInput,
    Stats, UnlinkInput,
};

const DEFAULT_EDGES: usize = 100_000;
const RECALL_WARMUP: usize = 3;
const RECALL_SAMPLES: usize = 20;
const ANCHOR: &str = "hub";
const REL: &str = "related_to";

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "plugmem-edge-bench-{}-{}",
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
        self.path.join("edges.plugmem")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct Options {
    edges: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run(parse_options()?)
}

fn parse_options() -> Result<Options, String> {
    let mut edges = DEFAULT_EDGES;
    let mut positional_seen = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "usage: bench_edges [EDGES]\n\n\
                     EDGES defaults to 100000. The benchmark creates EDGES \
                     leaf facts, links them from one hub, unlinks them, then \
                     measures historical graph recall and full maintenance."
                );
                std::process::exit(0);
            }
            value if !value.starts_with('-') && !positional_seen => {
                edges = value
                    .parse()
                    .map_err(|_| format!("edges must be a positive integer, got `{value}`"))?;
                positional_seen = true;
            }
            value => return Err(format!("unrecognized argument `{value}`")),
        }
    }
    if edges == 0 {
        return Err("edges must be greater than zero".into());
    }
    Ok(Options { edges })
}

fn run(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let corpus = corpus_label(options.edges);
    let (db, _) = Database::builder(Config::default())
        .fsync(FsyncPolicy::OnSnapshot)
        .snapshot_every_ops(0)
        .snapshot_journal_bytes(0)
        .open(temp.db())?;

    println!(
        "# plugmem edge lifecycle benchmark: edges={}",
        options.edges
    );
    println!("# persistence: OnSnapshot, no auto-snapshot");
    println!("# metric format: #DB<TAB>corpus<TAB>runtime<TAB>phase<TAB>metric<TAB>value");

    let mut names = Vec::with_capacity(options.edges);
    let mut provenance = Vec::with_capacity(options.edges);
    let setup_start = Instant::now();
    db.remember(RememberInput {
        now: 1,
        text: "edge lifecycle hub anchor",
        entity: Some(ANCHOR),
        tags: &[],
        links: &[],
        vector: None,
        valid_from: None,
        metadata: None,
    })?;
    for i in 0..options.edges {
        let name = format!("leaf-{i}");
        let text = format!("edge lifecycle leaf {i}");
        let out = db.remember(RememberInput {
            now: 2 + i as u64,
            text: &text,
            entity: Some(&name),
            tags: &[],
            links: &[],
            vector: None,
            valid_from: None,
            metadata: None,
        })?;
        names.push(name);
        provenance.push(out.id);
    }
    emit_ms(&corpus, "setup", "elapsed_ms", setup_start.elapsed());
    emit_stats(&corpus, "after_setup", db.stats());
    emit_edge_counts(&corpus, "after_setup", db.stats());

    let link_base = 1_000_000u64;
    let link_start = Instant::now();
    for (i, (name, fact)) in names.iter().zip(provenance.iter()).enumerate() {
        db.link(LinkInput {
            now: link_base + i as u64,
            src: ANCHOR,
            rel: REL,
            dst: name,
            provenance: Some(*fact),
        })?;
    }
    let link_elapsed = link_start.elapsed();
    emit_ms(&corpus, "link", "elapsed_ms", link_elapsed);
    emit_us_per_op(
        &corpus,
        "link",
        "latency_us_per_op",
        link_elapsed,
        options.edges,
    );
    emit_stats(&corpus, "after_link", db.stats());
    emit_edge_counts(&corpus, "after_link", db.stats());

    let anchors = [ANCHOR];
    let current_now = link_base + options.edges as u64 + 1;
    measure_query(&corpus, "current_graph_recall/open_edges", || {
        db.recall(graph_query(current_now, None, &anchors))
    })?;

    let unlink_base = 2_000_000u64;
    let unlink_start = Instant::now();
    for (i, name) in names.iter().enumerate() {
        let fresh = db.unlink(UnlinkInput {
            now: unlink_base + i as u64,
            src: ANCHOR,
            rel: REL,
            dst: name,
        })?;
        black_box(fresh);
    }
    let unlink_elapsed = unlink_start.elapsed();
    emit_ms(&corpus, "unlink", "elapsed_ms", unlink_elapsed);
    emit_us_per_op(
        &corpus,
        "unlink",
        "latency_us_per_op",
        unlink_elapsed,
        options.edges,
    );
    emit_stats(&corpus, "after_unlink", db.stats());
    emit_edge_counts(&corpus, "after_unlink", db.stats());

    let after_unlink_now = unlink_base + options.edges as u64 + 1;
    measure_query(&corpus, "current_graph_recall/after_unlink", || {
        db.recall(graph_query(after_unlink_now, None, &anchors))
    })?;
    measure_query(&corpus, "historical_graph_recall/as_of_open", || {
        db.recall(graph_query(
            after_unlink_now,
            Some(unlink_base.saturating_sub(1)),
            &anchors,
        ))
    })?;

    let maintain_start = Instant::now();
    let maintain = db.maintain_with_options(after_unlink_now + 1, MaintenanceOptions::full())?;
    let maintain_elapsed = maintain_start.elapsed();
    emit_ms(&corpus, "full_maintain", "elapsed_ms", maintain_elapsed);
    emit_us_per_op(
        &corpus,
        "full_maintain",
        "latency_us_per_op",
        maintain_elapsed,
        options.edges,
    );
    emit_usize(
        &corpus,
        "full_maintain",
        "bytes_before",
        maintain.bytes_before,
    );
    emit_usize(
        &corpus,
        "full_maintain",
        "bytes_after",
        maintain.bytes_after,
    );
    emit_usize(&corpus, "full_maintain", "purged_facts", maintain.purged);
    emit_stats(&corpus, "after_full_maintain", db.stats());
    emit_edge_counts(&corpus, "after_full_maintain", db.stats());

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
    let mut last_facts = 0usize;
    let mut last_edges = 0usize;
    for _ in 0..RECALL_SAMPLES {
        let started = Instant::now();
        let result = call()?;
        last_facts = result.facts.len();
        last_edges = result.edges.len();
        black_box((last_facts, last_edges));
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    emit_us_duration(corpus, phase, "p50_us", samples[samples.len() / 2]);
    emit_us_duration(corpus, phase, "p95_us", samples[samples.len() * 95 / 100]);
    emit_usize(corpus, phase, "facts", last_facts);
    emit_usize(corpus, phase, "edges", last_edges);
    Ok(())
}

fn emit_stats(corpus: &str, phase: &str, stats: Stats) {
    emit_usize(corpus, phase, "facts", stats.facts);
    emit_usize(corpus, phase, "entities", stats.entities);
    emit_usize(corpus, phase, "edges", stats.edges);
    emit_usize(corpus, phase, "edge_versions", stats.edge_versions);
    emit_u64(corpus, phase, "next_edge", u64::from(stats.next_edge));
    emit_usize(corpus, phase, "pool_bytes", stats.pool_bytes);
}

fn emit_edge_counts(corpus: &str, phase: &str, stats: Stats) {
    emit_usize(
        corpus,
        &format!("current_edges/{phase}"),
        "count",
        stats.edges,
    );
    emit_usize(
        corpus,
        &format!("edge_history/{phase}"),
        "count",
        stats.edge_versions,
    );
}

fn corpus_label(edges: usize) -> String {
    if edges >= 1_000_000 && edges.is_multiple_of(1_000_000) {
        format!("edge-{}m", edges / 1_000_000)
    } else if edges >= 1_000 && edges.is_multiple_of(1_000) {
        format!("edge-{}k", edges / 1_000)
    } else {
        format!("edge-{edges}")
    }
}

fn emit_ms(corpus: &str, phase: &str, metric: &str, duration: Duration) {
    emit_f64(corpus, phase, metric, duration.as_secs_f64() * 1_000.0);
}

fn emit_us_duration(corpus: &str, phase: &str, metric: &str, duration: Duration) {
    emit_f64(corpus, phase, metric, duration.as_secs_f64() * 1_000_000.0);
}

fn emit_us_per_op(corpus: &str, phase: &str, metric: &str, duration: Duration, count: usize) {
    emit_f64(
        corpus,
        phase,
        metric,
        duration.as_secs_f64() * 1_000_000.0 / count as f64,
    );
}

fn emit_usize(corpus: &str, phase: &str, metric: &str, value: usize) {
    emit_u64(corpus, phase, metric, value as u64);
}

fn emit_u64(corpus: &str, phase: &str, metric: &str, value: u64) {
    emit_f64(corpus, phase, metric, value as f64);
}

fn emit_f64(corpus: &str, phase: &str, metric: &str, value: f64) {
    println!("#DB\t{corpus}\tnative\t{phase}\t{metric}\t{value:.3}");
}
