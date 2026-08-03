//! Million-scale, file-backed database benchmark.
//!
//! This is an end-to-end host measurement, not a Criterion micro-benchmark.
//! It streams a deterministic [`plugmem_testgen::Gen`] operation by operation,
//! so the million-operation corpus is never held in a `Vec<GenOp>`. The run
//! measures the writer path, checkpoint, maintain, reopen, read-only mmap,
//! recall, verification and snapshot scrub.
//!
//! The default corpus is lexical/structural (`dim = 0`) and uses no network or
//! embedding provider. Pass `--dim 384` to exercise deterministic synthetic
//! vectors and the HNSW maintenance path; those vectors are generated locally
//! by `plugmem-testgen`, not by a real embedding model.
//!
//! ```text
//! cargo run --release -p plugmem-host --example bench_database -- 1000000
//! cargo run --release -p plugmem-host --example bench_database -- 1000000 --dim 384
//! ```

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use plugmem_host::{Config, Database, FactId, FsyncPolicy, LinkInput, RecallQuery, RememberInput};
use plugmem_testgen::{Gen, GenOp, Profile, word_for};

const DEFAULT_OPS: usize = 1_000_000;
const RECALL_WARMUP: usize = 3;
const RECALL_SAMPLES: usize = 20;
const SCRUB_BUDGET: usize = 4 << 20;

/// A temporary directory with a stable database path for the whole run.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "plugmem-database-bench-{}-{}",
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
        self.path.join("database.plugmem")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Counts the generated operation mix, which makes a benchmark result
/// auditable instead of reporting only the requested stream length.
#[derive(Default)]
struct OpCounts {
    remember: usize,
    revise: usize,
    forget: usize,
    link: usize,
    maintain: usize,
}

impl OpCounts {
    fn observe(&mut self, op: &GenOp) {
        match op {
            GenOp::Remember { .. } => self.remember += 1,
            GenOp::Revise { .. } => self.revise += 1,
            GenOp::Forget { .. } => self.forget += 1,
            GenOp::Link { .. } => self.link += 1,
            GenOp::Maintain { .. } => self.maintain += 1,
        }
    }
}

/// A single benchmark configuration parsed from the command line.
struct Options {
    operations: usize,
    dim: usize,
    diagnose_recall: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    run(options)
}

fn parse_options() -> Result<Options, String> {
    let mut operations = DEFAULT_OPS;
    let mut dim = 0usize;
    let mut diagnose_recall = false;
    let mut positional_seen = false;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "usage: bench_database [OPERATIONS] [--dim DIM]\n\n\
                     OPERATIONS defaults to 1000000. DIM defaults to 0; use\n\
                     --dim 384 for deterministic synthetic vector generation.\n\
                     Pass --diagnose-recall for per-source query timings."
                );
                std::process::exit(0);
            }
            "--dim" => dim = parse_next(&mut args, "--dim")?,
            value if value.starts_with("--dim=") => {
                dim = parse_value(&value["--dim=".len()..], "--dim")?;
            }
            "--diagnose-recall" => diagnose_recall = true,
            value if !value.starts_with('-') && !positional_seen => {
                operations = parse_value(value, "operations")?;
                positional_seen = true;
            }
            value => return Err(format!("unrecognized argument `{value}`")),
        }
    }

    if operations == 0 {
        return Err("operations must be greater than zero".into());
    }
    if dim > 4096 {
        return Err("dim must be at most 4096".into());
    }
    Ok(Options {
        operations,
        dim,
        diagnose_recall,
    })
}

fn parse_next(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
    parse_value(&value, flag)
}

fn parse_value(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be a positive integer, got `{value}`"))
}

fn run(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let path = temp.db();
    let mut config = Config::default();
    config.dim = options.dim;

    println!(
        "# plugmem database benchmark: operations={} dim={} synthetic_vectors={} ",
        options.operations,
        options.dim,
        options.dim > 0
    );
    println!("# persistence: OnSnapshot, one final checkpoint, no auto-snapshot");
    println!("# metric format: #DB<TAB>corpus<TAB>runtime<TAB>phase<TAB>metric<TAB>value");

    let rss_before = rss();
    let (db, _) = Database::builder(config.clone())
        // This benchmark measures the database and snapshot machinery without
        // turning one million journal appends into one million syscall/fsync
        // samples. The import path still gets its own bounded batch behavior
        // in the host API and the durability policy is recorded explicitly.
        .fsync(FsyncPolicy::OnSnapshot)
        .snapshot_every_ops(0)
        .snapshot_journal_bytes(0)
        .open(&path)?;

    let profile = Profile {
        dim: options.dim,
        ..Profile::default()
    };
    let query_text = format!("{} {} {}", word_for(0), word_for(400), word_for(2500));
    // Rank 0 of the Zipf vocabulary: the term almost every document carries.
    let common_text = word_for(0);
    let mut query_entity = None;
    let mut query_tag = None;
    let mut query_vector = None;
    let mut counts = OpCounts::default();
    let mut last_now = 0u64;

    let load_start = Instant::now();
    for (index, op) in Gen::new(0x5EED_0000_0000_0007, profile)
        .take(options.operations)
        .enumerate()
    {
        last_now = op_now(&op);
        counts.observe(&op);
        if query_entity.is_none()
            && let Some(entity) = op_entity(&op)
        {
            query_entity = Some(entity.to_owned());
        }
        if query_tag.is_none()
            && let Some(tag) = op_tag(&op)
        {
            query_tag = Some(tag.to_owned());
        }
        if query_vector.is_none()
            && let Some(vector) = op_vector(&op)
        {
            query_vector = Some(vector.to_vec());
        }
        apply_op(&db, &op).map_err(|error| format!("operation {index} failed: {error}"))?;
    }
    let load_elapsed = load_start.elapsed();
    emit_ms("mixed_stream", "load_ms", load_elapsed);
    emit_f64(
        "mixed_stream",
        "load_ops_per_sec",
        options.operations as f64 / load_elapsed.as_secs_f64(),
    );
    emit_us(
        "mixed_stream",
        "load_us_per_op",
        load_elapsed,
        options.operations,
    );
    emit_usize("mixed_stream", "remember_ops", counts.remember);
    emit_usize("mixed_stream", "revise_ops", counts.revise);
    emit_usize("mixed_stream", "forget_ops", counts.forget);
    emit_usize("mixed_stream", "link_ops", counts.link);
    emit_usize("mixed_stream", "maintain_ops", counts.maintain);

    let after_load = db.stats();
    emit_stats("after_load", after_load, rss().zip(rss_before));

    let checkpoint_start = Instant::now();
    db.checkpoint(last_now + 1)?;
    emit_ms("checkpoint", "elapsed_ms", checkpoint_start.elapsed());
    emit_u64("checkpoint", "snapshot_bytes", snapshot_bytes(&path)?);

    let maintain_start = Instant::now();
    let maintain = db.maintain(last_now + 2)?;
    emit_ms("maintain", "elapsed_ms", maintain_start.elapsed());
    emit_usize("maintain", "purged_facts", maintain.purged);
    emit_usize("maintain", "bytes_before", maintain.bytes_before);
    emit_usize("maintain", "bytes_after", maintain.bytes_after);
    emit_usize("maintain", "no_op", usize::from(maintain.no_op));
    emit_usize(
        "maintain",
        "structural_compacted",
        usize::from(maintain.structural_compacted),
    );
    emit_usize(
        "maintain",
        "bm25_compacted",
        usize::from(maintain.bm25_compacted),
    );
    emit_usize(
        "maintain",
        "bm25_reindexed",
        usize::from(maintain.bm25_reindexed),
    );
    emit_usize(
        "maintain",
        "hnsw_rebuilt",
        usize::from(maintain.hnsw_rebuilt),
    );
    emit_usize(
        "maintain",
        "hnsw_remapped",
        usize::from(maintain.hnsw_remapped),
    );
    emit_u64(
        "maintain",
        "hnsw_inserted",
        u64::from(maintain.hnsw_inserted),
    );

    let after_maintain = db.stats();
    emit_stats("after_maintain", after_maintain, None);
    let writer_verify_start = Instant::now();
    db.verify()?;
    emit_ms("writer_verify", "elapsed_ms", writer_verify_start.elapsed());

    let query_entity = query_entity.unwrap_or_else(|| word_for(1 << 25));
    let query_tag = query_tag.unwrap_or_else(|| word_for(1 << 24));
    let query_entities = [query_entity.as_str()];
    let query_tags = [query_tag.as_str()];
    let query_range = (
        last_now.saturating_sub(30 * 24 * 60 * 60 * 1000),
        last_now + 1,
    );
    let queries = QuerySpec {
        now: last_now + 3,
        text: &query_text,
        common_text: &common_text,
        entities: &query_entities,
        tags: &query_tags,
        range: query_range,
        vector: query_vector.as_deref(),
    };

    measure_queries("writer", &db, &queries)?;
    if options.diagnose_recall {
        measure_recall_matrix("writer_diagnostic", &db, &queries)?;
    }
    drop(db);

    let reopen_start = Instant::now();
    let (reopened, report) = Database::builder(config.clone())
        .fsync(FsyncPolicy::OnSnapshot)
        .snapshot_every_ops(0)
        .snapshot_journal_bytes(0)
        .open(&path)?;
    emit_ms("reopen", "elapsed_ms", reopen_start.elapsed());
    emit_usize("reopen", "journal_replayed", report.replayed);
    emit_usize("reopen", "journal_skipped", report.skipped);
    let reopen_verify_start = Instant::now();
    reopened.verify()?;
    emit_ms("reopen_verify", "elapsed_ms", reopen_verify_start.elapsed());
    drop(reopened);

    let readonly_start = Instant::now();
    let readonly = Database::open_readonly(&path, config)?;
    emit_ms("readonly", "elapsed_ms", readonly_start.elapsed());
    emit_u64("readonly", "generation", readonly.generation());
    emit_stats("readonly", readonly.stats(), None);
    measure_readonly_queries(&readonly, &queries)?;

    let verify_start = Instant::now();
    readonly.verify()?;
    emit_ms("readonly_verify", "elapsed_ms", verify_start.elapsed());

    let scrub_start = Instant::now();
    let scrub = readonly.scrub_with_budget(SCRUB_BUDGET)?;
    let mut scrubbed = 0u64;
    for progress in scrub {
        scrubbed = progress?.done_bytes;
    }
    emit_ms("readonly_scrub", "elapsed_ms", scrub_start.elapsed());
    emit_u64("readonly_scrub", "bytes", scrubbed);

    Ok(())
}

fn apply_op(db: &Database, op: &GenOp) -> Result<(), plugmem_host::HostError> {
    match op {
        GenOp::Remember {
            now,
            text,
            entity,
            tags,
            links,
            vector,
        } => {
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let link_refs: Vec<(&str, &str)> = links
                .iter()
                .map(|(rel, dst)| (rel.as_str(), dst.as_str()))
                .collect();
            db.remember(RememberInput {
                now: *now,
                text,
                entity: entity.as_deref(),
                tags: &tag_refs,
                links: &link_refs,
                vector: vector.as_deref(),
                valid_from: None,
                metadata: None,
            })?;
        }
        GenOp::Revise {
            now,
            target,
            text,
            entity,
            tags,
            vector,
        } => {
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            db.revise(
                FactId(*target),
                RememberInput {
                    now: *now,
                    text,
                    entity: entity.as_deref(),
                    tags: &tag_refs,
                    links: &[],
                    vector: vector.as_deref(),
                    valid_from: None,
                    metadata: None,
                },
            )?;
        }
        GenOp::Forget { now, fact } => {
            db.forget(*now, FactId(*fact))?;
        }
        GenOp::Link {
            now,
            src,
            rel,
            dst,
            provenance,
        } => {
            db.link(LinkInput {
                now: *now,
                src,
                rel,
                dst,
                provenance: provenance.map(FactId),
            })?;
        }
        GenOp::Maintain { now } => {
            db.maintain(*now)?;
        }
    }
    Ok(())
}

struct QuerySpec<'a> {
    now: u64,
    text: &'a str,
    /// A single term the corpus uses everywhere — see [`QueryText::Common`].
    common_text: &'a str,
    entities: &'a [&'a str],
    tags: &'a [&'a str],
    range: (u64, u64),
    vector: Option<&'a [f32]>,
}

/// Which query text a variant asks with.
#[derive(Clone, Copy)]
enum QueryText {
    /// The three-word query the rest of the matrix shares: one frequent term,
    /// one mid-frequency, one rare.
    Mixed,
    /// One term the corpus uses in most documents — the degenerate lexical
    /// case. The stop-frequency guard drops such a term only when the query
    /// has a rarer one to fall back on, so a query made *only* of it is the
    /// engine's worst lexical input: its posting list covers the corpus.
    Common,
}

#[derive(Clone, Copy)]
struct QueryVariant {
    name: &'static str,
    text: QueryText,
    tags: bool,
    entities: bool,
    range: bool,
}

const RECALL_VARIANTS: &[QueryVariant] = &[
    QueryVariant {
        name: "text_only",
        text: QueryText::Mixed,
        tags: false,
        entities: false,
        range: false,
    },
    QueryVariant {
        name: "text_common",
        text: QueryText::Common,
        tags: false,
        entities: false,
        range: false,
    },
    QueryVariant {
        name: "text_tag",
        text: QueryText::Mixed,
        tags: true,
        entities: false,
        range: false,
    },
    QueryVariant {
        name: "text_entity",
        text: QueryText::Mixed,
        tags: false,
        entities: true,
        range: false,
    },
    QueryVariant {
        name: "text_range",
        text: QueryText::Mixed,
        tags: false,
        entities: false,
        range: true,
    },
    QueryVariant {
        name: "text_tag_entity",
        text: QueryText::Mixed,
        tags: true,
        entities: true,
        range: false,
    },
    QueryVariant {
        name: "text_tag_range",
        text: QueryText::Mixed,
        tags: true,
        entities: false,
        range: true,
    },
    QueryVariant {
        name: "text_entity_range",
        text: QueryText::Mixed,
        tags: false,
        entities: true,
        range: true,
    },
    QueryVariant {
        name: "full_hybrid",
        text: QueryText::Mixed,
        tags: true,
        entities: true,
        range: true,
    },
];

impl QueryVariant {
    fn request<'a>(self, query: &'a QuerySpec<'a>) -> RecallQuery<'a> {
        let text = match self.text {
            QueryText::Mixed => query.text,
            QueryText::Common => query.common_text,
        };
        let mut request = RecallQuery::text(query.now, text).with_k(8);
        if self.tags {
            request.tags = query.tags;
        }
        if self.entities {
            request.entities = query.entities;
        }
        if self.range {
            request.range = Some(query.range);
        }
        request
    }
}

fn measure_recall_matrix(
    phase: &str,
    db: &Database,
    query: &QuerySpec<'_>,
) -> Result<(), plugmem_host::HostError> {
    for &variant in RECALL_VARIANTS {
        measure_query(phase, variant.name, || db.recall(variant.request(query)))?;
    }
    Ok(())
}

fn measure_queries(
    phase: &str,
    db: &Database,
    query: &QuerySpec<'_>,
) -> Result<(), plugmem_host::HostError> {
    measure_query(phase, "text_recall", || {
        db.recall(RecallQuery::text(query.now, query.text).with_k(8))
    })?;
    measure_query(phase, "hybrid_recall", || {
        db.recall(RecallQuery {
            tags: query.tags,
            entities: query.entities,
            range: Some(query.range),
            ..RecallQuery::text(query.now, query.text).with_k(8)
        })
    })?;
    if let Some(vector) = query.vector {
        measure_query(phase, "vector_recall", || {
            db.recall(RecallQuery {
                vector: Some(vector),
                ..RecallQuery::text(query.now, "").with_k(8)
            })
        })?;
    }
    Ok(())
}

fn measure_readonly_queries(
    db: &plugmem_host::ReadOnlyDatabase,
    query: &QuerySpec<'_>,
) -> Result<(), plugmem_host::HostError> {
    measure_query("readonly", "text_recall", || {
        db.recall(RecallQuery::text(query.now, query.text).with_k(8))
    })?;
    measure_query("readonly", "hybrid_recall", || {
        db.recall(RecallQuery {
            tags: query.tags,
            entities: query.entities,
            range: Some(query.range),
            ..RecallQuery::text(query.now, query.text).with_k(8)
        })
    })?;
    if let Some(vector) = query.vector {
        measure_query("readonly", "vector_recall", || {
            db.recall(RecallQuery {
                vector: Some(vector),
                ..RecallQuery::text(query.now, "").with_k(8)
            })
        })?;
    }
    Ok(())
}

fn measure_query(
    phase: &str,
    name: &str,
    mut call: impl FnMut() -> Result<plugmem_host::RecallResult, plugmem_host::HostError>,
) -> Result<(), plugmem_host::HostError> {
    for _ in 0..RECALL_WARMUP {
        black_box(call()?.facts.len());
    }
    let mut samples = Vec::with_capacity(RECALL_SAMPLES);
    for _ in 0..RECALL_SAMPLES {
        let started = Instant::now();
        let result = call()?;
        black_box(result.facts.len());
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p95 = samples[samples.len() * 95 / 100];
    emit_us_duration(&format!("{phase}/{name}"), "p50_us", p50);
    emit_us_duration(&format!("{phase}/{name}"), "p95_us", p95);
    Ok(())
}

fn op_now(op: &GenOp) -> u64 {
    match op {
        GenOp::Remember { now, .. }
        | GenOp::Revise { now, .. }
        | GenOp::Forget { now, .. }
        | GenOp::Link { now, .. }
        | GenOp::Maintain { now } => *now,
    }
}

fn op_entity(op: &GenOp) -> Option<&str> {
    match op {
        GenOp::Remember { entity, .. } | GenOp::Revise { entity, .. } => entity.as_deref(),
        GenOp::Link { src, .. } => Some(src),
        GenOp::Forget { .. } | GenOp::Maintain { .. } => None,
    }
}

fn op_tag(op: &GenOp) -> Option<&str> {
    match op {
        GenOp::Remember { tags, .. } | GenOp::Revise { tags, .. } => {
            tags.first().map(String::as_str)
        }
        GenOp::Forget { .. } | GenOp::Link { .. } | GenOp::Maintain { .. } => None,
    }
}

fn op_vector(op: &GenOp) -> Option<&[f32]> {
    match op {
        GenOp::Remember { vector, .. } | GenOp::Revise { vector, .. } => vector.as_deref(),
        GenOp::Forget { .. } | GenOp::Link { .. } | GenOp::Maintain { .. } => None,
    }
}

fn snapshot_bytes(base: &Path) -> Result<u64, std::io::Error> {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let prefix = format!(
        "{}.snap.",
        base.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("database.plugmem")
    );
    std::fs::read_dir(parent)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if !name.starts_with(&prefix) || name.ends_with(".tmp") {
                return None;
            }
            entry.metadata().ok().map(|metadata| metadata.len())
        })
        .max()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "snapshot generation"))
}

fn rss() -> Option<usize> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem)
}

fn emit_stats(phase: &str, stats: plugmem_host::Stats, rss_delta: Option<(usize, usize)>) {
    emit_usize(phase, "facts", stats.facts);
    emit_usize(phase, "entities", stats.entities);
    emit_usize(phase, "terms", stats.terms);
    emit_usize(phase, "edges", stats.edges);
    emit_usize(phase, "edge_versions", stats.edge_versions);
    emit_usize(phase, "vectors", stats.vectors);
    emit_usize(phase, "tombstones", stats.tombstones);
    emit_u64(phase, "hnsw_indexed", u64::from(stats.hnsw_indexed));
    emit_usize(phase, "pool_bytes", stats.pool_bytes);
    emit_u64(phase, "next_fact", u64::from(stats.next_fact));
    emit_u64(phase, "next_edge", u64::from(stats.next_edge));
    if let Some((after, before)) = rss_delta {
        emit_usize(phase, "rss_delta_bytes", after.saturating_sub(before));
    }
}

fn emit_ms(phase: &str, metric: &str, duration: Duration) {
    emit_f64(phase, metric, duration.as_secs_f64() * 1_000.0);
}

fn emit_us_duration(phase: &str, metric: &str, duration: Duration) {
    emit_f64(phase, metric, duration.as_secs_f64() * 1_000_000.0);
}

fn emit_us(phase: &str, metric: &str, duration: Duration, count: usize) {
    emit_f64(
        phase,
        metric,
        duration.as_secs_f64() * 1_000_000.0 / count as f64,
    );
}

fn emit_usize(phase: &str, metric: &str, value: usize) {
    emit_u64(phase, metric, value as u64);
}

fn emit_u64(phase: &str, metric: &str, value: u64) {
    emit_f64(phase, metric, value as f64);
}

fn emit_f64(phase: &str, metric: &str, value: f64) {
    // Six-column rows are consumed by plugmem-bench-charts. The header records
    // the requested operation count; the fixed corpus label keeps the chart
    // schema stable for the normal 1M run.
    println!("#DB\tdatabase\tnative\t{phase}\t{metric}\t{value:.3}");
}

trait RecallQueryExt<'a> {
    fn with_k(self, k: usize) -> Self;
}

impl<'a> RecallQueryExt<'a> for RecallQuery<'a> {
    fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
}
