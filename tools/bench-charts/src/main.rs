//! Renders the benchmark `#TSV` rows into the README chart SVGs — the
//! arena charts from the [`plugmem-bench-matrix`](../bench-matrix) stand
//! and the core recall-latency chart from `plugmem-core`'s `bench_ops`
//! example, plus the native file-backed database and edge-lifecycle benchmarks.
//! A chart whose rows are absent from the input is left alone, so any source can
//! be rendered on its own or all can be piped together.
//!
//! Pure Rust: [plotters](https://github.com/plotters-rs/plotters) with its
//! SVG backend, so there is no browser, no WebDriver and nothing downloaded
//! — the same reproducibility contract as the zero-dependency bench stand
//! that produces the data. plotters is used directly (not through a
//! wrapper) so the allocation charts can use a **true logarithmic y-axis**:
//! their values span five orders of magnitude (≈40 for the arena vs
//! ≈1,000,000 for a per-element baseline), and on a linear axis the small
//! bars collapse to nothing — the whole point ("the arena allocates least")
//! would be invisible.
//!
//! ```text
//! # 1. collect data (run the stand as many times as you want — rows are
//! #    averaged by (n, runtime, structure, metric), so more runs = a
//! #    steadier average, not just the first pass):
//! for i in 1 2 3 4; do cargo run -qr -p plugmem-bench-matrix; done \
//!     | grep -P '^\d+\t' > bench.tsv
//! # 2. render (or, within the noise threshold, decline to rewrite):
//! cargo run -p plugmem-bench-charts -- bench.tsv
//! ```
//!
//! `bench.tsv` is the machine-readable block the stand prints (its `#TSV`
//! section: `n \t runtime \t structure \t metric \t value`). With no path
//! argument the tool reads that block from stdin.
//!
//! ## The noise gate (local only)
//!
//! Benchmarks jitter run to run, so a re-render would churn the SVGs even
//! when nothing really moved. Before rewriting a chart the tool compares
//! its new values against a committed [`baseline`](../baseline.tsv): if
//! every value moved less than the `threshold` in `config.toml`, the SVG
//! is left byte-identical (so `git status` stays clean); if any moved
//! more, the SVG is rewritten and the baseline updated in lockstep. This
//! only decides what lands on disk — you always choose what to commit. It
//! is never run in CI.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use plotters::coord::combinators::IntoLogRange;
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

/// The runtimes, in display order, with the colors used across every
/// chart (kept stable so the legend reads the same everywhere). Variant B
/// palette: navy / gold / rose.
const RUNTIMES: [(&str, RGBColor); 3] = [
    ("native", RGBColor(30, 58, 138)),   // navy
    ("wasmtime", RGBColor(202, 138, 4)), // gold
    ("wasmer", RGBColor(190, 24, 93)),   // rose
];

/// One chart: an output filename, a title, the corpus size and metric it
/// reads, the structures (bars) to include in order, the y-axis unit, and
/// whether the y-axis is logarithmic (for metrics spanning orders of
/// magnitude, so the small bars stay visible).
struct Chart {
    file: &'static str,
    title: &'static str,
    n: &'static str,
    metric: &'static str,
    structures: &'static [&'static str],
    y_title: &'static str,
    log: bool,
}

/// One same-workload size comparison chart.
struct ScaleChart {
    file: &'static str,
    title: &'static str,
    rows: &'static [(&'static str, &'static str, &'static str)],
    y_title: &'static str,
    log: bool,
}

/// Where each chart set is written (fixed repo paths, not user config).
const ARENA_OUT: &str = "crates/plugmem-arena/assets";
const CORE_OUT: &str = "crates/plugmem-core/assets";
const HOST_OUT: &str = "crates/plugmem-host/assets";

/// The core (engine) chart: per-source recall latency, native only. Its
/// rows come from `plugmem-core`'s `bench_ops` example (`n = core`,
/// `runtime = native`, `metric = latency_us`).
const CORE_CHARTS: &[Chart] = &[Chart {
    file: "recall-latency.svg",
    title: "recall source latency — native, log scale (lower is better)",
    n: "core",
    metric: "latency_us",
    structures: &[
        "BM25 (3 terms, 10k)",
        "tags (3 lists, 100k)",
        "flat vector (24k, d384)",
        "HNSW (30k, d384)",
    ],
    y_title: "µs (log)",
    // Logarithmic for the reason stated at the top of this file: the sources
    // span two and a half orders of magnitude (BM25 ~10 µs against HNSW
    // ~3300 µs), and on a linear axis the two cheap sources were a one-pixel
    // line — a chart comparing four sources that showed two of them.
    log: true,
}];

/// Native file-backed database charts. Rows come from the
/// `plugmem-host/examples/bench_database.rs` runner and use `n = database`.
const DATABASE_CHARTS: &[Chart] = &[
    Chart {
        file: "database-throughput-1m.svg",
        title: "file-backed database — streamed mixed-load throughput",
        n: "database-1m",
        metric: "load_ops_per_sec",
        structures: &["mixed_stream"],
        y_title: "operations / second",
        log: false,
    },
    Chart {
        file: "database-phases-1m.svg",
        title: "file-backed database — 1M lifecycle phase time",
        n: "database-1m",
        metric: "elapsed_ms",
        // No `mixed_stream`: the load phase reports `load_ms`, not
        // `elapsed_ms`, so naming it here charted nothing — and it has its own
        // throughput chart above, where ops/second is the number that means
        // something for a streamed load.
        structures: &[
            "checkpoint",
            "maintain",
            "writer_verify",
            "reopen",
            "reopen_verify",
            "readonly",
            "readonly_verify",
            "readonly_scrub",
        ],
        y_title: "milliseconds",
        log: true,
    },
    Chart {
        file: "database-recall-1m.svg",
        title: "file-backed database — recall p50 at 1M",
        n: "database-1m",
        metric: "p50_us",
        structures: &[
            "writer/text_recall",
            "writer/hybrid_recall",
            "writer/vector_recall",
            "readonly/text_recall",
            "readonly/hybrid_recall",
            "readonly/vector_recall",
        ],
        y_title: "microseconds",
        log: false,
    },
    Chart {
        file: "database-memory-1m.svg",
        title: "file-backed database — engine pool bytes at 1M",
        n: "database-1m",
        metric: "pool_bytes",
        structures: &["after_load", "after_maintain", "readonly"],
        y_title: "bytes",
        log: false,
    },
];

/// The like-for-like recall comparison chart. Its input contains one
/// `database-5k`, `database-100k` and `database-1m` series, all emitted by the
/// same file-backed runner and rendered in the same units.
///
/// The 5k point is the size an ordinary personal memory actually is, and until
/// the layout followed the data it was the size that behaved worst — so it
/// belongs on the chart rather than being extrapolated from the other two.
const DATABASE_SCALE_SERIES: [(&str, RGBColor); 3] = [
    ("5k operations", RGBColor(21, 128, 61)),
    ("100k operations", RGBColor(30, 58, 138)),
    ("1M operations", RGBColor(190, 24, 93)),
];
/// The corpus sizes those series read, in the same order.
const DATABASE_SCALE_SIZES: [&str; 3] = ["database-5k", "database-100k", "database-1m"];
const DATABASE_SCALE_ROWS: &[(&str, &str, &str)] = &[
    ("text_only", "writer_diagnostic/text_only", "p50_us"),
    ("tag + range", "writer_diagnostic/text_tag_range", "p50_us"),
    ("full hybrid", "writer_diagnostic/full_hybrid", "p50_us"),
    // The degenerate lexical query: one term the corpus uses everywhere, so
    // its posting list is the corpus. Charted next to the others because the
    // worst case is the number a caller has to budget for.
    ("common term", "writer_diagnostic/text_common", "p50_us"),
];

/// Like-for-like edge-lifecycle comparison series. Rows come from
/// `plugmem-host/examples/bench_edges.rs`; the SVGs are rendered only when both
/// 100k and 1M inputs are present.
const EDGE_SCALE_SERIES: [(&str, RGBColor); 2] = [
    ("100k edges", RGBColor(30, 58, 138)),
    ("1M edges", RGBColor(190, 24, 93)),
];
const EDGE_LATENCY_ROWS: &[(&str, &str, &str)] = &[
    ("link", "link", "latency_us_per_op"),
    ("unlink", "unlink", "latency_us_per_op"),
    ("full maintain", "full_maintain", "latency_us_per_op"),
];
const EDGE_RECALL_ROWS: &[(&str, &str, &str)] = &[
    ("current open", "current_graph_recall/open_edges", "p50_us"),
    (
        "history as_of",
        "historical_graph_recall/as_of_open",
        "p50_us",
    ),
    (
        "current after unlink",
        "current_graph_recall/after_unlink",
        "p50_us",
    ),
];
// There is no growth chart. Plotting "N edges linked, N history records
// retained, 0 current after unlink" draws the workload's own definition back
// at the reader: every bar is a number the benchmark was told to produce, not
// one it measured. The retention claim it was meant to make is a sentence, and
// the edge table states it.
const EDGE_SCALE_CHARTS: &[ScaleChart] = &[
    ScaleChart {
        file: "edge-lifecycle-latency-100k-1m.svg",
        title: "edge lifecycle — operation cost by corpus size",
        rows: EDGE_LATENCY_ROWS,
        y_title: "µs / edge",
        log: true,
    },
    ScaleChart {
        file: "edge-lifecycle-recall-100k-1m.svg",
        title: "edge lifecycle — graph recall by corpus size",
        rows: EDGE_RECALL_ROWS,
        y_title: "microseconds",
        log: true,
    },
];

/// The arena chart set. Every structure name matches a row the bench
/// stand emits.
const ARENA_CHARTS: &[Chart] = &[
    Chart {
        file: "arena-insert-100k.svg",
        title: "insert — ns/elem at 100k records (lower is better)",
        n: "100000",
        metric: "insert_ns",
        structures: &[
            "plugmem Arena (Ordered)",
            "std BTreeMap",
            "std HashMap",
            "sorted Vec (bulk)",
        ],
        y_title: "ns / element",
        log: false,
    },
    Chart {
        file: "arena-insert-1m.svg",
        title: "insert — ns/elem at 1M records (lower is better)",
        n: "1000000",
        metric: "insert_ns",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "ns / element",
        log: false,
    },
    Chart {
        file: "arena-lookup-100k.svg",
        title: "lookup — ns/op at 100k records (lower is better)",
        n: "100000",
        metric: "get_ns",
        structures: &["plugmem Arena (Uniform)", "std BTreeMap", "std HashMap"],
        y_title: "ns / op",
        log: false,
    },
    Chart {
        file: "arena-lookup-1m.svg",
        title: "lookup — ns/op at 1M records (lower is better)",
        n: "1000000",
        metric: "get_ns",
        structures: &["plugmem Arena (Uniform)", "std BTreeMap", "std HashMap"],
        y_title: "ns / op",
        log: false,
    },
    Chart {
        file: "arena-memory-1m.svg",
        title: "memory — bytes per element at 1M records",
        n: "1000000",
        metric: "mem_b",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "bytes / element",
        log: false,
    },
    Chart {
        file: "arena-allocs-1m.svg",
        title: "allocator calls to build 1M records — log scale (lower is better)",
        n: "1000000",
        metric: "allocs",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "allocator calls (log)",
        log: true,
    },
    Chart {
        file: "arena-tails-1m.svg",
        title: "insert tail latency (p99) at 1M records (lower is better)",
        n: "1000000",
        metric: "ins_p99",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "ns (p99)",
        log: false,
    },
    Chart {
        file: "arena-companions-insert-1m.svg",
        title: "companion structures — insert/push/intern ns at 1M (lower is better)",
        n: "1000000",
        metric: "insert_ns",
        structures: &[
            "plugmem BlobHeap",
            "Vec<Vec<u8>> (blob baseline)",
            "plugmem ChunkPool",
            "Vec<u8> per list (chunk baseline)",
            "plugmem Interner",
            "HashMap+Vec (intern baseline)",
        ],
        y_title: "ns / op",
        log: false,
    },
    Chart {
        file: "arena-companions-allocs-1m.svg",
        title: "companion structures — allocator calls at 1M — log scale (lower is better)",
        n: "1000000",
        metric: "allocs",
        structures: &[
            "plugmem BlobHeap",
            "Vec<Vec<u8>> (blob baseline)",
            "plugmem ChunkPool",
            "Vec<u8> per list (chunk baseline)",
            "plugmem Interner",
            "HashMap+Vec (intern baseline)",
        ],
        y_title: "allocator calls (log)",
        log: true,
    },
];

/// A cell key: `(n, runtime, structure, metric)`.
type Key = (String, String, String, String);
/// A cell table: key → (sum, count), so averaging is `sum / count`.
type Table = BTreeMap<Key, (f64, u32)>;

/// The rendering config (`config.toml`).
struct Config {
    threshold: f64,
    baseline: PathBuf,
}

fn main() {
    let cfg = load_config();

    // Args: an optional input path and an optional `--force`. `--force`
    // ignores the noise threshold and rewrites every chart — use it after
    // a style change (new colors/titles), where the values are unchanged
    // so the threshold would otherwise skip them. Re-rendering from the
    // committed baseline is then `-- tools/bench-charts/baseline.tsv --force`.
    let mut force = false;
    let mut input_path = None;
    for arg in std::env::args().skip(1) {
        if arg == "--force" {
            force = true;
        } else {
            input_path = Some(arg);
        }
    }

    // Input: the stand's #TSV rows, from a file argument or stdin.
    let raw = match input_path {
        Some(p) => std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {p}: {e}")),
        None => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .expect("reading stdin");
            s
        }
    };
    let new = parse(&raw);
    if new.is_empty() {
        eprintln!("no data rows (expected `n<TAB>runtime<TAB>structure<TAB>metric<TAB>value`)");
        std::process::exit(1);
    }
    let base = std::fs::read_to_string(&cfg.baseline)
        .map(|t| parse(&t))
        .unwrap_or_default();

    // Per chart: decide, render if it moved, and carry the right values
    // into the rewritten baseline (new for updated charts, old otherwise).
    // Charts absent from the input (e.g. the core chart when only arena
    // data was piped) are left entirely alone — SVG and baseline both.
    let mut next_baseline = base.clone();
    // `database` was the original label before the parser learned to retain
    // the operation count from the benchmark header. Drop those legacy rows
    // once a size-labelled database input is present.
    if new.keys().any(|key| key.0.starts_with("database-")) {
        next_baseline.retain(|key, _| key.0 != "database");
    }
    let mut updated = 0usize;
    let mut total = 0usize;
    for (out, charts) in [
        (ARENA_OUT, ARENA_CHARTS),
        (CORE_OUT, CORE_CHARTS),
        (HOST_OUT, DATABASE_CHARTS),
    ] {
        for chart in charts {
            let cells = chart_cells(chart, &new);
            if cells.is_empty() {
                continue; // no data for this chart in this input
            }
            total += 1;
            let verdict = if force {
                Verdict::Render { max_delta: 0.0 }
            } else {
                decide(&cells, &base, cfg.threshold)
            };
            match verdict {
                Verdict::Render { max_delta } => {
                    std::fs::create_dir_all(out).unwrap_or_else(|e| panic!("creating {out}: {e}"));
                    render(chart, &new, Path::new(out));
                    for (key, v) in &cells {
                        next_baseline.insert(key.clone(), (*v, 1));
                    }
                    updated += 1;
                    println!(
                        "{:32} rewritten (Δmax {:.0}%)",
                        chart.file,
                        max_delta * 100.0
                    );
                }
                Verdict::Skip { max_delta } => {
                    println!(
                        "{:32} unchanged (Δmax {:.0}% ≤ {:.0}%)",
                        chart.file,
                        max_delta * 100.0,
                        cfg.threshold * 100.0
                    );
                }
            }
        }
    }

    for chart in EDGE_SCALE_CHARTS {
        let cells = edge_scale_cells(chart, &new);
        if !scale_complete(&new, &["edge-100k", "edge-1m"], chart.rows) {
            continue;
        }
        total += 1;
        let verdict = if force {
            Verdict::Render { max_delta: 0.0 }
        } else {
            decide(&cells, &base, cfg.threshold)
        };
        match verdict {
            Verdict::Render { max_delta } => {
                std::fs::create_dir_all(HOST_OUT)
                    .unwrap_or_else(|e| panic!("creating {HOST_OUT}: {e}"));
                render_edge_scale(chart, &new, Path::new(HOST_OUT));
                for (key, value) in &cells {
                    next_baseline.insert(key.clone(), (*value, 1));
                }
                updated += 1;
                println!(
                    "{:32} rewritten (Δmax {:.0}%)",
                    chart.file,
                    max_delta * 100.0
                );
            }
            Verdict::Skip { max_delta } => {
                println!(
                    "{:32} unchanged (Δmax {:.0}% ≤ {:.0}%)",
                    chart.file,
                    max_delta * 100.0,
                    cfg.threshold * 100.0
                );
            }
        }
    }

    let scale_cells = database_scale_cells(&new);
    let has_100k = scale_cells.iter().any(|(key, _)| key.0 == "database-100k");
    let has_1m = scale_cells.iter().any(|(key, _)| key.0 == "database-1m");
    if has_100k && has_1m {
        total += 1;
        let verdict = if force {
            Verdict::Render { max_delta: 0.0 }
        } else {
            decide(&scale_cells, &base, cfg.threshold)
        };
        match verdict {
            Verdict::Render { max_delta } => {
                std::fs::create_dir_all(HOST_OUT)
                    .unwrap_or_else(|e| panic!("creating {HOST_OUT}: {e}"));
                render_database_scale(&new, Path::new(HOST_OUT));
                for (key, value) in &scale_cells {
                    next_baseline.insert(key.clone(), (*value, 1));
                }
                updated += 1;
                println!(
                    "{:32} rewritten (Δmax {:.0}%)",
                    "database-recall-scale.svg",
                    max_delta * 100.0
                );
            }
            Verdict::Skip { max_delta } => {
                println!(
                    "{:32} unchanged (Δmax {:.0}% ≤ {:.0}%)",
                    "database-recall-scale.svg",
                    max_delta * 100.0,
                    cfg.threshold * 100.0
                );
            }
        }
    }

    write_baseline(&cfg.baseline, &next_baseline);
    println!(
        "\n{updated}/{total} charts rewritten (threshold {:.0}%)",
        cfg.threshold * 100.0
    );
}

/// Loads `config.toml` from next to this crate.
fn load_config() -> Config {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let t: toml::Table = text.parse().expect("config.toml is not valid TOML");
    Config {
        threshold: t
            .get("threshold")
            .and_then(toml::Value::as_float)
            .unwrap_or(0.10),
        baseline: PathBuf::from(
            t.get("baseline")
                .and_then(toml::Value::as_str)
                .unwrap_or("tools/bench-charts/baseline.tsv"),
        ),
    }
}

/// The averaged value for a cell.
fn avg(table: &Table, key: &Key) -> Option<f64> {
    table.get(key).map(|(sum, n)| sum / f64::from(*n))
}

/// Parses `#TSV` rows and accumulates duplicates for averaging.
fn parse(raw: &str) -> Table {
    let mut table = Table::new();
    let mut database_label = String::from("database");
    for line in raw.lines() {
        if let Some(operations) = line
            .strip_prefix("# plugmem database benchmark: operations=")
            .and_then(|rest| rest.split_whitespace().next())
        {
            database_label = database_label_for(operations);
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (key, value): (Key, f64) = match c.as_slice() {
            [n, runtime, structure, metric, value] => {
                let Ok(value) = value.parse::<f64>() else {
                    continue;
                };
                (
                    (
                        (*n).into(),
                        (*runtime).into(),
                        (*structure).into(),
                        (*metric).into(),
                    ),
                    value,
                )
            }
            ["#DB", n, runtime, structure, metric, value] => {
                let Ok(value) = value.parse::<f64>() else {
                    continue;
                };
                let n = if *n == "database" {
                    database_label.as_str()
                } else {
                    *n
                };
                (
                    (
                        n.into(),
                        (*runtime).into(),
                        (*structure).into(),
                        (*metric).into(),
                    ),
                    value,
                )
            }
            _ => continue,
        };
        let cell = table.entry(key).or_insert((0.0, 0));
        cell.0 += value;
        cell.1 += 1;
    }
    table
}

/// Converts the runner's operation count into the stable corpus labels used
/// by database charts and the 100k-vs-1M comparison.
fn database_label_for(operations: &str) -> String {
    let Ok(operations) = operations.parse::<u64>() else {
        return "database".into();
    };
    if operations >= 1_000_000 && operations % 1_000_000 == 0 {
        format!("database-{}m", operations / 1_000_000)
    } else if operations >= 1_000 && operations % 1_000 == 0 {
        format!("database-{}k", operations / 1_000)
    } else {
        format!("database-{operations}")
    }
}

/// The (key, averaged value) cells one chart reads, skipping any absent
/// from the new data.
fn chart_cells(chart: &Chart, new: &Table) -> Vec<(Key, f64)> {
    let mut cells = Vec::new();
    for structure in chart.structures {
        for (runtime, _) in RUNTIMES {
            let key = (
                chart.n.into(),
                runtime.into(),
                (*structure).into(),
                chart.metric.into(),
            );
            if let Some(v) = avg(new, &key) {
                cells.push((key, v));
            }
        }
    }
    cells
}

/// Returns the rows consumed by the corpus-size comparison chart.
fn database_scale_cells(new: &Table) -> Vec<(Key, f64)> {
    scale_cells(new, &DATABASE_SCALE_SIZES, DATABASE_SCALE_ROWS)
}

/// Returns the rows consumed by an edge-lifecycle size comparison chart.
fn edge_scale_cells(chart: &ScaleChart, new: &Table) -> Vec<(Key, f64)> {
    scale_cells(new, &["edge-100k", "edge-1m"], chart.rows)
}

fn scale_cells(
    new: &Table,
    sizes: &[&str],
    rows: &[(&'static str, &'static str, &'static str)],
) -> Vec<(Key, f64)> {
    let mut cells = Vec::new();
    for &size in sizes {
        for &(_, structure, metric) in rows {
            let key = (
                size.into(),
                "native".into(),
                structure.into(),
                metric.into(),
            );
            if let Some(value) = avg(new, &key) {
                cells.push((key, value));
            }
        }
    }
    cells
}

fn scale_complete(new: &Table, sizes: &[&str], rows: &[(&str, &str, &str)]) -> bool {
    rows.iter().all(|&(_, structure, metric)| {
        sizes.iter().all(|&size| {
            avg(
                new,
                &(
                    size.into(),
                    "native".into(),
                    structure.into(),
                    metric.into(),
                ),
            )
            .is_some()
        })
    })
}

/// Whether a chart moved enough to rewrite.
enum Verdict {
    Render { max_delta: f64 },
    Skip { max_delta: f64 },
}

/// Rewrites when the baseline lacks a cell (a new chart) or any cell's
/// relative change exceeds the threshold.
fn decide(cells: &[(Key, f64)], base: &Table, threshold: f64) -> Verdict {
    let mut max_delta = 0.0f64;
    let mut missing = false;
    for (key, newv) in cells {
        match avg(base, key) {
            Some(bv) => {
                let denom = bv.abs().max(1e-9);
                max_delta = max_delta.max((newv - bv).abs() / denom);
            }
            None => missing = true,
        }
    }
    if missing || max_delta > threshold {
        Verdict::Render { max_delta }
    } else {
        Verdict::Skip { max_delta }
    }
}

/// Writes the baseline back, sorted and stable.
fn write_baseline(path: &Path, table: &Table) {
    let mut out = String::new();
    for ((n, runtime, structure, metric), (sum, count)) in table {
        let v = sum / f64::from(*count);
        out.push_str(&format!("{n}\t{runtime}\t{structure}\t{metric}\t{v:.1}\n"));
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}

/// The chart canvas size.
const CANVAS: (u32, u32) = (920, 520);

/// Assembles one chart's data and dispatches to the renderer with a linear
/// or logarithmic y-axis. Writes `out_dir/<file>`.
fn render(chart: &Chart, new: &Table, out_dir: &Path) {
    let cells = chart_cells(chart, new);

    // Runtimes that actually appear, in the canonical display order.
    let present: Vec<(&str, RGBColor)> = RUNTIMES
        .into_iter()
        .filter(|&(rt, _)| cells.iter().any(|(k, _)| k.1 == rt))
        .collect();

    // Categories in chart order that carry data, each a row of one optional
    // value per present runtime. Track the value spread for axis bounds.
    let mut categories: Vec<String> = Vec::new();
    let mut bars: Vec<Vec<Option<f64>>> = Vec::new();
    let (mut min_pos, mut max) = (f64::INFINITY, 0.0f64);
    for structure in chart.structures {
        let row: Vec<Option<f64>> = present
            .iter()
            .map(|&(rt, _)| {
                cells
                    .iter()
                    .find(|(k, _)| k.1 == rt && k.2 == *structure)
                    .map(|(_, v)| *v)
            })
            .collect();
        if row.iter().all(Option::is_none) {
            continue;
        }
        for &v in row.iter().flatten() {
            max = max.max(v);
            if v > 0.0 {
                min_pos = min_pos.min(v);
            }
        }
        categories.push(pretty(structure));
        bars.push(row);
    }

    let path = out_dir.join(chart.file);
    let (lo, hi) = log_bounds(min_pos, max);
    let data = BarData {
        path: &path,
        title: chart.title,
        y_title: chart.y_title,
        categories: &categories,
        series: &present,
        bars: &bars,
        y_base: if chart.log { lo } else { 0.0 },
    };
    if chart.log {
        draw_bars(&data, (lo..hi).log_scale());
    } else {
        draw_bars(&data, 0.0..(max * LINEAR_HEADROOM).max(1.0));
    }
}

/// Renders the same-workload recall comparison at 100k and 1M operations.
fn render_database_scale(new: &Table, out_dir: &Path) {
    render_scale(
        ScaleRender {
            file: "database-recall-scale.svg",
            title: "file-backed database — recall p50 by corpus size — 5k, 100k, 1M (log)",
            rows: DATABASE_SCALE_ROWS,
            y_title: "microseconds (log)",
            // The rows span two orders of magnitude once the degenerate query
            // is among them: on a linear axis its bar is the chart and the
            // other three are a flat line, which is the opposite of what a
            // comparison is for.
            log: true,
        },
        new,
        out_dir,
        &DATABASE_SCALE_SIZES,
        &DATABASE_SCALE_SERIES,
    );
}

fn render_edge_scale(chart: &ScaleChart, new: &Table, out_dir: &Path) {
    render_scale(
        ScaleRender {
            file: chart.file,
            title: chart.title,
            rows: chart.rows,
            y_title: chart.y_title,
            log: chart.log,
        },
        new,
        out_dir,
        &["edge-100k", "edge-1m"],
        &EDGE_SCALE_SERIES,
    );
}

struct ScaleRender {
    file: &'static str,
    title: &'static str,
    rows: &'static [(&'static str, &'static str, &'static str)],
    y_title: &'static str,
    log: bool,
}

fn render_scale(
    chart: ScaleRender,
    new: &Table,
    out_dir: &Path,
    sizes: &[&str],
    series: &[(&str, RGBColor)],
) {
    let path = out_dir.join(chart.file);
    let mut categories = Vec::new();
    let mut bars = Vec::new();
    let mut min_pos = f64::INFINITY;
    let mut max = 0.0f64;
    for &(label, structure, metric) in chart.rows {
        let mut row = Vec::new();
        for &size in sizes {
            let key = (
                size.into(),
                "native".into(),
                structure.into(),
                metric.into(),
            );
            let value = avg(new, &key);
            if let Some(value) = value {
                max = max.max(value);
                if value > 0.0 {
                    min_pos = min_pos.min(value);
                }
            }
            row.push(value);
        }
        if row.iter().any(Option::is_some) {
            categories.push(label.to_owned());
            bars.push(row);
        }
    }
    let data = BarData {
        path: &path,
        title: chart.title,
        y_title: chart.y_title,
        categories: &categories,
        series,
        bars: &bars,
        y_base: if chart.log {
            log_bounds(min_pos, max).0
        } else {
            0.0
        },
    };
    if chart.log {
        let (lo, hi) = log_bounds(min_pos, max);
        draw_bars(&data, (lo..hi).log_scale());
    } else {
        draw_bars(&data, 0.0..(max * LINEAR_HEADROOM).max(1.0));
    }
}

/// Nearest enclosing powers of ten for a logarithmic axis, with at least
/// one decade of span so a chart of equal values still renders.
///
/// `min` is the smallest **positive** value in the data (a log axis cannot show
/// zero, and the callers compute it that way); a run with no positive value at
/// all arrives as infinity and falls back to a single decade.
///
/// The floor follows the data rather than stopping at 1. It used to clamp
/// there, which was invisible until a measurement came in below a microsecond —
/// a graph recall after every edge was unlinked, at 0.7 µs — and its bar was
/// silently drawn beneath the axis. A chart that hides a real result reads as
/// missing data, which is worse than an unhelpful scale.
fn log_bounds(min: f64, max: f64) -> (f64, f64) {
    let lo = if min.is_finite() && min > 0.0 {
        10f64.powf(min.log10().floor())
    } else {
        1.0
    };
    let hi = 10f64.powf(max.max(lo * 10.0).log10().ceil());
    (lo, hi.max(lo * 10.0))
}

/// Everything a chart draw needs except the (generic) y-axis coordinate.
struct BarData<'a> {
    path: &'a Path,
    title: &'a str,
    y_title: &'a str,
    /// Category names, in x order.
    categories: &'a [String],
    /// Present runtimes (bar within each group), in legend order.
    series: &'a [(&'a str, RGBColor)],
    /// `bars[category][series_slot]` → value, `None` when absent.
    bars: &'a [Vec<Option<f64>>],
    /// The bar baseline: `0.0` for a linear axis, the axis floor for log.
    y_base: f64,
}

impl BarData<'_> {
    /// Where to put the legend: over whichever end of the chart has the
    /// shorter bars.
    ///
    /// A legend pinned to one corner sits on top of the data whenever the
    /// tallest bar is at that end, and a bar whose top is hidden cannot be
    /// read at all. Comparing the two halves costs nothing and is right far
    /// more often than a fixed corner.
    fn legend_position(&self) -> SeriesLabelPosition {
        let tallest = |group: &Vec<Option<f64>>| {
            group
                .iter()
                .flatten()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let mid = self.bars.len().div_ceil(2);
        let left = self.bars[..mid].iter().map(tallest).fold(0.0, f64::max);
        let right = self.bars[mid..].iter().map(tallest).fold(0.0, f64::max);
        if left <= right {
            SeriesLabelPosition::UpperLeft
        } else {
            SeriesLabelPosition::UpperRight
        }
    }
}

/// Headroom above the tallest bar on a linear axis.
///
/// Enough that the legend clears it: at 1.15 the tallest bar reached 87 % of
/// the height and the legend was drawn over its top, which is the one part of
/// a bar that carries information.
const LINEAR_HEADROOM: f64 = 1.35;

/// Draws a grouped bar chart (one bar per present runtime within each
/// category group) to `d.path`. Generic over the y-axis coordinate so the
/// same body serves a linear `Range<f64>` and a logarithmic
/// `(lo..hi).log_scale()`. Category names are drawn under each group.
fn draw_bars<YR>(d: &BarData, y_range: YR)
where
    YR: AsRangedCoord<Value = f64>,
    YR::CoordDescType: ValueFormatter<f64>,
{
    let root = SVGBackend::new(d.path, CANVAS).into_drawing_area();
    root.fill(&WHITE).expect("filling the canvas");

    let n = d.categories.len();
    let mut chart = ChartBuilder::on(&root)
        .caption(d.title, ("sans-serif", 16))
        .margin(18)
        .margin_right(30)
        .x_label_area_size(60)
        .y_label_area_size(72)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), y_range)
        .expect("building the chart area");

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(0)
        .x_label_formatter(&|_| String::new())
        .y_desc(d.y_title)
        .y_label_formatter(&fmt_axis)
        .axis_desc_style(("sans-serif", 13))
        .label_style(("sans-serif", 12))
        .draw()
        .expect("drawing the mesh");

    // Each category spans one x-unit. The group occupies `group_w` of it
    // (the rest is the gap between categories); within the group each
    // runtime gets a `slot_w` lane, and the bar fills `BAR_FILL` of its
    // lane so neighbouring bars are visibly separated rather than fused.
    let g = d.series.len().max(1);
    let group_w = 0.62f64;
    let slot_w = group_w / g as f64;
    const BAR_FILL: f64 = 0.80;
    let bar_w = slot_w * BAR_FILL;

    for (slot, (name, color)) in d.series.iter().enumerate() {
        let color = *color;
        let rects = (0..n).filter_map(|i| {
            d.bars[i][slot].map(|v| {
                let left = i as f64 - group_w / 2.0 + slot as f64 * slot_w + (slot_w - bar_w) / 2.0;
                Rectangle::new([(left, d.y_base), (left + bar_w, v)], color.filled())
            })
        });
        chart
            .draw_series(rects)
            .expect("drawing a runtime's bars")
            .label(*name)
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], color.filled()));
    }

    chart
        .configure_series_labels()
        .position(d.legend_position())
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.35))
        .label_font(("sans-serif", 12))
        .draw()
        .expect("drawing the legend");

    // Category names, centered under each group. Drawn on the root in pixel
    // space (from each group's axis-floor coordinate) so they land in the
    // x-label area below the axis, independent of the y-scale.
    let label = TextStyle::from(("sans-serif", 12))
        .pos(Pos::new(HPos::Center, VPos::Top))
        .color(&BLACK);
    for (i, name) in d.categories.iter().enumerate() {
        let (px, py) = chart.backend_coord(&(i as f64, d.y_base));
        root.draw(&Text::new(name.clone(), (px, py + 8), &label))
            .expect("drawing a category label");
    }

    root.present().expect("writing the SVG");
}

/// Formats a y-axis tick compactly: thousands as `k`, millions as `M`,
/// trailing `.0` dropped — so the log ticks read `10 / 100 / 1k / 100k /
/// 1M` (like the older charts) and linear ns/byte ticks stay clean.
fn fmt_axis(v: &f64) -> String {
    let v = *v;
    if v >= 1_000_000.0 {
        format!("{}M", trim(v / 1_000_000.0))
    } else if v >= 1_000.0 {
        format!("{}k", trim(v / 1_000.0))
    } else {
        trim(v)
    }
}

/// Renders a float without a redundant `.0`, one decimal otherwise.
fn trim(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{x:.1}")
    }
}

/// Shortens the long stand names to fit an x-axis tick.
fn pretty(structure: &str) -> String {
    match structure {
        "plugmem Arena (Ordered)" | "plugmem Arena (Uniform)" => "Arena",
        "std BTreeMap" => "BTreeMap",
        "std HashMap" => "HashMap",
        "sorted Vec (bulk)" => "Vec (bulk)",
        "plugmem BlobHeap" => "BlobHeap",
        "Vec<Vec<u8>> (blob baseline)" => "Vec<Vec<u8>>",
        "plugmem ChunkPool" => "ChunkPool",
        "Vec<u8> per list (chunk baseline)" => "Vec/list",
        "plugmem Interner" => "Interner",
        "HashMap+Vec (intern baseline)" => "HashMap+Vec",
        other => other,
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_axis_reaches_down_to_the_smallest_measurement() {
        // The case that exposed the old clamp: a sub-microsecond result next to
        // tens of microseconds. Its decade has to be on the axis, or the bar is
        // drawn below the floor and reads as no data at all.
        assert_eq!(log_bounds(0.735, 50.353), (0.1, 100.0));

        // Ordinary ranges are unchanged.
        assert_eq!(log_bounds(1.6, 3.0), (1.0, 10.0));
        assert_eq!(log_bounds(24.0, 900.0), (10.0, 1000.0));

        // At least one decade of span, even for a single repeated value.
        assert_eq!(log_bounds(5.0, 5.0), (1.0, 10.0));
        assert_eq!(log_bounds(0.02, 0.02), (0.01, 0.1));

        // No positive value at all (every sample zero): a usable default rather
        // than a log of zero.
        assert_eq!(log_bounds(f64::INFINITY, 0.0), (1.0, 10.0));
    }

    #[test]
    fn database_labels_are_stable_for_common_sizes() {
        assert_eq!(database_label_for("100000"), "database-100k");
        assert_eq!(database_label_for("1000000"), "database-1m");
        assert_eq!(database_label_for("123456"), "database-123456");
        assert_eq!(database_label_for("not-a-number"), "database");
    }

    #[test]
    fn database_headers_split_combined_runs_into_size_series() {
        let table = parse(
            "# plugmem database benchmark: operations=100000 dim=0\n\
             #DB\tdatabase\tnative\twriter_diagnostic/text_only\tp50_us\t52.2\n\
             # plugmem database benchmark: operations=1000000 dim=0\n\
             #DB\tdatabase\tnative\twriter_diagnostic/text_only\tp50_us\t1907.9\n",
        );
        assert_eq!(
            avg(
                &table,
                &(
                    "database-100k".into(),
                    "native".into(),
                    "writer_diagnostic/text_only".into(),
                    "p50_us".into()
                )
            ),
            Some(52.2)
        );
        assert_eq!(
            avg(
                &table,
                &(
                    "database-1m".into(),
                    "native".into(),
                    "writer_diagnostic/text_only".into(),
                    "p50_us".into()
                )
            ),
            Some(1907.9)
        );
    }

    #[test]
    fn scale_chart_collects_only_its_recall_rows() {
        let table = parse(
            "# plugmem database benchmark: operations=100000 dim=0\n\
             #DB\tdatabase\tnative\twriter_diagnostic/text_only\tp50_us\t52.2\n\
             #DB\tdatabase\tnative\twriter_diagnostic/full_hybrid\tp50_us\t636.4\n\
             # plugmem database benchmark: operations=1000000 dim=0\n\
             #DB\tdatabase\tnative\twriter_diagnostic/text_only\tp50_us\t1907.9\n\
             #DB\tdatabase\tnative\twriter_diagnostic/full_hybrid\tp50_us\t4891.6\n",
        );
        let cells = database_scale_cells(&table);
        assert_eq!(cells.len(), 4);
        assert!(cells.iter().all(|(key, _)| key.3 == "p50_us"));
    }

    #[test]
    fn edge_lifecycle_charts_collect_their_rows() {
        let table = parse(
            "# plugmem edge lifecycle benchmark: edges=100000\n\
             #DB\tedge-100k\tnative\tlink\tlatency_us_per_op\t24.2\n\
             #DB\tedge-100k\tnative\tunlink\tlatency_us_per_op\t18.1\n\
             #DB\tedge-100k\tnative\tfull_maintain\tlatency_us_per_op\t2.7\n\
             #DB\tedge-100k\tnative\tcurrent_graph_recall/open_edges\tp50_us\t2200.0\n\
             #DB\tedge-100k\tnative\thistorical_graph_recall/as_of_open\tp50_us\t7400.0\n\
             #DB\tedge-100k\tnative\tcurrent_edges/after_unlink\tcount\t0\n\
             #DB\tedge-100k\tnative\tedge_history/after_unlink\tcount\t100000\n",
        );

        // Two charts: operation cost and graph recall. The counts in the input
        // are still emitted by the benchmark and still read by the tables —
        // they are simply not charted, because they restate the workload.
        assert_eq!(EDGE_SCALE_CHARTS.len(), 2);
        assert_eq!(edge_scale_cells(&EDGE_SCALE_CHARTS[0], &table).len(), 3);
        assert_eq!(edge_scale_cells(&EDGE_SCALE_CHARTS[1], &table).len(), 2);
    }
}
