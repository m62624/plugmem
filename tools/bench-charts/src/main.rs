//! Renders the benchmark `#TSV` rows into the README chart SVGs — the
//! arena charts from the [`plugmem-bench-matrix`](../bench-matrix) stand
//! and the core recall-latency chart from `plugmem-core`'s `bench_ops`
//! example. A chart whose rows are absent from the input is left alone, so
//! either source can be rendered on its own or both piped together.
//!
//! Pure Rust: [Plotlars](https://github.com/alceal/plotlars) with the
//! `plotters` backend, so there is no browser, no WebDriver and nothing
//! downloaded — the same reproducibility contract as the zero-dependency
//! bench stand that produces the data.
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
//! every value moved less than the `threshold` in [`config.toml`], the SVG
//! is left byte-identical (so `git status` stays clean); if any moved
//! more, the SVG is rewritten and the baseline updated in lockstep. This
//! only decides what lands on disk — you always choose what to commit. It
//! is never run in CI.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use plotlars::polars::prelude::*;
use plotlars::{BarPlot, Legend, Plot, Rgb, Text};

/// The runtimes, in display order, with the colors used across every
/// chart (kept stable so the legend reads the same everywhere).
const RUNTIMES: [(&str, Rgb); 3] = [
    ("native", Rgb(30, 58, 138)),   // navy
    ("wasmtime", Rgb(202, 138, 4)), // gold
    ("wasmer", Rgb(190, 24, 93)),   // rose
];

/// One chart: an output filename, a title, the corpus size and metric it
/// reads, the structures (bars) to include in order, and the y-axis unit.
struct Chart {
    file: &'static str,
    title: &'static str,
    n: &'static str,
    metric: &'static str,
    structures: &'static [&'static str],
    y_title: &'static str,
}

/// Where each chart set is written (fixed repo paths, not user config).
const ARENA_OUT: &str = "crates/plugmem-arena/assets";
const CORE_OUT: &str = "crates/plugmem-core/assets";

/// The core (engine) chart: per-source recall latency, native only. Its
/// rows come from `plugmem-core`'s `bench_ops` example (`n = core`,
/// `runtime = native`, `metric = latency_us`).
const CORE_CHARTS: &[Chart] = &[Chart {
    file: "recall-latency.svg",
    title: "recall source latency — µs, native (lower is better)",
    n: "core",
    metric: "latency_us",
    structures: &[
        "BM25 (3 terms, 10k)",
        "tags (3 lists, 100k)",
        "flat vector (24k, d384)",
        "HNSW (30k, d384)",
    ],
    y_title: "µs",
}];

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
    },
    Chart {
        file: "arena-insert-1m.svg",
        title: "insert — ns/elem at 1M records (lower is better)",
        n: "1000000",
        metric: "insert_ns",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "ns / element",
    },
    Chart {
        file: "arena-lookup-100k.svg",
        title: "lookup — ns/op at 100k records (lower is better)",
        n: "100000",
        metric: "get_ns",
        structures: &["plugmem Arena (Uniform)", "std BTreeMap", "std HashMap"],
        y_title: "ns / op",
    },
    Chart {
        file: "arena-lookup-1m.svg",
        title: "lookup — ns/op at 1M records (lower is better)",
        n: "1000000",
        metric: "get_ns",
        structures: &["plugmem Arena (Uniform)", "std BTreeMap", "std HashMap"],
        y_title: "ns / op",
    },
    Chart {
        file: "arena-memory-1m.svg",
        title: "memory — bytes per element at 1M records",
        n: "1000000",
        metric: "mem_b",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "bytes / element",
    },
    Chart {
        file: "arena-allocs-1m.svg",
        title: "allocator calls to build 1M records (lower is better)",
        n: "1000000",
        metric: "allocs",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "allocator calls",
    },
    Chart {
        file: "arena-tails-1m.svg",
        title: "insert tail latency (p99) at 1M records (lower is better)",
        n: "1000000",
        metric: "ins_p99",
        structures: &["plugmem Arena (Ordered)", "std BTreeMap", "std HashMap"],
        y_title: "ns (p99)",
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
    },
    Chart {
        file: "arena-companions-allocs-1m.svg",
        title: "companion structures — allocator calls at 1M (lower is better)",
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
        y_title: "allocator calls",
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
    let mut updated = 0usize;
    let mut total = 0usize;
    for (out, charts) in [(ARENA_OUT, ARENA_CHARTS), (CORE_OUT, CORE_CHARTS)] {
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
    for line in raw.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 5 {
            continue;
        }
        let Ok(value) = c[4].parse::<f64>() else {
            continue;
        };
        let key = (c[0].into(), c[1].into(), c[2].into(), c[3].into());
        let cell = table.entry(key).or_insert((0.0, 0));
        cell.0 += value;
        cell.1 += 1;
    }
    table
}

/// The (key, averaged value) cells one chart reads, skipping any absent
/// from the new data.
fn chart_cells(chart: &Chart, new: &Table) -> Vec<(Key, f64)> {
    let mut cells = Vec::new();
    for structure in chart.structures {
        for (runtime, _) in RUNTIMES {
            let key = (
                chart.n.into(),
                (*runtime).into(),
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

/// Renders one chart to `out_dir/<file>`.
fn render(chart: &Chart, new: &Table, out_dir: &Path) {
    let (mut structures, mut runtimes, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for (key, v) in chart_cells(chart, new) {
        structures.push(pretty(&key.2));
        runtimes.push(key.1);
        values.push(v);
    }

    let df = df!(
        "structure" => structures,
        "runtime" => runtimes,
        "value" => values,
    )
    .expect("building the chart dataframe");

    let path = out_dir.join(chart.file);
    BarPlot::builder()
        .data(&df)
        .labels("structure")
        .values("value")
        .group("runtime")
        .colors(RUNTIMES.iter().map(|(_, c)| *c).collect())
        .plot_title(Text::from(chart.title).font("sans-serif").size(15))
        .x_title(Text::from("").font("sans-serif").size(12))
        .y_title(Text::from(chart.y_title).font("sans-serif").size(12))
        .legend(&Legend::new().orientation(plotlars::Orientation::Horizontal))
        .build()
        .save(path.to_str().expect("utf-8 output path"));
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
