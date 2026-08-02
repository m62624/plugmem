//! Renders the benchmark `#TSV` rows into the README chart SVGs — the
//! arena charts from the [`plugmem-bench-matrix`](../bench-matrix) stand
//! and the core recall-latency chart from `plugmem-core`'s `bench_ops`
//! example, plus the native file-backed database benchmark. A chart whose rows
//! are absent from the input is left alone, so any source can be rendered on
//! its own or all can be piped together.
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
//! every value moved less than the `threshold` in [`config.toml`], the SVG
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

/// Where each chart set is written (fixed repo paths, not user config).
const ARENA_OUT: &str = "crates/plugmem-arena/assets";
const CORE_OUT: &str = "crates/plugmem-core/assets";
const HOST_OUT: &str = "crates/plugmem-host/assets";

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
    log: false,
}];

/// Native file-backed database charts. Rows come from the
/// `plugmem-host/examples/bench_database.rs` runner and use `n = database`.
const DATABASE_CHARTS: &[Chart] = &[
    Chart {
        file: "database-throughput-1m.svg",
        title: "file-backed database — streamed mixed-load throughput",
        n: "database",
        metric: "load_ops_per_sec",
        structures: &["mixed_stream"],
        y_title: "operations / second",
        log: false,
    },
    Chart {
        file: "database-phases-1m.svg",
        title: "file-backed database — 1M lifecycle phase time",
        n: "database",
        metric: "elapsed_ms",
        structures: &[
            "mixed_stream",
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
        n: "database",
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
        n: "database",
        metric: "pool_bytes",
        structures: &["after_load", "after_maintain", "readonly"],
        y_title: "bytes",
        log: false,
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
            _ => continue,
        };
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
        draw_bars(&data, 0.0..(max * 1.15).max(1.0));
    }
}

/// Nearest enclosing powers of ten for a logarithmic axis, with at least
/// one decade of span so a chart of equal values still renders.
fn log_bounds(min: f64, max: f64) -> (f64, f64) {
    let lo = 10f64.powf(min.max(1.0).log10().floor());
    let hi = 10f64.powf(max.max(10.0).log10().ceil());
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
        .position(SeriesLabelPosition::UpperRight)
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
