//! Cross-runtime benchmark matrix — **zero dependencies**, fully scripted.
//!
//! This is the reproduction stand behind every number in the READMEs. One
//! command runs the whole thing:
//!
//! ```text
//! cargo run --release -p plugmem-bench-matrix
//! ```
//!
//! The runner builds `plugmem-arena`'s `bench_repro` example for native and
//! `wasm32-wasip1`, executes it on every runtime it can find (native
//! always; `wasmtime` and `wasmer` when installed — install them yourself,
//! the runner never downloads anything) at two corpus sizes (100k — the
//! design center, 1M — the scale ceiling), parses the `#M` metric lines
//! each run prints, and merges everything into per-metric comparison tables
//! (markdown + a machine-readable TSV at the end).
//!
//! Verification is the point: the workload is deterministic (seeded keys),
//! the harness is `std::process::Command` — nothing hidden, no network, no
//! extra crates. If a runtime is missing you get an install hint and the
//! matrix simply has fewer columns.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// structure -> metric -> value, as parsed from one `bench_repro` run.
type Run = BTreeMap<String, BTreeMap<String, f64>>;

/// Workspace root (this crate lives in `tools/bench-matrix`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("tools/bench-matrix sits two levels below the workspace root")
        .to_path_buf()
}

/// Runs a command in the workspace root; returns stdout on success.
fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(cmd)
        .args(args)
        .current_dir(workspace_root())
        .output()
        .map_err(|e| format!("failed to start `{cmd}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`{cmd} {}` failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parses `#M <structure> <metric> <value>` lines (tab-separated).
fn parse(output: &str) -> Run {
    let mut runs: Run = BTreeMap::new();
    for line in output.lines() {
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() == 4
            && cells[0] == "#M"
            && let Ok(value) = cells[3].parse::<f64>()
        {
            runs.entry(cells[1].to_string())
                .or_default()
                .insert(cells[2].to_string(), value);
        }
    }
    runs
}

/// Structures in display order (matches `bench_repro`).
const STRUCTURES: [&str; 6] = [
    "plugmem Arena (Ordered)",
    "plugmem Arena (Uniform)",
    "std BTreeMap",
    "std HashMap",
    "sorted Vec (bulk)",
    "sorted Vec (incremental)",
];

/// (metric key, display name) in display order.
const METRICS: [(&str, &str); 9] = [
    ("insert_ns", "insert ns/elem"),
    ("ins_p50", "insert latency p50 ns"),
    ("ins_p99", "insert latency p99 ns"),
    ("ins_max", "insert latency max ns"),
    ("get_ns", "point lookup ns/op"),
    ("scan_ns", "ordered scan ns/elem"),
    ("mem_b", "retained memory B/elem"),
    ("mem_peak_b", "peak memory B/elem"),
    ("allocs", "allocator calls per build"),
];

fn main() {
    let root = workspace_root();
    println!("bench-matrix: workspace {}", root.display());

    println!("building bench_repro (native release)...");
    run(
        "cargo",
        &[
            "build",
            "--release",
            "--example",
            "bench_repro",
            "-p",
            "plugmem-arena",
        ],
    )
    .expect("native build");
    println!("building bench_repro (wasm32-wasip1 release)...");
    run(
        "cargo",
        &[
            "build",
            "--release",
            "--example",
            "bench_repro",
            "-p",
            "plugmem-arena",
            "--target",
            "wasm32-wasip1",
        ],
    )
    .expect("wasm build (rustup target add wasm32-wasip1)");

    let native_bin = root.join("target/release/examples/bench_repro");
    let wasm_bin = root.join("target/wasm32-wasip1/release/examples/bench_repro.wasm");
    let wasm_path = wasm_bin.to_str().unwrap();

    // (runtime label, command, args-before-the-corpus-size, install hint).
    // Native always runs; wasm runtimes are optional and installed by the
    // verifier, never by this tool.
    let runtimes: [(&str, &str, Vec<&str>, &str); 3] = [
        ("native", native_bin.to_str().unwrap(), vec![], ""),
        (
            "wasmtime",
            "wasmtime",
            vec!["run", wasm_path],
            "https://wasmtime.dev/ (curl https://wasmtime.dev/install.sh -sSf | bash)",
        ),
        (
            "wasmer",
            "wasmer",
            vec!["run", wasm_path, "--"],
            "https://wasmer.io/ (curl https://get.wasmer.io -sSfL | sh)",
        ),
    ];

    let sizes = ["100000", "1000000"];
    // (size, [(runtime label, parsed run)]).
    let mut all: Vec<(&str, Vec<(&str, Run)>)> = Vec::new();
    for size in sizes {
        let mut per_runtime = Vec::new();
        for (label, cmd, args, hint) in &runtimes {
            print!("running N={size} on {label}... ");
            let mut argv = args.clone();
            argv.push(size);
            match run(cmd, &argv) {
                Ok(out) => {
                    println!("ok");
                    per_runtime.push((*label, parse(&out)));
                }
                Err(e) if e.contains("failed to start") => {
                    println!("not found — skipped. Install: {hint}");
                }
                Err(e) => panic!("{label} run failed: {e}"),
            }
        }
        all.push((size, per_runtime));
    }

    for (size, results) in &all {
        println!("\n## N = {size}");
        for (metric, display) in METRICS {
            println!("\n### {display}\n");
            print!("| structure |");
            for (label, _) in results {
                print!(" {label} |");
            }
            println!();
            print!("|---|");
            for _ in results {
                print!("---|");
            }
            println!();
            for s in STRUCTURES {
                // Skip rows absent at this size (incremental Vec at 1M).
                if results.iter().all(|(_, r)| !r.contains_key(s)) {
                    continue;
                }
                print!("| {s} |");
                for (_, runs) in results {
                    let cell = runs
                        .get(s)
                        .and_then(|m| m.get(metric))
                        .map(|v| format!("{v:.1}"))
                        .unwrap_or_else(|| "—".into());
                    print!(" {cell} |");
                }
                println!();
            }
        }
    }

    // Machine-readable dump (for charts / bench-history).
    println!("\n#TSV n\truntime\tstructure\tmetric\tvalue");
    for (size, results) in &all {
        for (label, runs) in results {
            for (name, metrics) in runs {
                for (metric, value) in metrics {
                    println!("{size}\t{label}\t{name}\t{metric}\t{value:.1}");
                }
            }
        }
    }
}
