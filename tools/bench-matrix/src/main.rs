//! Cross-runtime benchmark matrix — **zero dependencies**, fully scripted.
//!
//! One command reproduces every number in the README chart:
//!
//! ```text
//! cargo run --release -p plugmem-bench-matrix
//! ```
//!
//! The runner builds `plugmem-arena`'s `bench_repro` example for native and
//! `wasm32-wasip1`, executes it on every runtime it can find (native always;
//! `wasmtime` and `wasmer` when installed — install them yourself, the
//! runner never downloads anything), parses the TSV each run prints, and
//! merges everything into one per-metric comparison table (markdown +
//! machine-readable TSV on stdout).
//!
//! Verification is the point: the workload is deterministic (seeded keys),
//! the harness is `std::process::Command` — nothing hidden, no network, no
//! extra crates. If a runtime is missing you get an install hint and the
//! matrix simply has fewer columns.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One parsed measurement row from `bench_repro` output.
#[derive(Clone, Debug)]
struct Row {
    insert_ns: f64,
    get_ns: f64,
    /// `None` for structures without an ordered scan.
    scan_ns: Option<f64>,
    mem_bytes: f64,
}

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

/// Parses the TSV block of a `bench_repro` run into name -> row.
fn parse(output: &str) -> BTreeMap<String, Row> {
    let mut rows = BTreeMap::new();
    for line in output.lines() {
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() != 5 || cells[1].parse::<f64>().is_err() {
            continue; // header / banner lines
        }
        rows.insert(
            cells[0].to_string(),
            Row {
                insert_ns: cells[1].parse().unwrap(),
                get_ns: cells[2].parse().unwrap(),
                scan_ns: cells[3].parse().ok(),
                mem_bytes: cells[4].parse().unwrap(),
            },
        );
    }
    rows
}

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

    // (runtime label, command, args) — native always runs; wasm runtimes
    // are optional and must be installed by the verifier.
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
            vec!["run", wasm_path],
            "https://wasmer.io/ (curl https://get.wasmer.io -sSfL | sh)",
        ),
    ];

    let mut results: Vec<(&str, BTreeMap<String, Row>)> = Vec::new();
    for (label, cmd, args, hint) in &runtimes {
        print!("running on {label}... ");
        match run(cmd, args) {
            Ok(out) => {
                println!("ok");
                results.push((label, parse(&out)));
            }
            Err(e) if e.contains("failed to start") => {
                println!("not found — skipped. Install: {hint}");
            }
            Err(e) => panic!("{label} run failed: {e}"),
        }
    }

    // Structures in bench_repro's own print order (BTreeMap sorts them, so
    // re-list explicitly for readable output).
    let structures = [
        "plugmem Arena (Ordered)",
        "plugmem Arena (Uniform)",
        "std BTreeMap",
        "std HashMap",
        "sorted Vec (bulk)",
    ];
    /// How one metric is pulled out of a parsed row.
    type Extract = fn(&Row) -> Option<f64>;
    let metrics: [(&str, Extract); 4] = [
        ("insert ns/elem", |r| Some(r.insert_ns)),
        ("get ns/op", |r| Some(r.get_ns)),
        ("ordered scan ns/elem", |r| r.scan_ns),
        ("memory B/elem", |r| Some(r.mem_bytes)),
    ];

    for (metric, extract) in metrics {
        println!("\n### {metric}\n");
        print!("| structure |");
        for (label, _) in &results {
            print!(" {label} |");
        }
        println!();
        print!("|---|");
        for _ in &results {
            print!("---|");
        }
        println!();
        for s in structures {
            print!("| {s} |");
            for (_, rows) in &results {
                let cell = rows
                    .get(s)
                    .and_then(extract)
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "—".into());
                print!(" {cell} |");
            }
            println!();
        }
    }

    // Machine-readable dump (for charts / bench-history).
    println!("\n#TSV runtime\tstructure\tinsert_ns\tget_ns\tscan_ns\tmem_b");
    for (label, rows) in &results {
        for (name, r) in rows {
            println!(
                "{label}\t{name}\t{:.1}\t{:.1}\t{}\t{:.1}",
                r.insert_ns,
                r.get_ns,
                r.scan_ns.map(|v| format!("{v:.1}")).unwrap_or("-".into()),
                r.mem_bytes,
            );
        }
    }
}
