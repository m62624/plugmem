# Local guide: `bench-matrix`

## Role

`plugmem-bench-matrix` is a zero-dependency runner for the arena reproducibility benchmark. It builds `plugmem-arena/examples/bench_repro` in native release mode and for `wasm32-wasip1`, then runs native and any available wasm runtimes.

It is an orchestration/reporting tool, not a database engine benchmark. Its current stand measures arena containers and standard-library baselines. Do not label its output as full `plugmem-core` or `plugmem-host` performance.

## Execution model

The runner locates the workspace two levels above the tool, builds the native and wasm artifacts, and tries:

- native executable;
- `wasmtime run` when installed;
- `wasmer run` when installed.

Missing optional runtimes are reported with an install hint; native remains required. The corpus sizes are passed as command-line arguments to `bench_repro`.

The parser consumes `#M` metric lines into `(structure, metric) -> value` maps. The final output contains human-readable Markdown plus a `#TSV` section with `n`, runtime, structure, metric, and value. Preserve this schema because `bench-charts` consumes it.

## Benchmark discipline

Keep runtime labels, units, structure names, and corpus sizes stable. A native/wasm comparison is meaningful only when the same deterministic input and operation sequence are used. Report skipped runtimes rather than silently treating them as zero or equal to native.

Changes to the arena benchmark itself belong in `plugmem-arena`; changes here should remain runner/parsing/reporting changes. Do not add network downloads to the runner.

## Checks

```bash
cargo check -p plugmem-bench-matrix
cargo run --release -p plugmem-bench-matrix
```

Wasm execution additionally requires the target and an installed runtime. Inspect raw `#TSV` output before changing committed chart data.
