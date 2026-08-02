# Local guide: `bench-charts`

## Role

`plugmem-bench-charts` is a local pure-Rust TSV-to-SVG renderer. It uses Plotters' SVG and TTF backends; it does not run a browser or WebDriver. It renders arena, core recall, host database, and host edge-lifecycle charts into fixed repository asset directories.

## Input contract

Input is either a file argument or stdin. Valid rows have five tab-separated fields:

```text
n<TAB>runtime<TAB>structure<TAB>metric<TAB>value
```

Rows are accumulated and averaged by `(n, runtime, structure, metric)`. `Chart` definitions select the exact cells used by each SVG. Structure names and metric names must match the producer exactly; a spelling mismatch silently leaves that chart absent.

## Baseline/noise gate

`config.toml` defines the relative threshold (currently 10%) and the path to `baseline.tsv`. A chart is rewritten only when a required cell is missing or a value moves beyond the threshold. `--force` bypasses this decision for intentional style/title changes.

Charts absent from the input are left untouched. The baseline is rewritten in stable sorted order, carrying old values for charts not present in the current input. Review both SVG and TSV diffs before committing measurements.

The tool writes fixed paths under `crates/plugmem-arena/assets`, `crates/plugmem-core/assets`, and `crates/plugmem-host/assets`. It is not a benchmark runner and must not invent or extrapolate missing measurements.

## Checks and usage

```bash
cargo check -p plugmem-bench-charts
cargo run -p plugmem-bench-charts -- bench.tsv
cat bench.tsv | cargo run -p plugmem-bench-charts
cargo run -p plugmem-bench-charts -- tools/bench-charts/baseline.tsv --force
cargo run --release -p plugmem-host --example bench_edges -- 100000 | tee edge-benchmark-100k.tsv
cargo run --release -p plugmem-host --example bench_edges -- 1000000 | tee edge-benchmark-1m.tsv
cat edge-benchmark-100k.tsv edge-benchmark-1m.tsv > edge-benchmark-scale.tsv
cargo run -p plugmem-bench-charts -- edge-benchmark-scale.tsv --force
```

When changing chart scales, colors, titles, or structure lists, use `--force`, inspect generated SVGs, and keep the baseline/data units unchanged. `plotters` may require the system fontconfig development library on Linux.
