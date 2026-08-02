# Documentation and repository conventions

## README link policy

- The repository-root `README.md` is the workspace README. It may use relative
  links to files, crates, assets, and sections inside this repository.
- A crate-local `crates/<crate>/README.md` is package documentation: Cargo can
  publish it to crates.io, and readers may encounter it outside the checkout.
  Keep it self-contained and do not link to the workspace with paths such as
  `../../README.md` or to sibling crate files with `../...` paths.
- In crate-local READMEs, link public Rust APIs and sibling crates through their
  absolute `https://docs.rs/<crate>/latest` URLs. Link repository-only material
  (settings files, benchmarks, source files, and markdown documents from another
  crate) through absolute `https://github.com/m62624/plugmem/...` URLs.
- A crate-local README may refer to an SVG or other asset shipped by that same
  crate with a relative path such as `assets/chart.svg`. Cargo packages the
  README and the crate's `assets/` directory together, so this remains valid in
  the workspace, on GitHub, and on crates.io. Do not use relative paths to
  workspace files or sibling crates.

## Benchmark documentation

- Benchmark SVGs are generated artifacts. Keep their source data and renderer
  in `tools/bench-charts`; do not hand-draw a chart that the tool cannot
  reproduce.
- When a benchmark compares corpus sizes, the chart must be generated from the
  same tool input and the README must state the workload, platform, units, and
  whether embeddings are enabled.
