# Local guide: `wasm-probe`

## Role

`plugmem-wasm-probe` is a cross-target equivalence probe. It runs one deterministic scenario through the public `plugmem-core` API and returns `xxh3_64(snapshot_bytes)`. Native and wasm builds should produce the same decimal hash.

The scenario covers facts with vectors, entities, tags, links, forget, revise, maintenance with physical purge, recall, and snapshot serialization. The fixed workload is intentionally small and deterministic; this is a portability/integrity check, not a throughput benchmark.

## Targets and entry points

- Native uses `src/main.rs` and prints the library's `scenario_hash()`.
- Wasm exports `run` with `#[unsafe(no_mangle)]` for runtime invocation.
- The intended target matrix includes native, `wasm32-unknown-unknown`, `wasm64-unknown-unknown` with nightly/build-std, and wasm SIMD variants where configured.
- Wasm execution is performed by external runtimes such as wasmtime/wasmer in the surrounding verification workflow.

The scenario uses a fixed xorshift64* PRNG, dimension 64, a fixed fact count, fixed timestamps, and fixed string pools. Keep all of these deterministic and avoid platform-dependent time, locale, hash iteration, or floating-point randomness.

## Wasm allocator and unsafe code

Pure wasm has no OS allocator in this probe, so the wasm configuration installs a fixed 64 MiB bump allocator. It never frees because the process runs the scenario once. The `unsafe impl GlobalAlloc` proof depends on aligned monotonic allocation inside the static heap and exclusive atomic `NEXT` updates; `dealloc` is intentionally a no-op.

The wasm entry point's `unsafe(no_mangle)` is required for the runtime ABI. Keep the exported symbol and decimal result convention stable. A panic loops forever so the runtime watchdog exposes a failed probe instead of printing a misleading hash.

## Interpreting failures

A hash mismatch means target-dependent bytes or behavior changed somewhere in the engine/snapshot path. First compare compiler/target features and the exact scenario, then inspect core serialization and float/quantization code. Do not “fix” a mismatch by normalizing the hash or ignoring one target.

Useful commands depend on installed targets/runtimes, but the native check is:

```bash
cargo check -p plugmem-wasm-probe
cargo run --release -p plugmem-wasm-probe
```

For wasm checks, build the exact target and invoke the exported `run` entry point with the configured runtime; record the target, runtime, and printed hash.
