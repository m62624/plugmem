# 09 — Portability: WebAssembly 2.0 / 3.0 and capacity classes

**A "wasm version" for plugmem is a build's address-space class, not a code branch, a
feature flag, or a format field.** One core source builds for every target; one database
file is read by any build whose address space fits its limits.

| Class | Target | Addressing | Memory ceiling | Status |
|---|---|---|---|---|
| **v2-class** (Wasm 2.0) | `wasm32v1-none` / `wasm32-unknown-unknown` | 32-bit | 4 GiB (our budget ≤ 2 GiB) | default, runs everywhere |
| **v3-class** (Wasm 3.0, memory64) | `wasm64-unknown-unknown` | 64-bit | 2⁶⁴ (browsers today ~16 GiB) | opt-in; Tier 3, nightly + build-std |
| native | x86-64 / aarch64 | 64-bit | host RAM | as-is |

No "version choice" in the verb API: capabilities are chosen by **config** (`max_bytes`
and the other limits), not a switch. A database with limits ≤ 2 GiB opens everywhere; a
database whose `max_bytes` exceeds a 32-bit host's ceiling opens only on 64-bit builds.
**Migrating 2.0 → 3.0 is opening the same file** — the format is byte-for-byte common
(proof below). There is no reverse migration of a large database onto a small host; the
core answers such an attempt with a typed `ConfigMismatch("database requires a 64-bit
address space")` from `Config::decode` (the file is not corrupt; the host lacks the
address space — the test exercises both branches, the 32-bit one on `wasm32-wasip1`).

## Wasm 3.0 by the specification: what we take

From W3C WebAssembly Release 3.0, the engine needs exactly **memory64**; simd128 (from
2.0) is useful and compatible; everything else targets other language classes or tools.

| 3.0 feature | Verdict | Why |
|---|---|---|
| **64-bit address space (memory64)** | **take** — this is the v3-class | lifts the 4 GiB ceiling; the capacity contract scales from ~1M facts to tens of millions with no code change. The core built for `wasm64-unknown-unknown` with **no edits** — the payoff of fixed-width codecs everywhere |
| **Relaxed SIMD** | **not in deterministic paths** | the instructions are declared non-deterministic; HNSW build depends on dot-product results, and any non-determinism breaks "journal replay = byte-identical snapshot". (`relaxed_dot` also wants a 7-bit operand; our symmetric i8 quantization is [-127, 127].) |
| simd128 (from 2.0) | **compatible, verified** | integer SIMD is deterministic; building with `+simd128` gives an identical snapshot hash. The hamming/dot auto-vectorization already targets it |
| Multiple memories / GC / typed refs / EH / tail calls | not needed | addressed to managed-heap languages or call-indirect tables; our model is flat arenas, monomorphic code, `panic=abort`, iterative algorithms |
| **Deterministic profile** | **already followed** | integer distances, `total_cmp` orders, libm soft-math, no relaxed SIMD in state paths — our replay contract, now named by the standard |

"Wasm 3.0 support" for plugmem is capacity, not an instruction set.

## Why the format is one and the version is not written to the header

Every codec is fixed-width and platform-independent: the Config block writes all
`usize` fields as `u64 LE` (`ENCODED_LEN` = 188), decoding with a checked
`usize::try_from`; arena/structure headers use only `u32`/`u64`; snapshot section
offsets/lengths are `u64` with all bounds computed in `u64` arithmetic **before** any
`as usize`; the journal uses `u32` lengths and LEB128/fixed values. The only thing that
distinguishes a "64-bit database" is whether its stored limits fit the host's `usize` —
and that is fully derivable from the config block (`max_bytes`). A separate "wasm
version" header field would be derived state that could drift from config, so by the
canonical-format principle (`03-snapshot.md`: dump→load→dump byte-identical, no
redundant fields) it is not introduced; `max_bytes` is the class marker.

## Live probes (method and results)

A no_std scenario over the public API (200 facts with dim-64 vectors, entity/tags/links,
forget every 7th, revise every 11th, `maintain` with physical purge, `recall`,
`snapshot_bytes`) → `xxh3(snapshot bytes)`. One source, several builds:

| Build | Runtime | Result |
|---|---|---|
| native x86-64 | — | reference hash |
| `wasm32-unknown-unknown` | wasmtime | **same hash** |
| `wasm32-unknown-unknown` +simd128,+relaxed-simd | wasmtime | **same hash** (the compiler emits no relaxed instructions without explicit intrinsics) |
| `wasm64-unknown-unknown` (nightly, build-std) | wasmtime | **same hash**; memory64 on by default |
| `wasm64-unknown-unknown` | wasmer | ❌ does not run: wasmer cannot yet execute a Rust-built 64-bit table |
| `wasm32-unknown-unknown` | wasmer | **same hash** |

**The snapshot is byte-identical across 32-bit, 64-bit wasm and native builds** — the
file is portable across generations with no migrators. Runtime state: wasmtime supports
the v3-class out of the box, wasmer the v2-class only; browsers ship Wasm 3.0. The
product default stays the v2-class because it runs everywhere. The whole core contract
suite also runs on the real 32-bit `wasm32-wasip1` under wasmtime and wasmer (proptest
sections are native-only; the deterministic contract tests are shared).

## Reaching the wrappers

- **CLI / MCP** — native 64-bit binaries: both capacity classes are understood
  automatically (native = 64-bit addressing). The class is chosen at database *creation*
  (`max_bytes`); a binary just opens any file.
- **npm (napi)** — the Node addon is a **native** build, so it has host addressing and
  no wasm ceiling; the wasm capacity classes above are about the core's portability
  (and any future wasm embedding), not this package.
- One Rust source, no per-version templating: everything width-dependent is already
  expressed through `usize` + a checked decode.

## Reproduction gates

```sh
# v2-class (mandatory):
cargo build -p plugmem-arena -p plugmem-core --target wasm32v1-none --no-default-features
# the 32-bit suite run (wasmtime):
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=." \
  cargo test -p plugmem-core --target wasm32-wasip1
# v3-class (nightly gate, Tier 3):
cargo +nightly build -p plugmem-core --target wasm64-unknown-unknown \
  -Zbuild-std=core,alloc --no-default-features
```

`wasm64-unknown-unknown` is Tier 3; when it becomes Tier 2 with a precompiled std the
nightly gate becomes a stable gate.
