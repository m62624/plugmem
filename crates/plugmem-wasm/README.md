# plugmem-wasm

The WebAssembly build of the [plugmem](https://github.com/m62624/plugmem)
embedded memory engine, published to npm as `plugmem-wasm`.

**Status: distribution stub.** This release ships the version surface and
the companion Agent Skill (`skill()`, stripped for wasm consumers;
`skillVersion()` always matches `version()`); the engine class —
`remember / recall / revise / forget / link / maintain` over storage and
embedder callbacks (specs/06) — lands in an upcoming release and will
extend the module without breaking the current exports.

```js
const plugmem = require("plugmem-wasm");
console.log(plugmem.version()); // engine version, e.g. "0.1.0"
console.log(plugmem.about());   // what this is + where the skill lives
// Persist plugmem.skill() next to your agent so it knows how to use
// the memory once the engine class ships.
```

The underlying engine is a `no_std` Rust core (BM25, quantized vectors with
HNSW, an entity graph, bitemporal facts) whose snapshot format is
pointer-width independent — the same database file also opens in native
builds and in wasm64 (memory64) builds. See the
[core README](https://github.com/m62624/plugmem/tree/main/crates/plugmem-core)
for the engine itself and the Rust-native
[host layer](https://github.com/m62624/plugmem/tree/main/crates/plugmem-host)
for the "point at a file and go" experience.

## Building from source

```sh
cd crates/plugmem-wasm
node scripts/build-npm.mjs   # wasm-pack build + package assembly into pkg/
node --test test/smoke.test.mjs
```

## License

MIT.
