# Local guide: `plugmem-napi`

## Role

`plugmem-napi` is the Node.js native addon boundary. It publishes a `cdylib` `.node` module and an `rlib` so wrapper logic can be tested natively. The implementation mirrors `plugmem-host::Database` and `ReadOnlyDatabase`; storage and persistence behavior must not be reimplemented here.

`build.rs` performs the napi build integration. `src/db.rs` owns the `Plugmem` class, input objects, writer/reader dispatch, and async maintenance tasks. `src/types.rs` contains `#[napi(object)]` result mirrors. `src/lib.rs` exposes version/about/skill metadata and module registration.

## Public boundary

The `Plugmem` constructor accepts a path and optional open options, including read-only mode and config. Methods cover remember, remember-many, revise, recall, forget, link, get, tags, collected and paged export, verify, maintain, checkpoint, generation, refresh, and close.

The wrapper has two internal modes:

- writer: owns `plugmem_host::Database`;
- reader: owns `ReadOnlyDatabase` and must reject write operations.

`remember`, `revise`, `recall`, `rememberMany`, `exportPage`, `maintain`, and `checkpoint` use napi-rs async tasks/libuv. The rule for which verbs are async is *blocking work*, not verb size: with an `[embedder]` configured, `remember`/`revise`/`recall` each cost an HTTP round trip, and Node has one thread for all JavaScript — running that on it stalls the whole process. Verbs that only touch mapped memory stay synchronous. Do not move a verb across that line without the same reasoning. Preserve the rule that the task owns the necessary database handle and that errors cross the boundary as JS exceptions/promises rather than panics. Paged export is pull-based and item-bounded; do not replace it with an unbounded threadsafe-function callback queue or wait for JavaScript while holding the host read lock.

## Type and error rules

Keep TypeScript-visible field names and optionality stable. The typed objects in `types.rs` are deliberately separate from core structs so Rust lifetimes and borrowed inputs do not leak into JavaScript. Convert `HostError` into napi errors with useful context; never expose internal debug formatting as the API contract.

`src/error.rs` owns the thrown-error contract and is the single place to change it. Every failure the wrapper decides carries a stable `PLUGMEM_*` `code`; an engine failure inside a verb carries `GenericFailure` and the host's message on **both** the sync and the async path, because napi-rs' `Task` fixes its error type and a code that appeared only on synchronous verbs would make `code` mean different things on different verbs. Do not code engine failures on one path only.

Every verb that reaches the engine's vector layer accepts an optional caller-supplied `vector`, which replaces the embedder for that call — the host embeds only when the field is absent. The length rule belongs to the engine (`dim`); validate only that the numbers are finite and the array is non-empty. The CLI (`--vector`) and MCP (`vector`) expose the same thing; keep the three in step.

An argument that shapes an answer must be refused, never dropped. `range` is exactly `[from, to]`; `range`, `asOf` and `validFrom` are checked with `checked_ms` rather than cast, because `f64 as u64` saturates in Rust and would silently turn a caller's mistake into a different, plausible query. Validate on the JS thread — before an `AsyncTask` is constructed — so a refusal throws where the caller stands instead of rejecting a promise.

`serde-json` is used to reuse JSON-shaped argument/result paths where appropriate. Validate user input at the boundary, especially ids, dimensions, metadata, tags, links, and read-only restrictions.

This crate is published to npm, not crates.io. Do not add platform-specific assumptions to the Rust wrapper; native package selection is handled by the npm package layout.

## Verification

```bash
cargo test -p plugmem-napi
npm run build --prefix crates/plugmem-napi   # regenerates index.js / index.d.ts
npm test --prefix crates/plugmem-napi
npm run typecheck --prefix crates/plugmem-napi
```

The JavaScript tests cover smoke operations, verbs, metadata, config, read-only behavior, the argument/error contract (`contract.test.mjs`) and read-only query embedding (`readonly-embedding.test.mjs`). A test that needs an embedder must mock it **in a worker thread**: `recall` is a synchronous native call, so while it waits on the embedder it owns the JS thread and a mock server on the same event loop can never answer it — the test would wait on itself. When changing an exported class/method/type, update the npm package declarations and JS tests together. Keep Node runtime requirements aligned with the configured Node-API version.
