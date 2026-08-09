# Local guide: `plugmem-napi`

## Role

`plugmem-napi` is the Node.js native addon boundary. It publishes a `cdylib` `.node` module and an `rlib` so wrapper logic can be tested natively. The implementation mirrors `plugmem-host::Database` and `ReadOnlyDatabase`; storage and persistence behavior must not be reimplemented here.

`build.rs` performs the napi build integration. `src/db.rs` owns the `Plugmem` class, input objects, writer/reader dispatch, and async maintenance tasks. `src/types.rs` contains `#[napi(object)]` result mirrors. `src/lib.rs` exposes version/about/skill metadata and module registration.

## Public boundary

`Plugmem` has **no JavaScript constructor**: it is opened with the static, asynchronous `Plugmem.open(path?, options?)`. A constructor must evaluate to its object immediately, so it cannot hand the open — the exclusive lock, the journal replay, the snapshot mapping — to a worker thread; a static method returning a promise can. `Workspace` keeps its constructor, which only reads `config.toml` and builds a struct; every database it touches opens lazily through its (async) verbs. Do not reintroduce a synchronous open. Methods cover remember, remember-many, revise, recall, forget, link, get, per-fact tags, bounded tag listing, global tag removal, collected and paged export, verify, maintain, checkpoint, generation, refresh, and close.

The wrapper has two internal modes:

- writer: owns `plugmem_host::Database`;
- reader: owns `ReadOnlyDatabase` and must reject write operations.

`remember`, `revise`, `recall`, `rememberMany`, `listTags`, `removeTag`, `exportPage`, `maintain`, and `checkpoint` use napi-rs async tasks/libuv. The rule for which verbs are async is *blocking work*, not verb size: with an `[embedder]` configured, `remember`/`revise`/`recall` each cost an HTTP round trip; `removeTag` can revise many facts; and Node has one thread for all JavaScript — running that work on it stalls the whole process. Verbs that only touch mapped memory stay synchronous. Do not move a verb across that line without the same reasoning. Preserve the rule that the task owns the necessary database handle and that errors cross the boundary as JS exceptions/promises rather than panics. Paged export and tag listing are pull-based and item-bounded; do not replace them with an unbounded threadsafe-function callback queue or wait for JavaScript while holding the host read lock.

## Type and error rules

Keep TypeScript-visible field names and optionality stable. The typed objects in `types.rs` are deliberately separate from core structs so Rust lifetimes and borrowed inputs do not leak into JavaScript. Convert `HostError` into napi errors with useful context; never expose internal debug formatting as the API contract.

`src/error.rs` owns the thrown-error contract and is the single place to change it. Every failure carries a stable `PLUGMEM_*` `code`, on a synchronous throw and on a promise rejection alike. The async half takes a trick: napi-rs' `Task` fixes its error type to `napi::Error<Status>`, whose status is a closed enum, so a task carries its failure inside its `Output` (`error::Produced`) and `resolve` — which runs on the JS thread and has an `Env` — rebuilds it as a real JS `Error` with `code` via `error::to_js`. napi's rejection path returns that stored object verbatim. **Every fallible `Task::resolve` must go through `error::to_js`**; a new task that skips it silently drops the code for that verb.

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

The JavaScript tests cover smoke operations, verbs, metadata, config, read-only behavior, the argument/error contract (`contract.test.mjs`), read-only query embedding (`readonly-embedding.test.mjs`) and the event-loop contract (`event-loop.test.mjs`). A test that needs an embedder may mock it **on the test's own event loop**: `recall` and `remember` embed inside a libuv task, so a server on this thread answers normally. (Before 0.5.0 they embedded inline and owned the JS thread while waiting, which is why `readonly-embedding.test.mjs` runs its mock in a worker; that is now belt-and-braces, not a requirement.) When changing an exported class/method/type, update the npm package declarations and JS tests together. Keep Node runtime requirements aligned with the configured Node-API version.

Two properties are load-bearing and easy to break silently. **No verb may hold the JS thread for work proportional to the database** — `Task::compute` runs on a worker, but `Task::resolve` runs on the main thread, so building a large result there is a stall (this is why `export` is documented as the unbounded one and `exportPage` exists). **No lock may be held across an embedder round trip**: `Embedder::embed` takes `&self` precisely so callers do not need one, and reintroducing a `Mutex` in front of an embedder would serialize every concurrent recall. `event-loop.test.mjs` asserts both.
