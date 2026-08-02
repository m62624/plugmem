# Local guide: `plugmem-napi`

## Role

`plugmem-napi` is the Node.js native addon boundary. It publishes a `cdylib` `.node` module and an `rlib` so wrapper logic can be tested natively. The implementation mirrors `plugmem-host::Database` and `ReadOnlyDatabase`; storage and persistence behavior must not be reimplemented here.

`build.rs` performs the napi build integration. `src/db.rs` owns the `Plugmem` class, input objects, writer/reader dispatch, and async maintenance tasks. `src/types.rs` contains `#[napi(object)]` result mirrors. `src/lib.rs` exposes version/about/skill metadata and module registration.

## Public boundary

The `Plugmem` constructor accepts a path and optional open options, including read-only mode and config. Methods cover remember, revise, recall, forget, link, get, stats, export, verify, maintain, checkpoint, generation, refresh, and close.

The wrapper has two internal modes:

- writer: owns `plugmem_host::Database`;
- reader: owns `ReadOnlyDatabase` and must reject write operations.

Async `maintain` and `checkpoint` use napi-rs async tasks/libuv. Preserve the rule that the task owns the necessary database handle and that errors cross the boundary as JS exceptions/promises rather than panics.

## Type and error rules

Keep TypeScript-visible field names and optionality stable. The typed objects in `types.rs` are deliberately separate from core structs so Rust lifetimes and borrowed inputs do not leak into JavaScript. Convert `HostError` into napi errors with useful context; never expose internal debug formatting as the API contract.

`serde-json` is used to reuse JSON-shaped argument/result paths where appropriate. Validate user input at the boundary, especially ids, dimensions, metadata, tags, links, and read-only restrictions.

This crate is published to npm, not crates.io. Do not add platform-specific assumptions to the Rust wrapper; native package selection is handled by the npm package layout.

## Verification

```bash
cargo test -p plugmem-napi
npm test --prefix crates/plugmem-napi
npm run build --prefix crates/plugmem-napi
```

The JavaScript tests cover smoke operations, verbs, metadata, config, and read-only behavior. When changing an exported class/method/type, update the npm package declarations and JS tests together. Keep Node runtime requirements aligned with the configured Node-API version.
