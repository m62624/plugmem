//! napi-rs build glue: emits the linker args a Node addon needs (undefined
//! `napi_*` symbols are resolved by the Node process that loads the `.node`),
//! and enables the generated `.d.ts` type surface.

fn main() {
    napi_build::setup();
}
