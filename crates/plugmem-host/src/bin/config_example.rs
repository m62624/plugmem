//! Writes `config.example.toml` at the repository root from the settings
//! catalogue.
//!
//! A binary rather than a build script for the same reason `plugmem-py`'s
//! `stub_gen` is one: the artifact is committed, so a human runs this and
//! commits the result, and CI fails on the difference. A build script would
//! rewrite the file during every unrelated build and turn an honest gate into
//! a dirty working tree.

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    // `CARGO_MANIFEST_DIR` is this crate; the file belongs at the workspace
    // root, where somebody looking for "how do I configure this" will find it
    // without knowing which crate owns the parser.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let path = root.join("config.example.toml");
    std::fs::write(
        &path,
        plugmem_host::settings_help().render_config_example(false),
    )?;
    println!("wrote {}", path.display());
    Ok(())
}
