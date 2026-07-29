//! `plugmem` — the CLI binary. A one-liner over [`plugmem_cli::run`]; all
//! logic (and its tests) live in the library so this file needs no
//! coverage.

use std::process::ExitCode;

fn main() -> ExitCode {
    plugmem_cli::run()
}
