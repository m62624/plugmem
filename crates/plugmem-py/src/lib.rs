//! Python extension module for plugmem, published to PyPI via maturin.
//!
//! The surface mirrors [`plugmem-napi`](https://github.com/m62624/plugmem/tree/main/crates/plugmem-napi)
//! verb for verb, and that is deliberate: napi narrowed `plugmem-host` in a few
//! places (one `export_page` instead of also `export_each`, one `maintain(mode)`
//! instead of also `maintain_with_options`, `scrub(budget)` instead of also
//! `scrub_with_budget`), and a third surface that re-derived those choices from
//! host would diverge from the second. The parity test in `tests/` reads napi's
//! generated `index.d.ts` and fails on any name that exists on one side only.
//!
//! **No async layer, and that is not a gap.** Node has one thread that runs
//! JavaScript, so napi had to push every blocking verb onto a libuv worker. The
//! Python equivalent of "one thread" is the GIL, and the equivalent move is to
//! *release* it: every verb here is a plain synchronous method whose body runs
//! inside [`pyo3::Python::detach`]. That gives real parallelism even on a build
//! with a GIL, because no bytecode executes while we are inside Rust, and it
//! makes `asyncio.to_thread(db.recall, ...)` work correctly rather than block
//! the event loop.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// The `Plugmem` class — the Python mirror of `plugmem-host`'s `Database`.
mod db;
/// The exception hierarchy: which failures carry which `code`.
mod error;
/// The `Scrub` class — a resumable byte-level check of one snapshot generation.
mod scrub;
/// Typed result mirrors, so verbs return attribute-addressable objects.
mod types;
/// The `Workspace` class — many memories in one directory, addressed by name.
mod workspace;

/// The companion skill, embedded so a consumer can persist it next to the
/// engine without a second download. Single source of truth: the repo-root
/// `skills/plugmem/SKILL.md`.
const SKILL_MD: &str = include_str!("../../../skills/plugmem/SKILL.md");

/// Version-free pointer to the skill (the version lives in [`version`]).
const ABOUT: &str = "plugmem is an embedded bitemporal memory and retrieval engine for \
local-first applications and agents: remember / recall / revise / forget over one local \
snapshot-plus-journal file, with hybrid retrieval (BM25, optional embedding vectors, an \
entity graph, time) and bounded rendered context. You'll get \
markedly better results with the matching `plugmem` skill loaded — see `skill()` and \
load the one matching `version()`: https://github.com/m62624/plugmem";

/// The engine/package version (the workspace version; the PyPI package tracks
/// it release-for-release).
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A short, version-free description pointing the caller at the skill.
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
fn about() -> String {
    ABOUT.to_string()
}

/// The companion skill for Python consumers: the canonical `SKILL.md` with the
/// CLI/MCP "Run it" appendix removed (an embedding application has one
/// transport and always ships skill and engine from the same release, so that
/// ceremony never applies).
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
fn skill() -> String {
    strip_excluded(SKILL_MD)
}

/// The canonical, unstripped `SKILL.md` (what CLI/MCP consumers read).
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
fn skill_full() -> String {
    SKILL_MD.to_string()
}

/// The `<!-- skill-version: X.Y.Z -->` marker value from the canonical skill
/// (read from the raw text, so the marker living inside the stripped block is
/// still visible here).
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
fn skill_version() -> String {
    skill_version_of(SKILL_MD).unwrap_or_default()
}

/// Return the complete `config.toml` settings catalogue without opening a
/// database.
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
fn settings_help() -> types::SettingsHelpResult {
    types::SettingsHelpResult::collect()
}

/// Remove the block fenced by `<!-- wasm-strip:begin -->` /
/// `<!-- wasm-strip:end -->` (markers included). Returns the input unchanged if
/// either marker is absent or they are out of order, so a skill without the
/// fence still round-trips. (The fence keeps its `wasm-strip` name: it is a
/// `SKILL.md` marker shared with the retired wasm build, and renaming it is a
/// skill-doc change tracked separately.)
fn strip_excluded(skill: &str) -> String {
    const BEGIN: &str = "<!-- wasm-strip:begin -->";
    const END: &str = "<!-- wasm-strip:end -->";
    let (Some(start), Some(end)) = (skill.find(BEGIN), skill.find(END)) else {
        return skill.to_string();
    };
    if end < start {
        return skill.to_string();
    }
    let head = skill[..start].trim_end();
    let tail = skill[end + END.len()..].trim_start();
    if tail.is_empty() {
        format!("{head}\n")
    } else {
        format!("{head}\n\n{tail}")
    }
}

/// Extract the `<!-- skill-version: X -->` marker value from skill text.
fn skill_version_of(skill: &str) -> Option<String> {
    let marker = "<!-- skill-version:";
    let start = skill.find(marker)? + marker.len();
    let rest = &skill[start..];
    let end = rest.find("-->")?;
    let value = rest[..end].trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// The compiled half of the `plugmem` package.
///
/// Named with a leading underscore because it is not the import surface: the
/// pure-Python `plugmem/__init__.py` re-exports from here, which is what lets
/// the package also ship `py.typed`, the generated stubs, and the two
/// generators that are better written in Python than in Rust.
#[pymodule]
fn _plugmem(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(about, module)?)?;
    module.add_function(wrap_pyfunction!(skill, module)?)?;
    module.add_function(wrap_pyfunction!(skill_full, module)?)?;
    module.add_function(wrap_pyfunction!(skill_version, module)?)?;
    module.add_function(wrap_pyfunction!(settings_help, module)?)?;
    module.add_function(wrap_pyfunction!(db::recover, module)?)?;

    module.add_class::<db::Plugmem>()?;
    module.add_class::<scrub::Scrub>()?;
    module.add_class::<workspace::Workspace>()?;
    module.add_class::<workspace::WorkspaceMemory>()?;

    types::register(module)?;
    error::register(module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_marker_is_a_semver() {
        assert_eq!(skill_version().split('.').count(), 3);
    }

    #[test]
    fn stripped_skill_drops_the_cli_appendix_but_keeps_the_body() {
        let stripped = skill();
        assert!(!stripped.contains("wasm-strip:begin"));
        assert!(!stripped.contains("Step 0c"));
        assert!(stripped.contains("# plugmem"));
        // The full skill keeps both.
        assert!(skill_full().contains("wasm-strip:begin"));
        assert!(skill_full().contains("Step 0c"));
    }

    #[test]
    fn strip_is_identity_without_a_fence_and_on_a_reversed_fence() {
        assert_eq!(strip_excluded("no fence here"), "no fence here");
        let reversed = "<!-- wasm-strip:end --> x <!-- wasm-strip:begin -->";
        assert_eq!(strip_excluded(reversed), reversed);
    }

    #[test]
    fn version_marker_parsing_handles_malformed_markers() {
        assert_eq!(
            skill_version_of("<!-- skill-version: 1.2.3 -->"),
            Some("1.2.3".into())
        );
        assert_eq!(skill_version_of("<!-- skill-version: -->"), None);
        assert_eq!(skill_version_of("no marker"), None);
        assert_eq!(
            skill_version_of("<!-- skill-version: 9 (unterminated)"),
            None
        );
    }

    #[test]
    fn about_points_at_the_skill_and_version_is_semver() {
        assert!(about().contains("skill()"));
        assert_eq!(version().split('.').count(), 3);
    }
}

// Collects the type information the `gen_stub_*` macros registered, so the
// `stub_gen` binary can write `python/plugmem/__init__.pyi`.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
