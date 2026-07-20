//! WebAssembly bridge for plugmem, published to npm via wasm-pack.
//!
//! **Stage-5 stub with a finished distribution pipeline.** The engine
//! contract (the `Plugmem` class over storage/embedder callbacks,
//! specs/06) is not implemented yet; what IS final is everything the
//! release machinery needs: the companion-skill accessors, the version
//! surface, and the npm packaging (`scripts/build-npm.mjs`) that CI
//! builds, smoke-tests and publishes. Adding the engine class later
//! extends this file without touching the pipeline.

use wasm_bindgen::prelude::*;

/// The companion skill, embedded so a consumer can persist it next to
/// the engine without a second download. Single source of truth: the
/// repo-root `skill/SKILL.md`.
const SKILL_MD: &str = include_str!("../../../skill/SKILL.md");

/// Version-free pointer to the skill (the version lives in [`version`]).
const ABOUT: &str = "plugmem is an embedded long-term memory engine for LLM agents: \
remember / recall / revise / forget over one local snapshot-plus-journal file, with \
hybrid retrieval (BM25, optional embedding vectors, an entity graph, time). You'll get \
markedly better results with the matching `plugmem` skill loaded — see `skill()` and \
load the one matching `version()`: https://github.com/m62624/plugmem";

/// The engine/package version (the workspace version; the npm package
/// tracks it release-for-release).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A short, version-free description pointing the caller at the skill.
#[wasm_bindgen]
pub fn about() -> String {
    ABOUT.to_string()
}

/// The companion skill for wasm consumers: the canonical `SKILL.md` with
/// the CLI/MCP "Run it" appendix removed (a wasm host has one transport
/// and always ships skill and engine from the same release, so that
/// ceremony never applies).
#[wasm_bindgen]
pub fn skill() -> String {
    strip_wasm_excluded(SKILL_MD)
}

/// The canonical, unstripped `SKILL.md` (what CLI/MCP consumers read).
#[wasm_bindgen]
pub fn skill_full() -> String {
    SKILL_MD.to_string()
}

/// The `<!-- skill-version: X.Y.Z -->` marker value from the canonical
/// skill (read from the raw text, so the marker living inside the
/// stripped block is still visible here).
#[wasm_bindgen]
pub fn skill_version() -> String {
    skill_version_of(SKILL_MD).unwrap_or_default()
}

/// Remove the block fenced by `<!-- wasm-strip:begin -->` /
/// `<!-- wasm-strip:end -->` (markers included). Returns the input
/// unchanged if either marker is absent or they are out of order, so a
/// skill without the fence still round-trips.
fn strip_wasm_excluded(skill: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    // The skill gate, executed on every plain `cargo test`: the marker in
    // skill/SKILL.md must track the workspace version, so skill and engine
    // can never drift silently between releases (the release workflow
    // re-checks the same equality against the version it is cutting).
    #[test]
    fn skill_marker_matches_the_workspace_version() {
        assert_eq!(skill_version(), env!("CARGO_PKG_VERSION"));
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
        assert_eq!(strip_wasm_excluded("no fence here"), "no fence here");
        let reversed = "<!-- wasm-strip:end --> x <!-- wasm-strip:begin -->";
        assert_eq!(strip_wasm_excluded(reversed), reversed);
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
