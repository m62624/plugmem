//! Native Node.js addon for plugmem, published to npm via napi-rs.
//!
//! **N0: skill + version surface.** The companion-skill accessors and the
//! version function are ported 1:1 from the retired `plugmem-wasm` crate
//! (`#[wasm_bindgen]` → `#[napi]`); they carry `SKILL.md` and re-home the
//! skill-version gate that lived in that crate's unit tests. The engine
//! surface — a `Plugmem` class mirroring `plugmem-host`'s `Database` verb
//! for verb — arrives in the next milestones and extends this file without
//! touching the release/skill pipeline.

use napi_derive::napi;

/// The companion skill, embedded so a consumer can persist it next to the
/// engine without a second download. Single source of truth: the repo-root
/// `skill/SKILL.md`.
const SKILL_MD: &str = include_str!("../../../skill/SKILL.md");

/// Version-free pointer to the skill (the version lives in [`version`]).
const ABOUT: &str = "plugmem is an embedded long-term memory engine for LLM agents: \
remember / recall / revise / forget over one local snapshot-plus-journal file, with \
hybrid retrieval (BM25, optional embedding vectors, an entity graph, time). You'll get \
markedly better results with the matching `plugmem` skill loaded — see `skill()` and \
load the one matching `version()`: https://github.com/m62624/plugmem";

/// The engine/package version (the workspace version; the npm package tracks
/// it release-for-release).
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A short, version-free description pointing the caller at the skill.
#[napi]
pub fn about() -> String {
    ABOUT.to_string()
}

/// The companion skill for napi consumers: the canonical `SKILL.md` with the
/// CLI/MCP "Run it" appendix removed (a napi host has one transport and always
/// ships skill and engine from the same release, so that ceremony never applies).
#[napi]
pub fn skill() -> String {
    strip_excluded(SKILL_MD)
}

/// The canonical, unstripped `SKILL.md` (what CLI/MCP consumers read).
#[napi]
pub fn skill_full() -> String {
    SKILL_MD.to_string()
}

/// The `<!-- skill-version: X.Y.Z -->` marker value from the canonical skill
/// (read from the raw text, so the marker living inside the stripped block is
/// still visible here).
#[napi]
pub fn skill_version() -> String {
    skill_version_of(SKILL_MD).unwrap_or_default()
}

/// Remove the block fenced by `<!-- wasm-strip:begin -->` /
/// `<!-- wasm-strip:end -->` (markers included). Returns the input unchanged if
/// either marker is absent or they are out of order, so a skill without the
/// fence still round-trips. (The fence keeps its `wasm-strip` name: it is a
/// SKILL.md marker shared with the retired wasm build; renaming it is a
/// skill-doc change, tracked separately.)
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

#[cfg(test)]
mod tests {
    use super::*;

    // The skill gate, executed on every plain `cargo test`: the marker in
    // skill/SKILL.md must track the workspace version, so skill and engine can
    // never drift silently between releases (the release workflow re-checks the
    // same equality against the version it is cutting). This test re-homes the
    // gate that lived in the retired plugmem-wasm crate.
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
