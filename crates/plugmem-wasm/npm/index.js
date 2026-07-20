"use strict";

// Hand-written Node entry layered over the wasm-pack output (`./plugmem.js`).
// It re-exports the engine surface with camelCase names.
// `scripts/build-npm.mjs` copies this next to `plugmem.js` inside `pkg/`.
//
// Stage-5 stub: the engine class (open/remember/recall/... over storage
// callbacks, specs/06) is not wired yet — today the package ships the version
// surface and the companion skill; the pipeline (build, smoke test, publish)
// is final and the class will slot in here without changing consumers of the
// existing exports.

const wasm = require("./plugmem.js");

module.exports = {
  /** Engine/package version (tracks the workspace release). */
  version: wasm.version,
  /** Short, version-free description pointing at the skill. */
  about: wasm.about,
  /** Companion skill for wasm consumers (CLI/MCP appendix stripped). */
  skill: wasm.skill,
  /** The canonical, unstripped SKILL.md. */
  skillFull: wasm.skill_full,
  /** The `<!-- skill-version -->` marker value from the canonical skill. */
  skillVersion: wasm.skill_version,
};
