// N0 smoke test: prove the built `.node` loads through the generated loader
// and the skill/version surface answers. `napi build` emits index.js (CJS) +
// the platform `.node`; we require it from an ESM test via createRequire.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const addon = require("../index.js");

// The workspace version, read straight from Cargo.toml, is the ground truth the
// addon's version() and the SKILL.md marker must both match.
const crateDir = dirname(fileURLToPath(import.meta.url)).replace(/[/\\]__test__$/, "");
const cargoToml = readFileSync(join(crateDir, "Cargo.toml"), "utf8");
// version.workspace = true, so the number lives in the root manifest.
const rootCargo = readFileSync(join(crateDir, "..", "..", "Cargo.toml"), "utf8");
const WORKSPACE_VERSION = rootCargo.match(/version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"/)[1];

test("version() equals the workspace version", () => {
  assert.equal(addon.version(), WORKSPACE_VERSION);
});

test("skillVersion() marker tracks the version", () => {
  assert.equal(addon.skillVersion(), WORKSPACE_VERSION);
});

test("skill() ships the body, stripped of the CLI appendix", () => {
  const s = addon.skill();
  assert.match(s, /# plugmem/);
  assert.doesNotMatch(s, /wasm-strip:begin/);
});

test("skillFull() keeps the fenced appendix", () => {
  assert.match(addon.skillFull(), /wasm-strip:begin/);
});

test("about() points at the skill", () => {
  assert.match(addon.about(), /skill\(\)/);
});
