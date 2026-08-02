// N0 smoke test: prove the built `.node` loads through the generated loader
// and the skill/version surface answers. `napi build` emits index.js (CJS) +
// the platform `.node`; we require it from an ESM test via createRequire.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const addon = require("../index.js");

test("version() exposes a semantic version", () => {
  assert.match(addon.version(), /^\d+\.\d+\.\d+$/);
});

test("skillVersion() exposes a semantic version marker", () => {
  assert.match(addon.skillVersion(), /^\d+\.\d+\.\d+$/);
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
