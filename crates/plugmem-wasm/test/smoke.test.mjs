// Smoke test for the assembled npm package (run `npm run build` first, or let
// CI do it): the wasm module loads under Node, the version surface answers,
// and the packaged skill is the wasm-stripped variant while the marker stays
// readable. The engine-class tests join here at stage 5.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const crateDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const pkg = require(join(crateDir, "pkg", "index.js"));

test("version matches the packaged package.json", () => {
  const manifest = JSON.parse(readFileSync(join(crateDir, "pkg", "package.json"), "utf8"));
  assert.equal(pkg.version(), manifest.version);
  assert.equal(manifest.name, process.env.NPM_PACKAGE_NAME ?? "plugmem-wasm");
});

test("skill marker tracks the engine version", () => {
  // The same lockstep rule the release gate enforces: skill and engine
  // must never drift.
  assert.equal(pkg.skillVersion(), pkg.version());
});

test("wasm skill is stripped, canonical skill is not", () => {
  const stripped = pkg.skill();
  assert.ok(!stripped.includes("wasm-strip:begin"));
  assert.ok(!stripped.includes("Step 0c"));
  assert.ok(stripped.includes("# plugmem"));
  const full = pkg.skillFull();
  assert.ok(full.includes("wasm-strip:begin"));
  assert.ok(full.includes(`skill-version: ${pkg.version()}`));
});

test("packaged SKILL.md equals the skill() accessor", () => {
  const onDisk = readFileSync(join(crateDir, "pkg", "SKILL.md"), "utf8");
  assert.equal(onDisk, pkg.skill());
});

test("about points at the skill and the repo", () => {
  const about = pkg.about();
  assert.ok(about.includes("skill()"));
  assert.ok(about.includes("github.com/m62624/plugmem"));
});
