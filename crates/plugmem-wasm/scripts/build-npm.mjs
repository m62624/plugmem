#!/usr/bin/env node
// Assemble the publishable npm package from the wasm-pack output:
//   1. wasm-pack build (nodejs target) -> pkg/           (skip with --no-build)
//   2. drop in the hand-written Node entry (index.js/.d.ts) and the skill
//   3. rewrite pkg/package.json (name, entry points, file list)
//
// Run from the crate dir:  node scripts/build-npm.mjs
// Override the published name with NPM_PACKAGE_NAME (default "plugmem-wasm").

import { execFileSync } from "node:child_process";
import { copyFileSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const crateDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = join(crateDir, "..", "..");
const pkg = join(crateDir, "pkg");

if (!process.argv.includes("--no-build")) {
  execFileSync(
    "wasm-pack",
    ["build", crateDir, "--target", "nodejs", "--out-dir", "pkg", "--out-name", "plugmem"],
    { stdio: "inherit" },
  );
}

/** Drop the CLI/MCP "Run it" appendix fenced by `<!-- wasm-strip:begin/end -->`
 * (markers included) from the canonical skill, matching the Rust `skill()`
 * accessor: a wasm host has one transport and always ships skill + engine from
 * the same release, so that ceremony never applies. Returns the text unchanged
 * if the fence is absent. */
function stripWasmExcluded(text) {
  const begin = "<!-- wasm-strip:begin -->";
  const end = "<!-- wasm-strip:end -->";
  const start = text.indexOf(begin);
  const stop = text.indexOf(end);
  if (start === -1 || stop === -1 || stop < start) return text;
  const head = text.slice(0, start).replace(/\s+$/, "");
  const tail = text.slice(stop + end.length).replace(/^\s+/, "");
  return tail ? `${head}\n\n${tail}` : `${head}\n`;
}

// Node entry + companion skill, copied next to the wasm-pack artifacts. The
// skill is stripped to the wasm-relevant subset (see stripWasmExcluded).
copyFileSync(join(crateDir, "npm", "index.js"), join(pkg, "index.js"));
copyFileSync(join(crateDir, "npm", "index.d.ts"), join(pkg, "index.d.ts"));
writeFileSync(
  join(pkg, "SKILL.md"),
  stripWasmExcluded(readFileSync(join(repoRoot, "skill", "SKILL.md"), "utf8")),
);
copyFileSync(join(crateDir, "README.md"), join(pkg, "README.md"));
copyFileSync(join(crateDir, "LICENSE"), join(pkg, "LICENSE"));

// Point the package at the curated Node entry and list everything we ship.
const pkgJsonPath = join(pkg, "package.json");
const pkgJson = JSON.parse(readFileSync(pkgJsonPath, "utf8"));
// Native, unscoped name matching the crate (override with NPM_PACKAGE_NAME).
pkgJson.name = process.env.NPM_PACKAGE_NAME ?? "plugmem-wasm";
pkgJson.main = "index.js";
pkgJson.types = "index.d.ts";
// Normalize the repository URL to the `git+...git` form npm expects, so
// `npm publish` doesn't warn and auto-correct it.
pkgJson.repository = {
  type: "git",
  url: "git+https://github.com/m62624/plugmem.git",
};
pkgJson.files = [
  "plugmem_bg.wasm",
  "plugmem.js",
  "plugmem.d.ts",
  "index.js",
  "index.d.ts",
  "SKILL.md",
  "README.md",
  "LICENSE",
];
writeFileSync(pkgJsonPath, `${JSON.stringify(pkgJson, null, 2)}\n`);

console.log(`Assembled ${pkgJson.name}@${pkgJson.version} in ${pkg}`);
