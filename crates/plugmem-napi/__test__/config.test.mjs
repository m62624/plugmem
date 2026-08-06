// Config plumbing: the `config` open option resolves settings exactly like the
// CLI/MCP surface — the `[embedder]` section is what would make a text-only
// recall auto-embed. Here we prove the file is read and its dimension governs,
// without standing up an embedder server (no verb embeds in these tests).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

/** Run `fn(dir)` in a throwaway directory, cleaned up afterwards. */
async function withDir(fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-cfg-"));
  try {
    await fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("a config file opens the writer; lexical recall works with no embedder", async () => {
  await withDir(async (dir) => {
    const cfg = join(dir, "config.toml");
    writeFileSync(cfg, "[engine]\ndim = 0\n");
    const db = await Plugmem.open(join(dir, "m.plugmem"), { config: cfg });
    await db.remember({ text: "prefers native addons" });
    const res = await db.recall({ query: "native" });
    assert.match(res.rendered, /native/);
    db.close();
  });
});

test("a dim option disagreeing with the config embedder throws", async () => {
  await withDir(async (dir) => {
    const cfg = join(dir, "config.toml");
    writeFileSync(
      cfg,
      '[engine]\ndim = 8\n[embedder]\nurl = "http://127.0.0.1:1/v1/embeddings"\nmodel = "dummy"\n',
    );
    // The dim conflict is decided before any work is scheduled: a synchronous
    // throw, not a rejection.
    assert.throws(
      () => Plugmem.open(join(dir, "m.plugmem"), { dim: 16, config: cfg }),
      /disagrees/,
    );
  });
});

test("an explicit but missing config path throws", async () => {
  await withDir(async (dir) => {
    assert.throws(() => Plugmem.open(join(dir, "m.plugmem"), { config: join(dir, "nope.toml") }));
  });
});

test("a misspelled setting is reported rather than ignored", async () => {
  await withDir(async (dir) => {
    const cfg = join(dir, "config.toml");
    writeFileSync(
      cfg,
      [
        "[engine]",
        "dim = 0",
        "max_txt = 4096", // a key nothing claims
        "",
        "[engin]", // a section nothing claims
        "dim = 4",
        "",
        "[maintenance]",
        "batch_size = 256", // the CLI's, not host's — must NOT be warned about
      ].join("\n"),
    );

    const db = await Plugmem.open(join(dir, "m.plugmem"), { config: cfg });
    const warnings = db.configWarnings();

    // A value, not a printed line: an addon has no business writing to the
    // host application's stderr.
    assert.equal(warnings.length, 2, warnings.join(" | "));
    assert.match(warnings[0], /unknown config section \[engin\].*did you mean `engine`/);
    assert.match(warnings[1], /unknown setting \[engine\]\.max_txt.*did you mean `max_text`/);

    // The typo changed nothing, which is exactly why it had to be reported.
    assert.equal(db.stats().facts, 0);
    db.close();
  });
});

test("a clean config warns about nothing", async () => {
  await withDir(async (dir) => {
    const cfg = join(dir, "config.toml");
    writeFileSync(cfg, "[engine]\ndim = 0\n");
    const db = await Plugmem.open(join(dir, "m.plugmem"), { config: cfg });
    assert.deepEqual(db.configWarnings(), []);
    db.close();
  });
});
