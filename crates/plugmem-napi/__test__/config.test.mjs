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
    const db = new Plugmem(join(dir, "m.plugmem"), { config: cfg });
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
      '[engine]\ndim = 8\n[embedder]\nkind = "openai"\nurl = "http://127.0.0.1:1/v1/embeddings"\nmodel = "dummy"\n',
    );
    assert.throws(
      () => new Plugmem(join(dir, "m.plugmem"), { dim: 16, config: cfg }),
      /disagrees/,
    );
  });
});

test("an explicit but missing config path throws", async () => {
  await withDir(async (dir) => {
    assert.throws(
      () => new Plugmem(join(dir, "m.plugmem"), { config: join(dir, "nope.toml") }),
    );
  });
});
