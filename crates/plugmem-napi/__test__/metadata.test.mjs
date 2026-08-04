// Metadata surface: a remember carries a key→value object, and get/export
// return it. Values are strings; the engine stores it opaquely.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

async function withDb(fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-meta-"));
  try {
    await fn(await Plugmem.open(join(dir, "m.plugmem")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("metadata round-trips through remember, get and export", async () => {
  await withDb(async (db) => {
    const meta = { uri: "s3://b/x", mime: "application/pdf", page: "3" };
    const out = await db.remember({ text: "a scanned contract", metadata: meta });
    assert.equal(out.id, 0);

    // get returns the same map, with keys in one canonical (ascending) order —
    // the same order core and host use, so nothing sorts differently per layer.
    const card = db.get(0);
    assert.deepEqual(card.metadata, meta);
    assert.deepEqual(Object.keys(card.metadata), ["mime", "page", "uri"]);

    // A fact without metadata gets an empty object, not undefined.
    await db.remember({ text: "no metadata" });
    assert.deepEqual(db.get(1).metadata, {});

    // export carries it too.
    const exported = await db.export();
    const withMeta = exported.find((f) => Object.keys(f.metadata).length > 0);
    assert.deepEqual(withMeta.metadata, meta);
    db.close();
  });
});
