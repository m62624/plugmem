// N2 read-only parity: a writer checkpoints and releases the file, a read-only
// instance then observes the snapshot — read verbs answer, write verbs throw,
// and the freshness verbs (generation/refresh) work. Also covers close().
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

/** A throwaway directory (kept across a writer close + read-only reopen). */
async function withDir(fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ro-"));
  try {
    await fn(join(dir, "m.plugmem"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("read-only observes a writer's checkpointed snapshot", async () => {
  await withDir(async (path) => {
    // Writer stores + checkpoints (clears the journal) + closes (drops the lock).
    const w = await Plugmem.open(path);
    await w.remember({ text: "the sky is blue", entity: "sky", tags: ["color"] });
    await w.checkpoint(); // async — must finish before close/reopen
    w.close();

    // Read-only open over the published snapshot.
    const ro = await Plugmem.open(path, { readOnly: true });

    // Read verbs answer.
    assert.ok(Array.isArray((await ro.recall({ query: "sky" })).facts));
    assert.match(ro.get(0).text, /sky/);
    assert.deepEqual(ro.tagsOf(0), ["color"]);
    assert.deepEqual((await ro.listTags()).items, [{ name: "color", count: 1 }]);
    assert.equal(ro.stats().facts, 1);
    assert.equal((await ro.export()).length, 1);
    const page = await ro.exportPage();
    assert.deepEqual(page.facts.map(({ text }) => text), ["the sky is blue"]);
    assert.equal(page.nextCursor, undefined);
    await assert.doesNotReject(() => ro.verify());

    // Freshness verbs work; nothing newer to adopt.
    assert.equal(typeof ro.generation(), "number");
    assert.equal(ro.refresh(), false);

    // Write verbs are refused.
    assert.throws(() => ro.remember({ text: "x" }), /read-only/);
    assert.throws(() => ro.revise(0, { text: "x" }), /read-only/);
    assert.throws(() => ro.forget(0), /read-only/);
    assert.throws(() => ro.removeTag("color"), /read-only/);
    assert.throws(() => ro.link({ src: "a", rel: "r", dst: "b" }), /read-only/);
    assert.throws(() => ro.unlink({ src: "a", rel: "r", dst: "b" }), /read-only/);
    assert.throws(() => ro.checkpoint(), /read-only/);
    assert.throws(() => ro.maintain(), /read-only/);

    ro.close();
  });
});

test("generation/refresh are read-only-only; verbs throw after close", async () => {
  await withDir(async (path) => {
    const w = await Plugmem.open(path);
    assert.throws(() => w.generation(), /read-only mode/);
    assert.throws(() => w.refresh(), /read-only mode/);

    w.close();
    assert.throws(() => w.stats(), /closed/);
    assert.throws(() => w.remember({ text: "x" }), /closed/);
    assert.doesNotThrow(() => w.close()); // idempotent
  });
});
