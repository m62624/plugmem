// N1 verb parity: drive the Plugmem class (the napi mirror of host::Database)
// through the full writer flow and assert it behaves like the host verbs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

/** A fresh Plugmem over a throwaway db file, cleaned up after `fn`. */
function withDb(fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    fn(new Plugmem(join(dir, "m.plugmem")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("remember returns an id, recall finds it, get reads it", () => {
  withDb((db) => {
    const out = db.remember({
      text: "prefers tokio",
      entity: "user",
      tags: ["pref"],
      links: [{ rel: "at", entity: "acme" }],
    });
    assert.equal(out.id, 0);

    const res = db.recall({ query: "tokio" });
    assert.ok(Array.isArray(res.facts));
    assert.match(res.rendered, /tokio/);

    const card = db.get(0);
    assert.match(card.text, /prefers tokio/);
    assert.equal(db.get(999), null);
  });
});

test("revise closes a fact and opens its successor", () => {
  withDb((db) => {
    assert.equal(db.remember({ text: "prefers tokio" }).id, 0);
    const rv = db.revise(0, { text: "prefers async-std" });
    assert.equal(rv.id, 1);
  });
});

test("rememberMany, tagsOf and exportEach cover batch/read/stream verbs", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = new Plugmem(join(dir, "m.plugmem"));
    const promise = db.rememberMany([
      { text: "batch one", tags: ["batch", "one"], metadata: { source: "test" } },
      { text: "batch two", tags: ["batch", "two"] },
    ]);
    assert.ok(promise instanceof Promise);
    const outcomes = await promise;
    assert.deepEqual(outcomes.map(({ id }) => id), [0, 1]);
    assert.deepEqual(db.tagsOf(0), ["batch", "one"]);
    assert.deepEqual(db.tagsOf(999), []);

    const seen = [];
    await db.exportEach((error, fact) => {
      assert.equal(error, null);
      seen.push(fact);
    });
    // The Promise marks the native scan/queue complete; TSFN callbacks drain
    // separately on Node's event loop.
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(seen.map(({ text }) => text), ["batch one", "batch two"]);
    assert.deepEqual(seen[0].metadata, { source: "test" });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("async tasks retain their host handle after close", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = new Plugmem(join(dir, "m.plugmem"));
    const pending = db.rememberMany([{ text: "survives close" }]);
    db.close();
    assert.deepEqual((await pending).map(({ id }) => id), [0]);
    assert.throws(() => db.stats(), /closed/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("forget tombstones a live fact and reports freshness", () => {
  withDb((db) => {
    db.remember({ text: "the sky is blue", entity: "sky" });
    assert.equal(db.forget(0), true);
    assert.equal(db.forget(0), false); // already gone
  });
});

test("link upserts and unlink closes a typed edge", () => {
  withDb((db) => {
    assert.doesNotThrow(() => db.link({ src: "user", rel: "works_at", dst: "acme" }));
    assert.equal(db.stats().edges, 1);
    assert.equal(db.stats().edgeVersions, 1);
    assert.equal(db.unlink({ src: "user", rel: "works_at", dst: "acme" }), true);
    assert.equal(db.unlink({ src: "user", rel: "works_at", dst: "acme" }), false);
    assert.equal(db.stats().edges, 0);
    assert.equal(db.stats().edgeVersions, 1);
  });
});

test("stats / export / maintain / checkpoint / verify", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = new Plugmem(join(dir, "m.plugmem"));
    db.remember({ text: "one", entity: "a" });
    db.remember({ text: "two", entity: "b" });

    assert.equal(db.stats().facts, 2);
    assert.equal(db.export().length, 2);
    // maintain / checkpoint are async (libuv worker): they return Promises.
    const report = await db.maintain();
    assert.equal(typeof report.purged, "number");
    await assert.doesNotReject(db.checkpoint());
    assert.doesNotThrow(() => db.verify());
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("maintain takes an explicit mode and full repacks the edges", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = new Plugmem(join(dir, "m.plugmem"));
    db.remember({ text: "anchor", entity: "hub" });
    for (let round = 0; round < 4; round += 1) {
      for (let target = 0; target < 8; target += 1) {
        db.link({ src: "hub", rel: "assigned_to", dst: `t-${target}` });
        db.unlink({ src: "hub", rel: "assigned_to", dst: `t-${target}` });
      }
    }
    const versions = db.stats().edgeVersions;
    assert.equal(versions, 32);

    // The default is still `auto`, which does not touch the edge arenas.
    const auto = await db.maintain();
    assert.equal(auto.edgesCompacted, false);

    const full = await db.maintain("full");
    assert.equal(full.edgesCompacted, true);
    assert.equal(full.edgeVersionsBefore, versions);
    // History is never dropped by any mode.
    assert.equal(db.stats().edgeVersions, versions);
    assert.doesNotThrow(() => db.verify());

    // An unknown mode is rejected at the boundary, before any work starts.
    assert.throws(() => db.maintain("nonsense"), /MaintainMode/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("typed outputs are fully populated (serde round-trip + camelCase)", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = new Plugmem(join(dir, "m.plugmem"));
    const out = db.remember({ text: "prefers tokio", entity: "user" });
    assert.equal(typeof out.id, "number");
    assert.ok(Array.isArray(out.similar));

    const res = db.recall({ query: "tokio" });
    assert.equal(typeof res.truncated, "boolean");
    assert.equal(typeof res.rendered, "string");
    const f = res.facts[0];
    assert.equal(typeof f.id, "number");
    assert.equal(typeof f.score, "number");
    assert.equal(typeof f.recordedAt, "number"); // camelCase from recorded_at
    assert.equal(typeof f.validFrom, "number");

    const card = db.get(0);
    assert.equal(typeof card.text, "string");
    assert.equal(typeof card.record.id, "number");
    assert.equal(typeof card.record.recordedAt, "number");
    assert.equal(typeof card.record.flags, "number");

    const s = db.stats();
    for (const key of ["facts", "entities", "terms", "edges", "edgeVersions", "vectors", "nextFact", "nextEntity", "nextEdge", "poolBytes"]) {
      assert.equal(typeof s[key], "number", `stats.${key}`);
    }

    const m = await db.maintain();
    assert.equal(typeof m.purged, "number");
    assert.equal(typeof m.bytesBefore, "number");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("maintain/checkpoint are async and don't block the event loop", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = new Plugmem(join(dir, "m.plugmem"));
    db.remember({ text: "one" });

    const p = db.maintain();
    assert.ok(p instanceof Promise);
    // The libuv worker runs the pass; the event loop stays free to tick.
    let ticked = false;
    setImmediate(() => (ticked = true));
    const report = await p;
    assert.equal(typeof report.bytesAfter, "number");
    assert.ok(ticked, "event loop advanced while maintain ran");

    assert.equal(await db.checkpoint(), undefined); // Promise<void>
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a missing required arg throws, not crashes", () => {
  withDb((db) => {
    // `text` is required by the RememberArgs interface; napi rejects the call.
    assert.throws(() => db.remember({}));
  });
});
