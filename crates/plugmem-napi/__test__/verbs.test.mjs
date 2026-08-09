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
async function withDb(fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    await fn(await Plugmem.open(join(dir, "m.plugmem")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("remember returns an id, recall finds it, get reads it", async () => {
  await withDb(async (db) => {
    const out = await db.remember({
      text: "prefers tokio",
      entity: "user",
      tags: ["pref"],
      links: [{ rel: "at", entity: "acme" }],
    });
    assert.equal(out.id, 0);

    const res = await db.recall({ query: "tokio" });
    assert.ok(Array.isArray(res.facts));
    assert.match(res.rendered, /tokio/);

    const card = db.get(0);
    assert.match(card.text, /prefers tokio/);
    assert.equal(db.get(999), null);
  });
});

test("revise closes a fact and opens its successor", async () => {
  await withDb(async (db) => {
    assert.equal((await db.remember({ text: "prefers tokio" })).id, 0);
    const rv = await db.revise(0, { text: "prefers async-std" });
    assert.equal(rv.id, 1);
  });
});

test("rememberMany and tagsOf cover the host batch/read verbs", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    const promise = db.rememberMany([
      { text: "batch one", tags: ["batch", "one"], metadata: { source: "test" } },
      { text: "batch two", tags: ["batch", "two"] },
    ]);
    assert.ok(promise instanceof Promise);
    const outcomes = await promise;
    assert.deepEqual(outcomes.map(({ id }) => id), [0, 1]);
    assert.deepEqual(db.tagsOf(0), ["batch", "one"]);
    assert.deepEqual(db.tagsOf(999), []);

    const page = await db.exportPage();
    assert.equal(page.nextCursor, undefined);
    assert.deepEqual(page.facts.map(({ text }) => text), ["batch one", "batch two"]);
    assert.deepEqual(page.facts[0].metadata, { source: "test" });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("exportPage is bounded, pull-driven and releases the lock between pages", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    await db.rememberMany(
      Array.from({ length: 260 }, (_, i) => ({ text: `page fact ${i}` })),
    );

    const first = await db.exportPage();
    assert.equal(first.facts.length, 128);
    assert.equal(typeof first.nextCursor, "number");

    // The native read guard is gone when the Promise resolves. A write between
    // pulls therefore progresses instead of forming a callback/read-lock cycle.
    assert.equal((await db.remember({ text: "written between pages" })).id, 260);

    const second = await db.exportPage(first.nextCursor);
    const third = await db.exportPage(second.nextCursor);
    assert.equal(second.facts.length, 128);
    assert.equal(third.facts.length, 5);
    assert.equal(third.nextCursor, undefined);
    assert.equal(
      new Set([...first.facts, ...second.facts, ...third.facts].map(({ text }) => text)).size,
      261,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("async tasks retain their host handle after close", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    const pending = db.rememberMany([{ text: "survives close" }]);
    db.close();
    assert.deepEqual((await pending).map(({ id }) => id), [0]);
    assert.throws(() => db.stats(), /closed/);

    const reopened = await Plugmem.open(join(dir, "m.plugmem"));
    const page = reopened.exportPage();
    reopened.close();
    assert.deepEqual((await page).facts.map(({ text }) => text), ["survives close"]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("forget tombstones a live fact and reports freshness", async () => {
  await withDb(async (db) => {
    await db.remember({ text: "the sky is blue", entity: "sky" });
    assert.equal(await db.forget(0), true);
    assert.equal(await db.forget(0), false); // already gone
  });
});

test("listTags is bounded and removeTag preserves facts", async () => {
  await withDb(async (db) => {
    await db.rememberMany([
      { text: "one", tags: ["drop", "keep"] },
      { text: "two", tags: ["drop"] },
      { text: "three", tags: ["project:plugmem"] },
    ]);
    const first = await db.listTags({ limit: 1 });
    assert.deepEqual(first.items, [{ name: "drop", count: 2 }]);
    const second = await db.listTags({ cursor: first.nextCursor, limit: 2 });
    assert.deepEqual(second.items.map(({ name }) => name), ["keep", "project:plugmem"]);
    assert.deepEqual(
      (await db.listTags({ prefix: "project" })).items.map(({ name }) => name),
      ["project:plugmem"],
    );

    assert.deepEqual(await db.removeTag("drop"), { affected: 2 });
    assert.deepEqual((await db.listTags()).items.map(({ name }) => name), ["keep", "project:plugmem"]);
    assert.equal((await db.export()).length, 3, "removing a tag keeps every current fact");
    await assert.rejects(
      () => db.listTags({ cursor: first.nextCursor }),
      (err) => err.code === "PLUGMEM_STALE_CURSOR",
    );
  });
});

test("link upserts and unlink closes a typed edge", async () => {
  await withDb(async (db) => {
    await assert.doesNotReject(() => db.link({ src: "user", rel: "works_at", dst: "acme" }));
    assert.equal(db.stats().edges, 1);
    assert.equal(db.stats().edgeVersions, 1);
    assert.equal(await db.unlink({ src: "user", rel: "works_at", dst: "acme" }), true);
    assert.equal(await db.unlink({ src: "user", rel: "works_at", dst: "acme" }), false);
    assert.equal(db.stats().edges, 0);
    assert.equal(db.stats().edgeVersions, 1);
  });
});

test("stats / export / maintain / checkpoint / verify", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    await db.remember({ text: "one", entity: "a" });
    await db.remember({ text: "two", entity: "b" });

    assert.equal(db.stats().facts, 2);
    assert.equal((await db.export()).length, 2);
    // maintain / checkpoint are async (libuv worker): they return Promises.
    const report = await db.maintain();
    assert.equal(typeof report.purged, "number");
    await assert.doesNotReject(db.checkpoint());
    await assert.doesNotReject(() => db.verify());
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("maintain takes an explicit mode and full repacks the edges", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    await db.remember({ text: "anchor", entity: "hub" });
    for (let round = 0; round < 4; round += 1) {
      for (let target = 0; target < 8; target += 1) {
        await db.link({ src: "hub", rel: "assigned_to", dst: `t-${target}` });
        await db.unlink({ src: "hub", rel: "assigned_to", dst: `t-${target}` });
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
    await assert.doesNotReject(() => db.verify());

    // An unknown mode is rejected at the boundary, before any work starts.
    assert.throws(() => db.maintain("nonsense"), /MaintainMode/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("typed outputs are fully populated (serde round-trip + camelCase)", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    const out = await db.remember({ text: "prefers tokio", entity: "user" });
    assert.equal(typeof out.id, "number");
    assert.ok(Array.isArray(out.similar));

    const res = await db.recall({ query: "tokio" });
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
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    await db.remember({ text: "one" });

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

test("a missing required arg throws, not crashes", async () => {
  await withDb(async (db) => {
    // `text` is required by the RememberArgs interface; napi rejects the call.
    assert.throws(() => db.remember({}));
  });
});

// The knobs that reached the engine but no wrapper: `token_budget` and `ef` on
// recall, `provenance` on link. Every binding hardcoded them to `None`, so the
// engine supported them and nobody could use them. These lock the parity.
test("the token budget bounds the rendered block", async () => {
  await withDb(async (db) => {
    for (let i = 0; i < 40; i++) {
      await db.remember({
        text: `fact number ${i} about the deployment pipeline and its many stages`,
      });
    }

    const generous = await db.recall({ query: "deployment pipeline", k: 40 });
    const tight = await db.recall({
      query: "deployment pipeline",
      k: 40,
      tokenBudget: 40,
    });

    assert.ok(
      tight.rendered.length < generous.rendered.length,
      `a tight budget must shrink the block: ${tight.rendered.length} vs ${generous.rendered.length}`,
    );
  });
});

test("ef is accepted and leaves a lexical answer alone", async () => {
  await withDb(async (db) => {
    await db.remember({ text: "the release ships on friday" });

    const plain = await db.recall({ query: "release" });
    const withEf = await db.recall({ query: "release", ef: 64 });
    assert.deepEqual(withEf.facts.map((f) => f.id), plain.facts.map((f) => f.id));
  });
});

test("link records provenance and recall returns it", async () => {
  await withDb(async (db) => {
    const source = await db.remember({ text: "ann hired bob in march" });

    await db.link({ src: "ann", rel: "hires", dst: "bob", provenance: source.id });
    const res = await db.recall({ entities: ["ann"] });

    assert.ok(res.edges.length > 0, "the graph source walked the edge");
    assert.ok(
      res.edges.some((e) => e.provenance === source.id),
      `the edge names the fact it follows from: ${JSON.stringify(res.edges)}`,
    );

    // unlink takes the same argument shape and ignores provenance: closing an
    // edge has no source fact to name.
    assert.equal(await db.unlink({ src: "ann", rel: "hires", dst: "bob" }), true);
  });
});

test("graphDepth is a per-call knob over the configured default", async () => {
  // A chain a -> b -> c -> d with one fact each, so the number of facts a
  // recall returns is the number of hops it took.
  await withDb(async (db) => {
    for (const [entity, next] of [
      ["a", "b"],
      ["b", "c"],
      ["c", "d"],
      ["d", "e"],
    ]) {
      await db.remember({
        text: `fact on ${entity}`,
        entity,
        links: [{ rel: "leads_to", entity: next }],
      });
    }

    const reached = async (graphDepth) =>
      (
        await db.recall({ entities: ["a"], k: 64, tokenBudget: 4096, graphDepth })
      ).facts.length;

    assert.equal(await reached(undefined), 3, "the configured default is 2 hops");
    assert.equal(await reached(0), 1, "no expansion: the anchor's own fact");
    assert.equal(await reached(1), 2);
    assert.equal(await reached(3), 4);
    // The ceiling is the engine's, not a suggestion the caller may exceed.
    assert.equal(await reached(99), await reached(4));
  });
});
