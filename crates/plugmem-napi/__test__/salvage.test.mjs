// The salvage and dump-completeness verbs: `exportEdges`, `scrub`, `recover`.
//
// These three exist because a memory is a file that can rot, and because a dump
// of facts alone is not a backup. What they have in common is that all of them
// read the *file* rather than the handle's view of it, which is why `scrub` and
// `recover` are available whatever handle you hold.
//
// Blocking is not asserted here — `event-loop.test.mjs` owns that. This file
// asserts they do the right thing.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem, recover } = require("../index.js");

/** A throwaway directory, removed after `fn`. */
async function withDir(tag, fn) {
  const dir = mkdtempSync(join(tmpdir(), `plugmem-salvage-${tag}-`));
  try {
    await fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * The generation file behind a database, read the way any reader would: the
 * manifest is a fixed 24-byte record whose second u64 is the current
 * generation, and the image lives beside it as `<base>.snap.<N>`.
 */
function snapshotOf(base) {
  const manifest = readFileSync(base);
  assert.equal(manifest.length, 24, "the manifest is a fixed 24-byte record");
  return `${base}.snap.${Number(manifest.readBigUInt64LE(8))}`;
}

/** Flips one byte inside `needle`, which must sit in a section body. */
function flipByteAt(path, needle, offset = 0) {
  const bytes = readFileSync(path);
  const at = bytes.indexOf(Buffer.from(needle));
  assert.ok(at >= 0, `${needle} is present in the image`);
  bytes[at + offset] ^= 0xff;
  writeFileSync(path, bytes);
}

// ---------------------------------------------------------------------------
// exportEdges
// ---------------------------------------------------------------------------

test("exportEdges streams every edge, in batches, with provenance", async () => {
  await withDir("edges", async (dir) => {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    const fact = await db.remember({ text: "kim works at acme", entity: "kim" });
    await db.link({ src: "kim", rel: "works_at", dst: "acme", provenance: fact.id });
    await db.link({ src: "kim", rel: "knows", dst: "ann" });
    // Past two batch boundaries, so batching is exercised rather than assumed.
    for (let i = 0; i < 2600; i++) {
      await db.link({ src: `e${i}`, rel: "r", dst: `d${i}` });
    }

    const sizes = [];
    const all = [];
    const total = await db.exportEdges((edges) => {
      sizes.push(edges.length);
      all.push(...edges);
    });

    assert.equal(total, 2602, "every edge is streamed");
    assert.equal(all.length, total, "the callback saw exactly what was counted");
    assert.equal(total, db.stats().edges, "and it agrees with the engine's own count");
    assert.ok(sizes.length > 1, `more than one batch, got ${sizes.length}`);
    assert.ok(
      sizes.slice(0, -1).every((n) => n === sizes[0]),
      `every batch but the last is full: ${sizes.join(",")}`,
    );
    assert.ok(sizes.at(-1) <= sizes[0], "the last batch is a remainder");

    // Provenance is present when it was recorded and absent when it was not —
    // absent rather than a sentinel, so it cannot be mistaken for fact 0.
    const named = all.filter((e) => e.src === "kim");
    assert.deepEqual(
      named.map((e) => [e.rel, e.dst, e.provenance]),
      [
        ["works_at", "acme", 0],
        ["knows", "ann", undefined],
      ],
    );
    db.close();
  });
});

test("exportEdges works on a read-only handle and reports an empty graph", async () => {
  await withDir("edges-ro", async (dir) => {
    const base = join(dir, "m.plugmem");
    const db = await Plugmem.open(base);
    await db.remember({ text: "no edges here" });
    await db.checkpoint();

    let batches = 0;
    assert.equal(
      await db.exportEdges(() => batches++),
      0,
      "a memory with no edges streams none",
    );
    assert.equal(batches, 0, "and the callback is never invoked");

    const ro = await Plugmem.open(base, { readOnly: true });
    assert.equal(await ro.exportEdges(() => {}), 0);
    ro.close();
    db.close();
  });
});

// ---------------------------------------------------------------------------
// scrub
// ---------------------------------------------------------------------------

/** Runs a scrub to completion, returning the last progress and the step count. */
async function drain(scrub) {
  let steps = 0;
  let last = null;
  for (;;) {
    const step = await scrub.next();
    if (step === null) return { steps, last };
    last = step;
    steps++;
  }
}

test("scrub verifies a clean image, and the budget sets only the grain", async () => {
  await withDir("scrub-clean", async (dir) => {
    const base = join(dir, "m.plugmem");
    const db = await Plugmem.open(base);
    for (let i = 0; i < 40; i++) await db.remember({ text: `fact number ${i} about tokio` });
    await db.checkpoint();
    const size = statSync(snapshotOf(base)).size;

    // A writer scrubs its own published generation: no second handle needed.
    const fine = await drain(await db.scrub({ budget: 64 }));
    assert.ok(fine.steps > 1, `a small budget takes many steps, got ${fine.steps}`);
    assert.equal(fine.last.totalBytes, size);
    assert.equal(fine.last.doneBytes, size, "the scan reached EOF");

    // The default budget covers this image in one step — same answer, fewer
    // crossings, which is the whole of what the knob controls.
    const coarse = await drain(await db.scrub());
    assert.ok(coarse.steps < fine.steps);
    assert.equal(coarse.last.doneBytes, size);
    db.close();
  });
});

test("scrub catches a flipped byte and is finished afterwards", async () => {
  await withDir("scrub-corrupt", async (dir) => {
    const base = join(dir, "m.plugmem");
    const db = await Plugmem.open(base);
    for (let i = 0; i < 40; i++) await db.remember({ text: `fact number ${i} about tokio` });
    await db.checkpoint();

    // Inside a section body: the structure still parses, so only the checksum
    // can tell. This is exactly what `verify` would not catch.
    flipByteAt(snapshotOf(base), "fact number 7 about tokio");

    const scrub = await db.scrub();
    await assert.rejects(
      async () => {
        while ((await scrub.next()) !== null) {
          /* to the first mismatch */
        }
      },
      (e) => {
        assert.equal(e.code, "PLUGMEM_ENGINE");
        assert.match(e.message, /checksum mismatch/);
        return true;
      },
    );
    assert.equal(await scrub.next(), null, "a failed scrub is finished, not retried");
    assert.equal(scrub.active(), false);
    db.close();
  });
});

test("a scrub releases its generation on close, and refuses a bad budget", async () => {
  await withDir("scrub-close", async (dir) => {
    const base = join(dir, "m.plugmem");
    const db = await Plugmem.open(base);
    await db.remember({ text: "one" });
    await db.checkpoint();

    const scrub = await db.scrub({ budget: 64 });
    assert.notEqual(await scrub.next(), null);
    assert.equal(scrub.active(), true);
    scrub.close();
    assert.equal(scrub.active(), false);
    assert.equal(await scrub.next(), null, "a closed scrub yields nothing");
    scrub.close(); // idempotent

    // A refused argument throws where the caller stands, not at the promise.
    for (const budget of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.throws(
        () => db.scrub({ budget }),
        (e) => e.code === "PLUGMEM_INVALID_ARG",
        `budget ${budget} must be refused`,
      );
    }
    db.close();
  });
});

test("scrub needs something published", async () => {
  await withDir("scrub-fresh", async (dir) => {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    await db.remember({ text: "never checkpointed" });
    await assert.rejects(db.scrub(), (e) => e.code === "PLUGMEM_NEEDS_CHECKPOINT");
    db.close();
  });
});

// ---------------------------------------------------------------------------
// recover
// ---------------------------------------------------------------------------

test("recover salvages the survivors and leaves the source alone", async () => {
  await withDir("recover", async (dir) => {
    const src = join(dir, "broken.plugmem");
    const dst = join(dir, "fixed.plugmem");
    {
      const db = await Plugmem.open(src);
      for (let i = 0; i < 12; i++) {
        await db.remember({ text: `fact ${i} about работа`, entity: "kim" });
      }
      await db.link({ src: "kim", rel: "knows", dst: "ann" });
      await db.checkpoint();
      db.close();
    }

    // Break one fact's text into invalid UTF-8: content damage, which is what
    // recover is for. The container still parses.
    const snap = snapshotOf(src);
    flipByteAt(snap, "fact 5 about работа", 13);
    const sizeBefore = statSync(snap).size;

    const report = await recover(src, dst);
    assert.deepEqual(report, {
      kept: 11,
      droppedText: 1,
      droppedVector: 0,
      droppedMetadata: 0,
    });
    assert.equal(statSync(snap).size, sizeBefore, "the source is evidence, not a workspace");

    const fixed = await Plugmem.open(dst, { readOnly: true });
    assert.equal(fixed.stats().facts, 11);
    assert.equal(fixed.stats().edges, 1, "edges survive the salvage");
    await fixed.verify();
    fixed.close();
  });
});

test("recover refuses to write over its own source", async () => {
  await withDir("recover-same", async (dir) => {
    const src = join(dir, "m.plugmem");
    {
      const db = await Plugmem.open(src);
      await db.remember({ text: "one" });
      await db.checkpoint();
      db.close();
    }
    await assert.rejects(recover(src, src), (e) => {
      assert.equal(e.code, "PLUGMEM_ENGINE");
      assert.match(e.message, /must differ from the source/);
      return true;
    });
  });
});

test("recover reports a bad config where the caller stands", async () => {
  await withDir("recover-config", async (dir) => {
    const config = join(dir, "config.toml");
    writeFileSync(config, "[engine\ndim = ");
    // Synchronous: the config is read before any work is scheduled, so this is
    // a throw rather than a rejected promise.
    assert.throws(
      () => recover(join(dir, "a.plugmem"), join(dir, "b.plugmem"), { config }),
      (e) => e.code === "PLUGMEM_CONFIG",
    );
  });
});
