// The input and error contract, from JS.
//
// Two properties, both about failures being *visible*:
//   1. An argument that shapes an answer is never silently dropped. `range`
//      drives the engine's temporal source and `asOf` is its visibility filter,
//      so a value the binding cannot use has to be an error — not an answer
//      computed without it, which looks exactly like a correct one.
//   2. Every failure plugmem itself decides carries a stable `code`, so a
//      caller branches on `err.code` instead of matching on prose.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

/** A throwaway memory, closed and removed afterwards. */
function withDb(fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-contract-"));
  const db = new Plugmem(join(dir, "m.plugmem"));
  try {
    return fn(db, join(dir, "m.plugmem"));
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

/** The error a call threw, or a failure if it did not throw. */
function thrown(fn) {
  try {
    fn();
  } catch (err) {
    return err;
  }
  assert.fail("the call was supposed to throw");
}

test("a malformed recall window is refused, not silently widened", () => {
  withDb((db) => {
    db.remember({ text: "the invoice was paid", entity: "acme" });

    // A well-formed window still reaches the temporal source. Asked with no
    // text, that source is the only one running, so what comes back is exactly
    // what the window covers.
    const now = Date.now();
    assert.equal(db.recall({ range: [0, now + 1000] }).facts.length, 1);
    assert.equal(db.recall({ range: [0, 1] }).facts.length, 0);

    // Every one of these used to mean "no window at all": the call answered
    // over the whole memory while looking like it had filtered.
    for (const range of [[0], [], [0, 1, 2], [1000, 0], [-1, 1000]]) {
      const err = thrown(() => db.recall({ query: "invoice", range }));
      assert.equal(err.code, "PLUGMEM_INVALID_ARG", `range ${JSON.stringify(range)}`);
      assert.match(err.message, /range/);
    }
  });
});

test("an instant that is not an instant is refused", () => {
  withDb((db) => {
    for (const asOf of [-1, NaN, Infinity, -Infinity]) {
      const err = thrown(() => db.recall({ query: "x", asOf }));
      assert.equal(err.code, "PLUGMEM_INVALID_ARG", `asOf ${asOf}`);
      assert.match(err.message, /asOf/);
    }

    for (const validFrom of [-1, NaN]) {
      const err = thrown(() => db.remember({ text: "x", validFrom }));
      assert.equal(err.code, "PLUGMEM_INVALID_ARG", `validFrom ${validFrom}`);
    }

    // A good value is still accepted, and reaches the fact.
    const at = 1_700_000_000_000;
    const { id } = db.remember({ text: "backdated", validFrom: at });
    assert.equal(db.get(id).record.validFrom, at);
  });
});

test("rememberMany refuses a bad instant synchronously, not as a rejected promise", () => {
  withDb((db) => {
    // The throw has to happen where the caller stands. If this returned a
    // promise, the `catch` below would never run and the test would fail.
    const err = thrown(() =>
      db.rememberMany([{ text: "fine" }, { text: "bad", validFrom: -1 }]),
    );
    assert.equal(err.code, "PLUGMEM_INVALID_ARG");
    assert.match(err.message, /\[1\]\.validFrom/, "the offending input is named");
  });
});

test("every refusal carries a code a program can branch on", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-codes-"));
  const path = join(dir, "m.plugmem");
  const db = new Plugmem(path);
  try {
    // Another writer on the same file: the one code a service actually retries.
    assert.equal(thrown(() => new Plugmem(path)).code, "PLUGMEM_LOCKED");

    // A config path that is not there.
    assert.equal(
      thrown(() => new Plugmem(join(dir, "other.plugmem"), { config: join(dir, "nope.toml") })).code,
      "PLUGMEM_CONFIG",
    );

    // Read-only-only verbs on a writer.
    assert.equal(thrown(() => db.generation()).code, "PLUGMEM_WRITER_ONLY");
    assert.equal(thrown(() => db.refresh()).code, "PLUGMEM_WRITER_ONLY");

    // Write verbs on a read-only handle.
    db.remember({ text: "published" });
    await db.checkpoint();
    db.close();
    const ro = new Plugmem(path, { readOnly: true });
    assert.equal(thrown(() => ro.remember({ text: "x" })).code, "PLUGMEM_READ_ONLY");
    assert.equal(thrown(() => ro.forget(0)).code, "PLUGMEM_READ_ONLY");
    ro.close();

    // Anything after close().
    assert.equal(thrown(() => ro.stats()).code, "PLUGMEM_CLOSED");
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the handle says which file it opened", () => {
  withDb((db, path) => {
    assert.equal(db.path(), path);
  });
});
