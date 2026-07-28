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

test("forget tombstones a live fact and reports freshness", () => {
  withDb((db) => {
    db.remember({ text: "the sky is blue", entity: "sky" });
    assert.equal(db.forget(0), true);
    assert.equal(db.forget(0), false); // already gone
  });
});

test("link upserts a typed edge", () => {
  withDb((db) => {
    assert.doesNotThrow(() => db.link({ src: "user", rel: "works_at", dst: "acme" }));
  });
});

test("stats / export / maintain / checkpoint / verify", () => {
  withDb((db) => {
    db.remember({ text: "one", entity: "a" });
    db.remember({ text: "two", entity: "b" });

    assert.equal(db.stats().facts, 2);
    assert.equal(db.export().length, 2);
    assert.ok(typeof db.maintain().purged === "number");
    assert.doesNotThrow(() => db.checkpoint());
    assert.doesNotThrow(() => db.verify());
  });
});

test("a missing required arg throws, not crashes", () => {
  withDb((db) => {
    // `text` is required by the RememberArgs interface; napi rejects the call.
    assert.throws(() => db.remember({}));
  });
});
