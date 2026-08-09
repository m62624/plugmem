import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem, Workspace } = require("../index.js");

function tempdir(tag) {
  return mkdtempSync(join(tmpdir(), `plugmem-guarded-${tag}-`));
}

async function deadline(promise, label) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`${label} timed out — possible deadlock`)),
          30_000,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

test("rememberGuarded stores a clear fact and blocks without allocating an id", async () => {
  const dir = tempdir("direct");
  const db = await Plugmem.open(join(dir, "m.plugmem"));
  try {
    const stored = await db.rememberGuarded({
      text: "likes green tea every morning",
      entity: "user",
    });
    assert.equal(stored.status, "stored");
    assert.equal(stored.outcome.id, 0);
    assert.deepEqual(stored.similar, []);

    const blocked = await db.rememberGuarded({
      text: "likes green tea each morning",
      entity: "user",
    });
    assert.equal(blocked.status, "blocked");
    assert.equal(blocked.outcome, undefined);
    assert.equal(blocked.similar.length, 1);
    assert.equal(blocked.similar[0].id, 0);
    assert.equal(db.stats().facts, 1);

    // Ordinary remember remains the explicit compatible-facts override.
    const kept = await db.remember({
      text: "likes green tea each morning",
      entity: "user",
    });
    assert.equal(kept.id, 1);
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("parallel guarded writes are race-free and never deadlock", async () => {
  const dir = tempdir("atomic");
  const db = await Plugmem.open(join(dir, "m.plugmem"));
  try {
    const outcomes = await deadline(
      Promise.all([
        db.rememberGuarded({ text: "same durable fact", entity: "user" }),
        db.rememberGuarded({ text: "same durable fact", entity: "user" }),
      ]),
      "parallel rememberGuarded",
    );
    assert.deepEqual(
      outcomes.map((outcome) => outcome.status).sort(),
      ["blocked", "stored"],
    );
    assert.equal(db.stats().facts, 1);
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("workspace memory exposes the same guarded contract", async () => {
  const dir = tempdir("workspace");
  const workspace = new Workspace(dir);
  try {
    const memory = workspace.memory("contact");
    assert.equal((await memory.rememberGuarded({ text: "one fact", entity: "user" })).status, "stored");
    assert.equal((await memory.rememberGuarded({ text: "one fact", entity: "user" })).status, "blocked");
  } finally {
    workspace.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
