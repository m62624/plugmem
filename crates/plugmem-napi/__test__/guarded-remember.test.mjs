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
    assert.equal(stored.checked, true, "an entity was named, so it compared");

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

test("a guarded write with no entity is stored unguarded, and says so", async () => {
  // The detector is scoped to the fact's entity, so an input naming none has
  // an empty candidate set and cannot block anything - now or after any number
  // of later writes. `checked` is what tells the caller that, because
  // otherwise "nothing similar was found" and "nothing could be compared" are
  // the same value. A consumer of this binding stored the same sentence six
  // times before anyone measured why.
  const dir = tempdir("no-entity");
  const db = await Plugmem.open(join(dir, "m.plugmem"));
  try {
    const text = "the cache is disabled because it raced with the warmup task";
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const outcome = await db.rememberGuarded({ text });
      assert.equal(outcome.status, "stored");
      assert.equal(outcome.checked, false, "no entity, so nothing compared");
    }
    assert.equal(db.stats().facts, 3, "identical facts, exactly as remember");

    // The same text with an entity: compared, and refused.
    const first = await db.rememberGuarded({ text, entity: "project" });
    assert.equal(first.status, "stored");
    assert.equal(first.checked, true);
    const second = await db.rememberGuarded({ text, entity: "project" });
    assert.equal(second.status, "blocked");
    assert.equal(second.checked, true, "a block always compared something");
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
