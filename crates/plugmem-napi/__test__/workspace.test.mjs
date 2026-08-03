// Workspace parity: many memories in one directory, addressed by name. The
// class is the napi mirror of host::Workspace, and the memory it hands back is
// the same Plugmem class a path-opened memory is — so these assert the routing
// and the registry, not the verbs (verbs.test.mjs already covers those).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem, Workspace } = require("../index.js");

/** A fresh Workspace over a throwaway directory, cleaned up after `fn`. */
async function withWorkspace(fn, options) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-"));
  const ws = new Workspace(dir, options);
  try {
    await fn(ws, dir);
  } finally {
    ws.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

test("a named memory is the same class as a path-opened one", async () => {
  await withWorkspace(async (ws) => {
    const chat = ws.open("chat-42");
    assert.ok(chat instanceof Plugmem);

    // Every verb of a single memory works on a named one, because it is the
    // same class — no second implementation to drift.
    const out = chat.remember({ text: "prefers tokio", entity: "user" });
    assert.equal(out.id, 0);
    assert.match(chat.recall({ query: "tokio" }).rendered, /tokio/);
    assert.equal(chat.stats().facts, 1);
    await chat.checkpoint();
    chat.close();
  });
});

test("memories in one workspace do not see each other", async () => {
  await withWorkspace(async (ws) => {
    ws.open("chat-42").remember({ text: "the sky is blue" });
    ws.open("chat-43").remember({ text: "the sky is red" });

    // The same query, the same fact id in each, and each answers only for
    // itself. Ids are per memory, which is why they stay plain numbers.
    assert.match(ws.open("chat-42").recall({ query: "sky" }).rendered, /blue/);
    assert.doesNotMatch(ws.open("chat-42").recall({ query: "sky" }).rendered, /is red/);
    assert.deepEqual(ws.list(), ["chat-42", "chat-43"]);
  });
});

test("a first write creates the memory; a read of an unknown name does not", async () => {
  await withWorkspace(async (ws, dir) => {
    // create defaults to true — that is what lets a new conversation get a
    // memory with no registration step.
    ws.open("chat-42").remember({ text: "a fact" });
    assert.ok(existsSync(join(dir, "db", "chat-42.plugmem.journal")));

    // Asked not to create, an unused name throws instead of quietly answering
    // nothing: such a name is a typo far more often than a new memory.
    assert.throws(() => ws.open("typo", false), /no database named typo/);
    assert.ok(!existsSync(join(dir, "db", "typo.plugmem.journal")));
  });
});

test("a name is not a path and cannot become one", async () => {
  await withWorkspace(async (ws, dir) => {
    for (const bad of ["../etc/passwd", "/abs", "a/b", "Chat", "chat.42", "", "con"]) {
      assert.throws(() => ws.open(bad), /not a usable database name/, bad);
    }
    // Nothing was created outside the workspace, and nothing inside it either.
    assert.ok(!existsSync(join(dir, "db")));
  });
});

test("describe, find and archive round-trip through the registry", async () => {
  await withWorkspace(async (ws) => {
    ws.describe("chat-42", {
      description: "release planning and engine performance",
      tags: ["kind:chat"],
      owner: "ann",
    });
    ws.describe("recipes", {
      description: "dinner ideas and shopping lists",
      owner: "bob",
    });

    // By description...
    assert.equal(ws.find("release planning")[0].db, "chat-42");
    // ...and by owner, which lives in a graph edge rather than in the text.
    assert.equal(ws.find("bob")[0].db, "recipes");

    const [entry] = ws.entries().filter((e) => e.db === "chat-42");
    assert.equal(entry.description, "release planning and engine performance");
    assert.deepEqual(entry.tags, ["kind:chat"]);
    assert.equal(entry.owner, "ann");
    assert.equal(entry.archived, false);

    // Archiving is a label and is idempotent; it does not close or move
    // anything.
    assert.equal(ws.archive("chat-42"), true);
    assert.equal(ws.archive("chat-42"), false);
    assert.equal(ws.entries().find((e) => e.db === "chat-42").archived, true);
    assert.throws(() => ws.archive("nope"), /no database named nope/);

    // Describing twice revises rather than duplicating.
    ws.describe("recipes", { description: "only dinner now" });
    assert.equal(ws.entries().filter((e) => e.db === "recipes").length, 1);
    assert.equal(ws.entries().find((e) => e.db === "recipes").description, "only dinner now");
  });
});

test("verify agrees, then reindex rebuilds a registry that was deleted", async () => {
  await withWorkspace(async (ws, dir) => {
    ws.describe("chat-42", { description: "release planning", tags: ["kind:chat"] });
    assert.deepEqual(await ws.verify(), []);

    // A memory nobody described is reported, not hidden — it works, it just
    // cannot be found by a description. The handle is closed: one this test
    // still held would be a *live writer*, and the checks below would see a
    // busy memory rather than an undescribed one.
    const scratch = ws.open("scratch");
    scratch.remember({ text: "a fact" });
    scratch.close();
    const issues = await ws.verify();
    assert.equal(issues.length, 1);
    assert.equal(issues[0].db, "scratch");
    assert.equal(issues[0].issue, "undescribed");

    // Lose the registry entirely, the way a botched restore would, and rebuild
    // it from the memories' own descriptions. That is the whole reason the
    // registry is allowed to be a cache.
    ws.close();
    const rebuilt = new Workspace(dir);
    for (const f of readdirSync(dir)) {
      if (f.startsWith("registry.plugmem")) rmSync(join(dir, f));
    }
    assert.deepEqual(rebuilt.entries(), []);

    const report = await rebuilt.reindex();
    assert.deepEqual(report.indexed, ["chat-42"]);
    assert.deepEqual(report.undescribed, ["scratch"]);
    assert.deepEqual(report.busy, []);
    const back = rebuilt.entries().find((e) => e.db === "chat-42");
    assert.equal(back.description, "release planning");
    assert.deepEqual(back.tags, ["kind:chat"]);
    rebuilt.close();
  });
});

test("opening the same name twice is one pooled memory, not two", async () => {
  await withWorkspace(async (ws) => {
    const first = ws.open("chat-42");
    first.remember({ text: "written through the first handle" });
    const second = ws.open("chat-42");

    // Two JS objects, one memory behind them: `open` is a pool lookup, so the
    // second sees the first's write immediately and no second file lock is
    // taken. (A *different process* holding it is what produces `busy` in a
    // reindex report; that path is covered where it belongs, in the host.)
    assert.equal(second.stats().facts, 1);
    assert.equal(ws.openCount(), 1);
    first.close();
    second.close();
  });
});

test("the pool bounds what is open, and closeIdle releases the rest", async () => {
  await withWorkspace(
    async (ws) => {
      for (const db of ["a", "b", "c"]) {
        const handle = ws.open(db);
        handle.remember({ text: `i am ${db}` });
        handle.close();
      }
      // Two at a time, so the third pushed one out.
      assert.equal(ws.openCount(), 2);

      // A zero timeout disables the sweep rather than closing everything.
      assert.equal(ws.closeIdle(), 0);
      assert.equal(ws.openCount(), 2);
    },
    { maxOpen: 2, idleTimeoutMs: 0 },
  );

  await withWorkspace(
    async (ws) => {
      const a = ws.open("a");
      a.remember({ text: "x" });
      a.close();
      assert.equal(ws.openCount(), 1);
      // Everything is idle the instant nothing is using it, so a 1 ms window
      // sweeps it — this is about releasing the file lock, not memory.
      await new Promise((r) => setTimeout(r, 20));
      assert.equal(ws.closeIdle(), 1);
      assert.equal(ws.openCount(), 0);
    },
    { idleTimeoutMs: 1 },
  );
});

test("a pool limit out of range is refused rather than clamped", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-"));
  try {
    // A number from JS reaching a pool limit is untrusted the same way a config
    // value is: it becomes a count of open file descriptors.
    assert.throws(() => new Workspace(dir, { maxOpen: 0 }), /maxOpen must be between/);
    assert.throws(() => new Workspace(dir, { maxOpen: 100000 }), /maxOpen must be between/);
    assert.throws(() => new Workspace(dir, { idleTimeoutMs: -1 }), /idleTimeoutMs must be/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a closed workspace throws, and a handle it gave out survives it", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-"));
  try {
    const ws = new Workspace(dir);
    const chat = ws.open("chat-42");
    chat.remember({ text: "still mine" });

    ws.close();
    assert.throws(() => ws.list(), /workspace is closed/);
    assert.throws(() => ws.open("chat-42"), /workspace is closed/);

    // The pool let go; this handle did not. Documented on the class, asserted
    // here — a caller parking a handle keeps that memory locked.
    assert.equal(chat.stats().facts, 1);
    chat.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a single memory is untouched by any of this", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-single-"));
  try {
    // The default: a path, and no workspace anywhere near it.
    const db = new Plugmem(join(dir, "m.plugmem"));
    db.remember({ text: "a plain single memory" });
    assert.equal(db.stats().facts, 1);
    db.close();

    assert.ok(!existsSync(join(dir, "registry.plugmem")));
    assert.ok(!existsSync(join(dir, "db")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
