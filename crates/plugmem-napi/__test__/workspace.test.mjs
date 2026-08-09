// The FFI ownership contract for named memories. A WorkspaceMemory is a name,
// not a native database handle: only an in-flight verb owns a scoped lease.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem, Workspace, WorkspaceMemory } = require("../index.js");

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

test("memory() returns a logical reference and a first write creates its file", async () => {
  await withWorkspace(async (ws, dir) => {
    const chat = ws.memory("chat-42");
    assert.ok(chat instanceof WorkspaceMemory);
    assert.equal(chat.name(), "chat-42");
    assert.equal(ws.openCount(), 0, "constructing a reference opens nothing");

    const out = await chat.remember({ text: "prefers tokio", entity: "user" });
    assert.equal(out.id, 0);
    assert.match((await chat.recall({ query: "tokio" })).rendered, /tokio/);
    assert.equal((await chat.stats()).facts, 1);
    assert.ok(existsSync(join(dir, "db", "chat-42.plugmem.journal")));
  });
});

test("every WorkspaceMemory verb executes through a scoped lease", async () => {
  await withWorkspace(async (ws) => {
    const memory = ws.memory("all-verbs");
    const [first, second] = await memory.rememberMany([
      { text: "first version", tags: ["old"] },
      { text: "temporary fact" },
    ]);
    const revised = await memory.revise(first.id, { text: "current version", tags: ["new"] });
    assert.equal((await memory.get(revised.id)).text, "current version");
    assert.deepEqual(await memory.tagsOf(revised.id), ["new"]);
    assert.deepEqual((await memory.listTags()).items, [
      { name: "new", count: 1 },
    ]);
    assert.deepEqual(await memory.removeTag("new"), { affected: 1 });
    assert.deepEqual((await memory.listTags()).items, []);
    assert.match((await memory.recall({ query: "current" })).rendered, /current version/);

    await memory.link({ src: "ann", rel: "owns", dst: "service", provenance: revised.id });
    const batches = [];
    assert.equal(await memory.exportEdges((edges) => batches.push(...edges)), 1);
    assert.equal(batches[0].provenance, revised.id);
    assert.equal(await memory.unlink({ src: "ann", rel: "owns", dst: "service" }), true);
    assert.equal(await memory.forget(second.id), true);

    assert.equal((await memory.export()).length, 1);
    assert.equal((await memory.exportPage()).facts.length, 1);
    assert.equal((await memory.stats()).facts, 4, "stats counts closed records before maintenance");
    await memory.verify();
    await memory.checkpoint();
    const scrub = await memory.scrub({ budget: 64 * 1024 });
    while ((await scrub.next()) !== null) {
      // sweep the published generation
    }
    const report = await memory.maintain("auto");
    assert.equal(typeof report.factsAfter, "number");
  });
});

test("a read of an unknown name fails and creates nothing", async () => {
  await withWorkspace(async (ws, dir) => {
    const typo = ws.memory("typo");
    await assert.rejects(() => typo.stats(), /no database named typo/);
    assert.ok(!existsSync(join(dir, "db", "typo.plugmem.journal")));
  });
});

test("a name is not a path and cannot become one", async () => {
  await withWorkspace(async (ws, dir) => {
    for (const bad of ["../etc/passwd", "/abs", "a/b", "Chat", "chat.42", "", "con"]) {
      assert.throws(() => ws.memory(bad), /not a usable database name/, bad);
    }
    assert.ok(!existsSync(join(dir, "db")));
  });
});

test("memories in one workspace remain independent", async () => {
  await withWorkspace(async (ws) => {
    const blue = ws.memory("chat-42");
    const red = ws.memory("chat-43");
    await blue.remember({ text: "the sky is blue" });
    await red.remember({ text: "the sky is red" });

    assert.match((await blue.recall({ query: "sky" })).rendered, /blue/);
    assert.doesNotMatch((await blue.recall({ query: "sky" })).rendered, /is red/);
    assert.deepEqual(await ws.list(), ["chat-42", "chat-43"]);
  });
});

test("eviction and release never invalidate an old logical reference", async () => {
  await withWorkspace(
    async (ws) => {
      const a = ws.memory("a");
      const b = ws.memory("b");
      await a.remember({ text: "memory a survives eviction" });
      await b.remember({ text: "memory b takes the only slot" });
      assert.equal(ws.openCount(), 1);

      assert.match((await a.recall({ query: "survives" })).rendered, /survives eviction/);
      assert.equal(ws.openCount(), 1);
      assert.equal(ws.release("a"), true);
      assert.equal(ws.openCount(), 0);
      assert.equal(ws.release("a"), false);
      assert.match((await a.recall({ query: "survives" })).rendered, /survives eviction/);
    },
    { maxOpen: 1, idleTimeoutMs: 0 },
  );
});

test("the pooled handle keeps the OS writer lock until release", async () => {
  await withWorkspace(async (ws, dir) => {
    const chat = ws.memory("chat");
    await chat.remember({ text: "the file is protected" });
    const path = join(dir, "db", "chat.plugmem");

    await assert.rejects(
      () => Plugmem.open(path),
      (err) => err.code === "PLUGMEM_LOCKED",
    );
    assert.equal(ws.release("chat"), true);

    const direct = await Plugmem.open(path);
    assert.equal(direct.stats().facts, 1);
    direct.close();
    assert.equal((await chat.stats()).facts, 1, "the logical reference reopens transparently");
  });
});

test("closeIdle uses elapsed time and ignores reachability of logical references", async () => {
  await withWorkspace(
    async (ws) => {
      const memory = ws.memory("a");
      await memory.remember({ text: "x" });
      assert.equal(ws.openCount(), 1);
      await new Promise((resolve) => setTimeout(resolve, 20));
      assert.equal(ws.closeIdle(), 1);
      assert.equal(ws.openCount(), 0);
      assert.equal((await memory.stats()).facts, 1);
    },
    { idleTimeoutMs: 1 },
  );
});

test("parallel references to one name serialize writes without leaking a lock", async () => {
  await withWorkspace(async (ws) => {
    const refs = Array.from({ length: 24 }, () => ws.memory("chat"));
    let timer;
    try {
      await Promise.race([
        Promise.all(refs.map((memory, index) => memory.remember({ text: `fact ${index}` }))),
        new Promise((_, reject) => {
          timer = setTimeout(
            () => reject(new Error("parallel workspace writes did not finish")),
            15_000,
          );
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
    assert.equal((await refs[0].stats()).facts, refs.length);
    assert.equal(ws.openCount(), 1);
    assert.equal(ws.release("chat"), true);
  });
});

test("describe, find, archive, verify and reindex use the registry", async () => {
  await withWorkspace(async (ws, dir) => {
    await ws.describe("chat-42", {
      description: "release planning and engine performance",
      tags: ["kind:chat"],
      owner: "ann",
    });
    await ws.describe("recipes", { description: "dinner ideas", owner: "bob" });
    assert.equal((await ws.find("release planning"))[0].db, "chat-42");
    assert.equal((await ws.find("bob"))[0].db, "recipes");
    assert.equal(await ws.archive("chat-42"), true);
    assert.equal(await ws.archive("chat-42"), false);

    await ws.memory("scratch").remember({ text: "not described" });
    assert.equal((await ws.verify()).find((issue) => issue.db === "scratch").issue, "undescribed");

    ws.close();
    for (const file of readdirSync(dir)) {
      if (file.startsWith("registry.plugmem")) rmSync(join(dir, file));
    }
    const rebuilt = new Workspace(dir);
    const report = await rebuilt.reindex();
    assert.deepEqual(report.indexed, ["chat-42", "recipes"]);
    assert.deepEqual(report.undescribed, ["scratch"]);
    rebuilt.close();
  });
});

test("closing a workspace invalidates every logical reference", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-close-"));
  try {
    const ws = new Workspace(dir);
    const chat = ws.memory("chat-42");
    await chat.remember({ text: "no stale native handle" });
    ws.close();
    ws.close();

    assert.throws(() => ws.memory("chat-42"), (err) => err.code === "PLUGMEM_CLOSED");
    assert.throws(() => chat.stats(), (err) => err.code === "PLUGMEM_CLOSED");

    const direct = await Plugmem.open(join(dir, "db", "chat-42.plugmem"));
    assert.equal(direct.stats().facts, 1, "close released the pooled file lock");
    direct.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("pool limits are checked and a direct database remains unchanged", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-single-"));
  try {
    assert.throws(() => new Workspace(dir, { maxOpen: 0 }), /maxOpen must be between/);
    assert.throws(() => new Workspace(dir, { maxOpen: 100000 }), /maxOpen must be between/);
    assert.throws(() => new Workspace(dir, { idleTimeoutMs: -1 }), /idleTimeoutMs must be/);

    const db = await Plugmem.open(join(dir, "m.plugmem"));
    await db.remember({ text: "a plain single memory" });
    assert.equal(db.stats().facts, 1);
    db.close();
    assert.ok(!existsSync(join(dir, "registry.plugmem")));
    assert.ok(!existsSync(join(dir, "db")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
