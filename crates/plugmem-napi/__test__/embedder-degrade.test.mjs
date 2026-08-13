// What happens to a memory when its embedder stops answering.
//
// The engine's own tests cover the policy; these cover the boundary, and the
// boundary is where the interesting half lives. A writer embeds inside the
// host, but a **read-only** handle embeds its own query out here in the
// binding — so "the provider is down" reaches the two handle kinds by two
// different code paths, and only one of them was ever exercised by a Rust
// test. A regression in the reader would show up as every read failing on the
// surface where a memory is only ever read.
//
// No network leaves the machine: the "provider" is a local server that is
// started, its port noted, and then stopped, so the address is guaranteed to
// refuse connections.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem, Workspace } = require("../index.js");

const DIM = 8;

/** A port nothing is listening on: bound, read, and released. */
async function deadEndpoint() {
  const server = createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return `http://127.0.0.1:${port}/v1/embeddings`;
}

/**
 * A live OpenAI-shaped embedder, plus a count of the texts it has embedded.
 *
 * `delayMs` holds the response back, which is how a test can be inside a
 * provider round trip while it calls something else.
 */
async function liveEndpoint(delayMs = 0) {
  const state = { embedded: 0 };
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", async () => {
      const inputs = JSON.parse(body).input;
      state.embedded += inputs.length;
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
      const data = inputs.map((text, index) => ({
        index,
        embedding: Array.from({ length: DIM }, (_, j) => Math.sin(text.length + j)),
      }));
      res.writeHead(200, {
        "content-type": "application/json",
        connection: "close",
      });
      res.end(JSON.stringify({ data }));
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  state.url = `http://127.0.0.1:${port}/v1/embeddings`;
  state.stop = () => new Promise((resolve) => server.close(resolve));
  return state;
}

/**
 * A throwaway directory with a config naming `url`, and a database path.
 *
 * `extra` is appended inside `[embedder]`, which is where every knob under
 * test lives.
 */
async function withConfig(url, extra, fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-degrade-"));
  const config = join(dir, "config.toml");
  writeFileSync(
    config,
    `[engine]\ndim = ${DIM}\n[embedder]\nurl = "${url}"\nmodel = "test"\n${extra}\n`,
  );
  try {
    return await fn({ config, path: join(dir, "m.plugmem") });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("the default is still to fail the verb, loudly and with a code", async () => {
  const url = await deadEndpoint();
  await withConfig(url, "", async ({ config, path }) => {
    const db = await Plugmem.open(path, { config });
    assert.equal(db.embedderState(), "active");
    await assert.rejects(
      () => db.remember({ text: "a fact" }),
      (err) => err.code === "PLUGMEM_ENGINE" && /embedder/.test(err.message),
    );
    // `fail` never suspends: there is nothing to resume once the provider is
    // back, which is the whole reason it is safe as a default.
    assert.equal(db.embedderState(), "active");
    db.close();
  });
});

test("degrade stores the fact without a vector and still finds it", async () => {
  const url = await deadEndpoint();
  await withConfig(url, 'on_error = "degrade"\nretry_after_ms = 0', async ({ config, path }) => {
    const db = await Plugmem.open(path, { config });
    await db.remember({ text: "the cache is off because it raced with the warmup" });

    const stats = db.stats();
    assert.equal(stats.facts, 1);
    assert.equal(stats.vectors, 0, "stored without a vector, as if no embedder existed");
    assert.equal(db.embedderState(), "suspended");

    // Still answerable: BM25, tags, graph and time never needed the provider.
    const found = await db.recall({ query: "cache" });
    assert.equal(found.facts.length, 1);
    db.close();
  });
});

test("a read-only handle degrades too, which is the path the host cannot cover", async () => {
  const live = await liveEndpoint();
  await withConfig(live.url, 'on_error = "degrade"\nretry_after_ms = 0', async ({ config, path }) => {
    const w = await Plugmem.open(path, { config });
    await w.remember({ text: "the deployment finished at noon" });
    await w.checkpoint();
    w.close();
    assert.ok(live.embedded >= 1);

    // The provider dies between the write and the read.
    await live.stop();

    const ro = await Plugmem.open(path, { config, readOnly: true });
    assert.equal(ro.embedderState(), "active");
    // Before the gate, this rejected: the binding embedded the query itself
    // and had nowhere to put the failure.
    const found = await ro.recall({ query: "deployment" });
    assert.equal(found.facts.length, 1);
    assert.equal(ro.embedderState(), "suspended");
    ro.close();
  });
});

test("suspend and resume are honoured on both handle kinds", async () => {
  const live = await liveEndpoint();
  try {
    await withConfig(live.url, "", async ({ config, path }) => {
      const db = await Plugmem.open(path, { config });
      db.suspendEmbedder();
      assert.equal(db.embedderState(), "suspended");

      const before = live.embedded;
      await db.remember({ text: "a fact stored while suspended" });
      assert.equal(live.embedded, before, "the provider was not called");
      assert.equal(db.stats().vectors, 0);

      db.resumeEmbedder();
      assert.equal(db.embedderState(), "active");
      await db.remember({ text: "a fact stored after resuming" });
      assert.equal(live.embedded, before + 1);
      assert.equal(db.stats().vectors, 1);

      // And the missing one is recoverable, which is what makes degrading safe.
      const report = await db.reembed();
      assert.equal(report.embedded, 2);
      assert.equal(db.stats().vectors, 2);
      await db.checkpoint();
      db.close();

      const ro = await Plugmem.open(path, { config, readOnly: true });
      ro.suspendEmbedder();
      assert.equal(ro.embedderState(), "suspended");
      const after = live.embedded;
      await ro.recall({ query: "fact" });
      assert.equal(live.embedded, after, "a suspended reader embeds nothing");
      ro.resumeEmbedder();
      assert.equal(ro.embedderState(), "active");
      ro.close();
    });
  } finally {
    await live.stop();
  }
});

test("a memory with no embedder says so, and both switches are no-ops", async () => {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-degrade-none-"));
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"));
    assert.equal(db.embedderState(), "absent");
    db.suspendEmbedder();
    assert.equal(db.embedderState(), "absent");
    db.resumeEmbedder();
    assert.equal(db.embedderState(), "absent");
    await db.remember({ text: "a fact" });
    assert.equal((await db.recall({ query: "fact" })).facts.length, 1);
    db.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a suspended embedder refuses a reembed instead of writing half a vector axis", async () => {
  const live = await liveEndpoint();
  try {
    await withConfig(live.url, "", async ({ config, path }) => {
      const db = await Plugmem.open(path, { config });
      await db.remember({ text: "a fact" });
      db.suspendEmbedder();
      await assert.rejects(
        () => db.reembed(),
        (err) => /suspended/.test(err.message),
      );
      db.resumeEmbedder();
      const report = await db.reembed();
      assert.equal(report.embedded, 1);
      db.close();
    });
  } finally {
    await live.stop();
  }
});

test("a workspace memory reports and switches its own embedder", async () => {
  // A workspace shares one provider between its memories and gives each its
  // own gate. Without these three verbs on `WorkspaceMemory`, a caller working
  // through a workspace could write vectorless facts for an hour and have no
  // way to ask why — `vectors < facts` looks the same as never having had an
  // embedder.
  const live = await liveEndpoint();
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-degrade-"));
  const config = join(dir, "config.toml");
  writeFileSync(
    config,
    `[engine]\ndim = ${DIM}\n[embedder]\nurl = "${live.url}"\nmodel = "test"\n`,
  );
  const ws = new Workspace(dir, { config });
  try {
    const chat = ws.memory("chat-1");
    const other = ws.memory("chat-2");
    await chat.remember({ text: "the cache is off" });
    await other.remember({ text: "the deploy is manual" });
    assert.equal(await chat.embedderState(), "active");
    assert.equal((await chat.stats()).vectors, 1);

    // Suspended: this memory writes without vectors, and says so.
    await chat.suspendEmbedder();
    assert.equal(await chat.embedderState(), "suspended");
    await chat.remember({ text: "the warmup runs first" });
    const suspended = await chat.stats();
    assert.equal(suspended.facts, 2);
    assert.equal(suspended.vectors, 1, "the second fact stored no vector");

    // Its sibling shares the provider, not the gate.
    assert.equal(await other.embedderState(), "active");

    // And back: a resumed memory embeds again, without reopening anything.
    await chat.resumeEmbedder();
    assert.equal(await chat.embedderState(), "active");
    await chat.remember({ text: "the queue drains on exit" });
    assert.equal((await chat.stats()).vectors, 2);
  } finally {
    ws.close();
    rmSync(dir, { recursive: true, force: true });
    await live.stop();
  }
});

test("a workspace memory degrades on an unreachable provider like any other", async () => {
  const url = await deadEndpoint();
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-dead-"));
  const config = join(dir, "config.toml");
  writeFileSync(
    config,
    `[engine]\ndim = ${DIM}\n[embedder]\nurl = "${url}"\nmodel = "test"\n` +
      'on_error = "degrade"\nretry_after_ms = 0\n',
  );
  const ws = new Workspace(dir, { config });
  try {
    const chat = ws.memory("chat-1");
    await chat.remember({ text: "a fact nobody could embed" });
    assert.equal(await chat.embedderState(), "suspended");
    const stats = await chat.stats();
    assert.equal(stats.facts, 1);
    assert.equal(stats.vectors, 0);
    assert.equal((await chat.recall({ query: "fact" })).facts.length, 1);
  } finally {
    ws.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the switches answer while a provider round trip is in flight", async () => {
  // The gate must never hold its lock across the HTTP call: `embed` takes
  // `&self` precisely so several verbs can be inside the provider at once, and
  // `embedderState()` is synchronous on this class. If the lock were ever held
  // through a round trip, this test would not fail — it would hang, with the
  // JavaScript main thread parked behind a network call.
  const live = await liveEndpoint(300);
  await withConfig(live.url, "", async ({ config, path }) => {
    const db = await Plugmem.open(path, { config });
    // Started, not awaited: the write is sitting in the provider right now.
    const writing = db.remember({ text: "a fact that takes its time" });

    // All three, from the main thread, mid-flight.
    assert.equal(db.embedderState(), "active");
    db.suspendEmbedder();
    assert.equal(db.embedderState(), "suspended");
    db.resumeEmbedder();
    assert.equal(db.embedderState(), "active");

    // And the write still lands: nothing above cancelled or corrupted it.
    const out = await writing;
    assert.equal(out.id, 0);
    assert.equal(db.stats().vectors, 1);
    db.close();
  });
  await live.stop();
});

test("a workspace memory answers its state while one of its writes is in flight", async () => {
  // Two scoped leases on the same pooled database at once: the write holds one
  // for 300 ms inside the provider, and `embedderState()` takes another. The
  // pool lock is released before either closure runs, so this returns rather
  // than queueing behind the round trip — and would hang, not fail, if that
  // ever changed.
  const live = await liveEndpoint(300);
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-ws-inflight-"));
  const config = join(dir, "config.toml");
  writeFileSync(
    config,
    `[engine]\ndim = ${DIM}\n[embedder]\nurl = "${live.url}"\nmodel = "test"\n`,
  );
  const ws = new Workspace(dir, { config });
  try {
    const chat = ws.memory("chat-1");
    // Create the file first, so the in-flight lease is a hit rather than an open.
    await chat.remember({ text: "a first fact" });

    const writing = chat.remember({ text: "a fact that takes its time" });
    assert.equal(await chat.embedderState(), "active");
    await writing;
    assert.equal((await chat.stats()).vectors, 2);
  } finally {
    ws.close();
    rmSync(dir, { recursive: true, force: true });
    await live.stop();
  }
});
