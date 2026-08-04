// Read-only recall reaches the vector source.
//
// A `ReadOnlyDatabase` carries no embedder by design — embedding inside it
// would write into a mapping opened zero-copy. The CLI and the MCP server solve
// that the same way: they embed the query themselves and hand the engine a
// vector. This binding used to do neither, so a text `recall` on a read-only
// handle silently had no vector source at all. These tests hold the parity.
//
// **The mock embedder must live off this thread.** `recall` is a synchronous
// native call: while it waits on the embedder's HTTP round trip it owns the JS
// thread, so a server on this event loop could never answer it and the test
// would wait on itself forever. The server therefore runs in a worker, and the
// request count comes back through a `SharedArrayBuffer` — memory the blocked
// main thread can still read the moment it is unblocked. No network leaves the
// machine and no real embedder is involved, so this runs anywhere CI does.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { Worker } from "node:worker_threads";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

const DIM = 8;

/**
 * The worker body: an OpenAI-shaped `/v1/embeddings` server that answers
 * deterministically (the vector is a function of the input length) and bumps a
 * shared counter per embedded text.
 */
const EMBEDDER_WORKER = `
import { createServer } from "node:http";
import { parentPort, workerData } from "node:worker_threads";

const { counter, dim } = workerData;
const seen = new Int32Array(counter);

const server = createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", () => {
    const inputs = JSON.parse(body).input;
    Atomics.add(seen, 0, inputs.length);
    const data = inputs.map((text, index) => ({
      index,
      embedding: Array.from({ length: dim }, (_, j) => Math.sin(text.length + j)),
    }));
    const payload = JSON.stringify({ data });
    // Close each connection: nothing here benefits from keep-alive, and a
    // lingering socket would hold the worker open past the test.
    res.writeHead(200, { "content-type": "application/json", connection: "close" });
    res.end(payload);
  });
});

server.listen(0, "127.0.0.1", () => {
  parentPort.postMessage(server.address().port);
});
`;

/** Runs `fn(base, seen)` against a worker-hosted embedder. */
async function withEmbedder(fn) {
  const counter = new SharedArrayBuffer(4);
  const seen = new Int32Array(counter);
  const worker = new Worker(EMBEDDER_WORKER, {
    eval: true,
    workerData: { counter, dim: DIM },
  });
  const port = await new Promise((resolve, reject) => {
    worker.once("message", resolve);
    worker.once("error", reject);
  });
  try {
    return await fn(`http://127.0.0.1:${port}/v1/embeddings`, seen);
  } finally {
    await worker.terminate();
  }
}

/**
 * A throwaway directory holding a config pointed at `url`, and a memory.
 *
 * `async` and awaiting on purpose: a synchronous wrapper would run its cleanup
 * the instant `fn` returned its promise, deleting the database out from under
 * the work still in flight.
 */
async function withConfig(url, fn) {
  const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-embed-"));
  const config = join(dir, "config.toml");
  writeFileSync(
    config,
    `[engine]\ndim = ${DIM}\n[embedder]\nkind = "openai"\nurl = "${url}"\nmodel = "test"\n`,
  );
  try {
    return await fn({ config, path: join(dir, "m.plugmem") });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("a read-only handle embeds the query itself, like the CLI and MCP do", async () => {
  await withEmbedder(async (url, seen) => {
    await withConfig(url, async ({ config, path }) => {
      // A writer with the same config: the engine embeds on `remember`.
      const w = new Plugmem(path, { config });
      w.remember({ text: "the deployment finished at noon" });
      await w.checkpoint();
      w.close();

      const afterWrite = Atomics.load(seen, 0);
      assert.ok(afterWrite >= 1, "the writer embedded the stored fact");

      // Read-only over the published snapshot. The engine cannot embed here,
      // so if the binding does not, this query never reaches the embedder.
      const ro = new Plugmem(path, { config, readOnly: true });
      const res = ro.recall({ query: "deployment" });

      assert.equal(
        Atomics.load(seen, 0),
        afterWrite + 1,
        "the read-only recall embedded its query exactly once",
      );
      assert.equal(res.facts.length, 1);
      ro.close();
    });
  });
});

test("a read-only recall with no text embeds nothing", async () => {
  await withEmbedder(async (url, seen) => {
    await withConfig(url, async ({ config, path }) => {
      const w = new Plugmem(path, { config });
      w.remember({ text: "a fact", tags: ["t"] });
      await w.checkpoint();
      w.close();

      const afterWrite = Atomics.load(seen, 0);
      const ro = new Plugmem(path, { config, readOnly: true });
      // Tags only: there is no query to embed, so no call must be made.
      ro.recall({ tags: ["t"] });
      assert.equal(Atomics.load(seen, 0), afterWrite, "no text, no embedder call");
      ro.close();
    });
  });
});

test("an unreachable embedder surfaces as a thrown error, not a silent miss", async () => {
  await withEmbedder(async (url) => {
    await withConfig(url, async ({ config, path }) => {
      const w = new Plugmem(path, { config });
      w.remember({ text: "stored while the embedder was up" });
      await w.checkpoint();
      w.close();

      // Point the read-only handle at a port nothing is listening on: the
      // failure has to be reported, not quietly downgraded to lexical-only.
      const dir = mkdtempSync(join(tmpdir(), "plugmem-napi-embed-down-"));
      const broken = join(dir, "config.toml");
      writeFileSync(
        broken,
        `[engine]\ndim = ${DIM}\n[embedder]\nkind = "openai"\nurl = "http://127.0.0.1:1/v1/embeddings"\nmodel = "test"\n`,
      );
      try {
        const ro = new Plugmem(path, { config: broken, readOnly: true });
        assert.throws(() => ro.recall({ query: "stored" }), /embedder/);
        ro.close();
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    });
  });
});
