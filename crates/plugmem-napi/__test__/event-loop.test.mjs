// The event-loop contract.
//
// Two properties hold this binding up, and both fail silently: nothing throws
// when they break, the process just gets slower in a way a unit test does not
// see. So they are asserted here.
//
//   1. No verb holds the JS thread for work proportional to the database.
//      `Task::compute` runs on a libuv worker, but `Task::resolve` runs on the
//      main thread, so a careless result conversion is a stall.
//   2. No lock is held across an embedder round trip. `Embedder::embed` takes
//      `&self` so no caller needs one; a `Mutex` reintroduced anywhere in that
//      chain would serialize concurrent recalls.
//
// **Every threshold here is relative.** A wall-clock budget calibrated on a
// developer's machine is a flake on a shared CI runner, so nothing below
// compares a measurement to a constant: each one is measured against a control
// taken in the same run on the same machine — a busy loop against a sleep,
// `export` against `exportPage`, four concurrent calls against one. The
// strongest assertion of the lot has no timing in it at all: it counts how many
// requests the embedder mock saw at once.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { Plugmem } = require("../index.js");

const ms = (t0) => Number(process.hrtime.bigint() - t0) / 1e6;

/**
 * Runs `fn` and reports how long the JS thread was unavailable, by counting how
 * many fires a fixed-interval timer lost. Blunt on purpose: it needs no
 * agreement about what a "loop delay" is, and it degrades gracefully on a busy
 * machine (a slow runner loses fires everywhere, including in the controls).
 */
const TICK = 5;

async function heldFor(fn) {
  await new Promise((r) => setTimeout(r, 40));
  let fires = 0;
  const beat = setInterval(() => fires++, TICK);
  const t0 = process.hrtime.bigint();
  try {
    var value = await fn();
  } finally {
    clearInterval(beat);
  }
  const total = ms(t0);
  return { value, total, held: Math.max(0, total / TICK - fires) * TICK };
}

/**
 * The most a call of `total` ms may hold the thread before it counts as
 * blocking, given a machine whose idle noise floor is `floor` ms.
 *
 * Two terms, neither of them a wall-clock constant. `floor` is what this
 * runner charges a call that provably blocks nothing, measured in the same
 * run. `TICK * 2` is the instrument's own granularity: below a couple of timer
 * periods a reading is a rounding artefact, not a signal, so a short verb is
 * only ever asked not to lose more than that. Above it, the bound is half the
 * call — loose enough to survive a noisy runner, tight enough that a verb which
 * moved its work back onto the JS thread (and would read near 100%) fails.
 */
const budget = (total, floor) => floor + Math.max(TICK * 2, total / 2);

/** What this machine charges an await that blocks nothing at all. */
async function noiseFloor() {
  const idle = await heldFor(() => new Promise((r) => setTimeout(r, 200)));
  return idle.held;
}

const spin = (duration) => async () => {
  const end = Date.now() + duration;
  while (Date.now() < end) {
    /* hold the thread */
  }
};

function tempdir(tag) {
  return mkdtempSync(join(tmpdir(), `plugmem-loop-${tag}-`));
}

const fact = (i) => ({
  text: `word${i % 8} other${(i * 7) % 8} fact number ${i} about subject ${i % 97}`,
  entity: `entity-${i % 97}`,
  tags: [`t${i % 8}`],
});

test("the instrument can tell a blocked thread from a busy one", async () => {
  // Without this, every assertion below could pass on a dead instrument.
  const blocked = await heldFor(spin(200));
  const idle = await heldFor(() => new Promise((r) => setTimeout(r, 200)));

  assert.ok(
    blocked.held > blocked.total / 2,
    `a busy loop must register as held: ${blocked.held.toFixed(0)} of ${blocked.total.toFixed(0)} ms`,
  );
  assert.ok(
    idle.held < blocked.held / 4,
    `sleeping must not register as held: ${idle.held.toFixed(0)} ms vs a blocked ${blocked.held.toFixed(0)} ms`,
  );
});

test("no verb holds the JS thread for database-sized work", async () => {
  const dir = tempdir("verbs");
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"), {});
    for (let i = 0; i < 20000; i += 2000) {
      await db.rememberMany(Array.from({ length: 2000 }, (_, j) => fact(i + j)));
    }
    const floor = await noiseFloor();

    // Every verb whose cost scales with the database. `export` is deliberately
    // absent: it materializes one JS object per fact in `resolve`, which is
    // main-thread work by construction and is documented as such — the next
    // test measures it rather than forbidding it.
    for (const [name, run] of [
      ["maintain('full')", () => db.maintain("full")],
      ["checkpoint", () => db.checkpoint()],
      ["verify", () => db.verify()],
      [
        "exportPage sweep",
        async () => {
          let cursor = 0;
          for (;;) {
            const page = await db.exportPage(cursor);
            if (page.nextCursor == null) return;
            cursor = page.nextCursor;
          }
        },
      ],
    ]) {
      const { total, held } = await heldFor(run);
      assert.ok(
        held <= budget(total, floor),
        `${name} held the JS thread for ${held.toFixed(0)} of its ${total.toFixed(0)} ms ` +
          `(budget ${budget(total, floor).toFixed(0)} ms on this machine)`,
      );
    }
    db.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("exportPage keeps the thread that export takes", async () => {
  // The two paths over identical data, in one run on one machine: whatever this
  // runner's absolute numbers are, the paged one must be far cheaper for the JS
  // thread. This is the regression that would otherwise land unnoticed.
  const dir = tempdir("export");
  try {
    const db = await Plugmem.open(join(dir, "m.plugmem"), {});
    for (let i = 0; i < 20000; i += 2000) {
      await db.rememberMany(Array.from({ length: 2000 }, (_, j) => fact(i + j)));
    }

    const whole = await heldFor(() => db.export());
    const paged = await heldFor(async () => {
      let cursor = 0;
      let facts = 0;
      for (;;) {
        const page = await db.exportPage(cursor);
        facts += page.facts.length;
        if (page.nextCursor == null) return facts;
        cursor = page.nextCursor;
      }
    });

    assert.equal(paged.value, whole.value.length, "both paths return the same facts");
    assert.ok(
      paged.held < whole.held / 4,
      `paging must be far cheaper for the JS thread: ${paged.held.toFixed(0)} ms paged ` +
        `vs ${whole.held.toFixed(0)} ms whole`,
    );
    db.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("an embedder on this very event loop can answer, and is not locked", async () => {
  // The sharpest form of the question. The mock replies only from a timer, so
  // it cannot answer at all unless the JS thread is free while the native call
  // waits. A binding that embedded inline would hang here, not fail.
  const dir = tempdir("embed");
  const DIM = 8;
  const DELAY = 150;
  let peak = 0;
  let inFlight = 0;
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      inFlight++;
      if (inFlight > peak) peak = inFlight;
      const inputs = JSON.parse(body).input;
      setTimeout(() => {
        inFlight--;
        res.writeHead(200, { "content-type": "application/json", connection: "close" });
        res.end(
          JSON.stringify({
            data: inputs.map((text, index) => ({
              index,
              embedding: Array.from({ length: DIM }, (_, j) => Math.sin(text.length + j)),
            })),
          }),
        );
      }, DELAY);
    });
  });
  const port = await new Promise((r) =>
    server.listen(0, "127.0.0.1", () => r(server.address().port)),
  );
  const config = join(dir, "config.toml");
  writeFileSync(
    config,
    `[engine]\ndim = ${DIM}\n[embedder]\nkind = "openai"\n` +
      `url = "http://127.0.0.1:${port}/v1/embeddings"\nmodel = "mock"\n`,
  );

  try {
    const db = await Plugmem.open(join(dir, "e.plugmem"), { config });

    const floor = await noiseFloor();

    const remembered = await heldFor(() =>
      db.remember({ text: "the release ships on friday", entity: "release" }),
    );
    assert.ok(
      remembered.held <= budget(remembered.total, floor),
      `remember held the JS thread for ${remembered.held.toFixed(0)} of ` +
        `${remembered.total.toFixed(0)} ms across the embedder call`,
    );

    const single = await heldFor(() => db.recall({ query: "when does it ship", k: 3 }));
    assert.ok(
      single.held <= budget(single.total, floor),
      `recall held the JS thread for ${single.held.toFixed(0)} of ${single.total.toFixed(0)} ms`,
    );

    // No lock anywhere in the chain, asserted two ways. The first has no timing
    // in it at all and so cannot flake: the mock counts how many requests were
    // open at once, and a `Mutex` in front of the embedder pins that at 1.
    peak = 0;
    const concurrent = await heldFor(() =>
      Promise.all(Array.from({ length: 4 }, (_, i) => db.recall({ query: `q${i}`, k: 2 }))),
    );
    assert.ok(peak > 1, `concurrent recalls serialized at the embedder (peak ${peak})`);
    // The second is relative to the single call just measured: serialized, four
    // of them cost four round trips; overlapped, barely more than one.
    assert.ok(
      concurrent.total < single.total * 2,
      `4 concurrent recalls took ${concurrent.total.toFixed(0)} ms against ` +
        `${single.total.toFixed(0)} ms for one, i.e. they queued`,
    );

    await db.checkpoint();
    db.close();

    // The read-only path embeds in this binding rather than in the engine, so
    // it needs its own proof.
    const ro = await Plugmem.open(join(dir, "e.plugmem"), { config, readOnly: true });
    const roSingle = await heldFor(() => ro.recall({ query: "when does it ship", k: 3 }));
    peak = 0;
    const roConcurrent = await heldFor(() =>
      Promise.all(Array.from({ length: 4 }, (_, i) => ro.recall({ query: `r${i}`, k: 2 }))),
    );
    assert.ok(peak > 1, `read-only recalls serialized at the embedder (peak ${peak})`);
    assert.ok(
      roConcurrent.total < roSingle.total * 2,
      `4 concurrent read-only recalls took ${roConcurrent.total.toFixed(0)} ms against ` +
        `${roSingle.total.toFixed(0)} ms for one`,
    );
    assert.ok(
      roConcurrent.held <= budget(roConcurrent.total, floor),
      `read-only recall held the JS thread for ${roConcurrent.held.toFixed(0)} ms`,
    );
    ro.close();
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
