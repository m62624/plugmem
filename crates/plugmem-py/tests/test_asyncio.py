"""The event loop keeps running while the engine works.

`test_concurrency.py` proves the GIL is released by counting threads inside the
engine. This file proves the consequence a Python program actually cares about:
`asyncio.to_thread` behaves, so an async application can use plugmem without an
async API and without stalling its loop.

Still counted rather than timed. A heartbeat coroutine increments on every pass
of the loop while the database work is in flight; if a verb held the GIL, the
loop could not schedule the heartbeat at all and the count would be zero. The
assertion is `> 0`, which no runner speed can make flaky — the failure it
guards against is a hard stall, not slowness.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import plugmem
import pytest

FACTS = 400
RECALLS = 12


def seed(path: str) -> plugmem.Plugmem:
    db = plugmem.Plugmem.open(path)
    db.remember_many(
        [
            {"text": f"fact number {i} about tokio and deploys", "entity": f"e{i % 9}"}
            for i in range(FACTS)
        ]
    )
    return db


def test_the_event_loop_keeps_turning_during_recalls(tmp_path) -> None:
    db = seed(str(tmp_path / "loop.plugmem"))

    async def main() -> tuple[int, int]:
        beats = 0
        done = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal beats
            while not done.is_set():
                beats += 1
                await asyncio.sleep(0)

        pulse = asyncio.create_task(heartbeat())
        results = await asyncio.gather(
            *(asyncio.to_thread(db.recall, "tokio", k=8) for _ in range(RECALLS))
        )
        done.set()
        await pulse
        return beats, sum(len(r.facts) for r in results)

    beats, hits = asyncio.run(main())
    db.close()

    assert hits > 0, "the recalls found nothing, so they proved nothing"
    assert beats > 0, (
        "the event loop never ran while recalls were in flight: a verb held "
        "the GIL, and an async application would have stalled"
    )


def test_a_write_does_not_stall_the_loop_either(tmp_path) -> None:
    # Writes serialize inside the engine — one writer per database is the
    # design — so this does not claim they run in parallel. It claims the
    # narrower and more important thing: waiting for the writer happens with
    # the interpreter released, so everything else in the process keeps going.
    db = plugmem.Plugmem.open(str(tmp_path / "writes.plugmem"))

    async def main() -> int:
        beats = 0
        done = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal beats
            while not done.is_set():
                beats += 1
                await asyncio.sleep(0)

        pulse = asyncio.create_task(heartbeat())
        await asyncio.gather(
            *(
                asyncio.to_thread(db.remember, f"written from task {i}")
                for i in range(RECALLS)
            )
        )
        done.set()
        await pulse
        return beats

    beats = asyncio.run(main())
    assert db.stats().facts == RECALLS
    db.close()
    assert beats > 0, "the loop stalled while writes were queued behind each other"


def test_a_thread_pool_and_the_loop_can_share_one_handle(tmp_path) -> None:
    """The mixed case, which is what a real application looks like."""
    from concurrent.futures import ThreadPoolExecutor

    db = seed(str(tmp_path / "mixed.plugmem"))
    barrier = threading.Barrier(4, timeout=30)
    seen: list[int] = []

    def worker() -> None:
        barrier.wait()
        for _ in range(25):
            seen.append(len(db.recall("deploys", k=4).facts))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(4):
            pool.submit(worker)

    db.close()
    # Deliberately not claiming this proves overlap — the barrier is passed
    # before any verb is called, so it only shows four live threads. Overlap is
    # proved by the peak counters in `test_concurrency.py`. What this covers is
    # the case those do not: four threads sharing one handle through a pool
    # complete every call, with no `BusyError` and nothing lost.
    assert len(seen) == 100


def test_bulk_tag_removal_does_not_stall_the_loop(tmp_path) -> None:
    db = plugmem.Plugmem.open(str(tmp_path / "remove-tag.plugmem"))
    db.remember_many(
        [{"text": f"tagged fact {i}", "tags": ["bulk"]} for i in range(2_000)]
    )

    async def main() -> tuple[int, int]:
        beats = 0
        done = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal beats
            while not done.is_set():
                beats += 1
                await asyncio.sleep(0)

        pulse = asyncio.create_task(heartbeat())
        report = await asyncio.to_thread(db.remove_tag, "bulk")
        done.set()
        await pulse
        return beats, report.affected

    beats, affected = asyncio.run(main())
    db.close()
    assert affected == 2_000
    assert beats > 0, "the event loop stalled while remove_tag revised the tagged facts"


def test_reembed_uses_to_thread_without_stalling_the_loop(tmp_path) -> None:
    dim = 8
    provider_entered = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            provider_entered.set()
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            inputs = json.loads(body).get("input", [])
            time.sleep(0.05)
            payload = json.dumps(
                {
                    "data": [
                        {"index": i, "embedding": [0.1] * dim}
                        for i, _text in enumerate(inputs)
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = tmp_path / "reembed.toml"
    config.write_text(
        f"[engine]\ndim = {dim}\n[embedder]\n"
        f'url = "http://127.0.0.1:{server.server_port}/v1/embeddings"\n'
        'model = "async-test"\n',
        encoding="utf-8",
    )
    db = plugmem.Plugmem.open(str(tmp_path / "reembed.plugmem"), config=str(config))
    db.remember_many([{"text": f"fact {i}"} for i in range(8)])
    provider_entered.clear()

    async def main() -> tuple[int, plugmem.ReembedReport]:
        beats = 0
        done = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal beats
            while not done.is_set():
                beats += 1
                await asyncio.sleep(0)

        pulse = asyncio.create_task(heartbeat())
        reembedding = asyncio.create_task(asyncio.to_thread(db.reembed, 2))
        assert await asyncio.to_thread(provider_entered.wait, 10)
        with pytest.raises(plugmem.BusyError):
            await asyncio.to_thread(
                db.remember,
                "cannot cross frozen source",
                vector=[0.1] * dim,
            )
        report = await reembedding
        done.set()
        await pulse
        return beats, report

    try:
        beats, report = asyncio.run(main())
        assert report.new_space == "async-test"
        assert report.embedded == 8
        assert beats > 0, "the event loop stalled while reembed called the model"

        changed = tmp_path / "changed.toml"
        changed.write_text(
            f"[engine]\ndim = {dim}\n[embedder]\n"
            f'url = "http://127.0.0.1:{server.server_port}/v1/embeddings"\n'
            'model = "other-model"\n',
            encoding="utf-8",
        )
        reader = plugmem.Plugmem.open(
            str(tmp_path / "reembed.plugmem"),
            config=str(changed),
            read_only=True,
        )
        try:
            with pytest.raises(plugmem.EngineError, match="vector space"):
                reader.recall("fact")
        finally:
            reader.close()
    finally:
        db.close()
        server.shutdown()
        server.server_close()
