"""FFI ownership tests for logical workspace memories."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import plugmem
import pytest


def test_reference_opens_nothing_and_survives_eviction_and_release(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path), max_open=1, idle_timeout_ms=0)
    a = ws.memory("a")
    b = ws.memory("b")
    assert isinstance(a, plugmem.WorkspaceMemory)
    assert a.name() == "a"
    assert ws.open_count() == 0

    a.remember("memory a survives eviction")
    b.remember("memory b takes the only slot")
    assert ws.open_count() == 1
    assert "survives eviction" in a.recall("survives").rendered
    assert ws.release("a") is True
    assert ws.release("a") is False
    assert a.stats().facts == 1
    ws.close()


def test_reads_do_not_create_and_the_pool_holds_the_real_file_lock(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    missing = ws.memory("missing")
    with pytest.raises(plugmem.EngineError, match="no database named missing"):
        missing.stats()
    assert not (tmp_path / "db" / "missing.plugmem.journal").exists()

    chat = ws.memory("chat")
    chat.remember("the file is protected")
    path = tmp_path / "db" / "chat.plugmem"
    with pytest.raises(plugmem.LockedError) as raised:
        plugmem.Plugmem.open(str(path))
    assert raised.value.code == "PLUGMEM_LOCKED"

    assert ws.release("chat") is True
    direct = plugmem.Plugmem.open(str(path))
    assert direct.stats().facts == 1
    direct.close()
    assert chat.stats().facts == 1
    ws.close()


def test_every_workspace_memory_verb_uses_the_scoped_path(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    memory = ws.memory("all-verbs")
    first, second = memory.remember_many(
        [{"text": "first version", "tags": ["old"]}, {"text": "temporary fact"}]
    )
    revised = memory.revise(first.id, "current version", tags=["new"])
    snapshot = memory.get(revised.id)
    assert snapshot is not None and snapshot.text == "current version"
    assert memory.tags_of(revised.id) == ["new"]
    assert "current version" in memory.recall("current").rendered

    memory.link("ann", "owns", "service", provenance=revised.id)
    edges: list[plugmem.ExportedEdge] = []
    assert memory.export_edges(edges.extend) == 1
    assert edges[0].provenance == revised.id
    assert memory.unlink("ann", "owns", "service") is True
    assert memory.forget(second.id) is True

    assert len(memory.export()) == 1
    assert len(memory.export_page().facts) == 1
    assert memory.stats().facts == 3  # closed records remain until maintenance
    memory.verify()
    memory.checkpoint()
    scrub = memory.scrub(64 * 1024)
    while scrub.next() is not None:
        pass
    assert isinstance(memory.maintain("auto").facts_after, int)
    ws.close()


def test_parallel_references_serialize_writes_without_a_deadlock(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    refs = [ws.memory("chat") for _ in range(24)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(memory.remember, f"fact {index}")
            for index, memory in enumerate(refs)
        ]
        for future in futures:
            future.result(timeout=30)

    assert refs[0].stats().facts == len(refs)
    assert ws.open_count() == 1
    assert ws.release("chat") is True
    ws.close()


def test_workspace_writes_do_not_stall_asyncio(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    memory = ws.memory("chat")

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
            *(asyncio.to_thread(memory.remember, f"fact {i}") for i in range(24))
        )
        done.set()
        await pulse
        return beats

    assert asyncio.run(main()) > 0
    assert memory.stats().facts == 24
    ws.close()


def test_active_capacity_is_busy_without_waiting_or_losing_the_lease(tmp_path) -> None:
    dim = 8
    arrived = threading.Event()
    may_answer = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            inputs = json.loads(body).get("input", [])
            arrived.set()
            may_answer.wait(timeout=30)
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
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    config = tmp_path / "config.toml"
    config.write_text(
        f"[engine]\ndim = {dim}\n[embedder]\n"
        f'url = "http://127.0.0.1:{server.server_port}/v1/embeddings"\n'
        'model = "test"\n',
        encoding="utf-8",
    )

    ws = plugmem.Workspace(str(tmp_path), config=str(config), max_open=1)
    errors: list[BaseException] = []
    reports: list[plugmem.ReembedReport] = []

    def hold_lease() -> None:
        try:
            memory = ws.memory("active")
            memory.remember("holds the scoped lease")
            reports.append(memory.reembed(1))
        except BaseException as exc:  # noqa: BLE001 — asserted below
            errors.append(exc)

    worker = threading.Thread(target=hold_lease)
    worker.start()
    try:
        assert arrived.wait(timeout=30), "the detached verb never reached the embedder"
        with pytest.raises(plugmem.BusyError) as capacity:
            ws.memory("other").remember("must not wait")
        assert capacity.value.code == "PLUGMEM_BUSY"
        with pytest.raises(plugmem.BusyError):
            ws.release("active")
    finally:
        may_answer.set()
        worker.join(timeout=30)
        ws.close()
        server.shutdown()

    assert not worker.is_alive(), "the active workspace verb deadlocked"
    assert not errors
    assert reports[0].new_space == "test"
    assert reports[0].embedded == 1


def test_close_invalidates_references_and_releases_the_lock(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    memory = ws.memory("chat")
    memory.remember("no stale native handle")
    ws.close()
    ws.close()

    with pytest.raises(plugmem.ClosedError):
        ws.memory("chat")
    with pytest.raises(plugmem.ClosedError):
        memory.stats()

    direct = plugmem.Plugmem.open(str(tmp_path / "db" / "chat.plugmem"))
    assert direct.stats().facts == 1
    direct.close()
