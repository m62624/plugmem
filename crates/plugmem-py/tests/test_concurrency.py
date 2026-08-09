"""Proof that the GIL is actually released, without measuring time.

The Node binding tests the same property against a measured noise floor, because
what it must show is "the event loop kept turning" and that is inherently about
elapsed time. Python allows a stronger statement: instrument the embedder,
**count** how many threads are inside it at once, and require the peak to exceed
one. A counter cannot be flaky on a slow runner, cannot depend on core count,
and says exactly what we mean — two threads were inside the engine at the same
instant, therefore no interpreter lock was held across the call.

The embedder is an HTTP endpoint the engine calls while a `remember` is in
flight, which makes it the one place a test can observe the inside of a verb.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import plugmem
import pytest

DIM = 8
WRITERS = 4


class Counter:
    """Peak simultaneous occupancy of the embedder."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        # Every request waits for this, so a peak of 1 can only mean the calls
        # were serialized — never that they merely failed to overlap by luck.
        self.all_arrived = threading.Barrier(WRITERS, timeout=30)


COUNTER = Counter()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        with COUNTER.lock:
            COUNTER.inside += 1
            COUNTER.peak = max(COUNTER.peak, COUNTER.inside)
        try:
            # Hold every request here until all of them have arrived. If the GIL
            # were held across `remember`, the second thread could never get
            # here and this would time out — which is the failure we want.
            COUNTER.all_arrived.wait()
        except threading.BrokenBarrierError:
            pass
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        count = len(json.loads(body or b"{}").get("input", [""]))
        # `index` is required: the engine reorders the provider's answers by it
        # rather than trusting arrival order, so a response without one is
        # refused with "embedding without an index".
        payload = json.dumps(
            {"data": [{"index": i, "embedding": [0.1] * DIM} for i in range(count)]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        with COUNTER.lock:
            COUNTER.inside -= 1

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture(scope="module")
def embedder_url() -> str:
    # Threading, not the plain HTTPServer: that one answers requests one at a
    # time, so it would report a peak of 1 no matter what the binding does.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/embeddings"
    server.shutdown()


def test_remember_releases_the_interpreter(embedder_url: str, tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[engine]\ndim = {DIM}\n[embedder]\nurl = "{embedder_url}"\nmodel = "test"\n',
        encoding="utf-8",
    )
    db_path = tmp_path / "concurrent.plugmem"

    with plugmem.Plugmem.open(str(db_path), config=str(config)) as db:
        errors: list[BaseException] = []

        def writer(index: int) -> None:
            try:
                db.remember_guarded(f"fact number {index}", entity=f"e{index}")
            except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(WRITERS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, f"a writer failed: {errors[0]!r}"
        assert COUNTER.peak > 1, (
            "only one thread was ever inside the embedder: the GIL was held "
            "across remember(), so nothing else in the process could run"
        )
        assert db.stats().facts == WRITERS


def test_parallel_guarded_writes_are_race_free_and_do_not_deadlock(tmp_path) -> None:
    with plugmem.Plugmem.open(str(tmp_path / "guarded.plugmem")) as db:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    db.remember_guarded,
                    "same durable fact",
                    entity="user",
                )
                for _ in range(2)
            ]
            outcomes = [future.result(timeout=30) for future in futures]

        assert sorted(outcome.status for outcome in outcomes) == ["blocked", "stored"]
        blocked = next(outcome for outcome in outcomes if outcome.status == "blocked")
        assert blocked.outcome is None
        assert len(blocked.similar) == 1
        assert db.stats().facts == 1


def test_reads_run_concurrently_on_one_handle(tmp_path) -> None:
    """Many threads recalling the same handle is not serialized by this layer.

    Counted, not timed: each thread records the maximum number of threads it saw
    inside `recall` alongside itself. The engine takes its own read lock, so the
    only way this can be 1 is if the wrapper's handle lock or the GIL made it 1.
    """
    db_path = tmp_path / "readers.plugmem"
    with plugmem.Plugmem.open(str(db_path)) as db:
        for i in range(200):
            db.remember(f"fact about tokio number {i}", entity=f"e{i % 7}")

        lock = threading.Lock()
        inside = 0
        peak = 0
        start = threading.Barrier(6, timeout=30)

        def reader() -> None:
            nonlocal inside, peak
            start.wait()
            for _ in range(50):
                with lock:
                    inside += 1
                    peak = max(peak, inside)
                db.recall("tokio", k=8)
                with lock:
                    inside -= 1

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert peak > 1, "recalls on one handle never overlapped"


def test_close_waits_for_readers_rather_than_racing_them(tmp_path) -> None:
    """`close()` takes the handle exclusively, so it cannot free it mid-read.

    The observable form of that guarantee: after `close()` returns, every verb
    raises `ClosedError` — never a crash, and never a stale answer.
    """
    db_path = tmp_path / "closing.plugmem"
    db = plugmem.Plugmem.open(str(db_path))
    db.remember("one fact")
    db.close()
    db.close()  # idempotent

    for call in (
        lambda: db.recall("one"),
        lambda: db.remember("two"),
        lambda: db.stats(),
        lambda: db.export(),
    ):
        with pytest.raises(plugmem.ClosedError) as raised:
            call()
        assert raised.value.code == "PLUGMEM_CLOSED"


def test_the_database_path_is_reported_even_when_resolved(tmp_path) -> None:
    db_path = tmp_path / "named.plugmem"
    with plugmem.Plugmem.open(str(db_path)) as db:
        assert db.path() == str(db_path)

    os.environ["PLUGMEM_DB"] = str(tmp_path / "from-env.plugmem")
    try:
        with plugmem.Plugmem.open() as db:
            assert db.path() == str(tmp_path / "from-env.plugmem")
    finally:
        del os.environ["PLUGMEM_DB"]
