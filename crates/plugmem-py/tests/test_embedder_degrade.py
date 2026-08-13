"""What a memory does when its embedder stops answering.

The policy itself is the host's, and the host tests it. What these cover is the
half only a binding has: a **read-only** handle embeds its own query out here,
so "the provider is unreachable" reaches the two handle kinds by two different
code paths. Only one of them is exercised by a Rust test, and the untested one
is the surface where a memory is only ever read.

Nothing leaves the machine: the "provider" is a local server, either answering
on a thread or already stopped so its address refuses connections.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import plugmem
import pytest

DIM = 8


class _Handler(BaseHTTPRequestHandler):
    """An OpenAI-shaped `/v1/embeddings` endpoint that counts what it embeds."""

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        inputs = json.loads(self.rfile.read(length))["input"]
        self.server.embedded += len(inputs)
        data = [
            {
                "index": index,
                "embedding": [math.sin(len(text) + j) for j in range(DIM)],
            }
            for index, text in enumerate(inputs)
        ]
        body = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Silence: the test output is the assertions, not an access log."""


class _Embedder:
    """A live endpoint, until `stop()`."""

    def __init__(self) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.embedded = 0
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/embeddings"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def embedded(self) -> int:
        return self.server.embedded

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def embedder() -> _Embedder:
    live = _Embedder()
    yield live
    try:
        live.stop()
    except OSError:
        pass  # Already stopped by a test that wanted a dead address.


def dead_url() -> str:
    """An address nothing is listening on: bound, read, released."""
    live = _Embedder()
    url = live.url
    live.stop()
    return url


def config_file(tmp_path: Path, url: str, extra: str = "") -> str:
    path = tmp_path / "config.toml"
    path.write_text(
        f'[engine]\ndim = {DIM}\n[embedder]\nurl = "{url}"\nmodel = "test"\n{extra}\n'
    )
    return str(path)


def test_the_default_still_fails_the_verb(tmp_path: Path) -> None:
    config = config_file(tmp_path, dead_url())
    with plugmem.Plugmem.open(str(tmp_path / "m.plugmem"), config=config) as db:
        assert db.embedder_state() == "active"
        with pytest.raises(plugmem.EngineError, match="embedder"):
            db.remember("a fact")
        # `fail` never suspends, so nothing has to be resumed once the provider
        # is back — which is what makes it safe as the default.
        assert db.embedder_state() == "active"


def test_degrade_stores_without_a_vector_and_still_finds_it(tmp_path: Path) -> None:
    config = config_file(
        tmp_path, dead_url(), 'on_error = "degrade"\nretry_after_ms = 0'
    )
    with plugmem.Plugmem.open(str(tmp_path / "m.plugmem"), config=config) as db:
        db.remember("the cache is off because it raced with the warmup")
        stats = db.stats()
        assert stats.facts == 1
        assert stats.vectors == 0
        assert db.embedder_state() == "suspended"
        # Lexical, tag, graph and time never needed the provider.
        assert len(db.recall("cache").facts) == 1


def test_a_read_only_handle_degrades_too(tmp_path: Path, embedder: _Embedder) -> None:
    config = config_file(
        tmp_path, embedder.url, 'on_error = "degrade"\nretry_after_ms = 0'
    )
    path = str(tmp_path / "m.plugmem")
    with plugmem.Plugmem.open(path, config=config) as writer:
        writer.remember("the deployment finished at noon")
        writer.checkpoint()
    assert embedder.embedded >= 1

    # The provider dies between the write and the read.
    embedder.stop()

    with plugmem.Plugmem.open(path, config=config, read_only=True) as reader:
        assert reader.embedder_state() == "active"
        # Before the shared gate this raised: the binding embedded the query
        # itself and had nowhere to put the failure.
        assert len(reader.recall("deployment").facts) == 1
        assert reader.embedder_state() == "suspended"


def test_suspend_and_resume_on_both_handle_kinds(
    tmp_path: Path, embedder: _Embedder
) -> None:
    config = config_file(tmp_path, embedder.url)
    path = str(tmp_path / "m.plugmem")
    with plugmem.Plugmem.open(path, config=config) as db:
        db.suspend_embedder()
        assert db.embedder_state() == "suspended"

        before = embedder.embedded
        db.remember("a fact stored while suspended")
        assert embedder.embedded == before
        assert db.stats().vectors == 0

        db.resume_embedder()
        assert db.embedder_state() == "active"
        db.remember("a fact stored after resuming")
        assert embedder.embedded == before + 1
        assert db.stats().vectors == 1

        # The missing vector is recoverable, which is what makes degrading safe.
        assert db.reembed().embedded == 2
        assert db.stats().vectors == 2
        db.checkpoint()

    with plugmem.Plugmem.open(path, config=config, read_only=True) as reader:
        reader.suspend_embedder()
        after = embedder.embedded
        reader.recall("fact")
        assert embedder.embedded == after
        reader.resume_embedder()
        assert reader.embedder_state() == "active"


def test_a_memory_without_an_embedder_says_absent(tmp_path: Path) -> None:
    with plugmem.Plugmem.open(str(tmp_path / "m.plugmem")) as db:
        assert db.embedder_state() == "absent"
        db.suspend_embedder()
        assert db.embedder_state() == "absent"
        db.resume_embedder()
        assert db.embedder_state() == "absent"
        db.remember("a fact")
        assert len(db.recall("fact").facts) == 1


def test_a_suspended_embedder_refuses_a_reembed(
    tmp_path: Path, embedder: _Embedder
) -> None:
    config = config_file(tmp_path, embedder.url)
    with plugmem.Plugmem.open(str(tmp_path / "m.plugmem"), config=config) as db:
        db.remember("a fact")
        db.suspend_embedder()
        with pytest.raises(plugmem.EngineError, match="suspended"):
            db.reembed()
        db.resume_embedder()
        assert db.reembed().embedded == 1
