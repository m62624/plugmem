"""`forget_many` parity: batching several tombstones under one write."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import plugmem


def test_direct_forget_many_matches_single_forgets(tmp_path: Path) -> None:
    path = str(tmp_path / "forget.plugmem")
    with plugmem.Plugmem.open(path) as db:
        a, b, c = db.remember_many(
            [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
        )

        assert db.forget_many([a.id, b.id]) == [True, True]
        assert db.get(a.id) is None
        assert db.get(b.id) is None
        assert db.get(c.id) is not None

        # Idempotent, same as single forget: a repeat reports False.
        assert db.forget_many([a.id, c.id]) == [False, True]
        assert db.forget_many([]) == []


def test_workspace_memory_forget_many_uses_a_scoped_lease(tmp_path: Path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    memory = ws.memory("forget-many")
    a, b = memory.remember_many([{"text": "alpha"}, {"text": "beta"}])

    assert memory.forget_many([a.id, b.id]) == [True, True]
    assert memory.get(a.id) is None
    assert memory.forget_many([]) == []
    ws.close()


def test_forget_many_from_parallel_threads_does_not_deadlock(tmp_path: Path) -> None:
    # Same `py.detach` pattern as `forget`/`remember_many`, exercised from
    # several threads at once with a timeout: if the GIL (or the host write
    # lock) were held across the call instead of released, this would hang
    # rather than fail loudly.
    with plugmem.Plugmem.open(str(tmp_path / "concurrent-forget.plugmem")) as db:
        ids = [
            outcome.id
            for outcome in db.remember_many([{"text": f"f{i}"} for i in range(8)])
        ]
        batches = [ids[0:2], ids[2:4], ids[4:6], ids[6:8]]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(db.forget_many, batch) for batch in batches]
            results = [future.result(timeout=30) for future in futures]

        assert all(all(r) for r in results)
        assert db.stats().facts == 8  # tombstoned, not purged
