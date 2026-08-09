"""Tag-catalog parity: bounded discovery, history-safe removal, and workspaces."""

from __future__ import annotations

import plugmem
import pytest


def test_direct_catalog_pages_prefixes_and_removes_without_deleting_facts(tmp_path) -> None:
    path = str(tmp_path / "tags.plugmem")
    with plugmem.Plugmem.open(path) as db:
        db.remember_many(
            [
                {"text": "one", "tags": ["drop", "keep"]},
                {"text": "two", "tags": ["drop"]},
                {"text": "three", "tags": ["project:plugmem"]},
            ]
        )
        first = db.list_tags(limit=1)
        assert [(item.name, item.count) for item in first.items] == [("drop", 2)]
        assert first.next_cursor is not None
        second = db.list_tags(cursor=first.next_cursor, limit=2)
        assert [item.name for item in second.items] == ["keep", "project:plugmem"]
        assert [item.name for item in db.list_tags(prefix="project").items] == [
            "project:plugmem"
        ]

        report = db.remove_tag("drop")
        assert report.affected == 2
        assert [item.name for item in db.list_tags().items] == ["keep", "project:plugmem"]
        assert len(db.export()) == 3
        with pytest.raises(plugmem.StaleCursorError) as raised:
            db.list_tags(cursor=first.next_cursor)
        assert raised.value.code == "PLUGMEM_STALE_CURSOR"
        db.checkpoint()

    with plugmem.Plugmem.open(path, read_only=True) as reader:
        assert [item.name for item in reader.list_tags().items] == [
            "keep",
            "project:plugmem",
        ]
        with pytest.raises(plugmem.ReadOnlyError):
            reader.remove_tag("keep")


def test_workspace_tag_verbs_use_scoped_leases(tmp_path) -> None:
    ws = plugmem.Workspace(str(tmp_path))
    memory = ws.memory("chat")
    memory.remember("one", tags=["drop"])
    assert [(item.name, item.count) for item in memory.list_tags().items] == [
        ("drop", 1)
    ]
    assert memory.remove_tag("drop").affected == 1
    assert memory.list_tags().items == []
    assert memory.stats().facts == 2  # old closed revision + current successor
    ws.close()
