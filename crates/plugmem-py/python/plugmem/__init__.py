"""plugmem: an embedded bitemporal memory and retrieval engine.

The whole surface comes from the compiled `plugmem._plugmem` module and is
re-exported here. Only what genuinely belongs in Python lives in this file:
generators that wrap a native paging verb, and nothing else. A verb implemented
twice is a verb that will eventually mean two things.
"""

from __future__ import annotations

from collections.abc import Iterator

from ._plugmem import (
    # Exceptions
    BusyError,
    ClosedError,
    ConfigError,
    # Result mirrors
    DbEntry,
    EngineError,
    ExportedEdge,
    ExportedFact,
    ExportPage,
    FactRecord,
    FactSnapshot,
    InternalError,
    InvalidArgError,
    InvalidNameError,
    LockedError,
    MaintainReport,
    NeedsCheckpointError,
    OpenError,
    # Classes
    Plugmem,
    PlugmemError,
    ReadOnlyError,
    RecalledEdge,
    RecalledFact,
    RecallResult,
    RecoverReport,
    ReindexReport,
    RememberOutcome,
    Scrub,
    ScrubProgress,
    SettingHelpItem,
    SettingsHelpResult,
    Similar,
    Stats,
    Workspace,
    WorkspaceProblem,
    WriterOnlyError,
    # Module functions
    about,
    recover,
    settings_help,
    skill,
    skill_full,
    skill_version,
    version,
)


def export_pages(db: Plugmem) -> Iterator[ExportPage]:
    """Walk every open fact one bounded page at a time.

    Sugar over `Plugmem.export_page`, not a second export: it calls the same
    verb and threads the same cursor. Written in Python because a generator is
    the natural shape for "keep asking until the cursor runs out", and because
    building one in Rust would add a class whose only job is to remember a
    number.

        for page in plugmem.export_pages(db):
            for fact in page.facts:
                ...
    """
    cursor: int | None = None
    while True:
        page = db.export_page(cursor)
        yield page
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


__all__ = [
    "BusyError",
    "ClosedError",
    "ConfigError",
    "DbEntry",
    "EngineError",
    "ExportPage",
    "ExportedEdge",
    "ExportedFact",
    "FactRecord",
    "FactSnapshot",
    "InternalError",
    "InvalidArgError",
    "InvalidNameError",
    "LockedError",
    "MaintainReport",
    "NeedsCheckpointError",
    "OpenError",
    "Plugmem",
    "PlugmemError",
    "ReadOnlyError",
    "RecallResult",
    "RecalledEdge",
    "RecalledFact",
    "RecoverReport",
    "ReindexReport",
    "RememberOutcome",
    "Scrub",
    "ScrubProgress",
    "SettingHelpItem",
    "SettingsHelpResult",
    "Similar",
    "Stats",
    "Workspace",
    "WorkspaceProblem",
    "WriterOnlyError",
    "about",
    "export_pages",
    "recover",
    "settings_help",
    "skill",
    "skill_full",
    "skill_version",
    "version",
]
