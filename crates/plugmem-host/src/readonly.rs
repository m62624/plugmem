//! [`ReadOnlyDatabase`]: a zero-copy read-only open over an mmap'd
//! snapshot (specs/16 §3).
//!
//! A normal [`Database`](crate::Database) open reads the whole snapshot
//! into RAM (every byte pool is copied into an arena). For a large,
//! read-mostly database that is wasteful: `open_readonly` maps the
//! snapshot file instead and lets the engine's byte pools *borrow* the
//! mapped pages, so the OS residents only the bytes `recall`/`get`
//! actually touch. An 8 GiB database opens in milliseconds with a few
//! pages resident, not 8 GiB.
//!
//! The handle is read-only by construction — it exposes `recall`/`get`/
//! `stats` and nothing that mutates. It requires a *checkpointed*
//! database (empty journal): replaying a journal would copy whole arenas
//! up (copy-on-write) and defeat the zero-copy intent, so a non-empty
//! journal is refused with [`HostError::NeedsCheckpoint`].
//!
//! Locking is a **shared** advisory lock held for the handle's whole life
//! (specs/13 §3): many read-only handles — in this process or others — map
//! the same file at once, so a read-mostly database serves concurrent
//! readers. A shared lock still excludes every exclusive (read-write)
//! owner, so no cooperating process writes or truncates the file while it
//! is mapped — which is exactly the safety argument for the mmap (see the
//! `unsafe` block in [`ReadOnlyDatabase::open`]).

use std::cell::RefCell;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(feature = "counters")]
use std::sync::Mutex;

use memmap2::Mmap;
use plugmem_core::snapshot::{DEFAULT_SCRUB_BUDGET, ScrubCursor, ScrubProgress, Snapshot};
use plugmem_core::{
    Config, FactId, Memory, RecallQuery, RecallResult, RecallScratch, Stats, Storage,
};

thread_local! {
    /// Per-thread recall scratch — the read-only analog of the one in
    /// [`crate::db`]. `recall` borrows the mapped engine shared (`&Memory`), so
    /// many threads recall one handle at once, each reusing its own scratch.
    static RECALL_SCRATCH: RefCell<RecallScratch> = RefCell::new(RecallScratch::new());
}

use crate::db::FactSnapshot;
use crate::error::HostError;
use crate::storage::{FileStorage, FsyncPolicy};

self_cell::self_cell!(
    /// Owns the memory map and the [`Memory`] that borrows it. `self_cell`
    /// keeps the self-reference safe: the only `unsafe` on this path is
    /// the inherent mmap call, not the borrow.
    struct MappedMemory {
        owner: Mmap,
        #[covariant]
        dependent: BorrowedMemory,
    }
);

/// The dependent type constructor `self_cell` reborrows per access.
/// [`Memory`] is covariant in its lifetime (its byte pools are
/// `Cow<'a, [u8]>`), so borrowing the map is sound.
type BorrowedMemory<'a> = Memory<'a>;

/// A read-only database handle backed by a memory-mapped snapshot
/// (specs/16). See the module docs. `Send + Sync` — share it across
/// threads behind a reference or an `Arc`.
pub struct ReadOnlyDatabase {
    /// The map and the engine borrowing it. Normally no lock: every verb
    /// borrows it shared (`&Memory`) — `recall` keeps its mutable scratch
    /// per-thread — so many threads read one handle concurrently. Under
    /// `counters` the engine embeds the arena's non-`Sync` counter `Cells`, so
    /// it is wrapped in a `Mutex` to stay `Sync` (readers serialize — fine for
    /// that single-threaded perf build). Purely internal: the public API is the
    /// same under every feature.
    #[cfg(not(feature = "counters"))]
    mapped: MappedMemory,
    #[cfg(feature = "counters")]
    mapped: Mutex<MappedMemory>,
    /// Holds the shared advisory lock for the handle's whole life (never
    /// read — the lock is the point). Shared with other readers, but it
    /// excludes every writer, which is what makes the mmap safe: no
    /// cooperating writer can touch the file under us.
    _store: FileStorage,
    /// The database base path.
    path: PathBuf,
}

impl ReadOnlyDatabase {
    /// Opens the database at `path` read-only over an mmap (specs/16 §3).
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] when the file is owned elsewhere;
    /// [`HostError::NeedsCheckpoint`] when the journal is non-empty (open
    /// read-write once to checkpoint, then retry); [`HostError::Io`] when
    /// the snapshot file is missing or cannot be mapped;
    /// [`HostError::Engine`] for a corrupt image or a config mismatch.
    pub(crate) fn open(path: impl Into<PathBuf>, cfg: Config) -> Result<Self, HostError> {
        // A shared lock: coexists with other readers, excludes every
        // writer (specs/13 §1). The fsync policy is irrelevant — we never
        // write — but the type needs one.
        let mut store = FileStorage::open_shared(path, FsyncPolicy::OnSnapshot)?;
        let base = store.path().to_path_buf();

        // A read-only open must not replay: require a checkpointed
        // database. A cleanly checkpointed journal is byte-empty (the
        // snapshot truncates it to zero); anything else means the caller
        // should fold it in read-write first.
        let journal = store.read_journal()?;
        if !journal.is_empty() {
            return Err(HostError::NeedsCheckpoint { path: base });
        }
        let Some(genp) = store.current_snapshot_path()? else {
            // No published generation to map — checkpoint the database first.
            return Err(HostError::NeedsCheckpoint { path: base });
        };

        let file = File::open(&genp).map_err(|e| HostError::io(&genp, e))?;
        // SAFETY: mapping a file is inherently unsafe — a concurrent truncate
        // or overwrite of the mapped file would fault the process
        // (SIGBUS/exception) on the next page access. Our correctness argument
        // (specs/16 §5): a generation file is **immutable** — a checkpoint
        // publishes a new generation and never rewrites this one — and this
        // handle holds a shared lock (`_store`) for its whole life. A foreign
        // `truncate`/`rm` under a live handle is out of contract — the same
        // caveat as corrupting any database file under a running engine.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&genp, e))?;
        // The `File` handle is no longer needed: `Mmap` owns the mapping.
        drop(file);

        // Borrow the mapped bytes into the engine. The journal is empty
        // (checked above), so no replay and no copy-on-write.
        let mapped =
            MappedMemory::try_new(map, |map| Memory::from_bytes_borrowed(&map[..], &[], cfg))?;

        Ok(Self {
            #[cfg(not(feature = "counters"))]
            mapped,
            #[cfg(feature = "counters")]
            mapped: Mutex::new(mapped),
            _store: store,
            path: base,
        })
    }

    /// Runs `f` over the mapped engine (`&Memory`). Normally a lock-free shared
    /// borrow (concurrent readers); under `counters` it takes the `Mutex` first.
    /// Private — the lock strategy never reaches the public API.
    #[cfg(not(feature = "counters"))]
    fn with_mem<R>(&self, f: impl FnOnce(&Memory<'_>) -> R) -> R {
        f(self.mapped.borrow_dependent())
    }

    #[cfg(feature = "counters")]
    fn with_mem<R>(&self, f: impl FnOnce(&Memory<'_>) -> R) -> R {
        let guard = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
        f(guard.borrow_dependent())
    }

    /// Runs a recall (specs/04). Same semantics as
    /// [`Database::recall`](crate::Database::recall) minus the embedder:
    /// a text-only query is not auto-embedded, so pass a vector for the
    /// vector source.
    pub fn recall(&self, q: RecallQuery<'_>) -> Result<RecallResult, HostError> {
        self.with_mem(|mem| {
            RECALL_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let mut out = RecallResult::default();
                mem.recall_into(q, &mut scratch, &mut out)?;
                Ok(out)
            })
        })
    }

    /// An owned copy of one fact, or `None` for unknown/tombstoned ids.
    pub fn get(&self, id: FactId) -> Option<FactSnapshot> {
        self.with_mem(|mem| {
            mem.get(id).map(|v| FactSnapshot {
                record: v.record,
                text: v.text.to_string(),
            })
        })
    }

    /// Engine size counters.
    pub fn stats(&self) -> Stats {
        self.with_mem(|mem| mem.stats())
    }

    /// Runs the on-demand integrity check (specs/16 §9) — the equivalent of
    /// SQLite's `integrity_check`. A read-only open validates only the metadata
    /// (the mapped text and vector pools stay non-resident); this sweeps them
    /// and reports any latent corruption. Reads the whole image, so it residents
    /// the pools it checks.
    ///
    /// # Errors
    ///
    /// [`HostError::Engine`] for the first inconsistency found.
    pub fn verify(&self) -> Result<(), HostError> {
        Ok(self.with_mem(|mem| mem.verify())?)
    }

    /// A resumable byte-level container scrub of the snapshot file, with the
    /// default slice budget (specs/16 §9 — the ZFS-scrub model). See
    /// [`Scrub`] and [`ReadOnlyDatabase::scrub_with_budget`].
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`]/[`HostError::Io`]/[`HostError::Engine`] if the
    /// file cannot be locked, mapped, or structurally parsed for the scan.
    pub fn scrub(&self) -> Result<Scrub, HostError> {
        self.scrub_with_budget(DEFAULT_SCRUB_BUDGET)
    }

    /// A resumable container scrub hashing at most `budget` bytes per
    /// [`Iterator::next`] (specs/16 §9).
    ///
    /// The returned [`Scrub`] owns its own map and its own shared advisory
    /// lock over the same file, so it holds a reader's lock for its whole
    /// life (a writer is refused with [`HostError::Locked`] while any scrub
    /// or read-only handle lives) and can be moved to its own thread — the
    /// caller paces the scan (`next`, pause, resume, cancel) exactly like
    /// the core [`ScrubCursor`]. Dropping it releases the lock.
    ///
    /// It is independent of `self`: the scrub keeps running after this handle
    /// is dropped. A non-empty journal is not an obstacle — the scrub checks
    /// the on-disk snapshot container as-is.
    ///
    /// # Errors
    ///
    /// As [`ReadOnlyDatabase::scrub`].
    pub fn scrub_with_budget(&self, budget: usize) -> Result<Scrub, HostError> {
        Scrub::open(&self.path, budget)
    }

    /// Dumps the currently-open facts for a human-readable backup
    /// (specs/06). See [`ExportedFact`](crate::ExportedFact).
    pub fn export(&self) -> Vec<crate::db::ExportedFact> {
        self.with_mem(crate::db::export_facts)
    }

    /// The database base path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for ReadOnlyDatabase {
    /// Summary only — the contents are the user's memory.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ReadOnlyDatabase")
            .field("path", &self.path)
            .field("facts", &stats.facts)
            .field("entities", &stats.entities)
            .finish()
    }
}

self_cell::self_cell!(
    /// Owns the memory map and the [`ScrubCursor`] that borrows it. As with
    /// [`MappedMemory`], the only `unsafe` is the inherent mmap call, not the
    /// self-reference.
    struct MappedScrub {
        owner: Mmap,
        #[covariant]
        dependent: BorrowedScrub,
    }
);

/// The dependent type constructor. [`ScrubCursor`] is covariant in its
/// lifetime (it borrows the mapped bytes as `&'a [u8]` and owns the rest),
/// so borrowing the map is sound.
type BorrowedScrub<'a> = ScrubCursor<'a>;

/// A resumable, byte-level container scrub over a memory-mapped snapshot
/// (specs/16 §9 — the ZFS-scrub model). Obtained from
/// [`ReadOnlyDatabase::scrub`].
///
/// It implements [`Iterator`]: each [`Iterator::next`] hashes up to the slice
/// budget and yields `Ok(ScrubProgress)`, verifying each section's stored
/// xxh3 as its body completes and the whole-file hash at EOF; the first
/// mismatch yields `Err(HostError::Engine(Error::Corrupt(..)))` and then
/// `None` (fused). Because it only reads the mapped bytes linearly, the pages
/// fault in, get hashed and stay reclaimable — a scrub never residents the
/// whole file.
///
/// It owns a shared advisory lock for its whole life (a reader's lock), so a
/// writer is refused while it lives, and it is [`Send`] — pace it on its own
/// thread. One-shot: obtain a new scrub to scan again.
pub struct Scrub {
    mapped: MappedScrub,
    /// Holds the shared advisory lock for the scrub's whole life (never read
    /// — the lock is the point), independent of the originating handle.
    _store: FileStorage,
}

impl Scrub {
    /// Maps the snapshot at `path` under a fresh shared lock and builds the
    /// cursor. See [`ReadOnlyDatabase::scrub_with_budget`].
    fn open(path: &Path, budget: usize) -> Result<Self, HostError> {
        // A shared lock: coexists with the originating read-only handle and
        // with other readers, excludes every writer. The fsync policy is
        // irrelevant — a scrub never writes — but the type needs one.
        let store = FileStorage::open_shared(path, FsyncPolicy::OnSnapshot)?;
        let base = store.path().to_path_buf();
        let Some(genp) = store.current_snapshot_path()? else {
            return Err(HostError::NeedsCheckpoint { path: base });
        };

        let file = File::open(&genp).map_err(|e| HostError::io(&genp, e))?;
        // SAFETY: identical to `ReadOnlyDatabase::open` — a generation file is
        // immutable (a checkpoint publishes a new one, never rewrites this), and
        // the shared lock in `store` is held for this scrub's whole life
        // (specs/16 §5).
        let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&genp, e))?;
        drop(file);

        let mapped = MappedScrub::try_new(map, |map| {
            Snapshot::parse(&map[..])
                .map(|snap| snap.scrub_with_budget(budget))
                .map_err(HostError::from)
        })?;

        Ok(Self {
            mapped,
            _store: store,
        })
    }
}

impl Iterator for Scrub {
    type Item = Result<ScrubProgress, HostError>;

    /// Hashes the next slice, mapping a core [`Error`](plugmem_core::Error)
    /// mismatch into [`HostError::Engine`]. `None` once complete or fused.
    fn next(&mut self) -> Option<Self::Item> {
        self.mapped
            .with_dependent_mut(|_map, cur| cur.next())
            .map(|step| step.map_err(HostError::from))
    }
}

impl std::fmt::Debug for Scrub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scrub")
            .field("path", &self._store.path())
            .finish_non_exhaustive()
    }
}
