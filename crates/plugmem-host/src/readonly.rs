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
//! Locking is identical to a read-write open (specs/13 §3): the handle
//! holds the same exclusive advisory lock for its whole life, so no
//! cooperating process writes or truncates the file while it is mapped —
//! which is exactly the safety argument for the mmap (see the `unsafe`
//! block in [`ReadOnlyDatabase::open`]).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use memmap2::Mmap;
use plugmem_core::{Config, FactId, Memory, RecallQuery, RecallResult, Stats, Storage};

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
    /// The map and the engine borrowing it. Behind a `Mutex` because
    /// `recall` reuses internal scratch buffers (`&mut Memory`), and to
    /// keep the handle `Sync`.
    mapped: Mutex<MappedMemory>,
    /// Holds the exclusive advisory lock for the handle's whole life
    /// (never read — the lock is the point). Also what makes the mmap
    /// safe: no cooperating writer can touch the file under us.
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
        // The same exclusive lock as a read-write open: one file, one
        // owner (specs/13 §1). The fsync policy is irrelevant — we never
        // write — but the type needs one.
        let mut store = FileStorage::open(path, FsyncPolicy::OnSnapshot)?;
        let base = store.path().to_path_buf();

        // A read-only open must not replay: require a checkpointed
        // database. A cleanly checkpointed journal is byte-empty (the
        // snapshot truncates it to zero); anything else means the caller
        // should fold it in read-write first.
        let journal = store.read_journal()?;
        if !journal.is_empty() {
            return Err(HostError::NeedsCheckpoint { path: base });
        }

        let file = File::open(&base).map_err(|e| HostError::io(&base, e))?;
        // SAFETY: mapping a file is inherently unsafe — a concurrent
        // truncate or overwrite of the mapped file would fault the
        // process (SIGBUS/exception) on the next page access. Our
        // correctness argument (specs/16 §5): this handle holds the
        // exclusive advisory lock (`_store`) for its whole life, and
        // plugmem is single-owner, so no cooperating process writes or
        // truncates the file while the map is live. A foreign `truncate`/
        // `rm` under a live handle is out of contract — the same caveat
        // as corrupting any database file under a running engine.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&base, e))?;
        // The `File` handle is no longer needed: `Mmap` owns the mapping.
        drop(file);

        // Borrow the mapped bytes into the engine. The journal is empty
        // (checked above), so no replay and no copy-on-write.
        let mapped =
            MappedMemory::try_new(map, |map| Memory::from_bytes_borrowed(&map[..], &[], cfg))?;

        Ok(Self {
            mapped: Mutex::new(mapped),
            _store: store,
            path: base,
        })
    }

    /// Runs a recall (specs/04). Same semantics as
    /// [`Database::recall`](crate::Database::recall) minus the embedder:
    /// a text-only query is not auto-embedded, so pass a vector for the
    /// vector source.
    pub fn recall(&self, q: RecallQuery<'_>) -> Result<RecallResult, HostError> {
        let mut mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
        Ok(mapped.with_dependent_mut(|_map, mem| mem.recall(q))?)
    }

    /// An owned copy of one fact, or `None` for unknown/tombstoned ids.
    pub fn get(&self, id: FactId) -> Option<FactSnapshot> {
        let mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
        mapped.borrow_dependent().get(id).map(|v| FactSnapshot {
            record: v.record,
            text: v.text.to_string(),
        })
    }

    /// Engine size counters.
    pub fn stats(&self) -> Stats {
        let mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
        mapped.borrow_dependent().stats()
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
