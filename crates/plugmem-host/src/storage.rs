//! `FileStorage`: the engine's `Storage` trait over two files with an
//! exclusive advisory lock (specs/13 §2).
//!
//! Layout for `base = "agent.plugmem"`:
//!
//! | file | role |
//! |---|---|
//! | `agent.plugmem` | the snapshot (the engine's memory image) |
//! | `agent.plugmem.journal` | the append-only journal since it |
//! | `agent.plugmem.lock` | the empty advisory-lock file |
//! | `agent.plugmem.tmp` | scratch for the atomic snapshot replace |
//!
//! Snapshot writes are atomic: the bytes go to the tmp file, the tmp is
//! fsynced, renamed over the snapshot, and the directory is fsynced
//! (unix) — a reader can observe the old image or the new one, never a
//! torn one. The lock is held from `open` until drop; the OS releases
//! it even on abnormal termination.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use plugmem_core::snapshot::SnapshotSink;
use plugmem_core::{Error, Scratch, Storage};

use crate::error::HostError;

/// Maps a filesystem error into the engine's storage-error variant so it can
/// cross the [`SnapshotSink`] boundary (which speaks [`plugmem_core::Error`]).
fn sink_io(e: std::io::Error) -> Error {
    Error::Storage(format!("{e}"))
}

/// When journal appends reach the disk (specs/13 §2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FsyncPolicy {
    /// Fsync after every appended journal record — every acknowledged
    /// mutation survives a power cut. The default: durability is worth
    /// microseconds at this write volume.
    #[default]
    EachOp,
    /// Fsync only at snapshot boundaries. Faster; an OS crash may lose
    /// the journal tail written since the last snapshot.
    OnSnapshot,
}

/// Which advisory lock a [`FileStorage`] takes on open (specs/13 §2).
///
/// `Exclusive` is the writer's lock — one owner, no other handle of either
/// kind. `Shared` is the reader's lock (used by
/// [`ReadOnlyDatabase`](crate::ReadOnlyDatabase)): any number of shared
/// holders coexist, but they mutually exclude every exclusive holder, so a
/// writer can never modify the file while a reader is live. This is the
/// safety guarantee the read-only mmap relies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LockMode {
    Exclusive,
    Shared,
}

/// File-backed [`Storage`] holding an advisory lock on its database.
#[derive(Debug)]
pub struct FileStorage {
    base: PathBuf,
    journal_path: PathBuf,
    tmp_path: PathBuf,
    /// Keeps the advisory lock alive; the handle itself is never read.
    _lock: File,
    /// The journal in append mode, kept open across appends.
    journal: File,
    fsync: FsyncPolicy,
}

impl FileStorage {
    /// Opens (creating as needed) the database files at `base` and takes
    /// the **exclusive** lock — the read-write owner. One handle of any
    /// kind at a time.
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] when another process (or handle) owns the
    /// lock; [`HostError::Io`] for filesystem failures.
    pub fn open(base: impl Into<PathBuf>, fsync: FsyncPolicy) -> Result<Self, HostError> {
        Self::open_with(base, fsync, LockMode::Exclusive)
    }

    /// Opens the database files at `base` and takes a **shared** lock — a
    /// reader. Any number of shared readers coexist; they exclude every
    /// exclusive writer, so the file cannot change under a live reader.
    /// Used by [`ReadOnlyDatabase`](crate::ReadOnlyDatabase); the returned
    /// storage must not be written through (a reader never mutates).
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] when an exclusive writer owns the lock;
    /// [`HostError::Io`] for filesystem failures.
    pub fn open_shared(base: impl Into<PathBuf>, fsync: FsyncPolicy) -> Result<Self, HostError> {
        Self::open_with(base, fsync, LockMode::Shared)
    }

    fn open_with(
        base: impl Into<PathBuf>,
        fsync: FsyncPolicy,
        mode: LockMode,
    ) -> Result<Self, HostError> {
        let base = base.into();
        let lock_path = suffixed(&base, "lock");
        let journal_path = suffixed(&base, "journal");
        let tmp_path = suffixed(&base, "tmp");

        let lock = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| HostError::io(&lock_path, e))?;
        let acquired = match mode {
            LockMode::Exclusive => lock.try_lock(),
            LockMode::Shared => lock.try_lock_shared(),
        };
        match acquired {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(HostError::Locked { path: base });
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(HostError::io(&lock_path, e));
            }
        }

        // A leftover tmp file is a crashed half-write: the rename never
        // happened, so the snapshot is intact — discard the scrap. This is
        // the writer's crash recovery: only an exclusive owner does it. A
        // shared reader must not mutate the directory, and concurrent
        // readers would race on the remove.
        if mode == LockMode::Exclusive && tmp_path.exists() {
            std::fs::remove_file(&tmp_path).map_err(|e| HostError::io(&tmp_path, e))?;
        }

        let journal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&journal_path)
            .map_err(|e| HostError::io(&journal_path, e))?;

        Ok(Self {
            base,
            journal_path,
            tmp_path,
            _lock: lock,
            journal,
            fsync,
        })
    }

    /// The database base path.
    pub fn path(&self) -> &Path {
        &self.base
    }

    /// Current journal size in bytes (drives the snapshot policy).
    pub fn journal_bytes(&self) -> u64 {
        self.journal.metadata().map(|m| m.len()).unwrap_or(0)
    }

    /// Streams a snapshot into the tmp file and fsyncs it, **without**
    /// renaming — the durable-but-not-yet-visible half of an atomic replace
    /// (specs/16 §9). `write` drives the engine's streaming snapshot writer
    /// against a buffered file sink, so the image never all lives in RAM at
    /// once. Split from [`FileStorage::commit_snapshot`] because the caller
    /// must drop the mmap of the old snapshot **between** the two: streaming
    /// reads through that map, but a mapped file cannot be renamed over on
    /// Windows. A staged tmp is cleaned up on the next exclusive open.
    pub(crate) fn stage_snapshot(
        &mut self,
        write: impl FnOnce(&mut FileSink) -> Result<(), HostError>,
    ) -> Result<(), HostError> {
        let file = File::create(&self.tmp_path).map_err(|e| HostError::io(&self.tmp_path, e))?;
        let mut sink = FileSink::new(file, self.tmp_path.clone());
        write(&mut sink)?;
        let file = sink.finish()?;
        file.sync_all()
            .map_err(|e| HostError::io(&self.tmp_path, e))?;
        Ok(())
    }

    /// Renames the staged tmp file over the snapshot and fsyncs the directory
    /// — the visible half of the atomic replace. Call only after
    /// [`FileStorage::stage_snapshot`] and after dropping any mmap of the old
    /// snapshot.
    pub(crate) fn commit_snapshot(&mut self) -> Result<(), HostError> {
        std::fs::rename(&self.tmp_path, &self.base).map_err(|e| HostError::io(&self.base, e))?;
        self.sync_dir()
    }

    /// Fsyncs the directory holding the database (unix only — the
    /// rename's durability point).
    fn sync_dir(&self) -> Result<(), HostError> {
        #[cfg(unix)]
        {
            let dir = self.base.parent().unwrap_or(Path::new("."));
            File::open(dir)
                .and_then(|d| d.sync_all())
                .map_err(|e| HostError::io(dir, e))?;
        }
        Ok(())
    }
}

/// A streaming [`SnapshotSink`] over a buffered file: sequential section
/// writes are buffered, and the single `patch` (the header file-hash, once
/// the running hash is known) flushes and seeks. Lets a snapshot stream to
/// disk without a full-image buffer (specs/16 §9).
pub(crate) struct FileSink {
    buf: BufWriter<File>,
    path: PathBuf,
}

impl FileSink {
    fn new(file: File, path: PathBuf) -> Self {
        Self {
            buf: BufWriter::new(file),
            path,
        }
    }

    /// Flushes the buffer and returns the underlying file for fsync.
    fn finish(self) -> Result<File, HostError> {
        self.buf
            .into_inner()
            .map_err(|e| HostError::io(&self.path, e.into_error()))
    }
}

impl SnapshotSink for &mut FileSink {
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.buf.write_all(bytes).map_err(sink_io)
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        // The one non-sequential write: flush buffered bytes, seek to the
        // header field, patch it, then restore the position to the end.
        self.buf.flush().map_err(sink_io)?;
        let file = self.buf.get_mut();
        file.seek(SeekFrom::Start(at)).map_err(sink_io)?;
        file.write_all(bytes).map_err(sink_io)?;
        file.seek(SeekFrom::End(0)).map_err(sink_io)?;
        Ok(())
    }
}

/// `"a.plugmem"` + `"lock"` → `"a.plugmem.lock"`.
fn suffixed(base: &Path, ext: &str) -> PathBuf {
    let mut s = base.as_os_str().to_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

impl Storage for FileStorage {
    type Error = HostError;

    fn read_snapshot(&mut self) -> Result<Option<Vec<u8>>, HostError> {
        match std::fs::read(&self.base) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(HostError::io(&self.base, e)),
        }
    }

    fn write_snapshot(&mut self, bytes: &[u8]) -> Result<(), HostError> {
        let mut tmp = File::create(&self.tmp_path).map_err(|e| HostError::io(&self.tmp_path, e))?;
        tmp.write_all(bytes)
            .and_then(|()| tmp.sync_all())
            .map_err(|e| HostError::io(&self.tmp_path, e))?;
        drop(tmp);
        std::fs::rename(&self.tmp_path, &self.base).map_err(|e| HostError::io(&self.base, e))?;
        self.sync_dir()
    }

    fn read_journal(&mut self) -> Result<Vec<u8>, HostError> {
        std::fs::read(&self.journal_path).map_err(|e| HostError::io(&self.journal_path, e))
    }

    fn append_journal(&mut self, entry: &[u8]) -> Result<(), HostError> {
        self.journal
            .write_all(entry)
            .map_err(|e| HostError::io(&self.journal_path, e))?;
        if self.fsync == FsyncPolicy::EachOp {
            self.journal
                .sync_data()
                .map_err(|e| HostError::io(&self.journal_path, e))?;
        }
        Ok(())
    }

    fn clear_journal(&mut self) -> Result<(), HostError> {
        // Truncate through a dedicated write handle, not `set_len` on the
        // append handle: Rust opens append handles with FILE_WRITE_DATA
        // masked off (append can only extend, never overwrite), so on
        // Windows `SetEndOfFile` is denied with ERROR_ACCESS_DENIED. A
        // `write + truncate` open empties the file portably; then the
        // append handle is re-established so later appends target the
        // fresh, empty journal.
        let truncated = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.journal_path)
            .map_err(|e| HostError::io(&self.journal_path, e))?;
        truncated
            .sync_data()
            .map_err(|e| HostError::io(&self.journal_path, e))?;
        drop(truncated);
        self.journal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.journal_path)
            .map_err(|e| HostError::io(&self.journal_path, e))?;
        Ok(())
    }
}

/// A host [`Scratch`] over a temp file (specs/16 §9, milestone H): sequential
/// appends go through a buffered writer; [`freeze`](Scratch::freeze) flushes
/// and memory-maps the file, so the staged pool is read (randomly and
/// sequentially) straight from the map instead of RAM. Dropping it unmaps and
/// deletes the temp file.
pub struct FileScratch {
    path: PathBuf,
    /// `Some` while writing, taken by the first `freeze`.
    writer: Option<BufWriter<File>>,
    /// `Some` after `freeze` — the read-back mapping the borrow points into.
    map: Option<Mmap>,
    len: u64,
}

impl FileScratch {
    /// Creates (truncating) a staging file at `path`, ready for appends.
    ///
    /// # Errors
    ///
    /// [`HostError::Io`] if the file cannot be created.
    pub fn create(path: impl Into<PathBuf>) -> Result<Self, HostError> {
        let path = path.into();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| HostError::io(&path, e))?;
        Ok(Self {
            path,
            writer: Some(BufWriter::new(file)),
            map: None,
            len: 0,
        })
    }
}

impl Scratch for FileScratch {
    type Error = HostError;

    fn write(&mut self, bytes: &[u8]) -> Result<(), HostError> {
        let Self {
            writer, path, len, ..
        } = self;
        let w = writer.as_mut().ok_or(HostError::Engine(Error::Invalid(
            "scratch write after freeze",
        )))?;
        w.write_all(bytes).map_err(|e| HostError::io(path, e))?;
        *len += bytes.len() as u64;
        Ok(())
    }

    fn len(&self) -> u64 {
        self.len
    }

    fn freeze(&mut self) -> Result<&[u8], HostError> {
        if self.map.is_none() {
            // Flush and fsync the staged bytes, then map the file fresh.
            let writer = self
                .writer
                .take()
                .ok_or(HostError::Engine(Error::Invalid("scratch frozen twice")))?;
            let file = writer
                .into_inner()
                .map_err(|e| HostError::io(&self.path, e.into_error()))?;
            file.sync_all().map_err(|e| HostError::io(&self.path, e))?;
            drop(file);
            let file = File::open(&self.path).map_err(|e| HostError::io(&self.path, e))?;
            // SAFETY: this is our private temp file — created by `create`,
            // owned by this `FileScratch` for its whole life, deleted on drop —
            // so no other process writes or truncates it under the map (the
            // same argument as the read-only snapshot map, specs/16 §5).
            let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&self.path, e))?;
            self.map = Some(map);
        }
        Ok(&self.map.as_ref().expect("just set")[..])
    }
}

impl Drop for FileScratch {
    fn drop(&mut self) {
        // Unmap before delete: Windows refuses to remove a mapped file (the
        // same constraint as renaming over one).
        self.map = None;
        self.writer = None;
        let _ = std::fs::remove_file(&self.path);
    }
}
