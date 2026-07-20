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
use std::io::Write as _;
use std::path::{Path, PathBuf};

use plugmem_core::Storage;

use crate::error::HostError;

/// When journal appends reach the disk (specs/13 §2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// File-backed [`Storage`] holding the exclusive lock on its database.
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
    /// the exclusive lock.
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] when another process (or handle) owns the
    /// lock; [`HostError::Io`] for filesystem failures.
    pub fn open(base: impl Into<PathBuf>, fsync: FsyncPolicy) -> Result<Self, HostError> {
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
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(HostError::Locked { path: base });
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(HostError::io(&lock_path, e));
            }
        }

        // A leftover tmp file is a crashed half-write: the rename never
        // happened, so the snapshot is intact — discard the scrap.
        if tmp_path.exists() {
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
