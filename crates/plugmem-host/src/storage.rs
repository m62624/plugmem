//! `FileStorage`: the engine's `Storage` trait over a **versioned** on-disk
//! layout — immutable snapshot generations named by a tiny manifest (specs/13,
//! specs/16). This is what lets a reader map a stable snapshot while a writer
//! keeps working: the writer never overwrites a live file, it publishes a new
//! generation and repoints the manifest.
//!
//! Layout for `base = "agent.plugmem"`:
//!
//! | file | role |
//! |---|---|
//! | `agent.plugmem` | the **manifest** — a tiny record naming the current snapshot generation |
//! | `agent.plugmem.snap.<N>` | **generation N** — an immutable full snapshot image; never rewritten |
//! | `agent.plugmem.journal` | the append-only journal since the current generation |
//! | `agent.plugmem.lock` | the advisory-lock file (writer-vs-writer) |
//! | `agent.plugmem.snap.<N>.tmp`, `agent.plugmem.manifest.tmp` | staging for the atomic writes |
//!
//! A checkpoint streams the fresh image to `…snap.<N+1>.tmp`, fsyncs it,
//! renames it to `…snap.<N+1>` (an immutable file, never overwritten), then
//! atomically repoints the manifest (tmp + fsync + rename + directory fsync).
//! The old generation is reclaimed once nothing maps it. A reader always
//! observes a manifest pointing at a generation that already exists on disk.
//! The lock is held from `open` until drop; the OS releases it even on
//! abnormal termination.

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
    /// The manifest path (what the caller points at).
    base: PathBuf,
    journal_path: PathBuf,
    /// Staging path for the atomic manifest publish.
    manifest_tmp: PathBuf,
    /// The generation the manifest currently names; `0` = no snapshot yet.
    current_gen: u64,
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
        let manifest_tmp = suffixed(&base, "manifest.tmp");

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

        let current_gen = read_manifest(&base)?.unwrap_or(0);

        // Crash recovery: discard scraps of a checkpoint that never published
        // (orphan generation files and staging tmps). Only the exclusive owner
        // does it — a shared reader must not mutate the directory, and the
        // manifest always names a generation that already exists on disk.
        if mode == LockMode::Exclusive {
            cleanup_orphans(
                &base,
                if current_gen == 0 {
                    None
                } else {
                    Some(current_gen)
                },
            )?;
        }

        let journal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&journal_path)
            .map_err(|e| HostError::io(&journal_path, e))?;

        Ok(Self {
            base,
            journal_path,
            manifest_tmp,
            current_gen,
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

    /// The path of the snapshot file the manifest currently names, or `None`
    /// for a fresh database with no published snapshot. Callers that map the
    /// snapshot (`open_engine`, `ReadOnlyDatabase`, `Scrub`, `recover`) resolve
    /// through this instead of mapping `base` (which is now the manifest).
    pub(crate) fn current_snapshot_path(&self) -> Result<Option<PathBuf>, HostError> {
        Ok(read_manifest(&self.base)?.map(|g| gen_path(&self.base, g)))
    }

    /// The next generation number this storage will publish.
    fn next_gen(&self) -> u64 {
        self.current_gen + 1
    }

    /// Streams a snapshot into the next generation's tmp file and fsyncs it,
    /// **without** publishing — the durable-but-not-yet-visible half of a
    /// checkpoint (specs/16 §9). `write` drives the engine's streaming snapshot
    /// writer against a buffered file sink, so the image never all lives in RAM
    /// at once. Split from [`FileStorage::commit_snapshot`] because the caller
    /// must drop the mmap of the *old* generation **between** the two: staging
    /// reads through that map, and the reclaim in `commit` deletes it. A staged
    /// tmp is cleaned up on the next exclusive open.
    pub(crate) fn stage_snapshot(
        &mut self,
        write: impl FnOnce(&mut FileSink) -> Result<(), HostError>,
    ) -> Result<(), HostError> {
        let tmp = gen_tmp_path(&self.base, self.next_gen());
        let file = File::create(&tmp).map_err(|e| HostError::io(&tmp, e))?;
        let mut sink = FileSink::new(file, tmp.clone());
        write(&mut sink)?;
        let file = sink.finish()?;
        file.sync_all().map_err(|e| HostError::io(&tmp, e))?;
        Ok(())
    }

    /// Publishes the staged generation: rename its tmp to the immutable
    /// `snap.<N+1>`, repoint the manifest, then reclaim the previous
    /// generation. Call only after [`FileStorage::stage_snapshot`] and after
    /// dropping any mmap of the old generation.
    pub(crate) fn commit_snapshot(&mut self) -> Result<(), HostError> {
        let next = self.next_gen();
        let tmp = gen_tmp_path(&self.base, next);
        let genp = gen_path(&self.base, next);
        std::fs::rename(&tmp, &genp).map_err(|e| HostError::io(&genp, e))?;
        sync_dir(&self.base)?;
        publish_manifest(&self.base, &self.manifest_tmp, next)?;
        let prev = self.current_gen;
        self.current_gen = next;
        self.reclaim_previous(prev);
        Ok(())
    }

    /// Reclaims a superseded generation (single-writer layout: nothing else
    /// maps it once the writer has re-mapped the new one). Best-effort — a
    /// failed delete just leaves a scrap the next exclusive open sweeps.
    fn reclaim_previous(&self, generation: u64) {
        if generation >= 1 {
            let _ = std::fs::remove_file(gen_path(&self.base, generation));
        }
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

/// Manifest magic ("PMGL" — distinct from the snapshot's own `MAGIC`).
const MANIFEST_MAGIC: u32 = 0x504D_474C;
/// On-disk manifest version (the layout, not the snapshot format).
const MANIFEST_VERSION: u16 = 1;
/// Manifest length: magic(4) + version(2) + pad(2) + gen(8) + checksum(8).
const MANIFEST_LEN: usize = 24;

/// 64-bit FNV-1a — a dependency-free integrity check for the manifest. The
/// manifest is written atomically (tmp + rename), so it can never be torn; this
/// only catches external garbage / bit-rot in the tiny fixed record.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The snapshot file for generation `n`: `base` + `.snap.<n>`.
fn gen_path(base: &Path, n: u64) -> PathBuf {
    suffixed(base, &format!("snap.{n}"))
}

/// The staging path for generation `n`: `base` + `.snap.<n>.tmp`.
fn gen_tmp_path(base: &Path, n: u64) -> PathBuf {
    suffixed(base, &format!("snap.{n}.tmp"))
}

/// Reads and validates the manifest at `base`. `Ok(None)` when it is absent (a
/// fresh database); `Err(Corrupt)` when it is present but malformed; `Err(Io)`
/// on a real filesystem failure. The returned generation is always ≥ 1.
fn read_manifest(base: &Path) -> Result<Option<u64>, HostError> {
    let bytes = match std::fs::read(base) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(HostError::io(base, e)),
    };
    let ok = bytes.len() == MANIFEST_LEN
        && u32::from_le_bytes(bytes[0..4].try_into().unwrap()) == MANIFEST_MAGIC
        && u16::from_le_bytes(bytes[4..6].try_into().unwrap()) == MANIFEST_VERSION
        && fnv1a(&bytes[0..16]) == u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if !ok {
        return Err(HostError::Engine(Error::Corrupt("manifest is corrupt")));
    }
    Ok(Some(u64::from_le_bytes(bytes[8..16].try_into().unwrap())))
}

/// Atomically publishes `gen` as the current generation: write a fresh manifest
/// to `manifest_tmp`, fsync, rename over `base`, fsync the directory.
fn publish_manifest(base: &Path, manifest_tmp: &Path, generation: u64) -> Result<(), HostError> {
    let mut buf = [0u8; MANIFEST_LEN];
    buf[0..4].copy_from_slice(&MANIFEST_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&MANIFEST_VERSION.to_le_bytes());
    buf[8..16].copy_from_slice(&generation.to_le_bytes());
    let sum = fnv1a(&buf[0..16]);
    buf[16..24].copy_from_slice(&sum.to_le_bytes());
    let mut f = File::create(manifest_tmp).map_err(|e| HostError::io(manifest_tmp, e))?;
    f.write_all(&buf)
        .and_then(|()| f.sync_all())
        .map_err(|e| HostError::io(manifest_tmp, e))?;
    drop(f);
    std::fs::rename(manifest_tmp, base).map_err(|e| HostError::io(base, e))?;
    sync_dir(base)
}

/// Fsyncs the directory holding the database (unix only — the rename's
/// durability point).
fn sync_dir(base: &Path) -> Result<(), HostError> {
    #[cfg(unix)]
    {
        let dir = base.parent().filter(|p| !p.as_os_str().is_empty());
        let dir = dir.unwrap_or_else(|| Path::new("."));
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| HostError::io(dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = base;
    Ok(())
}

/// Removes crash debris left by an interrupted checkpoint: every `snap.<n>`
/// (and its `.tmp`) whose generation is not `keep`, plus the manifest tmp. Only
/// the exclusive (writer) owner calls this — in the single-writer layout the
/// manifest's generation is the only live snapshot; anything else is a scrap
/// from a checkpoint that never published. `keep = None` (a fresh database with
/// no manifest) sweeps every generation file.
fn cleanup_orphans(base: &Path, keep: Option<u64>) -> Result<(), HostError> {
    let dir = base
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = base
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let prefix = format!("{name}.snap.");
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(HostError::io(dir, e)),
    };
    for entry in entries {
        let entry = entry.map_err(|e| HostError::io(dir, e))?;
        let fname = entry.file_name();
        let Some(fname) = fname.to_str() else {
            continue;
        };
        let Some(rest) = fname.strip_prefix(&prefix) else {
            continue;
        };
        let num = rest.strip_suffix(".tmp").unwrap_or(rest);
        match num.parse::<u64>() {
            // The live generation's committed file survives; its stale tmp does not.
            Ok(n) if keep == Some(n) && !rest.ends_with(".tmp") => {}
            Ok(_) => {
                let _ = std::fs::remove_file(entry.path());
            }
            Err(_) => {}
        }
    }
    let _ = std::fs::remove_file(suffixed(base, "manifest.tmp"));
    Ok(())
}

impl Storage for FileStorage {
    type Error = HostError;

    fn read_snapshot(&mut self) -> Result<Option<Vec<u8>>, HostError> {
        match self.current_snapshot_path()? {
            Some(p) => Ok(Some(std::fs::read(&p).map_err(|e| HostError::io(&p, e))?)),
            None => Ok(None),
        }
    }

    fn write_snapshot(&mut self, bytes: &[u8]) -> Result<(), HostError> {
        // Publish a new immutable generation (the non-streaming path): stage
        // its tmp, rename to snap.<N+1>, repoint the manifest, reclaim the old.
        let next = self.next_gen();
        let tmp = gen_tmp_path(&self.base, next);
        let mut f = File::create(&tmp).map_err(|e| HostError::io(&tmp, e))?;
        f.write_all(bytes)
            .and_then(|()| f.sync_all())
            .map_err(|e| HostError::io(&tmp, e))?;
        drop(f);
        let genp = gen_path(&self.base, next);
        std::fs::rename(&tmp, &genp).map_err(|e| HostError::io(&genp, e))?;
        sync_dir(&self.base)?;
        publish_manifest(&self.base, &self.manifest_tmp, next)?;
        let prev = self.current_gen;
        self.current_gen = next;
        self.reclaim_previous(prev);
        Ok(())
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
