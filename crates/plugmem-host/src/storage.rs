//! `FileStorage`: the engine's `Storage` trait over a **versioned** on-disk
//! layout — immutable snapshot generations named by a tiny manifest (
//! ). This is what lets a reader map a stable snapshot while a writer
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

/// When journal appends reach the disk.
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
    /// While `true`, `append_journal` skips its per-record fsync so a bulk write
    /// amortizes durability into one [`sync_journal`](Self::sync_journal) at the
    /// end (see [`set_batch`](Self::set_batch)). Only meaningful under `EachOp`.
    batch: bool,
}

impl FileStorage {
    /// Opens (creating as needed) the database at `base` and takes the
    /// **exclusive writer lock** — one writer at a time. Readers do *not* take
    /// this lock (they pin a generation file via
    /// `pin_current_generation` instead), so a writer and any number of
    /// readers coexist across threads or processes (the versioned
    /// MVCC layout).
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] when another **writer** owns the lock;
    /// [`HostError::Io`] for filesystem failures.
    pub fn open(base: impl Into<PathBuf>, fsync: FsyncPolicy) -> Result<Self, HostError> {
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
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(HostError::Locked { path: base });
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(HostError::io(&lock_path, e));
            }
        }

        let current_gen = read_manifest(&base)?.unwrap_or(0);

        // Crash recovery + GC: drop unpublished orphan generations and staging
        // tmps, and reclaim any unpinned superseded generation. Safe with live
        // readers — reclaim is pin-aware (a reader's shared lock keeps its
        // generation). Only the writer sweeps.
        sweep_generations(&base, current_gen)?;

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
            batch: false,
        })
    }

    /// Enters (`true`) or leaves (`false`) **batch mode**. While on,
    /// [`append_journal`](plugmem_core::Storage::append_journal) writes each
    /// record **without** its per-record fsync, so a bulk write (`remember_many`)
    /// amortizes durability into a single [`sync_journal`](Self::sync_journal) at
    /// the end. Only affects the `EachOp` policy — `OnSnapshot` never fsyncs per
    /// record anyway. The caller must pair `set_batch(true)` with a final
    /// `set_batch(false)` + `sync_journal`, even on error, so a later single
    /// write is durable again.
    pub(crate) fn set_batch(&mut self, on: bool) {
        self.batch = on;
    }

    /// Fsyncs the journal now — the durability point for records appended in
    /// batch mode. Idempotent (a plain `sync_data`), so calling it after a
    /// partially-written batch makes exactly the records that reached the file
    /// durable.
    pub(crate) fn sync_journal(&mut self) -> Result<(), HostError> {
        self.journal
            .sync_data()
            .map_err(|e| HostError::io(&self.journal_path, e))
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
    /// checkpoint. `write` drives the engine's streaming snapshot
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
    /// `snap.<N+1>`, repoint the manifest, then GC superseded generations
    /// (pin-aware). Call only after [`FileStorage::stage_snapshot`] and after
    /// dropping any mmap of the old generation.
    pub(crate) fn commit_snapshot(&mut self) -> Result<(), HostError> {
        let next = self.next_gen();
        let tmp = gen_tmp_path(&self.base, next);
        let genp = gen_path(&self.base, next);
        std::fs::rename(&tmp, &genp).map_err(|e| HostError::io(&genp, e))?;
        sync_dir(&self.base)?;
        publish_manifest(&self.base, &self.manifest_tmp, next)?;
        self.current_gen = next;
        // Reclaim every unpinned superseded generation (a reader on an old one
        // keeps it until it drops). Best-effort — leftovers go on the next pass.
        let _ = sweep_generations(&self.base, self.current_gen);
        Ok(())
    }
}

/// A streaming [`SnapshotSink`] over a buffered file: sequential section
/// writes are buffered, and the single `patch` (the header file-hash, once
/// the running hash is known) flushes and seeks. Lets a snapshot stream to
/// disk without a full-image buffer.
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

/// Whether a database exists at `base`.
///
/// Not `base.exists()`: the base path holds the published snapshot, and that is
/// written by the *first checkpoint*. A database created a moment ago and
/// written to is a journal and a lock with no base file at all, and it is
/// unmistakably a database — reopening it replays the journal and the facts are
/// there. Anything asking "is there a database here?" (a workspace listing a
/// directory, a caller deciding whether to create one) must ask this rather
/// than stat the base path, or it will overwrite live data.
///
/// The lock file alone does not count: it can outlive a database that was
/// deleted, and a bare lock has no content to lose.
pub fn database_exists(base: &Path) -> bool {
    base.exists() || suffixed(base, "journal").exists()
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

/// How many times a reader retries an open across the Windows manifest-swap
/// window. Publishing the manifest is a `rename(tmp, base)`, which on Windows
/// briefly leaves the old `base` *delete-pending*; a reader opening it in that
/// window gets a transient `ERROR_ACCESS_DENIED` (5) / `ERROR_SHARING_VIOLATION`
/// (32). POSIX renames are atomic with respect to a concurrent open, so this
/// never happens there. `100 * 1ms` dwarfs the microsecond swap while adding a
/// negligible delay to a genuine failure.
#[cfg(windows)]
const SHARE_RETRIES: u32 = 100;

/// True for the transient Windows sharing errors a rename-replace throws at a
/// concurrent open. Always `false` off Windows (codes 5/32 mean unrelated things
/// on POSIX), so the retry paths below collapse to the original single shot.
fn is_transient_share_error(e: &std::io::Error) -> bool {
    #[cfg(windows)]
    {
        matches!(e.raw_os_error(), Some(5) | Some(32))
    }
    #[cfg(not(windows))]
    {
        let _ = e;
        false
    }
}

/// `std::fs::read`, retried across the transient Windows rename-swap window so a
/// reader rides over a concurrent manifest publish instead of spuriously failing
/// with "Access is denied". On non-Windows it is a plain `std::fs::read`.
fn read_across_rename(path: &Path) -> std::io::Result<Vec<u8>> {
    #[cfg(windows)]
    {
        let mut attempts = 0u32;
        loop {
            match std::fs::read(path) {
                Err(e) if is_transient_share_error(&e) && attempts < SHARE_RETRIES => {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                other => return other,
            }
        }
    }
    #[cfg(not(windows))]
    {
        std::fs::read(path)
    }
}

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
pub(crate) fn read_manifest(base: &Path) -> Result<Option<u64>, HostError> {
    let bytes = match read_across_rename(base) {
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

/// Reclaims one superseded generation file, but only if nothing pins it. A
/// reader holds a **shared** lock on the generation file for as long as it maps
/// it (see `pin_current_generation`), so a successful **exclusive** try-lock
/// proves no reader is using it, and the delete under that lock cannot race a
/// new pin. Best-effort: a pinned (or, on Windows, an open-mapped) generation is
/// simply left for a later pass. Never deletes a live reader's snapshot.
fn try_reclaim_generation(genp: &Path) {
    if let Ok(f) = File::open(genp)
        && f.try_lock().is_ok()
    {
        let _ = std::fs::remove_file(genp);
    }
}

/// Sweeps the generation files around `current`: the current one stays; a
/// higher number is crash debris from a checkpoint that never published its
/// manifest (never pinned — delete it); a lower number is a superseded
/// generation reclaimed only if unpinned (a reader may still map it). Staging
/// `.tmp`s and the manifest tmp go unconditionally. Called by the exclusive
/// writer — on open (crash recovery) and after each checkpoint (GC).
fn sweep_generations(base: &Path, current: u64) -> Result<(), HostError> {
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
        if rest.ends_with(".tmp") {
            let _ = std::fs::remove_file(entry.path()); // staging scrap
            continue;
        }
        match rest.parse::<u64>() {
            Ok(n) if n == current => {} // the live generation
            Ok(n) if n > current => {
                let _ = std::fs::remove_file(entry.path()); // unpublished orphan
            }
            Ok(_) => try_reclaim_generation(&entry.path()), // superseded — pin-aware
            Err(_) => {}
        }
    }
    let _ = std::fs::remove_file(suffixed(base, "manifest.tmp"));
    Ok(())
}

/// Opens and **shared-locks** the current snapshot generation, pinning it
/// against the writer's GC for as long as the returned [`File`] is held; returns
/// it with the generation path and the generation number it pinned. `Ok(None)`
/// when there is no published generation (a fresh database). Readers do not take
/// the writer lock, so a reader and the writer coexist — this is what makes the
/// cross-process MVCC work.
///
/// Retries the open→lock race with the collector: if the manifest names a
/// generation that GC reclaims in the window between resolving and locking it,
/// the open (or the post-lock existence recheck) fails and we retry against the
/// fresh manifest. Once the shared lock is held and the file still exists, GC's
/// exclusive try-lock must fail, so the pin is stable. The returned number is the
/// generation actually pinned, which a caller can compare against a stale one to
/// tell whether the writer has published a newer snapshot (see
/// [`ReadOnlyDatabase::refresh`](crate::ReadOnlyDatabase::refresh)).
pub(crate) fn pin_current_generation(
    base: &Path,
) -> Result<Option<(File, PathBuf, u64)>, HostError> {
    loop {
        let Some(generation) = read_manifest(base)? else {
            return Ok(None);
        };
        let genp = gen_path(base, generation);
        let file = match File::open(&genp) {
            Ok(f) => f,
            // GC reclaimed it between the manifest read and the open — retry.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            // Windows: a gen file being GC-reclaimed can surface as a transient
            // delete-pending share error rather than NotFound — also a retry.
            Err(e) if is_transient_share_error(&e) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            Err(e) => return Err(HostError::io(&genp, e)),
        };
        match file.try_lock_shared() {
            Ok(()) => {}
            // A transient exclusive holder (GC probing) — retry.
            Err(std::fs::TryLockError::WouldBlock) => continue,
            Err(std::fs::TryLockError::Error(e)) => return Err(HostError::io(&genp, e)),
        }
        // Confirm the generation survived the open→lock window. If GC reclaimed
        // it just before we locked, the path is gone; drop the lock and retry
        // for the fresh generation. If it exists, our shared lock now blocks
        // GC's exclusive try-lock, so the pin holds.
        if genp.exists() {
            return Ok(Some((file, genp, generation)));
        }
    }
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
        self.current_gen = next;
        let _ = sweep_generations(&self.base, self.current_gen);
        Ok(())
    }

    fn read_journal(&mut self) -> Result<Vec<u8>, HostError> {
        std::fs::read(&self.journal_path).map_err(|e| HostError::io(&self.journal_path, e))
    }

    fn append_journal(&mut self, entry: &[u8]) -> Result<(), HostError> {
        self.journal
            .write_all(entry)
            .map_err(|e| HostError::io(&self.journal_path, e))?;
        // In batch mode the fsync is deferred to one `sync_journal` at the end
        // of the batch (durability amortized across the whole bulk write).
        if self.fsync == FsyncPolicy::EachOp && !self.batch {
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

/// A host [`Scratch`] over a temp file (milestone H): sequential
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
            // same argument as the read-only snapshot map).
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
