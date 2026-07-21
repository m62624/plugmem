//! `Database`: the engine + its file + the maintenance policy behind
//! one lock (specs/13 §1, §3).
//!
//! The orchestration model in one paragraph: a `Database` handle is
//! `Clone + Send + Sync` (an `Arc` around a mutex-guarded engine), so
//! any number of threads or agents in one process share one file by
//! cloning the handle — every verb serializes on the mutex, which at
//! microsecond engine calls is a queue, not a bottleneck. A second
//! *process* (or a second `Database` on the same path) is refused with
//! [`HostError::Locked`] by the file lock. Different files are fully
//! independent — open as many `Database`s as you have files.
//!
//! Everything expensive and external — computing embeddings over HTTP —
//! happens **before** the mutex is taken: while one agent waits for its
//! embedding provider, others keep reading and writing.
//!
//! ## Overlay write path (specs/16 §9)
//!
//! Opening a database does **not** copy its snapshot into RAM. `open`
//! memory-maps the snapshot file and the engine *borrows* the mapped pages
//! (an overlay over the base), replaying the journal into a small owned
//! overlay; a mutation lands its appends in an owned tail and copies only
//! the pages it rewrites (per-page copy-on-write in `plugmem-arena`). So a
//! multi-gigabyte database is opened and written to while resident only in
//! the pages it actually touches — the SQLite model (specs/00). A snapshot
//! materializes the base + overlay into a fresh file and **re-maps** it, so
//! the overlay collapses and a long write session stays bounded. A brand-new
//! database has no file to map yet: it opens *owned* and empty, and switches
//! to the mapped overlay at its first snapshot.

use std::fs::File;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use memmap2::Mmap;
use plugmem_core::{
    Config, Error, FactRecord, LinkInput, MaintainReport, Memory, OpenReport, RecallQuery,
    RecallResult, RememberInput, RememberOutcome, Stats, Storage,
};

use crate::embedder::Embedder;
use crate::error::HostError;
use crate::readonly::ReadOnlyDatabase;
use crate::storage::{FileStorage, FsyncPolicy};

self_cell::self_cell!(
    /// Owns the memory map and the overlay [`Memory`] that borrows it — the
    /// read-write sibling of `readonly::MappedMemory`. `self_cell` keeps the
    /// self-reference safe: the only `unsafe` on this path is the inherent
    /// mmap call, not the borrow.
    struct OverlayMap {
        owner: Mmap,
        #[covariant]
        dependent: OverlayMemory,
    }
);

/// The dependent type constructor `self_cell` reborrows per access.
/// [`Memory`] is covariant in its lifetime (its byte pools are
/// `Cow<'a, [u8]>`), so borrowing the map is sound.
type OverlayMemory<'a> = Memory<'a>;

/// The engine backing a live [`Database`]: either an owned in-RAM engine
/// (a brand-new database with no snapshot file yet) or an overlay over a
/// memory-mapped snapshot (the common case). Both are mutable; verbs reach
/// the engine through [`Engine::with`] / [`Engine::read`], which unify the
/// two lifetimes (`'static` vs the map's) behind one closure.
enum Engine {
    /// No snapshot file to map yet — owned and (initially) empty. Switches to
    /// `Mapped` at the first snapshot, once the file exists. Boxed so the
    /// common `Mapped` case does not carry the whole owned engine inline.
    Owned(Box<Memory<'static>>),
    /// Overlay over a memory-mapped snapshot: the base is borrowed, mutations
    /// live in the overlay (owned tail + per-page copy-on-write).
    Mapped(OverlayMap),
}

impl Engine {
    /// Reads through an immutable borrow of the engine (owned or mapped).
    fn read<R>(&self, f: impl for<'a> FnOnce(&Memory<'a>) -> R) -> R {
        match self {
            Engine::Owned(mem) => f(mem),
            Engine::Mapped(map) => f(map.borrow_dependent()),
        }
    }

    /// Mutates the engine and its store together (disjoint borrows). The
    /// closure is higher-ranked over the engine's lifetime so one body serves
    /// both the `'static` owned engine and the map-bound overlay.
    fn with<R>(
        &mut self,
        store: &mut FileStorage,
        f: impl for<'a> FnOnce(&mut Memory<'a>, &mut FileStorage) -> R,
    ) -> R {
        match self {
            Engine::Owned(mem) => f(mem, store),
            Engine::Mapped(map) => map.with_dependent_mut(|_owner, mem| f(mem, store)),
        }
    }
}

/// Opens the engine at `store`'s path (specs/16 §9): memory-maps the snapshot
/// and borrows it as an overlay, replaying the journal. A missing snapshot
/// file (a brand-new database) opens owned and empty — the file appears at the
/// first snapshot. `store` must already hold the exclusive lock.
fn open_engine(store: &mut FileStorage, cfg: &Config) -> Result<(Engine, OpenReport), HostError> {
    let base = store.path().to_path_buf();
    let file = match File::open(&base) {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            // No snapshot file to map yet. The database is owned until the
            // first snapshot writes the file (later opens map it) — but a
            // journal may already exist (mutations before any snapshot), so
            // still replay it into the owned engine.
            let journal = store.read_journal()?;
            let (mem, report) = Memory::from_bytes(None, &journal, cfg.clone())?;
            return Ok((Engine::Owned(Box::new(mem)), report));
        }
        Err(e) => return Err(HostError::io(&base, e)),
    };
    let journal = store.read_journal()?;
    // SAFETY: mapping a file is inherently unsafe — a concurrent truncate or
    // overwrite of the mapped file would fault the process (SIGBUS/exception)
    // on the next page access. Our correctness argument (specs/16 §5): the
    // `store` holds the **exclusive** advisory lock for the whole life of this
    // handle, so no cooperating process writes or truncates the file while the
    // map is live. A foreign `truncate`/`rm` under a live handle is out of
    // contract — the same caveat as corrupting any database file under a
    // running engine.
    let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&base, e))?;
    // The `File` handle is no longer needed: `Mmap` owns the mapping.
    drop(file);
    // Replay the journal into the overlay: no whole-arena clone, only the
    // touched pages copy up. `self_cell` builds the engine borrowing the map;
    // the replay report is captured out of the constructor closure.
    let mut report = None;
    let mapped = OverlayMap::try_new(map, |m| {
        let (mem, rep) = Memory::from_bytes_overlay(&m[..], &journal, cfg.clone())?;
        report = Some(rep);
        Ok::<_, Error>(mem)
    })?;
    Ok((Engine::Mapped(mapped), report.unwrap_or_default()))
}

/// An owned view of one fact — [`Memory::get`] returns borrows that
/// cannot cross the lock, so the database hands out copies.
#[derive(Clone, Debug, PartialEq)]
pub struct FactSnapshot {
    /// The raw record (temporality, flags, references).
    pub record: FactRecord,
    /// The fact text.
    pub text: String,
}

/// One exported fact — the human-readable, id-free shape [`Database::export`]
/// dumps and an importer re-`remember`s (specs/06). Internal ids and
/// `recorded_at` are the engine's bookkeeping and are *not* preserved across
/// a round-trip; the knowledge itself (text, subject name, tags, validity
/// start) is.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportedFact {
    /// The fact text.
    pub text: String,
    /// Subject entity name, if the fact had one.
    pub entity: Option<String>,
    /// Tag strings.
    pub tags: Vec<String>,
    /// When the memory learned it (informational; not restorable on import).
    pub recorded_at: u64,
    /// Validity start — preserved on import.
    pub valid_from: u64,
}

/// Dumps the currently-open facts (skipping closed revisions and
/// tombstones), resolving each subject name and tag string. Shared by the
/// read-write and read-only handles.
pub(crate) fn export_facts(mem: &Memory) -> Vec<ExportedFact> {
    use plugmem_core::{EntityId, FactId, VALID_TO_OPEN};
    let next = mem.stats().next_fact;
    let mut out = Vec::new();
    let mut terms = Vec::new();
    for i in 0..next {
        let id = FactId(i);
        let Some(view) = mem.get(id) else {
            continue; // unknown or tombstoned
        };
        if view.record.valid_to != VALID_TO_OPEN {
            continue; // a closed revision — export the current state only
        }
        let entity = (view.record.entity != EntityId::NONE)
            .then(|| mem.entity_name(view.record.entity))
            .flatten()
            .map(str::to_string);
        terms.clear();
        mem.tags_of(id, &mut terms);
        let tags = terms.iter().map(|t| mem.term(*t).to_string()).collect();
        out.push(ExportedFact {
            text: view.text.to_string(),
            entity,
            tags,
            recorded_at: view.record.recorded_at,
            valid_from: view.record.valid_from,
        });
    }
    out
}

/// Tuning knobs of a [`Database`] (specs/13 §3). Construct through
/// [`Database::builder`].
pub struct DatabaseBuilder {
    cfg: Config,
    fsync: FsyncPolicy,
    snapshot_every_ops: u64,
    snapshot_journal_bytes: u64,
    maintain_every_forgets: Option<u64>,
    embedder: Option<Box<dyn Embedder>>,
}

impl DatabaseBuilder {
    /// Journal fsync policy (default: every operation).
    pub fn fsync(mut self, policy: FsyncPolicy) -> Self {
        self.fsync = policy;
        self
    }

    /// Auto-snapshot after this many mutations (default 1024; `0`
    /// disables the count trigger).
    pub fn snapshot_every_ops(mut self, ops: u64) -> Self {
        self.snapshot_every_ops = ops;
        self
    }

    /// Auto-snapshot when the journal outgrows this many bytes (default
    /// 4 MiB; `0` disables the size trigger).
    pub fn snapshot_journal_bytes(mut self, bytes: u64) -> Self {
        self.snapshot_journal_bytes = bytes;
        self
    }

    /// Optional auto-`maintain` after this many forgets (default off —
    /// maintenance is O(database) and the first pass beyond the HNSW
    /// threshold pays the graph build; see specs/07).
    pub fn maintain_every_forgets(mut self, forgets: u64) -> Self {
        self.maintain_every_forgets = Some(forgets);
        self
    }

    /// The embedding provider. When set (and its `dim() > 0`),
    /// `remember` without a vector embeds the fact text and `recall`
    /// with a text but no vector embeds the query — both outside the
    /// database lock. `Config::dim` must equal the embedder's dimension.
    pub fn embedder(mut self, embedder: Box<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Opens (or creates) the database at `path`.
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] when the file is owned elsewhere;
    /// [`HostError::Engine`] for config/snapshot/journal problems
    /// (including an embedder dimension that disagrees with
    /// `Config::dim`); [`HostError::Io`] for filesystem failures.
    pub fn open(self, path: impl Into<PathBuf>) -> Result<(Database, OpenReport), HostError> {
        if let Some(embedder) = &self.embedder {
            let dim = embedder.dim();
            if dim != 0 && dim != self.cfg.dim {
                return Err(HostError::Engine(Error::ConfigMismatch(
                    "embedder dimension must equal Config::dim",
                )));
            }
        }
        let mut store = FileStorage::open(path, self.fsync)?;
        let (engine, report) = open_engine(&mut store, &self.cfg)?;
        let db = Database {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    engine,
                    store,
                    ops: 0,
                    forgets: 0,
                }),
                embedder: self.embedder.map(Mutex::new),
                cfg: self.cfg,
                snapshot_every_ops: self.snapshot_every_ops,
                snapshot_journal_bytes: self.snapshot_journal_bytes,
                maintain_every_forgets: self.maintain_every_forgets,
            }),
        };
        Ok((db, report))
    }
}

struct Inner {
    state: Mutex<State>,
    embedder: Option<Mutex<Box<dyn Embedder>>>,
    /// Kept to rebuild the overlay engine after a re-map on snapshot.
    cfg: Config,
    snapshot_every_ops: u64,
    snapshot_journal_bytes: u64,
    maintain_every_forgets: Option<u64>,
}

struct State {
    engine: Engine,
    store: FileStorage,
    /// Mutations since the last snapshot.
    ops: u64,
    /// Forgets since the last maintain.
    forgets: u64,
}

/// A clonable, thread-safe handle to one database file. See the module
/// docs for the concurrency model.
#[derive(Clone)]
pub struct Database {
    inner: Arc<Inner>,
}

impl Database {
    /// Opens `path` with every knob at its default and no embedder.
    pub fn open(path: impl Into<PathBuf>, cfg: Config) -> Result<(Self, OpenReport), HostError> {
        Self::builder(cfg).open(path)
    }

    /// Opens `path` read-only over a memory-mapped snapshot (specs/16):
    /// the engine borrows the mapped pages instead of copying the file
    /// into RAM, so a large read-mostly database residents only the pages
    /// `recall`/`get` touch. Requires a checkpointed database (empty
    /// journal) and takes a shared lock (N readers or one writer).
    /// See [`ReadOnlyDatabase`].
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`], [`HostError::NeedsCheckpoint`],
    /// [`HostError::Io`], [`HostError::Engine`] — see
    /// [`ReadOnlyDatabase::open`] semantics.
    pub fn open_readonly(
        path: impl Into<PathBuf>,
        cfg: Config,
    ) -> Result<ReadOnlyDatabase, HostError> {
        ReadOnlyDatabase::open(path, cfg)
    }

    /// Starts a configured open (specs/13 §3 knobs).
    pub fn builder(cfg: Config) -> DatabaseBuilder {
        DatabaseBuilder {
            cfg,
            fsync: FsyncPolicy::default(),
            snapshot_every_ops: 1024,
            snapshot_journal_bytes: 4 * 1024 * 1024,
            maintain_every_forgets: None,
            embedder: None,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panicked verb cannot leave the engine half-mutated (check
        // first, mutate last is the engine's own law), so a poisoned
        // lock is recoverable.
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Embeds `text` outside the state lock, when an embedder is
    /// configured. `None` = leave the input as it was.
    fn embed_one(&self, text: &str) -> Result<Option<Vec<f32>>, HostError> {
        let Some(embedder) = &self.inner.embedder else {
            return Ok(None);
        };
        let mut embedder = embedder.lock().unwrap_or_else(|e| e.into_inner());
        if embedder.dim() == 0 {
            return Ok(None);
        }
        let mut vs = embedder.embed(&[text])?;
        Ok(Some(vs.remove(0)))
    }

    /// Writes a full snapshot and re-maps the fresh file (specs/16 §9).
    ///
    /// Materializes the borrowed base + overlay into an owned buffer, drops
    /// the current map, writes the buffer (tmp + fsync + rename) and clears
    /// the journal, then maps the new file into a fresh overlay. The re-map
    /// collapses the overlay so a long write session stays bounded, and
    /// dropping the map **before** the rename keeps the write portable
    /// (a mapped file cannot be renamed over on Windows).
    fn resnapshot(&self, st: &mut State, now: u64) -> Result<(), HostError> {
        let bytes = st.engine.read(|mem| mem.snapshot_bytes(now));
        // Drop the current map before the rename: park a cheap empty engine.
        // It is replaced by the fresh overlay below — or, if the write fails,
        // rebuilt from the intact on-disk snapshot + journal.
        st.engine = Engine::Owned(Box::new(Memory::new(self.inner.cfg.clone())?));
        let write = st
            .store
            .write_snapshot(&bytes)
            .and_then(|()| st.store.clear_journal());
        // Re-open regardless: on success the fresh file, on failure the
        // untouched old file + journal (journal replay is idempotent, so a
        // failed `clear_journal` does not corrupt state). Then surface the
        // write error, if any.
        let (engine, _) = open_engine(&mut st.store, &self.inner.cfg)?;
        st.engine = engine;
        write
    }

    /// The post-mutation policy hook: counts the op, fires auto-maintain
    /// and auto-snapshot inside the same critical section (specs/13 §3).
    fn after_mutation(&self, st: &mut State, now: u64) -> Result<(), HostError> {
        st.ops += 1;
        if let Some(threshold) = self.inner.maintain_every_forgets
            && st.forgets >= threshold
        {
            let State { engine, store, .. } = &mut *st;
            engine.with(store, |mem, store| mem.maintain(store, now))?;
            st.forgets = 0;
        }
        let by_ops = self.inner.snapshot_every_ops > 0 && st.ops >= self.inner.snapshot_every_ops;
        let by_bytes = self.inner.snapshot_journal_bytes > 0
            && st.store.journal_bytes() >= self.inner.snapshot_journal_bytes;
        if by_ops || by_bytes {
            self.resnapshot(st, now)?;
            st.ops = 0;
        }
        Ok(())
    }

    /// Remembers a fact. Without an explicit vector and with an embedder
    /// configured, the text is embedded first — outside the lock.
    pub fn remember(&self, input: RememberInput<'_>) -> Result<RememberOutcome, HostError> {
        let embedded = match input.vector {
            Some(_) => None,
            None => self.embed_one(input.text)?,
        };
        let input = RememberInput {
            vector: embedded.as_deref().or(input.vector),
            ..input
        };
        let mut st = self.lock();
        let State { engine, store, .. } = &mut *st;
        let out = engine.with(store, |mem, store| mem.remember(store, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(out)
    }

    /// Runs a recall. With a text, no vector and an embedder configured,
    /// the query text is embedded first — outside the lock.
    pub fn recall(&self, q: RecallQuery<'_>) -> Result<RecallResult, HostError> {
        let embedded = match (q.vector, q.text) {
            (None, Some(text)) => self.embed_one(text)?,
            _ => None,
        };
        let q = RecallQuery {
            vector: embedded.as_deref().or(q.vector),
            ..q
        };
        let mut st = self.lock();
        let State { engine, store, .. } = &mut *st;
        Ok(engine.with(store, |mem, _store| mem.recall(q))?)
    }

    /// Revises `target` (same auto-embedding rule as `remember`).
    pub fn revise(
        &self,
        target: plugmem_core::FactId,
        input: RememberInput<'_>,
    ) -> Result<RememberOutcome, HostError> {
        let embedded = match input.vector {
            Some(_) => None,
            None => self.embed_one(input.text)?,
        };
        let input = RememberInput {
            vector: embedded.as_deref().or(input.vector),
            ..input
        };
        let mut st = self.lock();
        let State { engine, store, .. } = &mut *st;
        let out = engine.with(store, |mem, store| mem.revise(store, target, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(out)
    }

    /// Tombstones a fact.
    pub fn forget(&self, now: u64, id: plugmem_core::FactId) -> Result<bool, HostError> {
        let mut st = self.lock();
        let State { engine, store, .. } = &mut *st;
        let fresh = engine.with(store, |mem, store| mem.forget(store, now, id))?;
        st.forgets += 1;
        self.after_mutation(&mut st, now)?;
        Ok(fresh)
    }

    /// Upserts a typed edge.
    pub fn link(&self, input: LinkInput<'_>) -> Result<(), HostError> {
        let mut st = self.lock();
        let State { engine, store, .. } = &mut *st;
        engine.with(store, |mem, store| mem.link(store, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(())
    }

    /// An owned copy of one fact, or `None` for unknown/tombstoned ids.
    pub fn get(&self, id: plugmem_core::FactId) -> Option<FactSnapshot> {
        self.lock().engine.read(|mem| {
            mem.get(id).map(|v| FactSnapshot {
                record: v.record,
                text: v.text.to_string(),
            })
        })
    }

    /// Engine size counters.
    pub fn stats(&self) -> Stats {
        self.lock().engine.read(|mem| mem.stats())
    }

    /// Dumps the currently-open facts for a human-readable backup
    /// (specs/06). See [`ExportedFact`].
    pub fn export(&self) -> Vec<ExportedFact> {
        self.lock().engine.read(export_facts)
    }

    /// Runs a maintenance pass now (purge, compaction, HNSW build past
    /// the threshold — see specs/07 for the cost model).
    pub fn maintain(&self, now: u64) -> Result<MaintainReport, HostError> {
        let mut st = self.lock();
        let State { engine, store, .. } = &mut *st;
        let report = engine.with(store, |mem, store| mem.maintain(store, now))?;
        st.forgets = 0;
        Ok(report)
    }

    /// Writes a full snapshot and clears the journal now (re-mapping the
    /// fresh file — see [`Database::resnapshot`]).
    pub fn checkpoint(&self, now: u64) -> Result<(), HostError> {
        let mut st = self.lock();
        self.resnapshot(&mut st, now)?;
        st.ops = 0;
        Ok(())
    }
}

impl std::fmt::Debug for Database {
    /// Summary only — the contents are the user's memory.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("Database")
            .field("facts", &stats.facts)
            .field("entities", &stats.entities)
            .finish()
    }
}
