//! `Database`: the engine + its file-backed storage layout + the maintenance
//! policy behind one lock (§3).
//!
//! The orchestration model in one paragraph: a `Database` handle is
//! `Clone + Send + Sync` (an `Arc` around an `RwLock`-guarded engine), so
//! any number of threads or agents in one process share one local database by
//! cloning the handle — the read verbs (`recall`/`get`/`stats`/…) run
//! concurrently under a shared guard, the write verbs serialize under an
//! exclusive one; at microsecond engine calls neither is a bottleneck. A
//! second *process* (or a second `Database` on the same path) is refused
//! with [`HostError::Locked`] by the file lock. Different files are fully
//! independent — open as many `Database`s as you have files.
//!
//! Everything expensive and external — computing embeddings over HTTP —
//! happens **before** the lock is taken: while one agent waits for its
//! embedding provider, others keep reading and writing.
//!
//! (Under the `counters` perf-gate feature the engine's instrumentation
//! `Cell`s are not `Sync`, so the lock falls back to a `Mutex` and reads
//! serialize — a single-threaded measurement build; the public API is
//! unchanged. See `StateLock`.)
//!
//! ## Overlay write path
//!
//! Opening a database does **not** copy its snapshot into RAM. `open`
//! memory-maps the snapshot file and the engine *borrows* the mapped pages
//! (an overlay over the base), replaying the journal into a small owned
//! overlay; a mutation lands its appends in an owned tail and copies only
//! the pages it rewrites (per-page copy-on-write in `plugmem-arena`). So a
//! multi-gigabyte database is opened and written to while resident only in
//! the pages it actually touches. A snapshot
//! materializes the base + overlay into a fresh file and **re-maps** it, so
//! the overlay collapses and a long write session stays bounded. A brand-new
//! database has no file to map yet: it opens *owned* and empty, and switches
//! to the mapped overlay at its first snapshot.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock as EmbedderLock};
#[cfg(feature = "counters")]
use std::sync::{Mutex, MutexGuard};
#[cfg(not(feature = "counters"))]
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use memmap2::Mmap;
use plugmem_core::{
    Config, Error, FactFault, FactRecord, GuardedRememberOutcome, LinkInput, MaintainReport,
    MaintenanceMode, MaintenanceOptions, MemStorage, Memory, OpenReport, RecallQuery, RecallResult,
    RecallScratch, ReembedError, ReembedReport, RememberInput, RememberOutcome, RemoveTagReport,
    Stats, Storage, TagPage, TagQuery, UnlinkInput,
};

thread_local! {
    /// Per-thread recall scratch. `recall` takes `&self` on the engine, so many
    /// reader threads recall one [`Database`] at once; each reuses its own
    /// scratch here (zero re-alloc after warm-up, no lock on the hot path).
    static RECALL_SCRATCH: RefCell<RecallScratch> = RefCell::new(RecallScratch::new());
}

use crate::embedder::Embedder;
use crate::error::HostError;
use crate::readonly::{ReadOnlyDatabase, Scrub};
use crate::storage::{FileScratch, FileStorage, FsyncPolicy};

/// Default maximum number of fact texts sent to the embedding provider in one
/// explicit reembed request.
pub const DEFAULT_REEMBED_BATCH_SIZE: usize = 128;

/// Automatically produced vectors plus the provider identity that owns them.
type EmbeddedBatch = (Vec<Vec<f32>>, String);

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

/// Opens the engine at `store`'s path: memory-maps the snapshot
/// and borrows it as an overlay, replaying the journal. A missing snapshot
/// file (a brand-new database) opens owned and empty — the file appears at the
/// first snapshot. `store` must already hold the exclusive lock.
fn open_engine(store: &mut FileStorage, cfg: &Config) -> Result<(Engine, OpenReport), HostError> {
    let journal = store.read_journal()?;
    let Some(genp) = store.current_snapshot_path()? else {
        // No published generation yet. The database is owned until the first
        // checkpoint publishes one — but a journal may already exist (mutations
        // before any snapshot), so still replay it into the owned engine.
        let (mem, report) = Memory::from_bytes(None, &journal, cfg.clone())?;
        return Ok((Engine::Owned(Box::new(mem)), report));
    };
    let file = File::open(&genp).map_err(|e| HostError::io(&genp, e))?;
    // SAFETY: mapping a file is inherently unsafe — a concurrent truncate or
    // overwrite of the mapped file would fault the process (SIGBUS/exception)
    // on the next page access. Our correctness argument: the
    // generation file is **immutable** (a checkpoint publishes a new one and
    // never rewrites this), and the `store` holds the exclusive writer lock, so
    // nothing overwrites it under the map. A foreign `truncate`/`rm` under a
    // live handle is out of contract — the same caveat as corrupting any
    // database file under a running engine.
    let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&genp, e))?;
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FactSnapshot {
    /// The raw record (temporality, flags, references).
    pub record: FactRecord,
    /// The fact text.
    pub text: String,
    /// The fact's metadata as a sorted key→value map (empty when the fact
    /// carries none). The engine stores it opaquely; this is the decoded view.
    pub metadata: BTreeMap<String, String>,
}

/// One exported fact — the human-readable, id-free shape [`Database::export`]
/// dumps and an importer re-`remember`s. Internal ids and
/// `recorded_at` are the engine's bookkeeping and are *not* preserved across
/// a round-trip; the knowledge itself (text, subject name, tags, validity
/// start) is.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExportedFact {
    /// The fact's id **in the database it came from**.
    ///
    /// Informational, and deliberately not restored on import: a fresh database
    /// assigns its own. It is here because edges reference their provenance
    /// fact by id, so a dump that carries edges needs something for them to
    /// point at — an importer translates old id to new as it goes.
    pub id: u32,
    /// The fact text.
    pub text: String,
    /// Subject entity name, if the fact had one.
    pub entity: Option<String>,
    /// Tag strings.
    pub tags: Vec<String>,
    /// Metadata as a sorted key→value map (empty when none) — preserved on
    /// import.
    pub metadata: BTreeMap<String, String>,
    /// When the memory learned it (informational; not restorable on import).
    pub recorded_at: u64,
    /// Validity start — preserved on import.
    pub valid_from: u64,
}

/// One bounded page of currently-open facts.
///
/// `next_cursor` is the next fact id to inspect, not an offset into `facts`:
/// closed, tombstoned, and purged ids are skipped without making the caller
/// rescan them. `None` means the scan reached the database's current end.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExportPage {
    /// The open facts found in this page, in fact-id order.
    pub facts: Vec<ExportedFact>,
    /// Pass this to the next [`Database::export_page`] call.
    pub next_cursor: Option<u32>,
}

/// The outcome of a [`Database::recover`] salvage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecoverReport {
    /// Facts written to the destination (the survivors after the purge).
    pub kept: usize,
    /// Facts dropped because their stored text was not valid UTF-8.
    pub dropped_text: usize,
    /// Facts dropped because their vector slot was out of range or mismatched.
    pub dropped_vector: usize,
    /// Facts dropped because their metadata blob did not decode to a
    /// well-formed key→value map.
    pub dropped_metadata: usize,
}

/// Visits the currently-open facts (skipping closed revisions and tombstones),
/// resolving each subject name and tag string, calling `f` once per fact. The
/// streaming core of export: a caller that writes each fact out (CLI `export`)
/// never materializes the whole dump, so a huge database exports without a RAM
/// spike. Shared by the read-write and read-only handles.
/// Decodes a fact's metadata into an owned, sorted key→value map (empty when
/// the fact carries none). Shared by `get` (read-write and read-only) and
/// `export`; the pairs come back from the engine in canonical order, so the
/// resulting `BTreeMap` matches the raw core view key-for-key.
pub(crate) fn metadata_map(mem: &Memory, id: plugmem_core::FactId) -> BTreeMap<String, String> {
    let mut pairs = Vec::new();
    mem.metadata_of(id, &mut pairs);
    pairs
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn exported_fact(
    mem: &Memory,
    id: plugmem_core::FactId,
    terms: &mut Vec<plugmem_core::TermId>,
) -> Option<ExportedFact> {
    use plugmem_core::{EntityId, VALID_TO_OPEN};
    let view = mem.get(id)?;
    if view.record.valid_to != VALID_TO_OPEN {
        return None; // a closed revision — export the current state only
    }
    let entity = (view.record.entity != EntityId::NONE)
        .then(|| mem.entity_name(view.record.entity))
        .flatten()
        .map(str::to_string);
    terms.clear();
    mem.tags_of(id, terms);
    let tags = terms.iter().map(|t| mem.term(*t).to_string()).collect();
    Some(ExportedFact {
        id: id.0,
        text: view.text.to_string(),
        entity,
        tags,
        metadata: metadata_map(mem, id),
        recorded_at: view.record.recorded_at,
        valid_from: view.record.valid_from,
    })
}

pub(crate) fn export_facts_each(mem: &Memory, mut f: impl FnMut(ExportedFact)) {
    use plugmem_core::FactId;
    let next = mem.stats().next_fact;
    let mut terms = Vec::new();
    for i in 0..next {
        if let Some(fact) = exported_fact(mem, FactId(i), &mut terms) {
            f(fact);
        }
    }
}

pub(crate) fn export_facts_page(mem: &Memory, cursor: u32, limit: usize) -> ExportPage {
    use plugmem_core::FactId;
    let end = mem.stats().next_fact;
    let mut cursor = cursor.min(end);
    let stop = cursor
        .saturating_add(limit.min(u32::MAX as usize) as u32)
        .min(end);
    let mut terms = Vec::new();
    let mut facts = Vec::with_capacity((stop - cursor) as usize);
    while cursor < stop {
        let id = FactId(cursor);
        cursor += 1;
        if let Some(fact) = exported_fact(mem, id, &mut terms) {
            facts.push(fact);
        }
    }
    ExportPage {
        facts,
        next_cursor: (cursor < end).then_some(cursor),
    }
}

/// Collects the currently-open facts into a `Vec` (the owning form of
/// [`export_facts_each`]). Used where the whole dump is wanted in memory.
pub(crate) fn export_facts(mem: &Memory) -> Vec<ExportedFact> {
    let mut out = Vec::new();
    export_facts_each(mem, |e| out.push(e));
    out
}

/// Tuning knobs of a [`Database`]. Construct through
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
    /// threshold pays the graph build).
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
        let actual_cfg = engine.read(|mem| mem.cfg().clone());
        let db = Database {
            inner: Arc::new(Inner {
                state: StateLock::new(State {
                    engine,
                    store,
                    cfg: actual_cfg,
                    ops: 0,
                    forgets: 0,
                    reembedding: false,
                }),
                embedder: EmbedderLock::new(self.embedder.map(Arc::from)),
                snapshot_every_ops: self.snapshot_every_ops,
                snapshot_journal_bytes: self.snapshot_journal_bytes,
                maintain_every_forgets: self.maintain_every_forgets,
            }),
        };
        Ok((db, report))
    }
}

/// The engine lock. Normally an `RwLock` so read-only verbs run concurrently
/// (the whole point of Variant 1). Under `counters`, `State` embeds the arena's
/// non-`Sync` instrumentation `Cell`s, and `RwLock<T>` needs `T: Sync` to hand
/// out shared guards — so there we fall back to a `Mutex`. `counters` is a
/// single-threaded perf-gate build, so serialized readers cost nothing there,
/// and the `Mutex` keeps `Database: Send + Sync` so every test still builds.
#[cfg(not(feature = "counters"))]
type StateLock = RwLock<State>;
#[cfg(feature = "counters")]
type StateLock = Mutex<State>;

struct Inner {
    state: StateLock,
    /// Unlocked: [`Embedder::embed`] takes `&self`, so several verbs may be
    /// inside the provider at once. That is the point — the round trip is the
    /// slow part of a write, and a lock here would queue every concurrent
    /// caller behind one HTTP request.
    embedder: EmbedderLock<Option<Arc<dyn Embedder>>>,
    snapshot_every_ops: u64,
    snapshot_journal_bytes: u64,
    maintain_every_forgets: Option<u64>,
}

struct State {
    engine: Engine,
    store: FileStorage,
    /// Actual persisted structural configuration. `dim` may differ from the
    /// target settings while an explicit reembed is pending.
    cfg: Config,
    /// Mutations since the last snapshot.
    ops: u64,
    /// Forgets since the last maintain.
    forgets: u64,
    /// A reembed releases this lock during model calls; the flag makes every
    /// competing write fail fast instead of changing its frozen source.
    reembedding: bool,
}

/// A clonable, thread-safe handle to one local database. See the module
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

    /// Opens `path` read-only over a memory-mapped snapshot:
    /// the engine borrows the mapped pages instead of copying the file
    /// into RAM, so a large read-mostly database residents only the pages
    /// `recall`/`get` touch. Requires a **published snapshot generation** —
    /// checkpoint the database once — and takes a shared lock, so readers run
    /// alongside the writer rather than excluding it. A journal written since
    /// that checkpoint is not an obstacle and not visible either: the handle
    /// answers as of the generation it mapped. See [`ReadOnlyDatabase`].
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`], [`HostError::NeedsCheckpoint`],
    /// [`HostError::Io`], [`HostError::Engine`] — see
    /// `ReadOnlyDatabase::open` semantics.
    pub fn open_readonly(
        path: impl Into<PathBuf>,
        cfg: Config,
    ) -> Result<ReadOnlyDatabase, HostError> {
        ReadOnlyDatabase::open(path, cfg)
    }

    /// Starts a configured open (knobs).
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

    /// A shared (read) guard — for the read-only verbs (`recall`/`get`/
    /// `stats`/`export`/`verify`). Many run at once; they exclude only writers.
    /// (Under `counters` the lock is a `Mutex`, so reads serialize — see
    /// [`StateLock`].) A panicked verb cannot leave the engine half-mutated
    /// (check first, mutate last is the engine's own law), so a poisoned lock
    /// is recoverable.
    #[cfg(not(feature = "counters"))]
    fn read(&self) -> RwLockReadGuard<'_, State> {
        self.inner.state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// An exclusive (write) guard — for the mutating verbs. Serializes writers
    /// against each other and against every concurrent reader.
    #[cfg(not(feature = "counters"))]
    fn write(&self) -> RwLockWriteGuard<'_, State> {
        self.inner.state.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Under `counters` the engine lock is a `Mutex`: `read` and `write` both
    /// take the one exclusive guard (readers serialize — acceptable for the
    /// single-threaded perf-gate build). See [`StateLock`].
    #[cfg(feature = "counters")]
    fn read(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(feature = "counters")]
    fn write(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A short write guard for ordinary mutations. Reembed deliberately drops
    /// the guard during model calls, so the flag — not lock waiting — protects
    /// its frozen source generation.
    fn write_available(&self) -> Result<impl std::ops::DerefMut<Target = State> + '_, HostError> {
        let guard = self.write();
        if guard.reembedding {
            return Err(HostError::ReembedBusy);
        }
        Ok(guard)
    }

    fn embedder(&self) -> Option<Arc<dyn Embedder>> {
        self.inner
            .embedder
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn check_vector_space(&self, requested: &str) -> Result<(), HostError> {
        self.read().engine.read(|mem| match mem.vector_space() {
            Some(stored) if stored == requested => Ok(()),
            Some(_) if mem.stats().vectors == 0 => Ok(()),
            Some(stored) => Err(HostError::Engine(Error::VectorSpaceMismatch {
                stored: stored.into(),
                requested: requested.into(),
            })),
            None if mem.stats().vectors != 0 => Err(HostError::Engine(Error::UntrackedVectorSpace)),
            None => Ok(()),
        })
    }

    /// Replaces the complete vector axis with the currently configured
    /// embedder, in bounded provider batches.
    ///
    /// This operation is deliberately separate from [`Database::maintain`]:
    /// neither `Auto` nor any other maintenance mode can invoke a model. The
    /// source generation remains readable while the provider runs. Competing
    /// writes fail immediately with [`HostError::ReembedBusy`], and the new
    /// generation becomes visible in one atomic publication step.
    pub fn reembed(&self, now: u64) -> Result<ReembedReport, HostError> {
        self.reembed_with_batch(now, DEFAULT_REEMBED_BATCH_SIZE)
    }

    /// As [`Database::reembed`], with an explicit provider batch bound.
    pub fn reembed_with_batch(
        &self,
        now: u64,
        batch_size: usize,
    ) -> Result<ReembedReport, HostError> {
        let embedder = self.embedder().ok_or_else(|| {
            HostError::Embed("reembed requires a configured embedding provider".into())
        })?;
        self.reembed_arc(now, embedder, batch_size, false)
    }

    /// Replaces the vector axis with `target`, then installs that embedder on
    /// this handle after a successful atomic publication.
    ///
    /// `batch_size` bounds both the request body and the temporary vectors
    /// held in RAM. A provider output is validated against [`Embedder::dim`]
    /// for every fact; callers do not separately supply a database dimension.
    pub fn reembed_with(
        &self,
        now: u64,
        target: Box<dyn Embedder>,
        batch_size: usize,
    ) -> Result<ReembedReport, HostError> {
        self.reembed_arc(now, Arc::from(target), batch_size, true)
    }

    fn reembed_arc(
        &self,
        now: u64,
        target: Arc<dyn Embedder>,
        batch_size: usize,
        install_target: bool,
    ) -> Result<ReembedReport, HostError> {
        if batch_size == 0 {
            return Err(HostError::Engine(Error::Invalid(
                "reembed batch size must be nonzero",
            )));
        }

        // Freeze every acknowledged mutation into a private, unpublished
        // source image, reserve the next generation, and then release the
        // state lock before the first model call.
        let (source, source_cfg, base_path, mut staged) = {
            let mut st = self.write_available()?;
            st.reembedding = true;
            let setup = (|| {
                let source = {
                    let State { engine, store, .. } = &mut *st;
                    store.stage_reembed_source(|sink| {
                        engine
                            .read(|mem| mem.write_snapshot_to(now, &mut *sink))
                            .map_err(HostError::from)
                    })?
                };
                let source_cfg = st.cfg.clone();
                let base_path = st.store.path().to_path_buf();
                let staged = st.store.begin_detached_snapshot()?;
                Ok::<_, HostError>((source, source_cfg, base_path, staged))
            })();
            match setup {
                Ok(setup) => setup,
                Err(error) => {
                    st.reembedding = false;
                    return Err(error);
                }
            }
        };

        let build = (|| {
            let source_path = source.path()?;
            let file = File::open(source_path).map_err(|e| HostError::io(source_path, e))?;
            // SAFETY: snapshot generations are immutable. The writer lock is
            // held for this Database's lifetime and the reembed barrier keeps
            // this generation current until publication.
            let source_map =
                unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(source_path, e))?;
            let source = Memory::from_bytes_borrowed(&source_map, &[], source_cfg.clone())?;
            let scratch_path = tmp_sibling(&base_path, "reembed-vectors");
            let mut vec_scratch = FileScratch::create(scratch_path)?;
            let report = source
                .write_reembedded_snapshot(
                    now,
                    target.dim(),
                    target.space_id(),
                    batch_size,
                    &mut vec_scratch,
                    &mut staged,
                    |texts| target.embed(texts),
                )
                .map_err(|error| match error {
                    ReembedError::Engine(error) => HostError::Engine(error),
                    ReembedError::Embedder(error) => error,
                })?;
            Ok::<_, HostError>((report, staged.prepare()?))
        })();

        let (report, prepared) = match build {
            Ok(done) => done,
            Err(error) => {
                self.write().reembedding = false;
                return Err(error);
            }
        };

        // The expensive work is complete. Publication, journal clear and
        // re-map are short and serialized with readers/writers.
        let journal_clear = {
            let mut st = self.write();
            if let Err(error) = st.store.commit_detached_snapshot(prepared) {
                st.reembedding = false;
                return Err(error);
            }
            let journal_clear = st.store.clear_journal();
            let mut target_cfg = st.cfg.clone();
            target_cfg.dim = target.dim();
            let (engine, _) = match open_engine(&mut st.store, &target_cfg) {
                Ok(opened) => opened,
                Err(error) => {
                    // The new generation is already durable. Keep writes
                    // blocked rather than journal against an older in-memory
                    // image; reopening the Database recovers normally.
                    return Err(error);
                }
            };
            st.engine = engine;
            st.cfg = target_cfg;
            st.ops = 0;
            st.forgets = 0;
            st.reembedding = false;
            journal_clear
        };
        if install_target {
            *self
                .inner
                .embedder
                .write()
                .unwrap_or_else(|e| e.into_inner()) = Some(target);
        }
        journal_clear?;
        Ok(report)
    }

    /// Embeds `text` outside the state lock, when an embedder is
    /// configured. `None` = leave the input as it was.
    fn embed_one(&self, text: &str) -> Result<Option<(Vec<f32>, String)>, HostError> {
        let Some(embedder) = self.embedder() else {
            return Ok(None);
        };
        if embedder.dim() == 0 {
            return Ok(None);
        }
        let space = embedder.space_id().to_owned();
        self.check_vector_space(&space)?;
        let mut vs = embedder.embed(&[text])?;
        if vs.len() != 1 {
            return Err(HostError::Embed(format!(
                "expected 1 embedding, got {}",
                vs.len()
            )));
        }
        Ok(Some((vs.remove(0), space)))
    }

    /// Embeds a whole batch of texts in a **single** embedder call — outside the
    /// lock, like [`embed_one`](Self::embed_one). `Ok(None)` when no embedder is
    /// configured or `dim == 0`; otherwise a vector aligned one-to-one with
    /// `texts` (the provider contract, checked by [`OpenAiCompatEmbedder`]). An
    /// empty `texts` yields an empty vector without a round-trip. This is the one
    /// HTTP that [`remember_many`](Self::remember_many) makes for a bulk write.
    fn embed_many(&self, texts: &[&str]) -> Result<Option<EmbeddedBatch>, HostError> {
        let Some(embedder) = self.embedder() else {
            return Ok(None);
        };
        if embedder.dim() == 0 {
            return Ok(None);
        }
        if texts.is_empty() {
            return Ok(Some((Vec::new(), embedder.space_id().to_owned())));
        }
        let space = embedder.space_id().to_owned();
        self.check_vector_space(&space)?;
        let vectors = embedder.embed(texts)?;
        if vectors.len() != texts.len() {
            return Err(HostError::Embed(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                vectors.len()
            )));
        }
        Ok(Some((vectors, space)))
    }

    /// Writes a full snapshot and re-maps the fresh file.
    ///
    /// Materializes the borrowed base + overlay into an owned buffer, drops
    /// the current map, writes the buffer (tmp + fsync + rename) and clears
    /// the journal, then maps the new file into a fresh overlay. The re-map
    /// collapses the overlay so a long write session stays bounded, and
    /// dropping the map **before** the rename keeps the write portable
    /// (a mapped file cannot be renamed over on Windows).
    fn resnapshot(&self, st: &mut State, now: u64) -> Result<(), HostError> {
        // Stream the image straight to the tmp file — never a full-image Vec
        // This reads through the live map, so it happens
        // **before** the map is dropped.
        {
            let State { engine, store, .. } = &mut *st;
            store.stage_snapshot(|sink| {
                engine
                    .read(|mem| mem.write_snapshot_to(now, &mut *sink))
                    .map_err(HostError::from)
            })?;
        }
        // Drop the current map before the rename: park a cheap empty engine.
        // It is replaced by the fresh overlay below — or, if the commit fails,
        // rebuilt from the intact on-disk snapshot + journal.
        st.engine = Engine::Owned(Box::new(Memory::new(st.cfg.clone())?));
        let write = st
            .store
            .commit_snapshot()
            .and_then(|()| st.store.clear_journal());
        // Re-open regardless: on success the fresh file, on failure the
        // untouched old file + journal (journal replay is idempotent, so a
        // failed `clear_journal` does not corrupt state). Then surface the
        // commit error, if any.
        let cfg = st.cfg.clone();
        let (engine, _) = open_engine(&mut st.store, &cfg)?;
        st.engine = engine;
        write
    }

    /// The post-mutation policy hook: counts the op, fires auto-maintain
    /// and auto-snapshot inside the same critical section.
    fn after_mutation(&self, st: &mut State, now: u64) -> Result<(), HostError> {
        st.ops += 1;
        if let Some(threshold) = self.inner.maintain_every_forgets
            && st.forgets >= threshold
        {
            let State { engine, store, .. } = &mut *st;
            engine.with(store, |mem, store| mem.maintain(store, now))?;
            st.forgets = 0;
        }
        // A database that outgrows its shard layout re-shards itself. This is
        // on by default, unlike `maintain_every_forgets`, because without it
        // nothing would ever move a layout: a growing database would keep the
        // one it was created with until somebody ran `maintain` by hand, and
        // the cost of that is silent — memory, and a page directory that keeps
        // lengthening.
        //
        // Affordable because both halves are bounded. The question is O(1)
        // (stored record counts), so asking on every write is free; and the
        // answer is self-limiting — the thresholds are a doubling up and a
        // fourfold drop, so it says yes a handful of times over a database's
        // whole life. `resharding_settles_instead_of_asking_forever` in the
        // core suite is the test that keeps that true.
        let State { engine, store, .. } = &mut *st;
        if engine.with(store, |mem, _| mem.shard_layout_is_stale()) {
            engine.with(store, |mem, store| mem.maintain(store, now))?;
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
            vector: embedded
                .as_ref()
                .map(|(v, _)| v.as_slice())
                .or(input.vector),
            ..input
        };
        let mut st = self.write_available()?;
        let State { engine, store, .. } = &mut *st;
        if let Some((_, space)) = &embedded {
            engine.with(store, |mem, store| mem.claim_vector_space(store, space))?;
        }
        let out = engine.with(store, |mem, store| mem.remember(store, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(out)
    }

    /// Checks the ordinary `remember` similarity thresholds and stores only
    /// when no live same-entity candidate crosses one. Automatic embedding runs
    /// before the engine lock; no other write can slip between the check and
    /// possible journaled mutation because both share one exclusive guard.
    pub fn remember_guarded(
        &self,
        input: RememberInput<'_>,
    ) -> Result<GuardedRememberOutcome, HostError> {
        let embedded = match input.vector {
            Some(_) => None,
            None => self.embed_one(input.text)?,
        };
        let input = RememberInput {
            vector: embedded
                .as_ref()
                .map(|(vector, _)| vector.as_slice())
                .or(input.vector),
            ..input
        };
        let mut st = self.write_available()?;
        let vector_space = embedded.as_ref().map(|(_, space)| space.as_str());
        let State { engine, store, .. } = &mut *st;
        let out = engine.with(store, |mem, store| {
            mem.remember_guarded_with_vector_space(store, input, vector_space)
        })?;
        if matches!(out, GuardedRememberOutcome::Stored { .. }) {
            self.after_mutation(&mut st, input.now)?;
        }
        Ok(out)
    }

    /// Remembers a **batch** of facts in one shot — the bulk-write path (CLI
    /// `import`). Equivalent to [`remember`](Self::remember) on each input in
    /// order, but far cheaper for a batch: the texts that need embedding are
    /// embedded together in **one** embedder round-trip (outside the lock), and
    /// all facts are written under **one** write-guard with **one** post-mutation
    /// policy pass — instead of N HTTP calls and N critical sections.
    ///
    /// Inputs that already carry a `vector` are not re-embedded. **Chunking is
    /// the caller's job**: this writes the whole slice it is given, so a caller
    /// that needs bounded memory / a bounded HTTP body passes fixed-size batches
    /// (CLI `import` streams the file in `--batch`-sized slices).
    ///
    /// **Fail-fast:** the first engine error returns `Err`; the facts written
    /// before it stay written (exactly as separate `remember`s — the journal
    /// replay is idempotent, so a retried bulk load is safe). Returns one
    /// [`RememberOutcome`] per input, in order.
    pub fn remember_many(
        &self,
        inputs: Vec<RememberInput<'_>>,
    ) -> Result<Vec<RememberOutcome>, HostError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        // One embedder round-trip for every vector-less input's text, outside the
        // lock. `to_embed` is the vector-less inputs in order, so its result maps
        // back onto them by a running cursor below.
        let to_embed: Vec<&str> = inputs
            .iter()
            .filter(|i| i.vector.is_none())
            .map(|i| i.text)
            .collect();
        let embedded = if to_embed.is_empty() {
            None
        } else {
            self.embed_many(&to_embed)?
        };

        let mut st = self.write_available()?;
        if let Some((_, space)) = &embedded {
            let State { engine, store, .. } = &mut *st;
            engine.with(store, |mem, store| mem.claim_vector_space(store, space))?;
        }
        // Batch mode: journal appends skip their per-record fsync; one
        // `sync_journal` at the end makes the whole batch durable at once.
        st.store.set_batch(true);
        let mut out = Vec::with_capacity(inputs.len());
        let mut cursor = 0usize; // into `embedded`, over vector-less inputs in order
        let mut latest = 0u64;
        let mut failed = None;
        for input in inputs {
            latest = latest.max(input.now);
            let vector = if input.vector.is_some() {
                input.vector
            } else if let Some((vectors, _)) = &embedded {
                let v = vectors[cursor].as_slice();
                cursor += 1;
                Some(v)
            } else {
                None // no embedder — lexical/structural only, as single remember
            };
            let input = RememberInput { vector, ..input };
            let State { engine, store, .. } = &mut *st;
            match engine.with(store, |mem, store| mem.remember(store, input)) {
                Ok(o) => out.push(o),
                Err(e) => {
                    failed = Some(HostError::from(e));
                    break;
                }
            }
        }
        // Always leave batch mode and fsync — this is the batch's durability
        // point. On fail-fast it makes the facts written before the error durable
        // (they stay, exactly like separate remembers).
        st.store.set_batch(false);
        st.store.sync_journal()?;
        if let Some(e) = failed {
            return Err(e);
        }
        // One policy pass for the whole batch. The op counter advances by one per
        // batch; the journal-bytes threshold still fires on a large batch, so a
        // snapshot is not starved.
        self.after_mutation(&mut st, latest)?;
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
            vector: embedded.as_ref().map(|(v, _)| v.as_slice()).or(q.vector),
            ..q
        };
        // A shared guard: concurrent recalls run in parallel. `recall_into`
        // takes `&self` on the engine and a per-thread scratch, so there is no
        // writer path and no cross-reader contention on the hot path.
        let st = self.read();
        RECALL_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let mut out = RecallResult::default();
            st.engine
                .read(|mem| mem.recall_into(q, &mut scratch, &mut out))?;
            Ok(out)
        })
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
            vector: embedded
                .as_ref()
                .map(|(v, _)| v.as_slice())
                .or(input.vector),
            ..input
        };
        let mut st = self.write_available()?;
        let State { engine, store, .. } = &mut *st;
        if let Some((_, space)) = &embedded {
            engine.with(store, |mem, store| mem.claim_vector_space(store, space))?;
        }
        let out = engine.with(store, |mem, store| mem.revise(store, target, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(out)
    }

    /// Tombstones a fact.
    pub fn forget(&self, now: u64, id: plugmem_core::FactId) -> Result<bool, HostError> {
        let mut st = self.write_available()?;
        let State { engine, store, .. } = &mut *st;
        let fresh = engine.with(store, |mem, store| mem.forget(store, now, id))?;
        st.forgets += 1;
        self.after_mutation(&mut st, now)?;
        Ok(fresh)
    }

    /// Upserts a typed edge.
    pub fn link(&self, input: LinkInput<'_>) -> Result<(), HostError> {
        let mut st = self.write_available()?;
        let State { engine, store, .. } = &mut *st;
        engine.with(store, |mem, store| mem.link(store, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(())
    }

    /// Closes a typed edge. Returns `false` when the edge is already absent.
    pub fn unlink(&self, input: UnlinkInput<'_>) -> Result<bool, HostError> {
        let mut st = self.write_available()?;
        let State { engine, store, .. } = &mut *st;
        let fresh = engine.with(store, |mem, store| mem.unlink(store, input))?;
        self.after_mutation(&mut st, input.now)?;
        Ok(fresh)
    }

    /// An owned copy of one fact, or `None` for unknown/tombstoned ids.
    pub fn get(&self, id: plugmem_core::FactId) -> Option<FactSnapshot> {
        self.read().engine.read(|mem| {
            mem.get(id).map(|v| FactSnapshot {
                record: v.record,
                text: v.text.to_string(),
                metadata: metadata_map(mem, id),
            })
        })
    }

    /// One fact's tags, or an empty vector for an unknown or tombstoned id.
    ///
    /// [`FactSnapshot`] carries text and metadata but not tags, so without this
    /// the only way to read one fact's tags is [`Database::export`] — a full
    /// scan to answer a question about a single id.
    pub fn tags_of(&self, id: plugmem_core::FactId) -> Vec<String> {
        self.read().engine.read(|mem| {
            let mut terms = Vec::new();
            mem.tags_of(id, &mut terms);
            terms.iter().map(|t| mem.term(*t).to_string()).collect()
        })
    }

    /// One bounded, cursor-stable page of current tags.
    pub fn list_tags(&self, query: TagQuery<'_>) -> Result<TagPage, HostError> {
        self.read()
            .engine
            .read(|mem| mem.list_tags(query))
            .map_err(Into::into)
    }

    /// Removes a tag from every current fact by creating successor revisions.
    pub fn remove_tag(&self, now: u64, tag: &str) -> Result<RemoveTagReport, HostError> {
        let mut st = self.write_available()?;
        let State { engine, store, .. } = &mut *st;
        let report = engine.with(store, |mem, store| mem.remove_tag(store, now, tag))?;
        if report.affected != 0 {
            self.after_mutation(&mut st, now)?;
        }
        Ok(report)
    }

    /// Engine size counters.
    pub fn stats(&self) -> Stats {
        self.read().engine.read(|mem| mem.stats())
    }

    /// Dumps the currently-open facts for a human-readable backup
    /// See [`ExportedFact`]. Collects the whole set; for a large
    /// database prefer [`export_each`](Self::export_each), which streams.
    pub fn export(&self) -> Vec<ExportedFact> {
        self.read().engine.read(export_facts)
    }

    /// Streams the currently-open facts, calling `f` once per fact under the
    /// read guard — the whole dump is never materialized, so a huge database
    /// exports without a RAM spike (CLI `export` writes each line straight out).
    /// See [`ExportedFact`].
    pub fn export_each(&self, f: impl FnMut(ExportedFact)) {
        self.read().engine.read(|mem| export_facts_each(mem, f));
    }

    /// Streams the currently-open **edges**, calling `f` with
    /// `(source, relation, destination, provenance fact)` under the read guard.
    ///
    /// The companion to [`export_each`](Self::export_each): facts alone are not
    /// the memory, and a dump without edges silently drops one of the four
    /// recall sources. Names are borrowed, so a writer that formats them
    /// directly allocates nothing per edge.
    ///
    /// `provenance` is the fact id **as this database numbers it**. Ids do not
    /// survive a re-import, so a file format that wants to keep provenance has
    /// to translate it — see the CLI's `export`/`import`, which rewrite it as a
    /// position within the same file.
    pub fn export_edges_each(&self, mut f: impl FnMut(&str, &str, &str, plugmem_core::FactId)) {
        self.read().engine.read(|mem| {
            mem.edges_each(|src, rel, dst, fact| {
                f(src, rel, dst, fact);
                true
            });
        });
    }

    /// Inspects at most `limit` fact ids starting at `cursor` and returns the
    /// ones that are currently open. A sparse page may therefore contain fewer
    /// facts, including zero, while still carrying a `next_cursor`.
    ///
    /// This is the pull-based counterpart to [`export_each`](Self::export_each):
    /// it releases the database read guard before returning, so a boundary
    /// caller can process the page, apply backpressure, or write to the database
    /// without a callback running under this lock. Mutations between page calls
    /// are visible to later pages; use a stable read-only checkpoint when
    /// snapshot-consistent paging is required.
    pub fn export_page(&self, cursor: u32, limit: std::num::NonZeroUsize) -> ExportPage {
        self.read()
            .engine
            .read(|mem| export_facts_page(mem, cursor, limit.get()))
    }

    /// Runs a maintenance pass now (cheap no-op, purge/compaction, text
    /// reindex, and/or bounded HNSW work — for the cost model).
    ///
    /// **Disk-first** (milestone H): the compacted image is written by streaming
    /// the two big pools (vectors, text) through temp files and then re-mapped,
    /// so peak RAM tracks the record count (metadata + graph), not the image
    /// size — a database larger than RAM can be maintained. It writes a fresh
    /// snapshot and clears the journal (like a checkpoint). The optional
    /// auto-maintain policy (`maintain_every_forgets`) still runs in RAM inline
    /// — it is for databases that fit.
    ///
    /// The report's byte counts are the on-disk image size before and after.
    pub fn maintain(&self, now: u64) -> Result<MaintainReport, HostError> {
        self.maintain_with_options(now, MaintenanceOptions::auto())
    }

    /// Runs a maintenance pass with explicit policy.
    pub fn maintain_with_options(
        &self,
        now: u64,
        options: MaintenanceOptions,
    ) -> Result<MaintainReport, HostError> {
        let mut st = self.write_available()?;
        // The image size is the current snapshot generation's, not the tiny
        // manifest at the base path.
        let snap_len = |store: &FileStorage| -> usize {
            store
                .current_snapshot_path()
                .ok()
                .flatten()
                .and_then(|p| std::fs::metadata(&p).ok())
                .map(|m| m.len() as usize)
                .unwrap_or(0)
        };
        let bytes_before = snap_len(&st.store);
        if !st.engine.read(|mem| mem.maintenance_needed(options)) {
            let mut report = st
                .engine
                .read(|mem| mem.maintenance_preview(options, bytes_before));
            report.bytes_after = bytes_before;
            return Ok(report);
        }
        let stats = st.engine.read(|mem| mem.stats());
        let needs_disk_first =
            stats.tombstones != 0 || matches!(options.mode, MaintenanceMode::Full);
        if !needs_disk_first {
            let mut report = {
                let State { engine, store, .. } = &mut *st;
                engine.with(store, |mem, store| {
                    mem.maintain_with_options(store, now, options)
                })?
            };
            self.resnapshot(&mut st, now)?;
            st.forgets = 0;
            st.ops = 0;
            report.bytes_before = bytes_before;
            report.bytes_after = snap_len(&st.store);
            return Ok(report);
        }
        let text_tmp = tmp_sibling(st.store.path(), "mtext");
        let vec_tmp = tmp_sibling(st.store.path(), "mvec");

        // Stage a compacted snapshot, streaming the big pools through scratch;
        // this reads through the live map, so it happens before the map is
        // dropped (as in `resnapshot`).
        let mut purged = 0usize;
        let mut report = MaintainReport::default();
        {
            let State { engine, store, .. } = &mut *st;
            store.stage_snapshot(|sink| {
                engine.read(|mem| {
                    let mut text_scratch = FileScratch::create(&text_tmp)?;
                    let mut vec_scratch = FileScratch::create(&vec_tmp)?;
                    report = mem
                        .snapshot_disk_first_with_options(
                            now,
                            &mut text_scratch,
                            &mut vec_scratch,
                            &mut *sink,
                            options,
                        )
                        .map_err(HostError::from)?;
                    purged = report.purged;
                    Ok(())
                })
            })?;
        }
        // Drop the current map before the rename (park a cheap empty engine),
        // commit, clear the journal, then re-map the compacted file — exactly
        // the `resnapshot` dance, so the map is never renamed over on Windows.
        st.engine = Engine::Owned(Box::new(Memory::new(st.cfg.clone())?));
        st.store
            .commit_snapshot()
            .and_then(|()| st.store.clear_journal())?;
        let cfg = st.cfg.clone();
        let (engine, _) = open_engine(&mut st.store, &cfg)?;
        st.engine = engine;
        st.forgets = 0;
        st.ops = 0;
        let bytes_after = snap_len(&st.store);
        report.purged = purged;
        report.bytes_before = bytes_before;
        report.bytes_after = bytes_after;
        Ok(report)
    }

    /// Writes a full snapshot and clears the journal now (re-mapping the
    /// fresh file — see `Database::resnapshot`).
    pub fn checkpoint(&self, now: u64) -> Result<(), HostError> {
        let mut st = self.write_available()?;
        self.resnapshot(&mut st, now)?;
        st.ops = 0;
        Ok(())
    }

    /// Runs the on-demand full content-integrity check. An open validates only
    /// the metadata, so the
    /// large byte pools stay non-resident on an mmap'd base; this sweeps them
    /// (text UTF-8, vector self-consistency and the fact↔slot bijection) and
    /// reports any latent corruption. Skipping it is safe — the accessors never
    /// panic on bad bytes; `verify` only turns corruption into an explicit
    /// error.
    ///
    /// # Errors
    ///
    /// [`HostError::Engine`] wrapping [`Error::Corrupt`](plugmem_core::Error)
    /// for the first inconsistency found.
    pub fn verify(&self) -> Result<(), HostError> {
        Ok(self.read().engine.read(|mem| mem.verify())?)
    }

    /// A resumable byte-level container scrub of the current published
    /// generation, with the default slice budget. See
    /// [`Database::scrub_with_budget`].
    ///
    /// # Errors
    ///
    /// As [`Database::scrub_with_budget`].
    pub fn scrub(&self) -> Result<Scrub, HostError> {
        self.scrub_with_budget(plugmem_core::snapshot::DEFAULT_SCRUB_BUDGET)
    }

    /// A resumable container scrub hashing at most `budget` bytes per
    /// [`Iterator::next`] — the writer's counterpart to
    /// [`ReadOnlyDatabase::scrub_with_budget`], and the same [`Scrub`].
    ///
    /// It exists because a scrub is an operation on the **file**, not on this
    /// handle's view of it: it hashes the published container as it stands, and
    /// the journal belongs to the generation the writer has not published yet.
    /// A writer could always have reached one by opening a second, read-only
    /// handle on the same path — but that maps the whole image again, takes a
    /// second lock and reconciles the config, all to hash bytes this handle
    /// already knows the path of.
    ///
    /// The returned [`Scrub`] is independent of this handle: it owns its map
    /// and a shared lock on the generation it pins, so it outlives the
    /// database, can be moved to its own thread, and keeps this generation
    /// safe from the writer's own GC until it is dropped.
    ///
    /// # Errors
    ///
    /// [`HostError::NeedsCheckpoint`] when nothing has been published yet;
    /// [`HostError::Io`] if the generation cannot be opened or mapped;
    /// [`HostError::Engine`] if its container will not parse.
    pub fn scrub_with_budget(&self, budget: usize) -> Result<Scrub, HostError> {
        Scrub::open(self.read().store.path(), budget)
    }

    /// Salvages a content-corrupt database (Tier 2): opens `src`,
    /// drops the facts that fail the per-fact content checks (`verify`'s
    /// predicate), compacts the survivors and their indexes, and writes a clean
    /// image to `dst`. `src` on disk is left untouched — the evidence is
    /// preserved.
    ///
    /// It is **disk-first** (milestone H): `src` is opened as an mmap overlay
    /// (its pages are reclaimable) and the compacted image is written by
    /// streaming the two big pools (vectors, text) through temp files, so peak
    /// RAM tracks the record count (metadata + HNSW graph), not the image size.
    /// A database far larger than RAM can be recovered, as long as its graph
    /// fits.
    ///
    /// This handles *content* corruption (bad text bytes, a broken fact↔slot
    /// vector bijection). *Structural* damage — a snapshot that will not parse
    /// — is not salvageable here: `src` fails to open and recover returns the
    /// engine's typed error; restore from a backup instead (Tier 0).
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`] if `src` or `dst` is owned elsewhere;
    /// [`HostError::Engine`] if `src` will not parse (structural corruption) or
    /// `dst` equals `src`; [`HostError::Io`] for filesystem failures.
    pub fn recover(
        src: impl AsRef<Path>,
        dst: impl AsRef<Path>,
        cfg: Config,
        now: u64,
    ) -> Result<RecoverReport, HostError> {
        let src = src.as_ref();
        let dst = dst.as_ref();

        // Lock the source exclusively for the salvage's whole life. We never
        // write it — the lock only excludes a cooperating writer while we read.
        let mut src_store = FileStorage::open(src, FsyncPolicy::OnSnapshot)?;
        let src_base = src_store.path().to_path_buf();

        // The destination must be a different file: recover preserves the source
        // as evidence and writes the clean image elsewhere.
        let same = dst == src_base
            || matches!(
                (std::fs::canonicalize(dst), std::fs::canonicalize(&src_base)),
                (Ok(a), Ok(b)) if a == b
            );
        if same {
            return Err(HostError::Engine(Error::Invalid(
                "recover destination must differ from the source",
            )));
        }

        // Open the source as an overlay: borrow the mmap base (reclaimable
        // pages) and replay its journal into a small owned overlay — never an
        // owned copy of the image. A structurally corrupt image fails here —
        // that is Tier 0, not salvageable content corruption.
        let journal = src_store.read_journal()?;
        let Some(genp) = src_store.current_snapshot_path()? else {
            return Err(HostError::Engine(Error::Corrupt(
                "source database has no published snapshot to recover",
            )));
        };
        let file = File::open(&genp).map_err(|e| HostError::io(&genp, e))?;
        // SAFETY: as in `open_engine` — the generation file is immutable and
        // `src_store` holds the exclusive lock, so nothing touches it under us.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| HostError::io(&genp, e))?;
        drop(file);
        let (mut mem, _report) = Memory::from_bytes_overlay(&map[..], &journal, cfg.clone())?;

        // Drop each content-faulty fact into a throwaway store, so the source
        // file is never written. The disk-first rebuild below then physically
        // purges them and rebuilds clean indexes + HNSW from the survivors.
        let mut scratch = MemStorage::new();
        let mut dropped_text = 0usize;
        let mut dropped_vector = 0usize;
        let mut dropped_metadata = 0usize;
        for (id, fault) in mem.faulty_facts() {
            mem.forget(&mut scratch, now, id)?;
            match fault {
                FactFault::Text => dropped_text += 1,
                FactFault::Vector => dropped_vector += 1,
                FactFault::Metadata => dropped_metadata += 1,
            }
        }

        // Write the compacted image to `dst`, streaming the big pools through
        // temp scratch files (metadata + graph are the only things resident).
        let mut dst_store = FileStorage::open(dst, FsyncPolicy::OnSnapshot)?;
        let text_tmp = tmp_sibling(dst_store.path(), "rectext");
        let vec_tmp = tmp_sibling(dst_store.path(), "recvec");
        let mut purged = 0usize;
        dst_store.stage_snapshot(|sink| {
            let mut text_scratch = FileScratch::create(&text_tmp)?;
            let mut vec_scratch = FileScratch::create(&vec_tmp)?;
            purged = mem
                .snapshot_disk_first(now, &mut text_scratch, &mut vec_scratch, &mut *sink)
                .map_err(HostError::from)?;
            Ok(())
        })?;
        dst_store.commit_snapshot()?;

        let kept = mem.stats().facts.saturating_sub(purged);
        Ok(RecoverReport {
            kept,
            dropped_text,
            dropped_vector,
            dropped_metadata,
        })
    }
}

/// A temp-file path beside `base` with the given tag (for disk-first scratch).
fn tmp_sibling(base: &Path, tag: &str) -> PathBuf {
    let mut p = base.as_os_str().to_os_string();
    p.push(".");
    p.push(tag);
    p.push(".tmp");
    PathBuf::from(p)
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
