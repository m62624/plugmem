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

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use plugmem_core::{
    Config, Error, FactRecord, LinkInput, MaintainReport, Memory, OpenReport, RecallQuery,
    RecallResult, RememberInput, RememberOutcome, Stats,
};

use crate::embedder::Embedder;
use crate::error::HostError;
use crate::storage::{FileStorage, FsyncPolicy};

/// An owned view of one fact — [`Memory::get`] returns borrows that
/// cannot cross the lock, so the database hands out copies.
#[derive(Clone, Debug, PartialEq)]
pub struct FactSnapshot {
    /// The raw record (temporality, flags, references).
    pub record: FactRecord,
    /// The fact text.
    pub text: String,
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
        let (mem, report) = Memory::open(&mut store, self.cfg)?;
        let db = Database {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    mem,
                    store,
                    ops: 0,
                    forgets: 0,
                }),
                embedder: self.embedder.map(Mutex::new),
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
    snapshot_every_ops: u64,
    snapshot_journal_bytes: u64,
    maintain_every_forgets: Option<u64>,
}

struct State {
    mem: Memory,
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

    /// The post-mutation policy hook: counts the op, fires auto-maintain
    /// and auto-snapshot inside the same critical section (specs/13 §3).
    fn after_mutation(&self, st: &mut State, now: u64) -> Result<(), HostError> {
        st.ops += 1;
        if let Some(threshold) = self.inner.maintain_every_forgets
            && st.forgets >= threshold
        {
            let State { mem, store, .. } = &mut *st;
            mem.maintain(store, now)?;
            st.forgets = 0;
        }
        let by_ops = self.inner.snapshot_every_ops > 0 && st.ops >= self.inner.snapshot_every_ops;
        let by_bytes = self.inner.snapshot_journal_bytes > 0
            && st.store.journal_bytes() >= self.inner.snapshot_journal_bytes;
        if by_ops || by_bytes {
            let State { mem, store, .. } = &mut *st;
            mem.snapshot(store, now)?;
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
        let State { mem, store, .. } = &mut *st;
        let out = mem.remember(store, input)?;
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
        Ok(self.lock().mem.recall(q)?)
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
        let State { mem, store, .. } = &mut *st;
        let out = mem.revise(store, target, input)?;
        self.after_mutation(&mut st, input.now)?;
        Ok(out)
    }

    /// Tombstones a fact.
    pub fn forget(&self, now: u64, id: plugmem_core::FactId) -> Result<bool, HostError> {
        let mut st = self.lock();
        let State { mem, store, .. } = &mut *st;
        let fresh = mem.forget(store, now, id)?;
        st.forgets += 1;
        self.after_mutation(&mut st, now)?;
        Ok(fresh)
    }

    /// Upserts a typed edge.
    pub fn link(&self, input: LinkInput<'_>) -> Result<(), HostError> {
        let mut st = self.lock();
        let State { mem, store, .. } = &mut *st;
        mem.link(store, input)?;
        self.after_mutation(&mut st, input.now)?;
        Ok(())
    }

    /// An owned copy of one fact, or `None` for unknown/tombstoned ids.
    pub fn get(&self, id: plugmem_core::FactId) -> Option<FactSnapshot> {
        let st = self.lock();
        st.mem.get(id).map(|v| FactSnapshot {
            record: v.record,
            text: v.text.to_string(),
        })
    }

    /// Engine size counters.
    pub fn stats(&self) -> Stats {
        self.lock().mem.stats()
    }

    /// Runs a maintenance pass now (purge, compaction, HNSW build past
    /// the threshold — see specs/07 for the cost model).
    pub fn maintain(&self, now: u64) -> Result<MaintainReport, HostError> {
        let mut st = self.lock();
        let State { mem, store, .. } = &mut *st;
        let report = mem.maintain(store, now)?;
        st.forgets = 0;
        Ok(report)
    }

    /// Writes a full snapshot and clears the journal now.
    pub fn checkpoint(&self, now: u64) -> Result<(), HostError> {
        let mut st = self.lock();
        let State { mem, store, .. } = &mut *st;
        mem.snapshot(store, now)?;
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
