//! The `Workspace` class: many memories in one directory, addressed by name.
//!
//! **Optional.** A Node program that wants one memory constructs [`Plugmem`]
//! with a path and never touches this file. This is for a process that serves
//! many independent memories — one per conversation, per tenant, per project —
//! and wants them to stay independent.
//!
//! The same rule as [`crate::db`]: every method wraps the identically-named host
//! verb, and the engine logic is 100 % `plugmem-host`. [`Workspace::open`] hands
//! back a [`Plugmem`] over a pooled handle, so a named memory has exactly the
//! verbs a path-opened one has, with no second implementation to drift.
//!
//! Three host methods are deliberately not here: `registry()` (handing out the
//! raw registry database invites writing into it by hand), `entry(name)`
//! (`entries()` already answers it), and the path helpers (`path_of`, `exists`)
//! — a caller that has a name never needs the path.

use std::sync::Arc;

use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Task};
use napi_derive::napi;
use plugmem_host::{DbName, Description, IfMissing, Settings, WorkspaceError, WorkspaceIssue};

use crate::db::{Plugmem, now_ms};
use crate::error::{self, Result, code};
use crate::types::{DbEntry, ReindexReport, WorkspaceProblem};

/// Options for [`Workspace::new`].
#[napi(object)]
#[derive(Default)]
pub struct WorkspaceOptions {
    /// Embedding dimension, as in [`crate::db::OpenOptions`]. Applies to every
    /// memory in the workspace and to the registry.
    pub dim: Option<u32>,
    /// Path to a `config.toml`. Resolution is the standard one when omitted.
    /// Its `[workspace]` section supplies the pool defaults below.
    pub config: Option<String>,
    /// Memories kept open at once; the least recently used is closed to make
    /// room. Defaults to `[workspace].max_open`, else 16.
    pub max_open: Option<u32>,
    /// Milliseconds a memory may sit unused before `closeIdle()` closes it.
    /// `0` disables the sweep. Defaults to `[workspace].idle_timeout_ms`, else
    /// 60000.
    ///
    /// This is a liveness setting, not a memory one: an open memory holds the
    /// file's exclusive lock, so a long-running process that never let go would
    /// make its memories unreachable from anything else on the machine.
    pub idle_timeout_ms: Option<f64>,
}

/// What to record about a memory — the argument of [`Workspace::describe`].
#[napi(object)]
pub struct DescribeArgs {
    /// What this memory is for, in the words someone would search with. This is
    /// the text `find` matches.
    pub description: String,
    /// Tags to filter by.
    pub tags: Option<Vec<String>>,
    /// Who it belongs to. Recorded as a graph edge, so `find("ann")` returns
    /// what Ann owns even though no description mentions her.
    pub owner: Option<String>,
}

/// A directory of named memories — the napi mirror of
/// [`plugmem_host::Workspace`].
#[napi]
pub struct Workspace {
    /// `None` once `close()`d — every method then throws "workspace is closed".
    ///
    /// Behind an `Arc` because [`plugmem_host::Workspace`] owns the handle pool
    /// and is deliberately not clonable, while the async verbs need something
    /// that outlives the call on a libuv thread.
    inner: Option<Arc<plugmem_host::Workspace>>,
}

#[napi]
impl Workspace {
    /// Opens the workspace rooted at `root`. Creates nothing: the directories
    /// appear when a memory is first written.
    ///
    /// @throws on a config error.
    #[napi(constructor)]
    pub fn new(root: String, options: Option<WorkspaceOptions>) -> Result<Self> {
        let options = options.unwrap_or_default();
        let mut settings = Settings::load(options.config.as_deref().map(std::path::Path::new))
            .map_err(error::settings)?;

        if let Some(dim) = options.dim {
            let dim = dim as usize;
            if let Some(e) = &settings.embedder
                && e.dim() != dim
            {
                return Err(error::invalid_arg(format!(
                    "dim option {dim} disagrees with the configured embedder dimension {}",
                    e.dim()
                )));
            }
            settings.config.dim = dim;
        }

        // A JS number reaching a pool limit is untrusted the same way a config
        // value is: checked against the host's ceiling before it can become a
        // count of open file descriptors.
        if let Some(max_open) = options.max_open {
            settings.workspace.limits.max_open = checked_max_open(max_open)?;
        }
        if let Some(ms) = options.idle_timeout_ms {
            settings.workspace.limits.idle_timeout_ms = checked_timeout(ms)?;
        }

        let inner = settings
            .open_workspace(std::path::Path::new(&root))
            .map_err(error::workspace)?;
        Ok(Self {
            inner: Some(Arc::new(inner)),
        })
    }

    /// Opens the memory named `db` and returns it as a [`Plugmem`] — the same
    /// class, and the same verbs, as a memory opened by path.
    ///
    /// `create` defaults to `true`: a first use of an unused name brings that
    /// memory into being, which is what makes a new conversation need no
    /// registration step. Pass `false` to require that it already exists, which
    /// is what a read should do so a misspelled name is diagnosed rather than
    /// answered with nothing.
    ///
    /// @throws if the name is not a usable memory name, if it does not exist and
    /// `create` is false, or if another process holds it.
    /// **Async**: a first open replays the memory's journal and maps its
    /// snapshot, and making room in the pool closes another memory — file work
    /// that the one thread running JavaScript must not be holding.
    #[napi(ts_return_type = "Promise<Plugmem>")]
    pub fn open(&self, db: String, create: Option<bool>) -> Result<AsyncTask<OpenTask>> {
        let name = parse_name(&db)?;
        Ok(AsyncTask::new(OpenTask {
            workspace: self.shared()?,
            missing: if create.unwrap_or(true) {
                IfMissing::Create
            } else {
                IfMissing::Fail
            },
            name,
            now: now_ms(),
        }))
    }

    /// Every memory in the directory, sorted by name.
    ///
    /// Reads the filesystem, not the registry: a memory that exists but was
    /// never described still appears.
    /// **Async**: it reads the directory.
    #[napi(ts_return_type = "Promise<string[]>")]
    pub fn list(&self) -> Result<AsyncTask<RegistryTask>> {
        self.registry_task(Registry::List)
    }

    /// Every described memory, sorted by name.
    ///
    /// **Async**: the registry is itself a memory, so this opens and reads a
    /// database.
    #[napi(ts_return_type = "Promise<DbEntry[]>")]
    pub fn entries(&self) -> Result<AsyncTask<RegistryTask>> {
        self.registry_task(Registry::Entries)
    }

    /// The memories whose descriptions best match `query`, best first.
    ///
    /// This is the answer to "I do not know the name": ask in words, get names
    /// back, then open by name. A person's name works too — an owner is a graph
    /// edge, and the graph source reaches it.
    /// **Async**: this is a full recall against the registry memory — the same
    /// hybrid retrieval any other recall runs.
    #[napi(ts_return_type = "Promise<DbEntry[]>")]
    pub fn find(&self, query: String, k: Option<u32>) -> Result<AsyncTask<RegistryTask>> {
        self.registry_task(Registry::Find {
            query,
            k: k.unwrap_or(8) as usize,
        })
    }

    /// Records what a memory is for — in the memory itself and in the registry,
    /// so the registry can always be rebuilt from the memories. Creates the
    /// memory if it does not exist.
    ///
    /// Called again for the same memory this revises rather than duplicating,
    /// so the history of what it used to be for is kept.
    /// **Async**: it writes twice — into the memory itself and into the
    /// registry — and each write syncs a journal.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn describe(&self, db: String, args: DescribeArgs) -> Result<AsyncTask<RegistryTask>> {
        let name = parse_name(&db)?;
        self.registry_task(Registry::Describe { name, args })
    }

    /// Labels a memory archived, keeping its description. Returns whether
    /// anything changed. Archiving does not close, move or delete anything.
    ///
    /// @throws if the memory has no registry record to archive.
    /// **Async**: a registry write.
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn archive(&self, db: String) -> Result<AsyncTask<RegistryTask>> {
        let name = parse_name(&db)?;
        self.registry_task(Registry::Archive { name })
    }

    /// Rebuilds the registry from the memories' own descriptions.
    ///
    /// Runs on a libuv thread: it opens and reads every memory in the
    /// directory, which is not work for the main thread. A memory another
    /// process holds open cannot be read and is named in the report rather than
    /// skipped silently.
    #[napi(ts_return_type = "Promise<ReindexReport>")]
    pub fn reindex(&self) -> Result<AsyncTask<ReindexTask>> {
        Ok(AsyncTask::new(ReindexTask {
            workspace: self.shared()?,
            now: now_ms(),
        }))
    }

    /// Checks the registry against the directory. Reports every disagreement
    /// and repairs nothing — a workspace is a directory a person can edit, and
    /// guessing at their intent is how a consistency check loses data.
    #[napi(ts_return_type = "Promise<WorkspaceProblem[]>")]
    pub fn verify(&self) -> Result<AsyncTask<VerifyTask>> {
        Ok(AsyncTask::new(VerifyTask {
            workspace: self.shared()?,
            now: now_ms(),
        }))
    }

    /// Closes every memory unused for longer than the idle timeout, returning
    /// how many were closed. Call it on a timer: nothing else releases the file
    /// lock on a memory nobody is asking about.
    #[napi]
    pub fn close_idle(&self) -> Result<u32> {
        Ok(self.workspace()?.close_idle(now_ms()) as u32)
    }

    /// How many memories are open right now.
    #[napi]
    pub fn open_count(&self) -> Result<u32> {
        Ok(self.workspace()?.open_count() as u32)
    }

    /// Closes every pooled memory and the registry, releasing their file locks,
    /// and closes the workspace. Every method then throws.
    ///
    /// A [`Plugmem`] handed out by `open()` is **not** closed by this: it is its
    /// own handle and holds its own lock until it is closed or garbage
    /// collected.
    #[napi]
    pub fn close(&mut self) {
        if let Some(ws) = &self.inner {
            ws.close_all();
            ws.close_registry();
        }
        self.inner = None;
    }

    // ── internals ─────────────────────────────────────────────────────────

    /// The libuv task behind everything that touches the registry.
    fn registry_task(&self, what: Registry) -> Result<AsyncTask<RegistryTask>> {
        Ok(AsyncTask::new(RegistryTask {
            workspace: self.shared()?,
            what,
            now: now_ms(),
        }))
    }

    /// The open workspace, or a "closed" error.
    fn workspace(&self) -> Result<&plugmem_host::Workspace> {
        Ok(self.shared_ref()?.as_ref())
    }

    /// A shared handle for a task that runs off the main thread.
    fn shared(&self) -> Result<Arc<plugmem_host::Workspace>> {
        Ok(Arc::clone(self.shared_ref()?))
    }

    fn shared_ref(&self) -> Result<&Arc<plugmem_host::Workspace>> {
        self.inner.as_ref().ok_or_else(|| {
            error::coded(
                code::CLOSED,
                "workspace is closed (close() was called on this instance)",
            )
        })
    }
}

/// Opening a named memory, off the JS thread.
pub struct OpenTask {
    workspace: Arc<plugmem_host::Workspace>,
    name: DbName,
    missing: IfMissing,
    now: u64,
}

impl Task for OpenTask {
    type Output = (plugmem_host::Database, std::path::PathBuf);
    type JsValue = Plugmem;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let path = self.workspace.layout().path_of(&self.name);
        let db = self
            .workspace
            .get(&self.name, self.now, self.missing)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok((db, path))
    }

    fn resolve(&mut self, _env: Env, (db, path): Self::Output) -> napi::Result<Self::JsValue> {
        Ok(Plugmem::from_database(db, path))
    }
}

/// One thing to ask the registry.
enum Registry {
    List,
    Entries,
    Find { query: String, k: usize },
    Describe { name: DbName, args: DescribeArgs },
    Archive { name: DbName },
}

/// What a registry call resolves with.
pub enum RegistryOutput {
    Names(Vec<String>),
    Entries(Vec<DbEntry>),
    Changed(bool),
    Done,
}

/// The libuv-thread body of the registry verbs.
///
/// One task for all of them because the reason is the same in every case: the
/// registry is itself a plugmem memory, so listing, searching, describing and
/// archiving all open, read or write a database.
pub struct RegistryTask {
    workspace: Arc<plugmem_host::Workspace>,
    what: Registry,
    now: u64,
}

impl Task for RegistryTask {
    type Output = RegistryOutput;
    type JsValue = napi::bindgen_prelude::Either4<Vec<String>, Vec<DbEntry>, bool, ()>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let failed = |e: WorkspaceError| napi::Error::from_reason(e.to_string());
        match &self.what {
            Registry::List => Ok(RegistryOutput::Names(
                self.workspace
                    .layout()
                    .list()
                    .map_err(failed)?
                    .iter()
                    .map(DbName::to_string)
                    .collect(),
            )),
            Registry::Entries => Ok(RegistryOutput::Entries(
                self.workspace
                    .entries()
                    .map_err(failed)?
                    .iter()
                    .map(entry)
                    .collect(),
            )),
            Registry::Find { query, k } => Ok(RegistryOutput::Entries(
                self.workspace
                    .find(query, *k, self.now)
                    .map_err(failed)?
                    .iter()
                    .map(entry)
                    .collect(),
            )),
            Registry::Describe { name, args } => {
                let tags: Vec<&str> = args
                    .tags
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(String::as_str)
                    .collect();
                self.workspace
                    .describe(
                        name,
                        self.now,
                        Description {
                            text: &args.description,
                            tags: &tags,
                            owner: args.owner.as_deref(),
                        },
                    )
                    .map_err(failed)?;
                Ok(RegistryOutput::Done)
            }
            Registry::Archive { name } => self
                .workspace
                .archive(name, self.now)
                .map(RegistryOutput::Changed)
                .map_err(failed),
        }
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        use napi::bindgen_prelude::Either4;
        Ok(match output {
            RegistryOutput::Names(names) => Either4::A(names),
            RegistryOutput::Entries(entries) => Either4::B(entries),
            RegistryOutput::Changed(changed) => Either4::C(changed),
            RegistryOutput::Done => Either4::D(()),
        })
    }
}

/// The libuv-thread body of [`Workspace::reindex`].
pub struct ReindexTask {
    workspace: Arc<plugmem_host::Workspace>,
    now: u64,
}

impl Task for ReindexTask {
    type Output = plugmem_host::ReindexReport;
    type JsValue = ReindexReport;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.workspace
            .reindex(self.now)
            .map_err(error::workspace)
            .map_err(error::into_napi)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(ReindexReport {
            indexed: output.indexed.iter().map(DbName::to_string).collect(),
            undescribed: output.undescribed.iter().map(DbName::to_string).collect(),
            busy: output.busy.iter().map(DbName::to_string).collect(),
        })
    }
}

/// The libuv-thread body of [`Workspace::verify`].
pub struct VerifyTask {
    workspace: Arc<plugmem_host::Workspace>,
    now: u64,
}

impl Task for VerifyTask {
    type Output = Vec<WorkspaceIssue>;
    type JsValue = Vec<WorkspaceProblem>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.workspace
            .verify(self.now)
            .map_err(error::workspace)
            .map_err(error::into_napi)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output.iter().map(problem).collect())
    }
}

/// A host registry record as its typed mirror.
fn entry(e: &plugmem_host::DbEntry) -> DbEntry {
    DbEntry {
        db: e.name.to_string(),
        description: e.description.clone(),
        tags: e.tags.clone(),
        owner: e.owner.clone(),
        archived: e.is_archived(),
    }
}

/// A host issue as its typed mirror. The `issue` strings match what the CLI
/// prints in `--json`, so a script does not need a second vocabulary.
fn problem(issue: &WorkspaceIssue) -> WorkspaceProblem {
    let (db, kind, detail) = match issue {
        WorkspaceIssue::Missing { name } => (name.to_string(), "missing", None),
        WorkspaceIssue::Undescribed { name } => (name.to_string(), "undescribed", None),
        WorkspaceIssue::Stale { name } => (name.to_string(), "stale", None),
        WorkspaceIssue::Unreadable { name, why } => {
            (name.to_string(), "unreadable", Some(why.clone()))
        }
        WorkspaceIssue::AmbiguousSelf { name, facts } => (
            name.to_string(),
            "ambiguous-self",
            Some(format!("{facts} facts on the reserved anchor")),
        ),
        // `WorkspaceIssue` is non_exhaustive: a kind added later is reported as
        // itself rather than dropped.
        other => (String::new(), "unknown", Some(format!("{other:?}"))),
    };
    WorkspaceProblem {
        db,
        issue: kind.to_string(),
        detail,
    }
}

/// Validates a memory name, naming what was wrong with it.
fn parse_name(s: &str) -> Result<DbName> {
    DbName::parse(s).map_err(error::invalid_name)
}

/// A pool ceiling from JS, range-checked before it becomes a count of open
/// files. The comparison happens in `u32`/`usize` against the host's own
/// ceiling, so no value from a caller can talk the process out of descriptors.
fn checked_max_open(n: u32) -> Result<usize> {
    let ceiling = plugmem_host::MAX_OPEN_CEILING;
    if n == 0 || n as usize > ceiling {
        return Err(error::invalid_arg(format!(
            "maxOpen must be between 1 and {ceiling} (one open memory costs several file descriptors)"
        )));
    }
    Ok(n as usize)
}

/// An idle timeout from JS. A JS number is an `f64`, so a negative or
/// non-finite value has to be refused rather than cast.
fn checked_timeout(ms: f64) -> Result<u64> {
    if !ms.is_finite() || ms < 0.0 || ms > u64::MAX as f64 {
        return Err(error::invalid_arg(
            "idleTimeoutMs must be a non-negative number of milliseconds (0 disables the sweep)",
        ));
    }
    Ok(ms as u64)
}
