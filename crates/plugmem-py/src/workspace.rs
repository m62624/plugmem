//! The `Workspace` class: many memories in one directory, addressed by name.
//!
//! **Optional.** A program that wants one memory calls [`Plugmem::open`] with a
//! path and never touches this file. This is for a process serving many
//! independent memories — one per conversation, per tenant, per project — that
//! should stay independent.
//!
//! The same rule as [`crate::db`]: every method wraps the identically-named
//! host verb, and the engine logic is 100% `plugmem-host`. `open` hands back a
//! [`Plugmem`] over a pooled handle, so a named memory has exactly the verbs a
//! path-opened one has, with no second implementation to drift.
//!
//! Three host methods are deliberately not here, matching the Node binding:
//! `registry()` (handing out the raw registry database invites writing into it
//! by hand), `entry(name)` (`entries()` already answers it), and the path
//! helpers (`path_of`, `exists`) — a caller that has a name never needs a path.
//! `close_all()` is absent for the same reason it is absent there: dropping the
//! workspace does it, and a verb that closes memories other code is using is a
//! footgun rather than a feature.
//!
//! **A constructor, not an `open` classmethod**, again mirroring napi: opening
//! a workspace reads a config and takes no lock, so there is no expensive work
//! to hand anywhere. The per-memory open is where the file work happens, and
//! that is [`Workspace::open`].

use std::path::Path;
use std::sync::RwLock;

use plugmem_host::{DbName, Description, IfMissing, Settings, WorkspaceIssue};
use pyo3::prelude::*;

use crate::db::{Plugmem, now_ms};
use crate::error::{self, Result};
use crate::types::{DbEntry, ReindexReport, WorkspaceProblem};

/// A directory of named memories — the Python mirror of
/// `plugmem_host::Workspace`.
///
/// The lock guards the handle slot, not the workspace: `plugmem_host::Workspace`
/// synchronizes its own pool and registry, so what needs protecting here is only
/// the `Option` that `close()` empties while other threads may be inside a verb.
#[pyclass(frozen, module = "plugmem")]
pub struct Workspace {
    /// `None` once closed — every method then raises `ClosedError`.
    inner: RwLock<Option<plugmem_host::Workspace>>,
}

#[pymethods]
impl Workspace {
    /// Open the workspace rooted at `root`. Creates nothing: the directories
    /// appear when a memory is first written.
    ///
    /// `dim` applies to every memory in the workspace and to the registry.
    /// `config` points at a `config.toml` whose `[workspace]` section supplies
    /// the pool defaults; `max_open` and `idle_timeout_ms` override them.
    ///
    /// `idle_timeout_ms` is a liveness setting, not a memory one: an open
    /// memory holds that database's exclusive lock, so a long-running process
    /// that never let go would make its memories unreachable from anything else
    /// on the machine.
    #[new]
    #[pyo3(signature = (root, *, dim=None, config=None, max_open=None, idle_timeout_ms=None))]
    fn new(
        py: Python<'_>,
        root: String,
        dim: Option<u32>,
        config: Option<String>,
        max_open: Option<u32>,
        idle_timeout_ms: Option<u64>,
    ) -> Result<Self> {
        let mut settings =
            Settings::load(config.as_deref().map(Path::new)).map_err(error::settings)?;

        if let Some(dim) = dim {
            let dim = dim as usize;
            if let Some(e) = &settings.embedder
                && e.dim() != dim
            {
                return Err(error::invalid_arg(format!(
                    "dim {dim} disagrees with the configured embedder's dimension {}",
                    e.dim()
                )));
            }
            settings.config.dim = dim;
        }
        if let Some(max_open) = max_open {
            settings.workspace.limits.max_open = checked_max_open(max_open)?;
        }
        if let Some(ms) = idle_timeout_ms {
            settings.workspace.limits.idle_timeout_ms = ms;
        }

        let inner = py.detach(|| {
            settings
                .open_workspace(Path::new(&root))
                .map_err(error::workspace)
        })?;
        Ok(Self {
            inner: RwLock::new(Some(inner)),
        })
    }

    /// Open the memory named `db` and return it as a [`Plugmem`] — the same
    /// class, and the same verbs, as a memory opened by path.
    ///
    /// `create` defaults to `True`: a first use of an unused name brings that
    /// memory into being, which is what makes a new conversation need no
    /// registration step. Pass `False` to require that it already exists, which
    /// is what a read should do so a misspelled name is diagnosed rather than
    /// answered with nothing.
    #[pyo3(signature = (db, *, create=true))]
    fn open(&self, py: Python<'_>, db: String, create: bool) -> Result<Plugmem> {
        let name = parse_name(&db)?;
        let missing = if create {
            IfMissing::Create
        } else {
            IfMissing::Fail
        };
        let now = now_ms();
        py.detach(|| {
            let guard = self.read()?;
            let workspace = opened(&guard)?;
            let path = workspace.layout().path_of(&name);
            workspace
                .get(&name, now, missing)
                .map(|db| Plugmem::from_database(db, path))
                .map_err(error::workspace)
        })
    }

    /// Every memory name in the directory, whether or not the registry knows
    /// about it. This is the filesystem's answer, so a memory created by
    /// another process appears here immediately.
    fn list(&self, py: Python<'_>) -> Result<Vec<String>> {
        py.detach(|| {
            let guard = self.read()?;
            Ok(opened(&guard)?
                .layout()
                .list()
                .map_err(error::workspace)?
                .iter()
                .map(DbName::to_string)
                .collect())
        })
    }

    /// Every memory the registry has a record for, with its description, tags,
    /// owner and archived flag.
    fn entries(&self, py: Python<'_>) -> Result<Vec<DbEntry>> {
        py.detach(|| {
            let guard = self.read()?;
            Ok(opened(&guard)?
                .entries()
                .map_err(error::workspace)?
                .iter()
                .map(entry)
                .collect())
        })
    }

    /// Search the descriptions and return the best matches — the answer to
    /// "which memory is the one about X" when the caller does not know the name.
    #[pyo3(signature = (query, *, k=0))]
    fn find(&self, py: Python<'_>, query: String, k: usize) -> Result<Vec<DbEntry>> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.read()?;
            Ok(opened(&guard)?
                .find(&query, k, now)
                .map_err(error::workspace)?
                .iter()
                .map(entry)
                .collect())
        })
    }

    /// Record what a memory is for. `description` is the text `find` matches;
    /// `owner` is recorded as a graph edge, so `find("ann")` returns what Ann
    /// owns even though no description mentions her.
    #[pyo3(signature = (db, description, *, tags=None, owner=None))]
    fn describe(
        &self,
        py: Python<'_>,
        db: String,
        description: String,
        tags: Option<Vec<String>>,
        owner: Option<String>,
    ) -> Result<()> {
        let name = parse_name(&db)?;
        let tags = tags.unwrap_or_default();
        let now = now_ms();
        py.detach(|| {
            let guard = self.read()?;
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            opened(&guard)?
                .describe(
                    &name,
                    now,
                    Description {
                        text: &description,
                        tags: &tag_refs,
                        owner: owner.as_deref(),
                    },
                )
                .map_err(error::workspace)
        })
    }

    /// Label a memory archived. Returns `True` if the label changed.
    ///
    /// Nothing is deleted and the memory still opens: this is a filter for
    /// `find`, not a lifecycle.
    fn archive(&self, py: Python<'_>, db: String) -> Result<bool> {
        let name = parse_name(&db)?;
        let now = now_ms();
        py.detach(|| {
            let guard = self.read()?;
            opened(&guard)?
                .archive(&name, now)
                .map_err(error::workspace)
        })
    }

    /// Copy each memory's own description into the registry, and report which
    /// were indexed, which nobody has described, and which another process held
    /// open so this pass could not read them.
    fn reindex(&self, py: Python<'_>) -> Result<ReindexReport> {
        let now = now_ms();
        let report = py.detach(|| {
            let guard = self.read()?;
            opened(&guard)?.reindex(now).map_err(error::workspace)
        })?;
        let plugmem_host::ReindexReport {
            indexed,
            undescribed,
            busy,
        } = report;
        Ok(ReindexReport {
            indexed: indexed.iter().map(DbName::to_string).collect(),
            undescribed: undescribed.iter().map(DbName::to_string).collect(),
            busy: busy.iter().map(DbName::to_string).collect(),
        })
    }

    /// Report what disagrees between the directory and the registry. Findings
    /// only: nothing is repaired, because which side is right is a judgement.
    fn verify(&self, py: Python<'_>) -> Result<Vec<WorkspaceProblem>> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.read()?;
            Ok(opened(&guard)?
                .verify(now)
                .map_err(error::workspace)?
                .iter()
                .map(problem)
                .collect())
        })
    }

    /// Close memories idle longer than the configured timeout and return how
    /// many were closed. Each close releases that database's exclusive lock.
    fn close_idle(&self, py: Python<'_>) -> Result<usize> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.read()?;
            Ok(opened(&guard)?.close_idle(now))
        })
    }

    /// How many memories the pool currently holds open.
    fn open_count(&self, py: Python<'_>) -> Result<usize> {
        py.detach(|| {
            let guard = self.read()?;
            Ok(opened(&guard)?.open_count())
        })
    }

    /// Close the workspace and every memory it holds. Calling it twice is fine.
    fn close(&self, py: Python<'_>) -> Result<()> {
        py.detach(|| {
            let mut guard = self
                .inner
                .write()
                .map_err(|_| error::busy("this workspace"))?;
            let inner = guard.take();
            drop(guard);
            drop(inner);
            Ok(())
        })
    }

    /// `with Workspace(root) as ws:` — returns the workspace itself.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Closes the workspace when the `with` block ends, however it ends.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> Result<bool> {
        self.close(py)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        match self.inner.read() {
            Ok(guard) => match guard.as_ref() {
                Some(workspace) => format!(
                    "Workspace(root={:?}, open={})",
                    workspace.layout().root().display().to_string(),
                    workspace.open_count()
                ),
                None => "Workspace(closed)".to_string(),
            },
            Err(_) => "Workspace(poisoned)".to_string(),
        }
    }
}

impl Workspace {
    /// The shared side of the handle slot.
    fn read(&self) -> Result<std::sync::RwLockReadGuard<'_, Option<plugmem_host::Workspace>>> {
        self.inner.read().map_err(|_| error::busy("this workspace"))
    }
}

/// The open workspace, or the "closed" error.
fn opened(guard: &Option<plugmem_host::Workspace>) -> Result<&plugmem_host::Workspace> {
    guard.as_ref().ok_or_else(error::closed)
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

/// Validate a memory name, naming what was wrong with it.
fn parse_name(s: &str) -> Result<DbName> {
    DbName::parse(s).map_err(error::invalid_name)
}

/// A pool ceiling from Python, range-checked before it becomes a count of open
/// files. No value from a caller can talk the process out of descriptors.
fn checked_max_open(n: u32) -> Result<usize> {
    let ceiling = plugmem_host::MAX_OPEN_CEILING;
    if n == 0 || n as usize > ceiling {
        return Err(error::invalid_arg(format!(
            "max_open must be between 1 and {ceiling} (one open memory costs \
             several file descriptors)"
        )));
    }
    Ok(n as usize)
}
