//! The `Workspace` class: many memories in one directory, addressed by name.
//!
//! **Optional.** A program that wants one memory calls [`Plugmem::open`] with a
//! path and never touches this file. This is for a process serving many
//! independent memories — one per conversation, per tenant, per project — that
//! should stay independent.
//!
//! The same rule as [`crate::db`]: every method wraps the identically-named
//! host verb, and the engine logic is 100% `plugmem-host`. `memory` hands back
//! a logical [`WorkspaceMemory`] reference. It owns no database handle or file
//! lock; every verb borrows one from the pool for that call.
//!
//! Three host methods are deliberately not here, matching the Node binding:
//! `registry()` (handing out the raw registry database invites writing into it
//! by hand), `entry(name)` (`entries()` already answers it), and the path
//! helpers (`path_of`, `exists`) — a caller that has a name never needs a path.
//! `close_all()` is absent because [`Workspace::close`] owns that lifecycle.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Weak};

use plugmem_host::{
    DbName, Description, FactId, IfMissing, LinkInput, RecallQuery, Settings, UnlinkInput,
    WorkspaceIssue,
};
use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::db::{
    EXPORT_EDGE_BATCH, EXPORT_PAGE_LIMIT, RememberSpec, checked_vector, maintenance_options,
    now_ms, with_inputs,
};
use crate::error::{self, Result};
use crate::scrub::Scrub;
use crate::types::{
    DbEntry, ExportPage, ExportedEdge, ExportedFact, FactSnapshot, MaintainReport, RecallResult,
    ReindexReport, RememberOutcome, RemoveTagReport, Stats, TagPage, WorkspaceProblem,
};

/// Shared owner of the host workspace.
struct WorkspaceState {
    host: plugmem_host::Workspace,
    closed: AtomicBool,
}

impl WorkspaceState {
    fn new(host: plugmem_host::Workspace) -> Self {
        Self {
            host,
            closed: AtomicBool::new(false),
        }
    }

    fn host(&self) -> Result<&plugmem_host::Workspace> {
        if self.closed.load(Ordering::Acquire) {
            Err(workspace_closed_error())
        } else {
            Ok(&self.host)
        }
    }

    fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.host.close_all();
            self.host.close_registry();
        }
    }
}

#[derive(Clone)]
struct NamedMemoryTarget {
    state: Arc<WorkspaceState>,
    name: DbName,
}

impl NamedMemoryTarget {
    fn with_database<T>(
        &self,
        missing: IfMissing,
        f: impl FnOnce(&plugmem_host::Database) -> Result<T>,
    ) -> Result<T> {
        let lease = self
            .state
            .host()?
            .lease(&self.name, now_ms(), missing)
            .map_err(error::workspace)?;
        f(&lease)
    }
}

/// A logical reference to one named memory. It owns no open database and no
/// file lock. Each verb obtains one scoped
/// lease while the GIL is released. Dropping this Python object has no resource
/// meaning; closing the workspace invalidates every reference.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "plugmem._plugmem")]
pub struct WorkspaceMemory {
    workspace: Weak<WorkspaceState>,
    name: DbName,
}

impl WorkspaceMemory {
    fn target(&self) -> Result<NamedMemoryTarget> {
        let state = self
            .workspace
            .upgrade()
            .ok_or_else(workspace_closed_error)?;
        state.host()?;
        Ok(NamedMemoryTarget {
            state,
            name: self.name.clone(),
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl WorkspaceMemory {
    /// The stable workspace name this reference addresses.
    fn name(&self) -> String {
        self.name.to_string()
    }

    #[pyo3(signature = (text, *, entity=None, tags=None, links=None, metadata=None, valid_from=None, vector=None))]
    #[allow(clippy::too_many_arguments)]
    fn remember(
        &self,
        py: Python<'_>,
        text: String,
        entity: Option<String>,
        tags: Option<Vec<String>>,
        links: Option<Vec<(String, String)>>,
        metadata: Option<BTreeMap<String, String>>,
        valid_from: Option<u64>,
        vector: Option<Vec<f32>>,
    ) -> Result<RememberOutcome> {
        let spec = RememberSpec {
            text,
            entity,
            tags: tags.unwrap_or_default(),
            links: links.unwrap_or_default(),
            metadata,
            valid_from,
            vector: checked_vector(vector)?,
        };
        let target = self.target()?;
        let now = now_ms();
        let outcome = py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                spec.with_input(now, |input| db.remember(input))
                    .map_err(error::engine)
            })
        })?;
        RememberOutcome::build(py, outcome)
    }

    fn remember_many(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(
            type_repr = "collections.abc.Sequence[collections.abc.Mapping[builtins.str, typing.Any]]",
            imports = ("collections.abc", "builtins", "typing")
        ))]
        facts: &Bound<'_, PyAny>,
    ) -> Result<Vec<RememberOutcome>> {
        let sequence = facts
            .cast::<PySequence>()
            .map_err(|_| error::invalid_arg("remember_many takes a sequence of dicts"))?;
        let mut specs = Vec::with_capacity(sequence.len().unwrap_or(0));
        for index in 0..sequence.len().unwrap_or(0) {
            specs.push(RememberSpec::from_mapping(&sequence.get_item(index)?)?);
        }
        let target = self.target()?;
        let now = now_ms();
        let outcomes = py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                with_inputs(&specs, now, |inputs| db.remember_many(inputs)).map_err(error::engine)
            })
        })?;
        outcomes
            .into_iter()
            .map(|outcome| RememberOutcome::build(py, outcome))
            .collect()
    }

    #[pyo3(signature = (id, text, *, entity=None, tags=None, links=None, metadata=None, valid_from=None, vector=None))]
    #[allow(clippy::too_many_arguments)]
    fn revise(
        &self,
        py: Python<'_>,
        id: u32,
        text: String,
        entity: Option<String>,
        tags: Option<Vec<String>>,
        links: Option<Vec<(String, String)>>,
        metadata: Option<BTreeMap<String, String>>,
        valid_from: Option<u64>,
        vector: Option<Vec<f32>>,
    ) -> Result<RememberOutcome> {
        let spec = RememberSpec {
            text,
            entity,
            tags: tags.unwrap_or_default(),
            links: links.unwrap_or_default(),
            metadata,
            valid_from,
            vector: checked_vector(vector)?,
        };
        let target = self.target()?;
        let now = now_ms();
        let outcome = py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                spec.with_input(now, |input| db.revise(FactId(id), input))
                    .map_err(error::engine)
            })
        })?;
        RememberOutcome::build(py, outcome)
    }

    #[pyo3(signature = (query=None, *, tags=None, entities=None, as_of=None, range=None, k=0, closed=false, token_budget=None, ef=None, graph_depth=None, vector=None))]
    #[allow(clippy::too_many_arguments)]
    fn recall(
        &self,
        py: Python<'_>,
        query: Option<String>,
        tags: Option<Vec<String>>,
        entities: Option<Vec<String>>,
        as_of: Option<u64>,
        range: Option<(u64, u64)>,
        k: usize,
        closed: bool,
        token_budget: Option<usize>,
        ef: Option<usize>,
        graph_depth: Option<u32>,
        vector: Option<Vec<f32>>,
    ) -> Result<RecallResult> {
        let vector = checked_vector(vector)?;
        let tags = tags.unwrap_or_default();
        let entities = entities.unwrap_or_default();
        let target = self.target()?;
        let now = now_ms();
        let result = py.detach(move || {
            target.with_database(IfMissing::Fail, |db| {
                let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                let entity_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
                db.recall(RecallQuery {
                    now,
                    text: query.as_deref(),
                    vector: vector.as_deref(),
                    tags: &tag_refs,
                    entities: &entity_refs,
                    as_of,
                    range,
                    k,
                    token_budget,
                    include_closed: closed,
                    ef,
                    graph_depth,
                })
                .map_err(error::engine)
            })
        })?;
        RecallResult::build(py, result)
    }

    fn forget(&self, py: Python<'_>, id: u32) -> Result<bool> {
        let target = self.target()?;
        let now = now_ms();
        py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                db.forget(now, FactId(id)).map_err(error::engine)
            })
        })
    }

    #[pyo3(signature = (src, rel, dst, *, provenance=None))]
    fn link(
        &self,
        py: Python<'_>,
        src: String,
        rel: String,
        dst: String,
        provenance: Option<u32>,
    ) -> Result<()> {
        let target = self.target()?;
        let now = now_ms();
        py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                db.link(LinkInput {
                    now,
                    src: &src,
                    rel: &rel,
                    dst: &dst,
                    provenance: provenance.map(FactId),
                })
                .map_err(error::engine)
            })
        })
    }

    fn unlink(&self, py: Python<'_>, src: String, rel: String, dst: String) -> Result<bool> {
        let target = self.target()?;
        let now = now_ms();
        py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                db.unlink(UnlinkInput {
                    now,
                    src: &src,
                    rel: &rel,
                    dst: &dst,
                })
                .map_err(error::engine)
            })
        })
    }

    fn get(&self, py: Python<'_>, id: u32) -> Result<Option<FactSnapshot>> {
        let target = self.target()?;
        let snapshot =
            py.detach(move || target.with_database(IfMissing::Fail, |db| Ok(db.get(FactId(id)))))?;
        snapshot
            .map(|value| FactSnapshot::build(py, value))
            .transpose()
    }

    fn tags_of(&self, py: Python<'_>, id: u32) -> Result<Vec<String>> {
        let target = self.target()?;
        py.detach(move || target.with_database(IfMissing::Fail, |db| Ok(db.tags_of(FactId(id)))))
    }

    #[pyo3(signature = (*, prefix=None, cursor=None, limit=0))]
    fn list_tags(
        &self,
        py: Python<'_>,
        prefix: Option<String>,
        cursor: Option<String>,
        limit: usize,
    ) -> Result<TagPage> {
        let target = self.target()?;
        let page = py.detach(move || {
            target.with_database(IfMissing::Fail, |db| {
                db.list_tags(plugmem_host::TagQuery {
                    prefix: prefix.as_deref(),
                    cursor: cursor.as_deref(),
                    limit,
                })
                .map_err(error::engine)
            })
        })?;
        TagPage::build(py, page)
    }

    fn remove_tag(&self, py: Python<'_>, tag: String) -> Result<RemoveTagReport> {
        let target = self.target()?;
        let now = now_ms();
        let report = py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                db.remove_tag(now, &tag).map_err(error::engine)
            })
        })?;
        Ok(RemoveTagReport::from(report))
    }

    fn stats(&self, py: Python<'_>) -> Result<Stats> {
        let target = self.target()?;
        py.detach(move || target.with_database(IfMissing::Fail, |db| Ok(Stats::from(db.stats()))))
    }

    fn export(&self, py: Python<'_>) -> Result<Vec<ExportedFact>> {
        let target = self.target()?;
        py.detach(move || {
            target.with_database(IfMissing::Fail, |db| {
                Ok(db.export().into_iter().map(ExportedFact::from).collect())
            })
        })
    }

    #[pyo3(signature = (cursor=None))]
    fn export_page(&self, py: Python<'_>, cursor: Option<u32>) -> Result<ExportPage> {
        let target = self.target()?;
        let page = py.detach(move || {
            target.with_database(IfMissing::Fail, |db| {
                Ok(db.export_page(cursor.unwrap_or(0), EXPORT_PAGE_LIMIT))
            })
        })?;
        ExportPage::build(py, page)
    }

    fn export_edges(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(
            type_repr = "collections.abc.Callable[[builtins.list[ExportedEdge]], typing.Any]",
            imports = ("collections.abc", "builtins", "typing")
        ))]
        on_batch: Py<PyAny>,
    ) -> Result<usize> {
        let target = self.target()?;
        py.detach(move || {
            target.with_database(IfMissing::Fail, |db| {
                let mut batch = Vec::with_capacity(EXPORT_EDGE_BATCH);
                let mut total = 0usize;
                let mut failure = None;
                db.export_edges_each(|src, rel, dst, provenance| {
                    if failure.is_some() {
                        return;
                    }
                    total += 1;
                    batch.push((src.to_owned(), rel.to_owned(), dst.to_owned(), provenance));
                    if batch.len() >= EXPORT_EDGE_BATCH
                        && let Err(err) = deliver_edges(&on_batch, &mut batch)
                    {
                        failure = Some(err);
                    }
                });
                if let Some(err) = failure {
                    return Err(err);
                }
                deliver_edges(&on_batch, &mut batch)?;
                Ok(total)
            })
        })
    }

    fn verify(&self, py: Python<'_>) -> Result<()> {
        let target = self.target()?;
        py.detach(move || {
            target.with_database(IfMissing::Fail, |db| db.verify().map_err(error::engine))
        })
    }

    #[pyo3(signature = (budget=None))]
    fn scrub(&self, py: Python<'_>, budget: Option<usize>) -> Result<Scrub> {
        let target = self.target()?;
        let cursor = py.detach(move || {
            target.with_database(IfMissing::Fail, |db| {
                db.scrub_with_budget(budget.unwrap_or(plugmem_host::DEFAULT_SCRUB_BUDGET))
                    .map_err(error::engine)
            })
        })?;
        Ok(Scrub::new(cursor))
    }

    #[pyo3(signature = (mode="auto"))]
    fn maintain(&self, py: Python<'_>, mode: &str) -> Result<MaintainReport> {
        let options = maintenance_options(mode)?;
        let target = self.target()?;
        let now = now_ms();
        let report = py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                db.maintain_with_options(now, options)
                    .map_err(error::engine)
            })
        })?;
        Ok(MaintainReport::from(report))
    }

    fn checkpoint(&self, py: Python<'_>) -> Result<()> {
        let target = self.target()?;
        let now = now_ms();
        py.detach(move || {
            target.with_database(IfMissing::Create, |db| {
                db.checkpoint(now).map_err(error::engine)
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("WorkspaceMemory(name={:?})", self.name.to_string())
    }
}

fn deliver_edges(
    callback: &Py<PyAny>,
    batch: &mut Vec<(String, String, String, FactId)>,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let edges: Vec<ExportedEdge> = batch
        .drain(..)
        .map(|(src, rel, dst, provenance)| ExportedEdge {
            src,
            rel,
            dst,
            provenance: (provenance != FactId::NONE).then_some(provenance.0),
        })
        .collect();
    Python::attach(|py| callback.call1(py, (edges,)).map(|_| ()))
}

/// A directory of named memories — the Python mirror of
/// `plugmem_host::Workspace`.
///
/// The lock guards the handle slot, not the workspace: `plugmem_host::Workspace`
/// synchronizes its own pool and registry, so what needs protecting here is only
/// the `Option` that `close()` empties while other threads may be inside a verb.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "plugmem._plugmem")]
pub struct Workspace {
    /// `None` once closed — every method then raises `ClosedError`.
    inner: RwLock<Option<Arc<WorkspaceState>>>,
}

#[gen_stub_pymethods]
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
            inner: RwLock::new(Some(Arc::new(WorkspaceState::new(inner)))),
        })
    }

    /// Return a logical reference to the memory named `db`.
    ///
    /// This does not open a file, acquire a lock or create the memory. Read
    /// verbs require it to exist; write verbs create it on first use. The
    /// reference stays valid across pool eviction and `release()`.
    fn memory(&self, db: String) -> Result<WorkspaceMemory> {
        let name = parse_name(&db)?;
        let workspace = self.shared()?;
        Ok(WorkspaceMemory {
            workspace: Arc::downgrade(&workspace),
            name,
        })
    }

    /// Evict one inactive memory and release its file lock. Logical references
    /// remain valid. If a verb is using this memory, raises
    /// `BusyError` immediately instead of waiting.
    fn release(&self, py: Python<'_>, db: String) -> Result<bool> {
        let name = parse_name(&db)?;
        py.detach(|| {
            let guard = self.read()?;
            opened(&guard)?.release(&name).map_err(error::workspace)
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

    /// Invalidate every logical memory reference and close pooled memories.
    /// Calling it twice is fine. An already-running verb releases its scoped
    /// handle when that call returns.
    fn close(&self, py: Python<'_>) -> Result<()> {
        py.detach(|| {
            let mut guard = self
                .inner
                .write()
                .map_err(|_| error::busy("this workspace"))?;
            let inner = guard.take();
            drop(guard);
            if let Some(state) = &inner {
                state.close();
            }
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
                Some(state) => match state.host() {
                    Ok(workspace) => format!(
                        "Workspace(root={:?}, open={})",
                        workspace.layout().root().display().to_string(),
                        workspace.open_count()
                    ),
                    Err(_) => "Workspace(closed)".to_string(),
                },
                None => "Workspace(closed)".to_string(),
            },
            Err(_) => "Workspace(poisoned)".to_string(),
        }
    }
}

impl Workspace {
    /// The shared side of the handle slot.
    fn read(&self) -> Result<std::sync::RwLockReadGuard<'_, Option<Arc<WorkspaceState>>>> {
        self.inner.read().map_err(|_| error::busy("this workspace"))
    }

    fn shared(&self) -> Result<Arc<WorkspaceState>> {
        let guard = self.read()?;
        let state = guard.as_ref().ok_or_else(workspace_closed_error)?;
        state.host()?;
        Ok(Arc::clone(state))
    }
}

/// The open workspace, or the "closed" error.
fn opened(guard: &Option<Arc<WorkspaceState>>) -> Result<&plugmem_host::Workspace> {
    guard.as_ref().ok_or_else(workspace_closed_error)?.host()
}

fn workspace_closed_error() -> PyErr {
    error::ClosedError::new_err("workspace is closed (close() was called on this instance)")
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
