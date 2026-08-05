//! The `Plugmem` class: the Python surface over plugmem-host's [`Database`]
//! (and, in read-only mode, [`ReadOnlyDatabase`]).
//!
//! Every method wraps the identically-named host verb — the engine logic is
//! 100% `plugmem-host`; this layer only marshals arguments and results across
//! the CPython boundary. Its surface is `plugmem-napi`'s surface, including
//! where napi narrowed host: one `export_page` and no `export_each`, one
//! `maintain(mode)` and no `maintain_with_options`, `scrub(budget)` and no
//! `scrub_with_budget`. See the wrapper-parity rule in the repository's
//! `AGENTS.md`.
//!
//! **The GIL is released for every call that touches the engine.** The pattern
//! is the same three parts each time: convert Python values to owned Rust
//! values while attached, do the work inside [`Python::detach`], then build the
//! result once attached again. The middle part is what makes
//! `ThreadPoolExecutor` and `asyncio.to_thread` actually parallel rather than
//! merely concurrent — while it runs, no bytecode executes and the interpreter
//! is free.
//!
//! The handle lives behind an `RwLock` for a reason Node does not have: several
//! Python threads may hold the same object. Read verbs take the shared side and
//! genuinely run at once; `refresh` and `close` take the exclusive side, so a
//! reader never observes a half-swapped handle.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use plugmem_host::{
    Database, Embedder, FactId, LinkInput, MaintenanceMode, MaintenanceOptions, ReadOnlyDatabase,
    RecallQuery, RememberInput, Settings, SharedEmbedder, UnlinkInput,
};
use pyo3::prelude::*;
use pyo3::types::{PyMapping, PySequence};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::error::{self, Result};
use crate::scrub::Scrub;
use crate::types::{
    ExportPage, ExportedEdge, ExportedFact, FactSnapshot, MaintainReport, RecallResult,
    RecoverReport, RememberOutcome, Stats,
};

/// Keep one native→Python transfer bounded, matching the Node binding's page
/// grain so a paged export reads the same on both.
const EXPORT_PAGE_LIMIT: NonZeroUsize = NonZeroUsize::new(128).unwrap();

/// Edges handed to the Python callback per `export_edges` batch.
///
/// One call per *edge* would take and release the GIL a million times for a
/// million-edge dump. One call per batch makes it a thousand, and the walk
/// itself stays inside `detach` where it belongs.
const EXPORT_EDGE_BATCH: usize = 1024;

/// The system clock in unix milliseconds. The engine keeps no clock: every verb
/// that records time is told what time it is, exactly as the CLI and the MCP
/// server do it.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// What a `Plugmem` is open as.
enum Handle {
    /// The writer: every verb, exclusive lock on the file.
    Writer(Database),
    /// A reader over another process's published snapshot.
    Reader {
        /// The zero-copy handle.
        ///
        /// Boxed because a `ReadOnlyDatabase` owns its mmap and the self-cell
        /// borrowing it, which makes it far larger than a `Database` (an
        /// `Arc`). Unboxed, every `Handle` — including every writer — would pay
        /// the reader's size. The Node binding reaches the same shape through
        /// an `Arc`, which it needs anyway to hand the reader to a worker.
        db: Box<ReadOnlyDatabase>,
        /// The reader's own embedder, if the config built one.
        ///
        /// A [`ReadOnlyDatabase`] deliberately carries none — embedding inside
        /// it would mutate a mapping opened zero-copy — so the query is
        /// embedded out here and passed in as a vector. The CLI and MCP server
        /// do the same on their read-only paths, which is what lets a text
        /// `recall` reach the vector source on every surface.
        embedder: Option<SharedEmbedder>,
    },
}

/// A memory over one plugmem database — the Python mirror of
/// `plugmem_host::Database` (writer) or `ReadOnlyDatabase` (`read_only=True`).
///
/// Open it with [`Plugmem::open`], call the verbs, and `close()` it to release
/// the file. It is also a context manager, so `with Plugmem.open(path) as db:`
/// closes it for you.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "plugmem._plugmem")]
pub struct Plugmem {
    /// `None` once closed — every verb then raises `ClosedError`.
    handle: RwLock<Option<Handle>>,
    /// The database this handle resolved to. Kept because `open` may resolve it
    /// from `PLUGMEM_DB`, the config or the platform default, and a caller that
    /// passed no path otherwise has no way to learn what it just opened.
    path: PathBuf,
    /// Rendered `config.toml` warnings, carried from the open.
    warnings: Vec<String>,
}

/// One fact as `remember` / `revise` / `remember_many` take it, with every
/// string already owned so the work can leave the interpreter behind.
struct RememberSpec {
    text: String,
    entity: Option<String>,
    tags: Vec<String>,
    links: Vec<(String, String)>,
    metadata: Option<BTreeMap<String, String>>,
    valid_from: Option<u64>,
    vector: Option<Vec<f32>>,
}

impl RememberSpec {
    /// Borrow the owned strings back into the engine's input type.
    ///
    /// Split out because [`RememberInput`] borrows everything: the slices of
    /// `&str` it wants cannot outlive this call, so each caller builds them on
    /// its own stack right before handing the input to the engine.
    fn with_input<T>(&self, now: u64, f: impl FnOnce(RememberInput<'_>) -> T) -> T {
        let tags: Vec<&str> = self.tags.iter().map(String::as_str).collect();
        let links: Vec<(&str, &str)> = self
            .links
            .iter()
            .map(|(rel, entity)| (rel.as_str(), entity.as_str()))
            .collect();
        let metadata: Option<Vec<(&str, &str)>> = self.metadata.as_ref().map(|map| {
            map.iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .collect()
        });
        f(RememberInput {
            now,
            text: &self.text,
            entity: self.entity.as_deref(),
            tags: &tags,
            links: &links,
            vector: self.vector.as_deref(),
            valid_from: self.valid_from,
            metadata: metadata.as_deref(),
        })
    }

    /// Read one fact out of a mapping, for `remember_many`. The keys are the
    /// keyword-argument names of `remember`, so a caller writes the same thing
    /// either way.
    fn from_mapping(item: &Bound<'_, PyAny>) -> Result<Self> {
        let mapping = item.cast::<PyMapping>().map_err(|_| {
            error::invalid_arg("remember_many takes a sequence of dicts shaped like remember()'s keyword arguments")
        })?;
        let get = |key: &str| -> Result<Option<Bound<'_, PyAny>>> {
            match mapping.get_item(key) {
                Ok(value) if value.is_none() => Ok(None),
                Ok(value) => Ok(Some(value)),
                Err(_) => Ok(None),
            }
        };
        let text = match get("text")? {
            Some(value) => value
                .extract::<String>()
                .map_err(|_| error::invalid_arg("text must be a string"))?,
            None => return Err(error::invalid_arg("each fact needs a `text`")),
        };
        Ok(Self {
            text,
            entity: get("entity")?.map(|v| v.extract()).transpose()?,
            tags: get("tags")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default(),
            links: get("links")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default(),
            metadata: get("metadata")?.map(|v| v.extract()).transpose()?,
            valid_from: get("valid_from")?.map(|v| v.extract()).transpose()?,
            vector: checked_vector(get("vector")?.map(|v| v.extract()).transpose()?)?,
        })
    }
}

/// Refuse a vector Python can produce and the engine cannot judge: an empty one
/// (which would silently fall back to the embedder instead of using the
/// caller's model) and a non-finite component.
///
/// Length is deliberately *not* checked here — that is `dim`, and the engine
/// already refuses a vector that disagrees with it. Checking it twice would be
/// a second rule to keep in step.
fn checked_vector(vector: Option<Vec<f32>>) -> Result<Option<Vec<f32>>> {
    let Some(vector) = vector else {
        return Ok(None);
    };
    if vector.is_empty() {
        return Err(error::invalid_arg(
            "vector is empty: omit it to use the configured embedder",
        ));
    }
    if let Some(bad) = vector.iter().find(|v| !v.is_finite()) {
        return Err(error::invalid_arg(format!(
            "vector contains {bad}, which is not a finite number"
        )));
    }
    Ok(Some(vector))
}

#[gen_stub_pymethods]
#[pymethods]
impl Plugmem {
    /// Open (or create) the memory at `path` and return the handle.
    ///
    /// With `path` omitted, resolution is `PLUGMEM_DB`, then `[database].path`
    /// from the config, then the platform data path — and [`path`](Self::path)
    /// reports which it was.
    ///
    /// `dim` is the embedding dimension; omit it or pass 0 to keep vectors off.
    /// `read_only=True` opens over another process's writer and needs a
    /// published snapshot. `config` points at a `config.toml`; omitted, the
    /// standard discovery applies (`$PLUGMEM_CONFIG`, then the platform config
    /// path), exactly as the CLI and MCP server resolve it.
    ///
    /// Raises `LockedError` if another writer holds the lock,
    /// `NeedsCheckpointError` if `read_only` is asked of a database nobody has
    /// checkpointed, `ConfigError` for a bad `config.toml`, and `OpenError`
    /// otherwise.
    #[staticmethod]
    #[pyo3(signature = (path=None, *, dim=None, read_only=false, config=None))]
    fn open(
        py: Python<'_>,
        path: Option<String>,
        dim: Option<u32>,
        read_only: bool,
        config: Option<String>,
    ) -> Result<Self> {
        // Everything here is cheap and synchronous — reading `config.toml`,
        // resolving the path, checking the dimension — so a caller's mistake
        // raises where they stand rather than from inside the worker.
        let mut settings =
            Settings::load(config.as_deref().map(Path::new)).map_err(error::settings)?;

        let env_path = std::env::var_os("PLUGMEM_DB").map(PathBuf::from);
        let use_config_or_default_path = path.is_none() && env_path.is_none();
        let path = path
            .map(PathBuf::from)
            .or(env_path)
            .or_else(|| settings.database_path.clone())
            .or_else(plugmem_host::default_database_path)
            .unwrap_or_else(|| PathBuf::from("plugmem.db"));

        if use_config_or_default_path
            && !read_only
            && let Some(parent) = path.parent()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                error::OpenError::new_err(format!(
                    "cannot create database directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        apply_dim(&mut settings, dim)?;
        let warnings = settings.warnings.iter().map(|w| w.to_string()).collect();

        let handle = py.detach(|| -> Result<Handle> {
            if read_only {
                let mut settings = settings;
                let embedder = settings.embedder.take().map(SharedEmbedder::new);
                let db = Database::open_readonly(&path, settings.config).map_err(error::open)?;
                Ok(Handle::Reader {
                    db: Box::new(db),
                    embedder,
                })
            } else {
                let db = settings.open(&path).map_err(error::open)?;
                Ok(Handle::Writer(db))
            }
        })?;

        Ok(Self {
            handle: RwLock::new(Some(handle)),
            path,
            warnings,
        })
    }

    /// What `config.toml` said that nothing claimed — a misspelled key, a
    /// misspelled section — one human-readable line each, empty when the file
    /// was clean.
    ///
    /// It is a value rather than a printed warning because a library has
    /// nowhere sensible to print: stderr belongs to the application, which may
    /// be a server that logs as JSON. **Read it once after opening and log it
    /// your own way** — ignoring it puts you back where a typo silently
    /// changes nothing.
    fn config_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// The database this memory is open on.
    ///
    /// Worth having because `open` may resolve the path rather than be given
    /// one, and a caller that passed nothing otherwise cannot say what it just
    /// wrote to.
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// Store one fact and return its id plus similar/conflicting live facts.
    ///
    /// The engine never merges on its own: `similar` is the hint, and what to
    /// do about it — revise, forget, keep both — is the caller's decision.
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
        let now = now_ms();
        let outcome = py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            let db = writer(&guard)?;
            spec.with_input(now, |input| db.remember(input))
                .map_err(error::engine)
        })?;
        RememberOutcome::build(py, outcome)
    }

    /// Store a batch of facts and return one outcome per input.
    ///
    /// Each item is a mapping with `remember`'s keyword names (`text` is
    /// required). One journal write and one embedding round trip for the whole
    /// batch, which is the point.
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
            let item = sequence.get_item(index)?;
            specs.push(RememberSpec::from_mapping(&item)?);
        }

        let now = now_ms();
        let outcomes = py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            let db = writer(&guard)?;
            // Build every borrowed input first, then hand the whole batch over:
            // `remember_many` is one write, and splitting it would defeat that.
            with_inputs(&specs, now, |inputs| db.remember_many(inputs)).map_err(error::engine)
        })?;
        outcomes
            .into_iter()
            .map(|outcome| RememberOutcome::build(py, outcome))
            .collect()
    }

    /// Supersede a fact: close the old one's validity interval and record the
    /// successor, keeping the chain so an as-of query still answers about the
    /// old one.
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
        let now = now_ms();
        let outcome = py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            let db = writer(&guard)?;
            spec.with_input(now, |input| db.revise(FactId(id), input))
                .map_err(error::engine)
        })?;
        RememberOutcome::build(py, outcome)
    }

    /// Answer a query with ranked facts, the edges the graph source walked, and
    /// a rendered block bounded by `token_budget`.
    ///
    /// `as_of` is the truth axis ("what was true then"); `range` is a
    /// `(from, to)` window over the knowledge axis. `graph_depth` overrides the
    /// configured default for this call alone, because how wide a net to cast
    /// belongs to the question.
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
        let now = now_ms();

        let result = py.detach(|| -> Result<plugmem_host::RecallResult> {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let entity_refs: Vec<&str> = entities.iter().map(String::as_str).collect();

            // A read-only handle cannot embed, so the query is embedded out
            // here and handed in as a vector. A writer embeds for itself, and
            // giving it a vector would *replace* its embedder — so the two
            // branches differ in exactly one place and nowhere else.
            let embedded = match handle(&guard)? {
                Handle::Reader {
                    embedder: Some(embedder),
                    ..
                } if vector.is_none() => match query.as_deref() {
                    Some(text) => embedder
                        .embed(&[text])
                        .map_err(|e| error::EngineError::new_err(format!("embedding failed: {e}")))?
                        .into_iter()
                        .next(),
                    None => None,
                },
                _ => None,
            };
            let vector = vector.as_deref().or(embedded.as_deref());

            let q = RecallQuery {
                now,
                text: query.as_deref(),
                vector,
                tags: &tag_refs,
                entities: &entity_refs,
                as_of,
                range,
                k,
                token_budget,
                include_closed: closed,
                ef,
                graph_depth,
            };
            match handle(&guard)? {
                Handle::Writer(db) => db.recall(q).map_err(error::engine),
                Handle::Reader { db, .. } => db.recall(q).map_err(error::engine),
            }
        })?;
        RecallResult::build(py, result)
    }

    /// Tombstone a fact. Returns `True` if it was there to forget.
    ///
    /// The record is removed physically by `maintain`, not here — forgetting is
    /// a decision, purging is bookkeeping.
    fn forget(&self, py: Python<'_>, id: u32) -> Result<bool> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            writer(&guard)?
                .forget(now, FactId(id))
                .map_err(error::engine)
        })
    }

    /// Open a typed edge `src -[rel]-> dst`.
    ///
    /// `provenance` records the fact the edge follows from, which is what makes
    /// "why is this edge here" answerable later.
    #[pyo3(signature = (src, rel, dst, *, provenance=None))]
    fn link(
        &self,
        py: Python<'_>,
        src: String,
        rel: String,
        dst: String,
        provenance: Option<u32>,
    ) -> Result<()> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            writer(&guard)?
                .link(LinkInput {
                    now,
                    src: &src,
                    rel: &rel,
                    dst: &dst,
                    provenance: provenance.map(FactId),
                })
                .map_err(error::engine)
        })
    }

    /// Close an edge as of now. Returns `True` if an open one was there.
    ///
    /// The edge's history survives: an as-of query before this instant still
    /// walks it. That is why there is no "delete edge".
    fn unlink(&self, py: Python<'_>, src: String, rel: String, dst: String) -> Result<bool> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            writer(&guard)?
                .unlink(UnlinkInput {
                    now,
                    src: &src,
                    rel: &rel,
                    dst: &dst,
                })
                .map_err(error::engine)
        })
    }

    /// One fact's full card, or `None` if no such live fact exists.
    fn get(&self, py: Python<'_>, id: u32) -> Result<Option<FactSnapshot>> {
        let snapshot = py.detach(|| -> Result<Option<plugmem_host::FactSnapshot>> {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            Ok(match handle(&guard)? {
                Handle::Writer(db) => db.get(FactId(id)),
                Handle::Reader { db, .. } => db.get(FactId(id)),
            })
        })?;
        snapshot
            .map(|snapshot| FactSnapshot::build(py, snapshot))
            .transpose()
    }

    /// A fact's tags, as strings.
    fn tags_of(&self, py: Python<'_>, id: u32) -> Result<Vec<String>> {
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            Ok(match handle(&guard)? {
                Handle::Writer(db) => db.tags_of(FactId(id)),
                Handle::Reader { db, .. } => db.tags_of(FactId(id)),
            })
        })
    }

    /// Engine size counters.
    fn stats(&self, py: Python<'_>) -> Result<Stats> {
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            Ok(match handle(&guard)? {
                Handle::Writer(db) => Stats::from(db.stats()),
                Handle::Reader { db, .. } => Stats::from(db.stats()),
            })
        })
    }

    /// Every open fact, in one list.
    ///
    /// Fine for a memory you can hold in memory; for anything larger use
    /// [`export_page`](Self::export_page), which is the same dump in bounded
    /// pieces.
    fn export(&self, py: Python<'_>) -> Result<Vec<ExportedFact>> {
        let facts = py.detach(|| -> Result<Vec<plugmem_host::ExportedFact>> {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            Ok(match handle(&guard)? {
                Handle::Writer(db) => db.export(),
                Handle::Reader { db, .. } => db.export(),
            })
        })?;
        Ok(facts.into_iter().map(ExportedFact::from).collect())
    }

    /// One bounded page of open facts, in fact-id order.
    ///
    /// Pass `None` to start and then the page's `next_cursor` until it is
    /// `None`. `export_pages()` wraps this as a generator.
    #[pyo3(signature = (cursor=None))]
    fn export_page(&self, py: Python<'_>, cursor: Option<u32>) -> Result<ExportPage> {
        let page = py.detach(|| -> Result<plugmem_host::ExportPage> {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            let cursor = cursor.unwrap_or(0);
            Ok(match handle(&guard)? {
                Handle::Writer(db) => db.export_page(cursor, EXPORT_PAGE_LIMIT),
                Handle::Reader { db, .. } => db.export_page(cursor, EXPORT_PAGE_LIMIT),
            })
        })?;
        ExportPage::build(py, page)
    }

    /// Stream every edge to `on_batch`, a list of `ExportedEdge` at a time.
    /// Returns the total number of edges handed over.
    ///
    /// Batched rather than one call per edge because each call has to reattach
    /// to the interpreter: the walk itself runs with the GIL released, and the
    /// GIL is taken only to hand over a finished list.
    ///
    /// A dump is the two streams together — `export`/`export_page` for facts
    /// and this for edges. An edge is a statement *between* entities and
    /// outlives any single fact, so it is not part of a fact's record.
    fn export_edges(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(
            type_repr = "collections.abc.Callable[[builtins.list[ExportedEdge]], typing.Any]",
            imports = ("collections.abc", "builtins", "typing")
        ))]
        on_batch: Py<PyAny>,
    ) -> Result<usize> {
        let mut batch: Vec<(String, String, String, FactId)> =
            Vec::with_capacity(EXPORT_EDGE_BATCH);
        let mut total = 0usize;
        // The callback needs the interpreter, so the walk cannot simply sit
        // inside one `detach`: it collects a batch with the GIL released, then
        // reattaches to deliver it. `Python::attach` inside `detach` is the
        // supported way round, and the batch size is what keeps the number of
        // round trips down.
        let deliver = |batch: &mut Vec<(String, String, String, FactId)>| -> Result<()> {
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
            on_batch.call1(py, (edges,))?;
            Ok(())
        };

        let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
        let mut failure: Option<PyErr> = None;
        let walk = |src: &str, rel: &str, dst: &str, provenance: FactId| {
            if failure.is_some() {
                return;
            }
            total += 1;
            batch.push((src.to_owned(), rel.to_owned(), dst.to_owned(), provenance));
            if batch.len() >= EXPORT_EDGE_BATCH
                && let Err(e) = deliver(&mut batch)
            {
                failure = Some(e);
            }
        };
        match handle(&guard)? {
            Handle::Writer(db) => db.export_edges_each(walk),
            Handle::Reader { db, .. } => db.export_edges_each(walk),
        }
        if let Some(e) = failure {
            return Err(e);
        }
        deliver(&mut batch)?;
        Ok(total)
    }

    /// Check every index against the facts. Raises `EngineError` on the first
    /// inconsistency, returns `None` when the engine is coherent.
    ///
    /// This is the *logical* check. The byte-level one is [`scrub`](Self::scrub).
    fn verify(&self, py: Python<'_>) -> Result<()> {
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            match handle(&guard)? {
                Handle::Writer(db) => db.verify(),
                Handle::Reader { db, .. } => db.verify(),
            }
            .map_err(error::engine)
        })
    }

    /// A resumable byte-level check of the published snapshot.
    ///
    /// `verify` asks whether the indexes agree with the facts; this asks
    /// whether the bytes on disk are the bytes that were written. It holds a
    /// shared lock on the generation for as long as the returned object lives,
    /// so the writer's collector cannot recycle the file underneath it — drop
    /// the object, or call `close()`, to let go.
    #[pyo3(signature = (budget=None))]
    fn scrub(&self, py: Python<'_>, budget: Option<usize>) -> Result<Scrub> {
        let cursor = py.detach(|| -> Result<plugmem_host::Scrub> {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            let budget = budget.unwrap_or(plugmem_host::DEFAULT_SCRUB_BUDGET);
            match handle(&guard)? {
                Handle::Writer(db) => db.scrub_with_budget(budget),
                Handle::Reader { db, .. } => db.scrub_with_budget(budget),
            }
            .map_err(error::engine)
        })?;
        Ok(Scrub::new(cursor))
    }

    /// Run a maintenance pass and return what it did.
    ///
    /// `mode` is one of `"auto"` (the default: only pending work, cheap to run
    /// often), `"compact"`, `"reindex-text"`, `"optimize-vectors"` or `"full"`.
    #[pyo3(signature = (mode="auto"))]
    fn maintain(&self, py: Python<'_>, mode: &str) -> Result<MaintainReport> {
        let options = maintenance_options(mode)?;
        let now = now_ms();
        let report = py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            writer(&guard)?
                .maintain_with_options(now, options)
                .map_err(error::engine)
        })?;
        Ok(MaintainReport::from(report))
    }

    /// Publish a snapshot and truncate the journal.
    ///
    /// Also what makes the memory readable: a reader needs a published
    /// generation, so a database nobody has checkpointed cannot be opened
    /// `read_only=True`.
    fn checkpoint(&self, py: Python<'_>) -> Result<()> {
        let now = now_ms();
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            writer(&guard)?.checkpoint(now).map_err(error::engine)
        })
    }

    /// Which published generation this reader is looking at.
    ///
    /// Read-only handles only: a writer always sees its own latest state, so
    /// asking it raises `WriterOnlyError` rather than returning a number that
    /// would mean something different.
    fn generation(&self, py: Python<'_>) -> Result<u64> {
        py.detach(|| {
            let guard = self.handle.read().map_err(|_| error::busy("this memory"))?;
            Ok(reader(&guard)?.generation())
        })
    }

    /// Point this reader at the newest published generation. Returns `True` if
    /// it moved.
    ///
    /// Takes the handle exclusively, so it waits for in-flight reads on this
    /// object rather than swapping the mapping underneath them.
    fn refresh(&self, py: Python<'_>) -> Result<bool> {
        py.detach(|| {
            let mut guard = self
                .handle
                .write()
                .map_err(|_| error::busy("this memory"))?;
            match guard.as_mut() {
                Some(Handle::Reader { db, .. }) => db.refresh().map_err(error::engine),
                Some(Handle::Writer(_)) => Err(error::writer_only("refresh")),
                None => Err(error::closed()),
            }
        })
    }

    /// Release the database: unmap, drop the lock, and make every later verb
    /// raise `ClosedError`. Calling it twice is fine.
    fn close(&self, py: Python<'_>) -> Result<()> {
        py.detach(|| {
            let mut guard = self
                .handle
                .write()
                .map_err(|_| error::busy("this memory"))?;
            // Drop outside the lock's critical section is not needed here (the
            // handle's own teardown takes no plugmem lock), but taking the
            // value out rather than assigning `None` in place keeps the drop
            // ordering explicit.
            let handle = guard.take();
            drop(guard);
            drop(handle);
            Ok(())
        })
    }

    /// `with Plugmem.open(path) as db:` — returns the handle itself.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Closes the memory when the `with` block ends, however it ends.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> Result<bool> {
        self.close(py)?;
        // False: an exception inside the block propagates. Closing the memory
        // is cleanup, not a reason to swallow what went wrong.
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let state = match self.handle.read() {
            Ok(guard) => match guard.as_ref() {
                Some(Handle::Writer(_)) => "writer",
                Some(Handle::Reader { .. }) => "read-only",
                None => "closed",
            },
            Err(_) => "poisoned",
        };
        format!(
            "Plugmem(path={:?}, {state})",
            self.path.display().to_string()
        )
    }
}

impl Plugmem {
    /// Wrap a database a workspace already opened.
    ///
    /// Not a Python constructor: a memory in a workspace is reached by name
    /// through `Workspace.open`, which is what keeps a name from ever being a
    /// path. The class it hands back is this one, so a named memory has every
    /// verb a path-opened one does and there is no second implementation to
    /// drift.
    pub(crate) fn from_database(db: Database, path: PathBuf) -> Self {
        Self {
            handle: RwLock::new(Some(Handle::Writer(db))),
            path,
            // The workspace reported them once when it loaded the config; a
            // per-memory copy would repeat the same lines for every open.
            warnings: Vec::new(),
        }
    }
}

/// The open handle, or the "closed" error.
fn handle(guard: &Option<Handle>) -> Result<&Handle> {
    guard.as_ref().ok_or_else(error::closed)
}

/// The writer, or a read-only / closed error.
fn writer(guard: &Option<Handle>) -> Result<&Database> {
    match handle(guard)? {
        Handle::Writer(db) => Ok(db),
        Handle::Reader { .. } => Err(error::read_only()),
    }
}

/// The reader, or a writer-only / closed error.
fn reader(guard: &Option<Handle>) -> Result<&ReadOnlyDatabase> {
    match handle(guard)? {
        Handle::Reader { db, .. } => Ok(db),
        Handle::Writer(_) => Err(error::writer_only("generation")),
    }
}

/// Build every borrowed [`RememberInput`] for a batch and hand them over as one
/// call, so the whole batch is one write.
fn with_inputs<T>(
    specs: &[RememberSpec],
    now: u64,
    f: impl FnOnce(Vec<RememberInput<'_>>) -> T,
) -> T {
    // The borrowed slices must outlive the inputs that point at them, so they
    // are materialized here rather than inside `RememberSpec::with_input`.
    let tags: Vec<Vec<&str>> = specs
        .iter()
        .map(|s| s.tags.iter().map(String::as_str).collect())
        .collect();
    let links: Vec<Vec<(&str, &str)>> = specs
        .iter()
        .map(|s| {
            s.links
                .iter()
                .map(|(rel, entity)| (rel.as_str(), entity.as_str()))
                .collect()
        })
        .collect();
    let metadata: Vec<Option<Vec<(&str, &str)>>> = specs
        .iter()
        .map(|s| {
            s.metadata.as_ref().map(|map| {
                map.iter()
                    .map(|(key, value)| (key.as_str(), value.as_str()))
                    .collect()
            })
        })
        .collect();
    let inputs = specs
        .iter()
        .enumerate()
        .map(|(i, spec)| RememberInput {
            now,
            text: &spec.text,
            entity: spec.entity.as_deref(),
            tags: &tags[i],
            links: &links[i],
            vector: spec.vector.as_deref(),
            valid_from: spec.valid_from,
            metadata: metadata[i].as_deref(),
        })
        .collect();
    f(inputs)
}

/// Turn the mode string into engine options, naming the accepted values when it
/// is not one of them.
fn maintenance_options(mode: &str) -> Result<MaintenanceOptions> {
    let mode = match mode {
        "auto" => return Ok(MaintenanceOptions::auto()),
        "full" => return Ok(MaintenanceOptions::full()),
        "compact" => MaintenanceMode::Compact,
        "reindex-text" => MaintenanceMode::ReindexText,
        "optimize-vectors" => MaintenanceMode::OptimizeVectors,
        other => {
            return Err(error::invalid_arg(format!(
                "unknown maintain mode {other:?}: expected auto, compact, \
                 reindex-text, optimize-vectors or full"
            )));
        }
    };
    // An explicitly named mode keeps `auto`'s bounded vector budget.
    Ok(MaintenanceOptions {
        mode,
        ..MaintenanceOptions::auto()
    })
}

/// Reconcile the `dim` argument with what the config's embedder says.
///
/// The embedder's dimension is authoritative: a vector it produces has the
/// length it has. So a `dim` that disagrees is refused here rather than
/// silently overridden — a silent override would store vectors the embedder
/// cannot answer with.
fn apply_dim(settings: &mut Settings, dim: Option<u32>) -> Result<()> {
    let Some(dim) = dim.filter(|d| *d != 0) else {
        return Ok(());
    };
    let dim = dim as usize;
    if let Some(embedder) = &settings.embedder {
        let configured = embedder.dim();
        if configured != 0 && configured != dim {
            return Err(error::invalid_arg(format!(
                "dim {dim} disagrees with the configured embedder's dimension {configured}"
            )));
        }
    }
    settings.config.dim = dim;
    Ok(())
}

/// Salvage a content-corrupt memory into a clean copy.
///
/// Reads `src` fact by fact, writes what survives to `dst`, and reports what it
/// had to drop. The source is left untouched as evidence, and `dst` must not be
/// `src`.
///
/// This is not a repair for *structural* damage: a snapshot that will not parse
/// cannot be walked, and that case is a restore from backup, not a recovery.
#[gen_stub_pyfunction(module = "plugmem._plugmem")]
#[pyfunction]
#[pyo3(signature = (src, dst, *, dim=None, config=None))]
pub fn recover(
    py: Python<'_>,
    src: String,
    dst: String,
    dim: Option<u32>,
    config: Option<String>,
) -> Result<RecoverReport> {
    let mut settings = Settings::load(config.as_deref().map(Path::new)).map_err(error::settings)?;
    apply_dim(&mut settings, dim)?;
    let now = now_ms();
    let report =
        py.detach(|| Database::recover(&src, &dst, settings.config, now).map_err(error::engine))?;
    Ok(RecoverReport::from(report))
}
