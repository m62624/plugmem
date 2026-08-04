//! The `Plugmem` class: the napi surface over plugmem-host's [`Database`] (and,
//! in read-only mode, [`ReadOnlyDatabase`]).
//!
//! Every method wraps the identically-named host verb — the engine logic is
//! 100% `plugmem-host`; this layer only marshals arguments and results across
//! the Node boundary. `recover` is deliberately CLI-only alongside `import`
//! and `scrub` — salvage is a path-level operation on the host's own disk. Inputs are typed `#[napi(object)]` structs so napi emits
//! precise TypeScript interfaces (autocomplete in a TS host like Pi); results
//! come back as the typed mirrors in [`crate::types`]. A failure becomes a
//! thrown JS `Error` carrying a `code` — see [`crate::error`] for what is coded
//! and why the engine's own failures are not; the clock is the system clock,
//! read per call (the engine keeps none), exactly as the MCP server does.
//!
//! Opened `readOnly`, the instance observes another process's writer over a
//! shared snapshot: the read verbs answer, the write verbs throw, and the two
//! freshness verbs (`generation`/`refresh`) become available.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Task};
use napi_derive::napi;
use plugmem_host::{
    Database, Embedder, FactId, LinkInput, MaintenanceOptions, ReadOnlyDatabase, RecallQuery,
    RememberInput, Settings, UnlinkInput,
};

use crate::error::{self, Error, Result, code};
use crate::types::{
    self, ExportPage, ExportedFact, FactSnapshot, MaintainMode, MaintainReport, RecallResult,
    RememberOutcome, Stats,
};

/// Keep one native→JS transfer bounded. This matches the workspace's existing
/// conservative batch grain (CLI import) without exposing a tuning knob that
/// would let callers accidentally reconstruct an unbounded export.
const EXPORT_PAGE_LIMIT: NonZeroUsize = NonZeroUsize::new(128).unwrap();

/// Options for [`Plugmem::new`].
#[napi(object)]
#[derive(Default)]
pub struct OpenOptions {
    /// Embedding dimension (`Config::dim`). Omit or 0 to keep vectors off. When
    /// the config file configures an `[embedder]`, that embedder's dimension is
    /// authoritative and this — if given — must agree with it.
    pub dim: Option<u32>,
    /// Open read-only over another process's writer (requires a checkpointed
    /// database). The write verbs then throw; `generation`/`refresh` appear.
    ///
    /// A text `recall` still reaches the vector source: the engine cannot embed
    /// on a read-only handle (replaying into it would defeat the zero-copy
    /// open), so this binding embeds the query itself before asking — the same
    /// thing the CLI and the MCP server do for their read-only paths.
    pub read_only: Option<bool>,
    /// Path to a `config.toml` (`[database]` / `[engine]` / `[embedder]` /
    /// `[maintenance]`). When the constructor path is omitted, `[database].path`
    /// participates in database-path resolution.
    /// Omitted, the standard discovery applies — `$PLUGMEM_CONFIG`, then
    /// `$XDG_CONFIG_HOME/plugmem/config.toml` — exactly as the CLI and MCP
    /// server resolve it. The `[embedder]` section is what makes a text-only
    /// `remember`/`recall` auto-embed; with no config there is no embedder
    /// (lexical, tag, graph and time recall still answer).
    pub config: Option<String>,
}

/// One typed edge on a remembered fact: `entity` gains relation `rel`.
#[napi(object)]
pub struct LinkRef {
    /// The relation name.
    pub rel: String,
    /// The target entity name.
    pub entity: String,
}

/// Arguments for [`Plugmem::remember`] / [`Plugmem::revise`].
#[napi(object)]
pub struct RememberArgs {
    /// The fact text (required).
    pub text: String,
    /// Subject entity name.
    pub entity: Option<String>,
    /// Tag strings.
    pub tags: Option<Vec<String>>,
    /// Typed edges to attach.
    pub links: Option<Vec<LinkRef>>,
    /// Opaque metadata as a key→value map (a URI to the real payload, a mime
    /// type, an external key). The engine never interprets it.
    pub metadata: Option<BTreeMap<String, String>>,
    /// Validity start, unix milliseconds (default: the fact's record time).
    pub valid_from: Option<f64>,
}

/// Arguments for [`Plugmem::recall`] — every field optional; lexical/tag/graph/
/// time still answer with no query vector.
#[napi(object)]
#[derive(Default)]
pub struct RecallArgs {
    /// Free-text query (embedded by the engine when an embedder is configured).
    pub query: Option<String>,
    /// Restrict to facts carrying all these tags.
    pub tags: Option<Vec<String>>,
    /// Anchor entities for the graph source.
    pub entities: Option<Vec<String>>,
    /// "What was true at" this instant, unix milliseconds (bitemporal as-of).
    pub as_of: Option<f64>,
    /// Time window `[from, to)` over `recorded_at`, unix milliseconds.
    pub range: Option<Vec<f64>>,
    /// Max facts to return (0 = engine default).
    pub k: Option<u32>,
    /// Include closed revisions (default false).
    pub closed: Option<bool>,
}

impl RecallArgs {
    /// The `[from, to)` window as the engine wants it.
    ///
    /// Exactly two elements, or the call is refused. A short array used to make
    /// the window vanish silently, and with it the engine's whole temporal
    /// source: `recall({ range: [from] })` came back with no time-ranked
    /// candidates at all, indistinguishable from a call that never asked for
    /// any. A plausible wrong answer is worse than an error.
    fn window(&self) -> Result<Option<(u64, u64)>> {
        let Some(range) = self.range.as_deref() else {
            return Ok(None);
        };
        let [from, to] = range else {
            return Err(error::invalid_arg(format!(
                "range must be exactly [from, to] in unix milliseconds, got {} value(s)",
                range.len()
            )));
        };
        let (from, to) = (checked_ms(*from, "range[0]")?, checked_ms(*to, "range[1]")?);
        if from > to {
            return Err(error::invalid_arg(format!(
                "range start {from} is after its end {to}"
            )));
        }
        Ok(Some((from, to)))
    }
}

/// Arguments for [`Plugmem::link`].
#[napi(object)]
pub struct LinkArgs {
    /// Source entity name.
    pub src: String,
    /// Relation name.
    pub rel: String,
    /// Destination entity name.
    pub dst: String,
}

/// The open handle behind a [`Plugmem`]: a read-write writer, or a read-only
/// observer of another process's writer. The reader is reference-counted so an
/// async export task can outlive the JS object that scheduled it.
enum Handle {
    Writer(Database),
    Reader {
        db: Arc<ReadOnlyDatabase>,
        /// The read-only handle's own embedder, if the config built one.
        ///
        /// A [`ReadOnlyDatabase`] deliberately carries none — embedding inside
        /// it would mutate a mapping opened zero-copy — so the query is embedded
        /// out here and passed in as a vector.
        ///
        /// Owned outright, with no lock: [`Embedder::embed`] wants `&mut self`,
        /// and `recall` takes `&mut self` to give it one. A `Mutex` would be
        /// pure ceremony — one JS instance is only ever touched by its own
        /// thread, and the async verbs capture cloned engine handles, never
        /// this. (The MCP server does wrap its embedder, because there one
        /// reader really is shared by a pool of worker threads.)
        embedder: Option<Box<dyn Embedder>>,
    },
}

/// A memory over one plugmem file — the napi mirror of [`plugmem_host::Database`]
/// (writer) or [`plugmem_host::ReadOnlyDatabase`] (with `{ readOnly: true }`).
/// Construct it, call the verbs, and `close()` it to release the file when done.
#[napi]
pub struct Plugmem {
    /// `None` once `close()`d — every verb then throws "memory is closed".
    handle: Option<Handle>,
    /// The file this handle resolved to. Kept because the constructor may
    /// resolve it from `PLUGMEM_DB`, the config or the platform default, and a
    /// caller that passed no path otherwise has no way to learn which file it
    /// just opened.
    path: PathBuf,
}

#[napi]
impl Plugmem {
    /// Opens (or creates) the memory at `path`. If omitted, path resolution is
    /// `PLUGMEM_DB` > `[database].path` > the platform data path.
    ///
    /// @throws if the file is locked by another writer, if `readOnly` is set on a
    /// database with no published snapshot, or on a config/IO error.
    #[napi(constructor)]
    pub fn new(path: Option<String>, options: Option<OpenOptions>) -> Result<Self> {
        let options = options.unwrap_or_default();

        // Resolve settings exactly like the CLI and MCP server: an explicit
        // `config` path wins, else `$PLUGMEM_CONFIG`, else the platform config
        // path. This is what attaches the `[embedder]`, so a text-only
        // `remember`/`recall` auto-embeds at parity with the other surfaces.
        let mut settings =
            Settings::load(options.config.as_deref().map(Path::new)).map_err(error::settings)?;

        let env_path = std::env::var_os("PLUGMEM_DB").map(PathBuf::from);
        let use_config_or_default_path = path.is_none() && env_path.is_none();
        let path = path
            .map(PathBuf::from)
            .or(env_path)
            .or_else(|| settings.database_path.clone())
            .or_else(plugmem_host::default_database_path)
            .unwrap_or_else(|| PathBuf::from("plugmem.db"));

        if use_config_or_default_path
            && !options.read_only.unwrap_or(false)
            && let Some(parent) = path.parent()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                error::coded(
                    code::OPEN,
                    format!("cannot create database directory {}: {e}", parent.display()),
                )
            })?;
        }

        // A `dim` option sets the embedding dimension. If the config built an
        // embedder, that embedder's dimension is authoritative — `dim` must
        // agree, or the vectors would silently disagree in size.
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

        let handle = if options.read_only.unwrap_or(false) {
            // A read-only handle observes another writer's snapshot. The engine
            // cannot embed inside it, so the embedder is kept out here and the
            // query is embedded before the read — the same arrangement the CLI
            // and the MCP server use for their read-only paths, so a text
            // `recall` reaches the vector source on every surface.
            let embedder = settings.embedder.take();
            let ro = Database::open_readonly(&path, settings.config).map_err(error::open)?;
            Handle::Reader {
                db: Arc::new(ro),
                embedder,
            }
        } else {
            // The writer takes the embedder and maintenance policy from settings.
            let db = settings.open(&path).map_err(error::open)?;
            Handle::Writer(db)
        };
        Ok(Self {
            handle: Some(handle),
            path,
        })
    }

    /// Wraps a database handle a workspace already opened.
    ///
    /// Not a JS constructor: a memory in a workspace is reached by name through
    /// `Workspace.open`, which is what keeps a name from ever being a path.
    /// The class it hands back is this one, so a named memory has every verb a
    /// path-opened one does and there is no second implementation to drift.
    pub(crate) fn from_database(db: Database, path: PathBuf) -> Self {
        Self {
            handle: Some(Handle::Writer(db)),
            path,
        }
    }

    /// The file this memory is open on.
    ///
    /// Worth having because the constructor may resolve the path rather than be
    /// given one — `PLUGMEM_DB`, then `[database].path`, then the platform data
    /// path — and `new Plugmem()` with no argument otherwise leaves the caller
    /// unable to say which file it just wrote to.
    #[napi]
    pub fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// Stores a fact; returns its id plus similar/conflicting live facts.
    /// @throws in read-only mode.
    #[napi]
    pub fn remember(&self, args: RememberArgs) -> Result<RememberOutcome> {
        do_remember(self.writer()?, &args, None)
    }

    /// Stores a batch of facts and resolves with one outcome per input.
    ///
    /// A batch may call a remote embedder and always performs one journal sync,
    /// so it runs on napi-rs' libuv worker pool.
    #[napi(ts_return_type = "Promise<RememberOutcome[]>")]
    pub fn remember_many(&self, args: Vec<RememberArgs>) -> Result<AsyncTask<RememberManyTask>> {
        let db = self.writer()?.clone();
        // Validated here, on the JS thread, not inside `compute`: an argument
        // this wrapper refuses has to throw where the caller stands, and a
        // coded error cannot cross the `Task` boundary anyway (see `error`).
        let valid_from = args
            .iter()
            .enumerate()
            .map(|(i, input)| {
                input
                    .valid_from
                    .map(|value| checked_ms(value, &format!("[{i}].validFrom")))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(AsyncTask::new(RememberManyTask {
            db,
            args,
            valid_from,
            now: now_ms(),
        }))
    }

    /// Closes fact `id` and records `args` as its successor; returns the outcome.
    /// @throws in read-only mode.
    #[napi]
    pub fn revise(&self, id: u32, args: RememberArgs) -> Result<RememberOutcome> {
        do_remember(self.writer()?, &args, Some(FactId(id)))
    }

    /// Ranked, fused recall. Returns the structured result (its `rendered` field
    /// is the prompt-ready block; `facts`/`edges` are the structured hits).
    #[napi]
    pub fn recall(&mut self, args: Option<RecallArgs>) -> Result<RecallResult> {
        let args = args.unwrap_or_default();
        // Refuse a malformed window or instant before anything is read: these
        // are `f64`s from JS, and casting one blindly would turn a caller's
        // mistake into a different, plausible-looking query.
        let range = args.window()?;
        let as_of = args.as_of.map(|v| checked_ms(v, "asOf")).transpose()?;

        // A read-only handle carries the embedder itself, because the engine
        // cannot embed into a zero-copy mapping. Do it here so a text query
        // reaches the vector source, then read with `&self` as usual.
        let embedded = match self.handle_mut()? {
            Handle::Reader {
                embedder: Some(embedder),
                ..
            } => match args.query.as_deref() {
                Some(text) => embedder.embed(&[text]).map_err(error::engine)?.pop(),
                None => None,
            },
            _ => None,
        };

        let tags = str_refs(&args.tags);
        let entities = str_refs(&args.entities);
        let q = RecallQuery {
            now: now_ms(),
            text: args.query.as_deref(),
            vector: embedded.as_deref(),
            tags: &tags,
            entities: &entities,
            as_of,
            range,
            k: args.k.unwrap_or(0) as usize,
            token_budget: None,
            include_closed: args.closed.unwrap_or(false),
            ef: None,
        };
        let res = match self.handle()? {
            Handle::Writer(db) => db.recall(q),
            Handle::Reader { db, .. } => db.recall(q),
        }
        .map_err(error::engine)?;
        types::to_typed(&res)
    }

    /// Tombstones fact `id` (physically purged at the next `maintain`). Returns
    /// whether it was a live fact. @throws in read-only mode.
    #[napi]
    pub fn forget(&self, id: u32) -> Result<bool> {
        self.writer()?
            .forget(now_ms(), FactId(id))
            .map_err(error::engine)
    }

    /// Upserts a typed edge `src -rel-> dst`. @throws in read-only mode.
    #[napi]
    pub fn link(&self, args: LinkArgs) -> Result<()> {
        self.writer()?
            .link(LinkInput {
                now: now_ms(),
                src: &args.src,
                rel: &args.rel,
                dst: &args.dst,
                provenance: None,
            })
            .map_err(error::engine)
    }

    /// Closes the current typed edge `src -rel-> dst`. @throws in read-only mode.
    #[napi]
    pub fn unlink(&self, args: LinkArgs) -> Result<bool> {
        self.writer()?
            .unlink(UnlinkInput {
                now: now_ms(),
                src: &args.src,
                rel: &args.rel,
                dst: &args.dst,
            })
            .map_err(error::engine)
    }

    /// One fact's full card by `id`, or `null` if unknown/tombstoned.
    #[napi]
    pub fn get(&self, id: u32) -> Result<Option<FactSnapshot>> {
        let snap = match self.handle()? {
            Handle::Writer(db) => db.get(FactId(id)),
            Handle::Reader { db, .. } => db.get(FactId(id)),
        };
        match snap {
            Some(snap) => Ok(Some(types::to_typed(&snap)?)),
            None => Ok(None),
        }
    }

    /// Engine size counters.
    #[napi]
    pub fn stats(&self) -> Result<Stats> {
        let stats = match self.handle()? {
            Handle::Writer(db) => db.stats(),
            Handle::Reader { db, .. } => db.stats(),
        };
        types::to_typed(&stats)
    }

    /// Every currently-open fact, as an array (id-free, import-ready).
    ///
    /// **Synchronous and unbounded**: it materializes the whole memory into one
    /// array on the JS thread, so a large database blocks the event loop for as
    /// long as that takes. `exportPage` is the same data in bounded pages on a
    /// libuv thread — prefer it for anything but a small memory or a script.
    #[napi]
    pub fn export(&self) -> Result<Vec<ExportedFact>> {
        let facts = match self.handle()? {
            Handle::Writer(db) => db.export(),
            Handle::Reader { db, .. } => db.export(),
        };
        Ok(facts.into_iter().map(ExportedFact::from).collect())
    }

    /// Returns at most 128 inspected fact ids' open facts on a libuv worker
    /// thread. A sparse page can be empty and still carry `nextCursor`.
    ///
    /// Pass `nextCursor` back as `cursor` until it is absent. Each Promise owns
    /// exactly one bounded page and resolves only after its native scan has
    /// completed; there is no callback queue and no database lock held while JS
    /// processes the result. A writer may change between page calls, so do not
    /// mutate it during a snapshot-style backup; a read-only handle is stable.
    #[napi(ts_return_type = "Promise<ExportPage>")]
    pub fn export_page(&self, cursor: Option<u32>) -> Result<AsyncTask<ExportPageTask>> {
        let source = match self.handle()? {
            Handle::Writer(db) => ExportSource::Writer(db.clone()),
            Handle::Reader { db, .. } => ExportSource::Reader(Arc::clone(db)),
        };
        Ok(AsyncTask::new(ExportPageTask {
            source,
            cursor: cursor.unwrap_or(0),
        }))
    }

    /// One fact's tags, or an empty array for an unknown or tombstoned id.
    #[napi]
    pub fn tags_of(&self, id: u32) -> Result<Vec<String>> {
        match self.handle()? {
            Handle::Writer(db) => Ok(db.tags_of(FactId(id))),
            Handle::Reader { db, .. } => Ok(db.tags_of(FactId(id))),
        }
    }

    /// Content-integrity check; throws on the first inconsistency found.
    #[napi]
    pub fn verify(&self) -> Result<()> {
        match self.handle()? {
            Handle::Writer(db) => db.verify(),
            Handle::Reader { db, .. } => db.verify(),
        }
        .map_err(error::engine)
    }

    /// Runs policy-driven maintenance; resolves with the before/after report.
    /// **Async** (returns a `Promise`): the pass may do disk I/O (compaction,
    /// HNSW work), so it runs on a libuv worker thread and never blocks the
    /// event loop. @throws synchronously in read-only mode.
    ///
    /// `mode` defaults to `auto`, which does only what is pending. `full`
    /// rebuilds everything and repacks the edge arenas — O(database) work, and
    /// the only mode that reclaims edge-history page slack.
    #[napi(ts_return_type = "Promise<MaintainReport>")]
    pub fn maintain(
        &self,
        // The generated type for a napi string enum is `const enum`, which
        // TypeScript refuses to import from a declaration file under
        // `isolatedModules` — the setting Vite, esbuild and swc all imply. The
        // runtime has always accepted the plain string (the enum's values *are*
        // these strings), so the declared type says so and both worlds work.
        #[napi(ts_arg_type = "'auto' | 'compact' | 'reindex-text' | 'optimize-vectors' | 'full'")]
        mode: Option<MaintainMode>,
    ) -> Result<AsyncTask<MaintainTask>> {
        Ok(AsyncTask::new(MaintainTask {
            db: self.writer()?.clone(),
            now: now_ms(),
            options: mode.unwrap_or(MaintainMode::Auto).into(),
        }))
    }

    /// Flushes the journal into a fresh snapshot. **Async** (returns a `Promise`):
    /// it writes and fsyncs a snapshot file, so it runs on a libuv worker thread.
    /// @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn checkpoint(&self) -> Result<AsyncTask<CheckpointTask>> {
        Ok(AsyncTask::new(CheckpointTask {
            db: self.writer()?.clone(),
            now: now_ms(),
        }))
    }

    /// The pinned snapshot generation (read-only mode only).
    /// @throws on a writer.
    #[napi]
    pub fn generation(&self) -> Result<f64> {
        Ok(self.reader()?.generation() as f64)
    }

    /// Advance to the writer's latest published checkpoint (read-only mode only);
    /// returns whether a newer generation was adopted. @throws on a writer.
    #[napi]
    pub fn refresh(&mut self) -> Result<bool> {
        match self.handle_mut()? {
            Handle::Reader { db, .. } => Arc::get_mut(db)
                .ok_or_else(|| {
                    error::coded(code::BUSY, "cannot refresh while an export page is running")
                })?
                .refresh()
                .map_err(error::engine),
            Handle::Writer(_) => Err(writer_only_error("refresh")),
        }
    }

    /// Releases the file and its lock. Every verb afterwards throws; calling it
    /// again is a no-op. (The handle is also released when the object is GC'd,
    /// but `close()` makes the moment explicit — e.g. before a read-only reopen.)
    #[napi]
    pub fn close(&mut self) {
        self.handle = None;
    }

    // ── internals ─────────────────────────────────────────────────────────

    /// The open handle, or a "closed" error.
    fn handle(&self) -> Result<&Handle> {
        self.handle.as_ref().ok_or_else(closed_error)
    }

    /// The open handle by exclusive reference (for `refresh`), or "closed".
    fn handle_mut(&mut self) -> Result<&mut Handle> {
        self.handle.as_mut().ok_or_else(closed_error)
    }

    /// The writer handle, or a read-only / closed error.
    fn writer(&self) -> Result<&Database> {
        match self.handle()? {
            Handle::Writer(db) => Ok(db),
            Handle::Reader { .. } => Err(read_only_error()),
        }
    }

    /// The read-only handle, or a writer / closed error.
    fn reader(&self) -> Result<&ReadOnlyDatabase> {
        match self.handle()? {
            Handle::Reader { db, .. } => Ok(&**db),
            Handle::Writer(_) => Err(writer_only_error("generation")),
        }
    }
}

/// The libuv-thread body of [`Plugmem::maintain`]: holds a cloned `Database`
/// handle (cheap — an `Arc`) and the call-time clock, runs the pass off the main
/// thread, and resolves with the typed report.
pub struct MaintainTask {
    db: Database,
    now: u64,
    options: MaintenanceOptions,
}

impl Task for MaintainTask {
    type Output = plugmem_host::MaintainReport;
    type JsValue = MaintainReport;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.db
            .maintain_with_options(self.now, self.options)
            .map_err(to_napi_err)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        types::to_typed(&output).map_err(error::into_napi)
    }
}

/// The libuv-thread body of [`Plugmem::checkpoint`] — writes and fsyncs a fresh
/// snapshot off the main thread.
pub struct CheckpointTask {
    db: Database,
    now: u64,
}

impl Task for CheckpointTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.db.checkpoint(self.now).map_err(to_napi_err)
    }

    fn resolve(&mut self, _env: Env, (): Self::Output) -> napi::Result<Self::JsValue> {
        Ok(())
    }
}

/// The libuv-thread body of [`Plugmem::remember_many`].
pub struct RememberManyTask {
    db: Database,
    args: Vec<RememberArgs>,
    /// One pre-validated `validFrom` per input, checked before the task was
    /// scheduled (see [`Plugmem::remember_many`]).
    valid_from: Vec<Option<u64>>,
    now: u64,
}

impl Task for RememberManyTask {
    type Output = Vec<plugmem_host::RememberOutcome>;
    type JsValue = Vec<RememberOutcome>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let tags: Vec<Vec<&str>> = self.args.iter().map(|args| str_refs(&args.tags)).collect();
        let links: Vec<Vec<(&str, &str)>> = self
            .args
            .iter()
            .map(|args| {
                args.links
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|link| (link.rel.as_str(), link.entity.as_str()))
                    .collect()
            })
            .collect();
        let metadata: Vec<Vec<(&str, &str)>> = self
            .args
            .iter()
            .map(|args| {
                args.metadata
                    .as_ref()
                    .map(|metadata| {
                        metadata
                            .iter()
                            .map(|(key, value)| (key.as_str(), value.as_str()))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
        let inputs: Vec<RememberInput<'_>> = self
            .args
            .iter()
            .enumerate()
            .map(|(i, args)| RememberInput {
                entity: args.entity.as_deref(),
                tags: &tags[i],
                links: &links[i],
                metadata: (!metadata[i].is_empty()).then_some(metadata[i].as_slice()),
                valid_from: self.valid_from[i],
                ..RememberInput::text(self.now, &args.text)
            })
            .collect();

        self.db.remember_many(inputs).map_err(to_napi_err)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output.into_iter().map(RememberOutcome::from).collect())
    }
}

enum ExportSource {
    Writer(Database),
    Reader(Arc<ReadOnlyDatabase>),
}

/// The libuv-thread body of [`Plugmem::export_page`].
pub struct ExportPageTask {
    source: ExportSource,
    cursor: u32,
}

impl Task for ExportPageTask {
    type Output = plugmem_host::ExportPage;
    type JsValue = ExportPage;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(match &self.source {
            ExportSource::Writer(db) => db.export_page(self.cursor, EXPORT_PAGE_LIMIT),
            ExportSource::Reader(db) => db.export_page(self.cursor, EXPORT_PAGE_LIMIT),
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(ExportPage::from(output))
    }
}

/// `remember`/`revise` body (revise passes `Some(target)`): build the borrowed
/// [`RememberInput`] from the owned args and dispatch. The host embeds the text
/// outside its lock, so the wrapper passes only text.
fn do_remember(
    db: &Database,
    args: &RememberArgs,
    revise: Option<FactId>,
) -> Result<RememberOutcome> {
    let tags = str_refs(&args.tags);
    let links: Vec<(&str, &str)> = args
        .links
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|l| (l.rel.as_str(), l.entity.as_str()))
        .collect();
    // A `BTreeMap` gives the pairs already sorted and deduped, matching the one
    // canonical order core and host use; the engine canonicalizes again anyway.
    let meta: Vec<(&str, &str)> = args
        .metadata
        .as_ref()
        .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
        .unwrap_or_default();
    let input = RememberInput {
        entity: args.entity.as_deref(),
        tags: &tags,
        links: &links,
        metadata: (!meta.is_empty()).then_some(meta.as_slice()),
        valid_from: args
            .valid_from
            .map(|value| checked_ms(value, "validFrom"))
            .transpose()?,
        ..RememberInput::text(now_ms(), &args.text)
    };
    let outcome = match revise {
        Some(target) => db.revise(target, input),
        None => db.remember(input),
    }
    .map_err(error::engine)?;
    Ok(RememberOutcome::from(outcome))
}

/// Borrow an optional `Vec<String>` as `&[&str]` (empty when absent).
fn str_refs(v: &Option<Vec<String>>) -> Vec<&str> {
    v.as_deref()
        .unwrap_or(&[])
        .iter()
        .map(String::as_str)
        .collect()
}

/// A host error from inside a **libuv task**. Separate from [`error::engine`]
/// only because [`Task`] fixes its error type to `napi::Error<Status>` and so
/// cannot carry a custom code; both produce the same `GenericFailure` code and
/// the same host message, which is what keeps the contract seamless.
pub(crate) fn to_napi_err(e: plugmem_host::HostError) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

/// A unix-millisecond instant from JS.
///
/// A JS number is an `f64`, and `f64 as u64` in Rust *saturates*: `-1` would
/// become the epoch and `NaN` would become the epoch too, quietly answering a
/// different question than the caller asked. Refuse instead, naming the field.
pub(crate) fn checked_ms(value: f64, field: &str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(error::invalid_arg(format!(
            "{field} must be a finite, non-negative number of unix milliseconds, got {value}"
        )));
    }
    Ok(value as u64)
}

/// The error a write verb throws on a read-only handle.
fn read_only_error() -> Error {
    error::coded(
        code::READ_ONLY,
        "this memory is open read-only; writes are refused",
    )
}

/// The error a read-only-only verb (`generation`/`refresh`) throws on a writer.
fn writer_only_error(verb: &str) -> Error {
    error::coded(
        code::WRITER_ONLY,
        format!("`{verb}` is only available in read-only mode"),
    )
}

/// The error every verb throws after `close()`.
fn closed_error() -> Error {
    error::coded(code::CLOSED, "this memory is closed")
}

/// Wall-clock now in unix milliseconds (the engine keeps no clock).
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A unique temp directory holding the config file and database; removed on
    /// drop. Mirrors the host's own test helper (no `tempfile` dev-dependency).
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-napi-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn write(&self, name: &str, body: &str) -> String {
            let p = self.0.join(name);
            std::fs::write(&p, body).unwrap();
            p.to_str().unwrap().to_owned()
        }
        fn path(&self, name: &str) -> String {
            self.0.join(name).to_str().unwrap().to_owned()
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A config whose `[embedder]` builds against dimension 8. The embedder is
    /// constructed but never contacted (no verb embeds here), so no network.
    const EMBEDDER_DIM8: &str = "\
[engine]
dim = 8
[embedder]
kind = \"openai\"
url = \"http://127.0.0.1:1/v1/embeddings\"
model = \"dummy\"
";

    #[test]
    fn open_reads_config_and_dim_option_may_agree() {
        let tmp = TempDir::new("agree");
        let cfg = tmp.write("config.toml", EMBEDDER_DIM8);
        // `dim: 8` agrees with the config's embedder → the writer opens.
        let db = Plugmem::new(
            Some(tmp.path("mem.plugmem")),
            Some(OpenOptions {
                dim: Some(8),
                read_only: None,
                config: Some(cfg),
            }),
        );
        assert!(db.is_ok(), "matching dim should open: {:?}", db.err());
    }

    #[test]
    fn dim_option_disagreeing_with_config_embedder_throws() {
        let tmp = TempDir::new("conflict");
        let cfg = tmp.write("config.toml", EMBEDDER_DIM8);
        // `dim: 16` contradicts the config's 8-dim embedder → refused up front.
        let Err(err) = Plugmem::new(
            Some(tmp.path("mem.plugmem")),
            Some(OpenOptions {
                dim: Some(16),
                read_only: None,
                config: Some(cfg),
            }),
        ) else {
            panic!("mismatched dim must throw");
        };
        assert!(
            err.reason.contains("disagrees"),
            "unexpected message: {}",
            err.reason
        );
    }

    #[test]
    fn an_instant_from_js_is_checked_not_cast() {
        // `f64 as u64` saturates in Rust, so without this check `-1` and `NaN`
        // would both become the epoch — and `asOf` is a visibility filter
        // (`recorded_at <= as_of`), so "as of 1970" hides the whole memory and
        // returns an empty, entirely plausible answer. Every rejection names
        // its field and is coded, so a caller can tell "I passed nonsense"
        // from "the engine failed".
        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5] {
            let Err(err) = checked_ms(bad, "asOf") else {
                panic!("{bad} must be refused");
            };
            assert_eq!(err.status, code::INVALID_ARG, "for {bad}");
            assert!(
                err.reason.contains("asOf"),
                "the field is named: {}",
                err.reason
            );
        }
        assert_eq!(checked_ms(0.0, "asOf").unwrap(), 0);
        assert_eq!(
            checked_ms(1_700_000_000_000.0, "asOf").unwrap(),
            1_700_000_000_000
        );
    }

    #[test]
    fn a_malformed_recall_window_is_refused_instead_of_vanishing() {
        // The bug this locks out: `range: [from]` used to drop the window and
        // with it the temporal source, silently answering without it.
        let window = |range: Option<Vec<f64>>| {
            RecallArgs {
                range,
                ..RecallArgs::default()
            }
            .window()
        };

        assert_eq!(window(None).unwrap(), None, "no window is not a broken one");
        assert_eq!(window(Some(vec![10.0, 20.0])).unwrap(), Some((10, 20)));
        assert_eq!(
            window(Some(vec![7.0, 7.0])).unwrap(),
            Some((7, 7)),
            "empty is legal"
        );

        for bad in [
            vec![],
            vec![10.0],
            vec![10.0, 20.0, 30.0],
            vec![20.0, 10.0],
            vec![-1.0, 5.0],
        ] {
            let Err(err) = window(Some(bad.clone())) else {
                panic!("{bad:?} must be refused");
            };
            assert_eq!(err.status, code::INVALID_ARG, "for {bad:?}");
        }
    }

    #[test]
    fn a_locked_database_is_refused_with_the_code_a_caller_can_retry_on() {
        let tmp = TempDir::new("locked");
        let path = tmp.path("mem.plugmem");
        let _held = Plugmem::new(Some(path.clone()), None).expect("first open");

        let Err(err) = Plugmem::new(Some(path), None) else {
            panic!("a second writer must be refused");
        };
        assert_eq!(err.status, code::LOCKED);
    }

    #[test]
    fn a_closed_handle_and_a_read_only_one_refuse_with_their_own_codes() {
        let tmp = TempDir::new("codes");
        let mut db = Plugmem::new(Some(tmp.path("mem.plugmem")), None).expect("open");

        // `generation` belongs to a read-only handle only.
        let Err(generation) = db.generation() else {
            panic!("`generation` on a writer must be refused");
        };
        assert_eq!(generation.status, code::WRITER_ONLY);

        db.close();
        // `Err` only — the Ok types are napi mirrors without `Debug`.
        let Err(stats) = db.stats() else {
            panic!("a closed handle must refuse");
        };
        assert_eq!(stats.status, code::CLOSED);
        let Err(remembered) = db.remember(RememberArgs {
            text: "x".into(),
            entity: None,
            tags: None,
            links: None,
            metadata: None,
            valid_from: None,
        }) else {
            panic!("a closed handle must refuse a write");
        };
        assert_eq!(remembered.status, code::CLOSED);
    }

    #[test]
    fn the_handle_reports_the_file_it_resolved() {
        let tmp = TempDir::new("path");
        let path = tmp.path("mem.plugmem");
        let db = Plugmem::new(Some(path.clone()), None).expect("open");
        assert_eq!(db.path(), path);
    }

    #[test]
    fn missing_config_path_throws() {
        let tmp = TempDir::new("missing");
        let Err(err) = Plugmem::new(
            Some(tmp.path("mem.plugmem")),
            Some(OpenOptions {
                dim: None,
                read_only: None,
                config: Some(tmp.path("no-such-config.toml")),
            }),
        ) else {
            panic!("an explicit but missing config path must throw");
        };
        assert!(!err.reason.is_empty());
    }
}
