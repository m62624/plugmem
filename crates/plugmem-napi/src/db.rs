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
use napi::threadsafe_function::{
    ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode as CallMode,
};
use napi::{Env, JsFunction, Task};
use napi_derive::napi;
use plugmem_host::{
    Database, Embedder, FactId, LinkInput, MaintenanceOptions, ReadOnlyDatabase, RecallQuery,
    RememberInput, Settings, SharedEmbedder, UnlinkInput,
};

use crate::error::{self, Error, Produced, Result, code};
use crate::types::{
    self, ExportPage, ExportedEdge, ExportedFact, FactSnapshot, MaintainMode, MaintainReport,
    RecallResult, RememberOutcome, Stats,
};

/// Keep one native→JS transfer bounded. This matches the workspace's existing
/// conservative batch grain (CLI import) without exposing a tuning knob that
/// would let callers accidentally reconstruct an unbounded export.
const EXPORT_PAGE_LIMIT: NonZeroUsize = NonZeroUsize::new(128).unwrap();

/// Edges handed to the JS callback per `exportEdges` batch.
///
/// The same reasoning as [`EXPORT_PAGE_LIMIT`], for the other direction of the
/// boundary. One call per *edge* would make a million-edge dump a million
/// thread-boundary crossings and a million callback invocations on the event
/// loop — not a block, but a saturation. One call per batch makes it a thousand.
const EXPORT_EDGE_BATCH: usize = 1024;

/// Batches allowed in flight before the worker waits on the consumer.
///
/// This is the whole of `exportEdges`' memory bound: at most this many batches
/// exist at once, whatever the database's size. The queue being *full* is not a
/// failure — it is the callback being slower than the walk, and the worker
/// blocking is the correct response (see [`CallMode::Blocking`] at the call
/// site: it blocks the libuv worker, never the JS thread).
const EXPORT_EDGE_QUEUE: usize = 4;

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
    /// A precomputed embedding. Its length must equal the configured `dim`.
    ///
    /// Given, it **replaces** the embedder: nothing is sent to the provider.
    /// That is the host's own precedence — it embeds only when this is absent —
    /// and it is the route for a vector you already have, or for a model that
    /// is not an OpenAI-shaped HTTP endpoint.
    pub vector: Option<Vec<f64>>,
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
    /// Token budget of the `rendered` block (default 512). That block is what
    /// goes into a prompt, so this is the knob deciding how much of the
    /// context window a recall may spend.
    pub token_budget: Option<u32>,
    /// HNSW beam width for the vector source (default: the configured
    /// `hnsw_ef_search`). Higher is more accurate and slower; ignored while
    /// the engine is still in the flat regime, below `flat_to_hnsw`.
    pub ef: Option<u32>,
    /// A precomputed embedding. Its length must equal the configured `dim`.
    ///
    /// Given, it **replaces** the embedder: nothing is sent to the provider.
    /// That is the host's own precedence — it embeds only when this is absent —
    /// and it is the route for a vector you already have, or for a model that
    /// is not an OpenAI-shaped HTTP endpoint.
    pub vector: Option<Vec<f64>>,
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
    /// The fact this edge follows from, recorded on the edge and returned by
    /// graph recall — the answer to "why is this edge here". Ignored by
    /// `unlink`, which closes an edge rather than opening one.
    pub provenance: Option<u32>,
}

/// The open handle behind a [`Plugmem`]: a read-write writer, or a read-only
/// observer of another process's writer. The reader is reference-counted so an
/// async export task can outlive the JS object that scheduled it.
///
/// `pub` only because it is what [`OpenTask`] computes; it is not re-exported
/// and never reaches JavaScript.
pub enum Handle {
    Writer(Database),
    Reader {
        db: Arc<ReadOnlyDatabase>,
        /// The read-only handle's own embedder, if the config built one.
        ///
        /// A [`ReadOnlyDatabase`] deliberately carries none — embedding inside
        /// it would mutate a mapping opened zero-copy — so the query is embedded
        /// out here and passed in as a vector.
        ///
        /// [`SharedEmbedder`] is the host's own sharing primitive: a refcount,
        /// no lock. It is what a `recall` on a libuv worker needs, because the
        /// task outlives the call that scheduled it, and it is all it needs,
        /// because [`Embedder::embed`] takes `&self`. Several recalls may sit
        /// in the provider at once.
        embedder: Option<SharedEmbedder>,
    },
}

/// A memory over one plugmem file — the napi mirror of [`plugmem_host::Database`]
/// (writer) or [`plugmem_host::ReadOnlyDatabase`] (with `{ readOnly: true }`).
/// Construct it, call the verbs, and `close()` it to release the file when done.
#[napi]
pub struct Plugmem {
    /// `None` once `close()`d — every verb then throws "memory is closed".
    handle: Option<Handle>,
    /// The file this handle resolved to. Kept because `open` may resolve it
    /// from `PLUGMEM_DB`, the config or the platform default, and a caller that
    /// passed no path otherwise has no way to learn which file it just opened.
    path: PathBuf,
}

#[napi]
impl Plugmem {
    /// Opens (or creates) the memory at `path` and resolves with the handle. If
    /// `path` is omitted, resolution is `PLUGMEM_DB` > `[database].path` > the
    /// platform data path — and [`path()`](Plugmem::path) reports what that was.
    ///
    /// **A static method, not a constructor, and that is the point.** Opening
    /// takes the file's exclusive lock, replays the journal and maps the
    /// snapshot — work proportional to what is on disk. A JavaScript
    /// constructor must evaluate to its object immediately, so `new` has no way
    /// to hand that to a worker: it would run on the one thread that executes
    /// JavaScript and freeze the process for the length of the replay. A static
    /// method can return a `Promise`, so it does.
    ///
    /// @throws synchronously on a config error (the file is read before any
    /// work is scheduled); rejects if another writer holds the lock
    /// (`PLUGMEM_LOCKED`), if `readOnly` is set on a database with no published
    /// snapshot (`PLUGMEM_NEEDS_CHECKPOINT`), or on an IO error.
    #[napi(ts_return_type = "Promise<Plugmem>")]
    pub fn open(path: Option<String>, options: Option<OpenOptions>) -> Result<AsyncTask<OpenTask>> {
        Ok(AsyncTask::new(Self::open_task(path, options)?))
    }

    /// The checked, not-yet-scheduled half of [`open`](Plugmem::open). Separate
    /// so a native test can run the open body without a Node runtime.
    fn open_task(path: Option<String>, options: Option<OpenOptions>) -> Result<OpenTask> {
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

        // Everything above is cheap and synchronous — reading `config.toml`,
        // resolving the path, checking the dimension — so a caller's mistake
        // throws where they stand. The open itself goes to a worker.
        Ok(OpenTask {
            settings: Some(settings),
            path,
            read_only: options.read_only.unwrap_or(false),
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

    /// Stores a fact and resolves with its id plus similar/conflicting live
    /// facts.
    ///
    /// **Async** (returns a `Promise`): with an `[embedder]` configured this
    /// makes an HTTP call to the provider, and a write waits for the journal's
    /// durability policy. Both are blocking work, and Node has exactly one
    /// thread that runs JavaScript — doing them on it would freeze every timer,
    /// socket and callback in the process for the whole round trip. It runs on
    /// a libuv worker instead. Arguments are still checked synchronously, so a
    /// refused one throws here rather than rejecting later.
    /// @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<RememberOutcome>")]
    pub fn remember(&self, args: RememberArgs) -> Result<AsyncTask<RememberTask>> {
        self.remember_task(args, None)
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
        let vectors = args
            .iter()
            .enumerate()
            .map(|(i, input)| checked_vector(input.vector.as_ref(), &format!("[{i}].vector")))
            .collect::<Result<Vec<_>>>()?;
        Ok(AsyncTask::new(RememberManyTask {
            db,
            args,
            valid_from,
            vectors,
            now: now_ms(),
        }))
    }

    /// Closes fact `id`, records `args` as its successor, and resolves with the
    /// outcome. **Async** for the same reasons as
    /// [`remember`](Plugmem::remember).
    /// @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<RememberOutcome>")]
    pub fn revise(&self, id: u32, args: RememberArgs) -> Result<AsyncTask<RememberTask>> {
        self.remember_task(args, Some(FactId(id)))
    }

    /// Ranked, fused recall. Resolves with the structured result (its `rendered`
    /// field is the prompt-ready block; `facts`/`edges` are the structured hits).
    ///
    /// **Async** (returns a `Promise`): a text query with an `[embedder]`
    /// configured costs an HTTP round trip, and blocking the one thread that
    /// runs JavaScript for it would stall the whole process. The query shape is
    /// still validated synchronously.
    #[napi(ts_return_type = "Promise<RecallResult>")]
    pub fn recall(&self, args: Option<RecallArgs>) -> Result<AsyncTask<RecallTask>> {
        let args = args.unwrap_or_default();
        // Refused before anything is scheduled: these are `f64`s from JS, and
        // casting one blindly would turn a caller's mistake into a different,
        // plausible-looking query.
        let range = args.window()?;
        let as_of = args.as_of.map(|v| checked_ms(v, "asOf")).transpose()?;
        let vector = checked_vector(args.vector.as_ref(), "vector")?;
        let source = match self.handle()? {
            Handle::Writer(db) => RecallSource::Writer(db.clone()),
            Handle::Reader { db, embedder } => RecallSource::Reader {
                db: Arc::clone(db),
                embedder: embedder.clone(),
            },
        };
        Ok(AsyncTask::new(RecallTask {
            source,
            args,
            range,
            as_of,
            vector,
            now: now_ms(),
        }))
    }

    /// Tombstones fact `id` (physically purged at the next `maintain`) and
    /// resolves with whether it was a live fact.
    ///
    /// **Async**: every write syncs the journal, and the host's post-write
    /// policy can fire a whole maintenance pass or a reshard from here — work
    /// proportional to the database, which the JS thread must not be holding.
    /// @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn forget(&self, id: u32) -> Result<AsyncTask<WriteTask>> {
        self.write_task(Write::Forget(FactId(id)))
    }

    /// Upserts a typed edge `src -rel-> dst`. **Async** for the same reason as
    /// [`forget`](Plugmem::forget). @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn link(&self, args: LinkArgs) -> Result<AsyncTask<WriteTask>> {
        self.write_task(Write::Link(args))
    }

    /// Closes the current typed edge `src -rel-> dst`, resolving with whether
    /// one was open. **Async** for the same reason as
    /// [`forget`](Plugmem::forget). @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn unlink(&self, args: LinkArgs) -> Result<AsyncTask<WriteTask>> {
        self.write_task(Write::Unlink(args))
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

    /// Every currently-open fact, as one array (id-free, import-ready).
    ///
    /// **Async, but still unbounded, in two ways.** The scan runs on a worker,
    /// yet the whole memory is materialized into a single array before it
    /// resolves, so peak memory is the whole export. And `resolve` runs on the
    /// JS thread by definition, so building one JavaScript object per fact
    /// stalls it: measured at ~244 ms of a 289 ms call over 100 000 facts,
    /// against 0 ms for the same data through `exportPage`. A promise hides
    /// neither cost. Prefer `exportPage` for anything but a small memory or a
    /// script.
    #[napi(ts_return_type = "Promise<ExportedFact[]>")]
    pub fn export(&self) -> Result<AsyncTask<ReadTask>> {
        Ok(AsyncTask::new(ReadTask {
            source: self.read_source()?,
            what: Read::Export,
        }))
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

    /// Streams every current edge to `onBatch`, in batches, and resolves with
    /// the number streamed.
    ///
    /// **The other half of a backup.** `export`/`exportPage` dump facts; an edge
    /// is a statement between two entities and belongs to no single fact, so a
    /// dump of facts alone silently loses the graph.
    ///
    /// **Async and bounded.** The walk runs on a libuv worker; batches reach
    /// `onBatch` on the JS thread through a threadsafe function with a queue of
    /// four. When the callback is slower than the walk, the *worker* waits — the
    /// event loop never does, and peak memory is four batches whatever the
    /// database's size.
    ///
    /// `onBatch` throwing is not caught here: it surfaces as an uncaught error,
    /// as it would from any callback Node invokes on your behalf.
    #[napi(ts_return_type = "Promise<number>")]
    pub fn export_edges(
        &self,
        #[napi(ts_arg_type = "(edges: ExportedEdge[]) => void")] on_batch: JsFunction,
    ) -> Result<AsyncTask<ExportEdgesTask>> {
        let source = match self.handle()? {
            Handle::Writer(db) => ExportSource::Writer(db.clone()),
            Handle::Reader { db, .. } => ExportSource::Reader(Arc::clone(db)),
        };
        // Created here, on the JS thread, because it needs the function value;
        // it is `Send`, which is what lets the worker call back into JS at all.
        // `Fatal`: the callback takes the batch alone, with no node-style error
        // argument, because this call cannot fail *at* the callback.
        let sink: ThreadsafeFunction<Vec<ExportedEdge>, ErrorStrategy::Fatal> = on_batch
            .create_threadsafe_function(EXPORT_EDGE_QUEUE, |ctx| Ok(vec![ctx.value]))
            .map_err(|e| error::internal(format!("cannot bridge the callback to a worker: {e}")))?;
        Ok(AsyncTask::new(ExportEdgesTask {
            source,
            sink: Some(sink),
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

    /// Content-integrity check; rejects on the first inconsistency found.
    ///
    /// **Async**: this is a full sweep — every fact's text and metadata, the
    /// vector mapping, and both directions of every edge. On a large memory
    /// that is seconds of work, and it belongs on a worker.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn verify(&self) -> Result<AsyncTask<ReadTask>> {
        Ok(AsyncTask::new(ReadTask {
            source: self.read_source()?,
            what: Read::Verify,
        }))
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

    /// The libuv task behind a small write.
    fn write_task(&self, what: Write) -> Result<AsyncTask<WriteTask>> {
        Ok(AsyncTask::new(WriteTask {
            db: self.writer()?.clone(),
            what,
            now: now_ms(),
        }))
    }

    /// What a whole-database read runs against.
    fn read_source(&self) -> Result<RecallSource> {
        Ok(match self.handle()? {
            Handle::Writer(db) => RecallSource::Writer(db.clone()),
            Handle::Reader { db, .. } => RecallSource::Reader {
                db: Arc::clone(db),
                embedder: None,
            },
        })
    }

    /// The libuv task behind `remember` and `revise`: same work, same checks,
    /// one target id apart.
    fn remember_task(
        &self,
        args: RememberArgs,
        revise: Option<FactId>,
    ) -> Result<AsyncTask<RememberTask>> {
        let db = self.writer()?.clone();
        let valid_from = args
            .valid_from
            .map(|value| checked_ms(value, "validFrom"))
            .transpose()?;
        let vector = checked_vector(args.vector.as_ref(), "vector")?;
        Ok(AsyncTask::new(RememberTask {
            db,
            args,
            valid_from,
            vector,
            revise,
            now: now_ms(),
        }))
    }

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
    type Output = Produced<plugmem_host::MaintainReport>;
    type JsValue = MaintainReport;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(self
            .db
            .maintain_with_options(self.now, self.options)
            .map_err(error::engine))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        output
            .and_then(|report| types::to_typed(&report))
            .map_err(|e| error::to_js(env, e))
    }
}

/// The libuv-thread body of [`Plugmem::checkpoint`] — writes and fsyncs a fresh
/// snapshot off the main thread.
pub struct CheckpointTask {
    db: Database,
    now: u64,
}

impl Task for CheckpointTask {
    type Output = Produced<()>;
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(self.db.checkpoint(self.now).map_err(error::engine))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        output.map_err(|e| error::to_js(env, e))
    }
}

/// The libuv-thread body of [`Plugmem::remember_many`].
pub struct RememberManyTask {
    db: Database,
    args: Vec<RememberArgs>,
    /// One pre-validated `validFrom` per input, checked before the task was
    /// scheduled (see [`Plugmem::remember_many`]).
    valid_from: Vec<Option<u64>>,
    /// One pre-validated `vector` per input, likewise.
    vectors: Vec<Option<Vec<f32>>>,
    now: u64,
}

impl Task for RememberManyTask {
    type Output = Produced<Vec<plugmem_host::RememberOutcome>>;
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
                vector: self.vectors[i].as_deref(),
                ..RememberInput::text(self.now, &args.text)
            })
            .collect();

        Ok(self.db.remember_many(inputs).map_err(error::engine))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output
            .map_err(|e| error::to_js(env, e))?
            .into_iter()
            .map(RememberOutcome::from)
            .collect())
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

/// The libuv-thread body of [`Plugmem::export_edges`].
pub struct ExportEdgesTask {
    source: ExportSource,
    /// `Option` only so `compute` can take it by value: the sink must be
    /// **dropped** when the walk ends, and dropping it is what releases the
    /// threadsafe function's hold on the event loop. Left alive, the process
    /// would not exit.
    sink: Option<ThreadsafeFunction<Vec<ExportedEdge>, ErrorStrategy::Fatal>>,
}

impl ExportEdgesTask {
    /// Walks the edges, handing the callback one batch at a time.
    ///
    /// The host's visitor is per-edge; batching happens here, in Rust, so the
    /// per-edge cost stays a `push` and only the batch crosses to JavaScript.
    fn run(&mut self) -> usize {
        let Some(sink) = self.sink.take() else {
            return 0;
        };
        let mut batch: Vec<ExportedEdge> = Vec::with_capacity(EXPORT_EDGE_BATCH);
        let mut total = 0usize;
        let mut visit = |src: &str, rel: &str, dst: &str, provenance: FactId| {
            batch.push(ExportedEdge {
                src: src.to_string(),
                rel: rel.to_string(),
                dst: dst.to_string(),
                // Absent rather than the sentinel: `FactId::NONE` is `u32::MAX`,
                // which would arrive in JS as a plausible-looking id.
                provenance: (provenance != FactId::NONE).then(|| f64::from(provenance.0)),
            });
            total += 1;
            if batch.len() == EXPORT_EDGE_BATCH {
                // Blocking: when the queue is full this parks *this* worker
                // until the JS side drains one. That is the backpressure.
                sink.call(std::mem::take(&mut batch), CallMode::Blocking);
                batch = Vec::with_capacity(EXPORT_EDGE_BATCH);
            }
        };
        match &self.source {
            ExportSource::Writer(db) => db.export_edges_each(&mut visit),
            ExportSource::Reader(db) => db.export_edges_each(&mut visit),
        }
        if !batch.is_empty() {
            sink.call(batch, CallMode::Blocking);
        }
        // Explicit, because it is load-bearing rather than housekeeping: this
        // is the release that lets Node exit.
        drop(sink);
        total
    }
}

impl Task for ExportEdgesTask {
    type Output = usize;
    type JsValue = f64;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(self.run())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output as f64)
    }
}

/// The libuv-thread body of [`Plugmem::open`].
///
/// Everything expensive about opening lives here: taking the file's exclusive
/// lock, replaying the journal, mapping the snapshot and building the page
/// directory. `resolve` runs back on the JS thread and does nothing but wrap
/// the finished handle, which is why a `Promise` costs the caller nothing.
pub struct OpenTask {
    /// `Option` only so `compute` can take it by value: `Settings` owns a boxed
    /// embedder and is not `Clone`, and a task runs exactly once.
    settings: Option<Settings>,
    path: PathBuf,
    read_only: bool,
}

impl Task for OpenTask {
    type Output = Produced<Handle>;
    type JsValue = Plugmem;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(self.run())
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(Plugmem {
            handle: Some(output.map_err(|e| error::to_js(env, e))?),
            path: std::mem::take(&mut self.path),
        })
    }
}

impl OpenTask {
    /// The open itself. Its error is coded even though [`Task`] cannot carry
    /// the code to JavaScript, because a native test can and does assert on it.
    fn run(&mut self) -> Result<Handle> {
        let settings = self
            .settings
            .take()
            .ok_or_else(|| error::coded(code::OPEN, "open was already run"))?;
        if self.read_only {
            // A read-only handle observes another writer's snapshot. The engine
            // cannot embed inside it, so the embedder is kept out here and the
            // query is embedded before the read — the same arrangement the CLI
            // and the MCP server use for their read-only paths, so a text
            // `recall` reaches the vector source on every surface.
            let mut settings = settings;
            let embedder = settings.embedder.take().map(SharedEmbedder::new);
            let ro = Database::open_readonly(&self.path, settings.config).map_err(error::open)?;
            Ok(Handle::Reader {
                db: Arc::new(ro),
                embedder,
            })
        } else {
            // The writer takes the embedder and maintenance policy from settings.
            let db = settings.open(&self.path).map_err(error::open)?;
            Ok(Handle::Writer(db))
        }
    }
}

/// The libuv-thread body of [`Plugmem::remember`] and [`Plugmem::revise`].
///
/// Owns its arguments outright: the task outlives the call that scheduled it,
/// so nothing here may borrow from JS. The instant and the vector arrived
/// already checked — a refused argument threw on the JS thread, because a coded
/// error cannot cross this boundary (see [`crate::error`]).
pub struct RememberTask {
    db: Database,
    args: RememberArgs,
    valid_from: Option<u64>,
    vector: Option<Vec<f32>>,
    /// `Some(target)` makes this a revision of that fact.
    revise: Option<FactId>,
    now: u64,
}

impl Task for RememberTask {
    type Output = Produced<plugmem_host::RememberOutcome>;
    type JsValue = RememberOutcome;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let tags = str_refs(&self.args.tags);
        let links: Vec<(&str, &str)> = self
            .args
            .links
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|l| (l.rel.as_str(), l.entity.as_str()))
            .collect();
        // A `BTreeMap` gives the pairs already sorted and deduped, matching the
        // one canonical order core and host use; the engine canonicalizes again
        // anyway.
        let meta: Vec<(&str, &str)> = self
            .args
            .metadata
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();
        let input = RememberInput {
            entity: self.args.entity.as_deref(),
            tags: &tags,
            links: &links,
            metadata: (!meta.is_empty()).then_some(meta.as_slice()),
            valid_from: self.valid_from,
            // Authoritative when present: the host embeds only when this is
            // `None`, so a caller's own vector skips the provider entirely.
            vector: self.vector.as_deref(),
            ..RememberInput::text(self.now, &self.args.text)
        };
        Ok(match self.revise {
            Some(target) => self.db.revise(target, input),
            None => self.db.remember(input),
        }
        .map_err(error::engine))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(RememberOutcome::from(
            output.map_err(|e| error::to_js(env, e))?,
        ))
    }
}

/// One small write, named so [`WriteTask`] can carry any of them.
enum Write {
    Forget(FactId),
    Link(LinkArgs),
    Unlink(LinkArgs),
}

/// What a small write resolves with: `unlink`/`forget` report freshness,
/// `link` has nothing to say.
pub enum WriteOutput {
    Changed(bool),
    Done,
}

/// The libuv-thread body of [`Plugmem::forget`], [`Plugmem::link`] and
/// [`Plugmem::unlink`].
///
/// One task for the three because the reason they are async is the same and
/// belongs in one place: the journal sync, and the host's post-write policy,
/// which can fire a maintenance pass or a reshard from inside any of them.
pub struct WriteTask {
    db: Database,
    what: Write,
    now: u64,
}

impl Task for WriteTask {
    type Output = Produced<WriteOutput>;
    type JsValue = napi::bindgen_prelude::Either<bool, ()>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(match &self.what {
            Write::Forget(id) => self
                .db
                .forget(self.now, *id)
                .map(WriteOutput::Changed)
                .map_err(error::engine),
            Write::Link(args) => self
                .db
                .link(LinkInput {
                    now: self.now,
                    src: &args.src,
                    rel: &args.rel,
                    dst: &args.dst,
                    provenance: args.provenance.map(FactId),
                })
                .map(|()| WriteOutput::Done)
                .map_err(error::engine),
            Write::Unlink(args) => self
                .db
                .unlink(UnlinkInput {
                    now: self.now,
                    src: &args.src,
                    rel: &args.rel,
                    dst: &args.dst,
                })
                .map(WriteOutput::Changed)
                .map_err(error::engine),
        })
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(match output.map_err(|e| error::to_js(env, e))? {
            WriteOutput::Changed(changed) => napi::bindgen_prelude::Either::A(changed),
            WriteOutput::Done => napi::bindgen_prelude::Either::B(()),
        })
    }
}

/// One whole-database read.
enum Read {
    Verify,
    Export,
}

/// What a whole-database read resolves with.
pub enum ReadOutput {
    Verified,
    Exported(Vec<plugmem_host::ExportedFact>),
}

/// The libuv-thread body of [`Plugmem::verify`] and [`Plugmem::export`] — both
/// sweep every fact, which is work proportional to the database.
pub struct ReadTask {
    source: RecallSource,
    what: Read,
}

impl Task for ReadTask {
    type Output = Produced<ReadOutput>;
    type JsValue = napi::bindgen_prelude::Either<Vec<ExportedFact>, ()>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(match (&self.what, &self.source) {
            (Read::Verify, RecallSource::Writer(db)) => db
                .verify()
                .map(|()| ReadOutput::Verified)
                .map_err(error::engine),
            (Read::Verify, RecallSource::Reader { db, .. }) => db
                .verify()
                .map(|()| ReadOutput::Verified)
                .map_err(error::engine),
            (Read::Export, RecallSource::Writer(db)) => Ok(ReadOutput::Exported(db.export())),
            (Read::Export, RecallSource::Reader { db, .. }) => {
                Ok(ReadOutput::Exported(db.export()))
            }
        })
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(match output.map_err(|e| error::to_js(env, e))? {
            ReadOutput::Verified => napi::bindgen_prelude::Either::B(()),
            ReadOutput::Exported(facts) => napi::bindgen_prelude::Either::A(
                facts.into_iter().map(ExportedFact::from).collect(),
            ),
        })
    }
}

/// What a [`RecallTask`] reads: a writer, or a reader plus the embedder the
/// engine cannot carry for it.
enum RecallSource {
    Writer(Database),
    Reader {
        db: Arc<ReadOnlyDatabase>,
        embedder: Option<SharedEmbedder>,
    },
}

/// The libuv-thread body of [`Plugmem::recall`] — the embedder round trip and
/// the engine read both happen here, off the JS thread.
pub struct RecallTask {
    source: RecallSource,
    args: RecallArgs,
    range: Option<(u64, u64)>,
    as_of: Option<u64>,
    vector: Option<Vec<f32>>,
    now: u64,
}

impl Task for RecallTask {
    type Output = Produced<plugmem_host::RecallResult>;
    type JsValue = RecallResult;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        // Precedence, matching the host: a caller's vector wins outright. Next,
        // a read-only handle embeds its own query, because the engine cannot do
        // it inside a mapping opened zero-copy. A writer leaves this `None` and
        // the host embeds inside `recall`, outside its lock.
        let embedded = match (&self.vector, &self.source) {
            (Some(_), _) => None,
            (
                None,
                RecallSource::Reader {
                    embedder: Some(embedder),
                    ..
                },
            ) => match self.args.query.as_deref() {
                Some(text) => match embedder.embed(&[text]) {
                    Ok(mut vectors) => vectors.pop(),
                    Err(e) => return Ok(Err(error::engine(e))),
                },
                None => None,
            },
            _ => None,
        };

        let tags = str_refs(&self.args.tags);
        let entities = str_refs(&self.args.entities);
        let q = RecallQuery {
            now: self.now,
            text: self.args.query.as_deref(),
            vector: self.vector.as_deref().or(embedded.as_deref()),
            tags: &tags,
            entities: &entities,
            as_of: self.as_of,
            range: self.range,
            k: self.args.k.unwrap_or(0) as usize,
            token_budget: self.args.token_budget.map(|v| v as usize),
            include_closed: self.args.closed.unwrap_or(false),
            ef: self.args.ef.map(|v| v as usize),
        };
        Ok(match &self.source {
            RecallSource::Writer(db) => db.recall(q),
            RecallSource::Reader { db, .. } => db.recall(q),
        }
        .map_err(error::engine))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        let result = output.and_then(|res| types::to_typed(&res));
        result.map_err(|e| error::to_js(env, e))
    }
}

/// Borrow an optional `Vec<String>` as `&[&str]` (empty when absent).
fn str_refs(v: &Option<Vec<String>>) -> Vec<&str> {
    v.as_deref()
        .unwrap_or(&[])
        .iter()
        .map(String::as_str)
        .collect()
}

/// A precomputed embedding from JS, as the engine wants it.
///
/// Only the numbers are checked here; the engine owns the length rule and
/// refuses a vector that disagrees with `dim` (or any vector at all when `dim`
/// is 0), so duplicating that check would be a second place to keep in step.
/// An empty array is refused outright: it cannot mean "no vector" when omitting
/// the field already does, and passing one would silently swap the caller's
/// model for the configured embedder.
pub(crate) fn checked_vector(vector: Option<&Vec<f64>>, field: &str) -> Result<Option<Vec<f32>>> {
    let Some(vector) = vector else {
        return Ok(None);
    };
    if vector.is_empty() {
        return Err(error::invalid_arg(format!(
            "{field} must not be empty — omit it to let the embedder produce one"
        )));
    }
    if let Some(bad) = vector.iter().find(|value| !value.is_finite()) {
        return Err(error::invalid_arg(format!(
            "{field} must contain only finite numbers, got {bad}"
        )));
    }
    Ok(Some(vector.iter().map(|value| *value as f32).collect()))
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

    /// Opens the way `Plugmem.open` does, minus the promise: the checks run,
    /// then the task body runs inline. This is the whole open, so a native test
    /// still exercises config resolution, the lock, and the coded errors that
    /// JavaScript sees on a rejection.
    fn open_blocking(path: Option<String>, options: Option<OpenOptions>) -> Result<Plugmem> {
        let mut task = Plugmem::open_task(path, options)?;
        let handle = task.run()?;
        Ok(Plugmem {
            handle: Some(handle),
            path: std::mem::take(&mut task.path),
        })
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
        let db = open_blocking(
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
        let Err(err) = open_blocking(
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
    fn a_vector_from_js_is_checked_for_numbers_but_not_for_length() {
        // Length belongs to the engine (`dim`), which already refuses a
        // disagreeing vector — checking it here too would be a second rule to
        // keep in step. What JS can produce and the engine cannot see is a
        // non-number, and an empty array that would silently fall back to the
        // embedder instead of using the caller's model.
        assert_eq!(checked_vector(None, "vector").unwrap(), None);
        assert_eq!(
            checked_vector(Some(&vec![0.5, -0.25]), "vector").unwrap(),
            Some(vec![0.5f32, -0.25])
        );

        for bad in [vec![], vec![0.1, f64::NAN], vec![f64::INFINITY]] {
            let Err(err) = checked_vector(Some(&bad), "vector") else {
                panic!("{bad:?} must be refused");
            };
            assert_eq!(err.status, code::INVALID_ARG, "for {bad:?}");
            assert!(err.reason.contains("vector"), "{}", err.reason);
        }
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
        let _held = open_blocking(Some(path.clone()), None).expect("first open");

        let Err(err) = open_blocking(Some(path), None) else {
            panic!("a second writer must be refused");
        };
        assert_eq!(err.status, code::LOCKED);
    }

    #[test]
    fn a_closed_handle_and_a_read_only_one_refuse_with_their_own_codes() {
        let tmp = TempDir::new("codes");
        let mut db = open_blocking(Some(tmp.path("mem.plugmem")), None).expect("open");

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
            vector: None,
        }) else {
            panic!("a closed handle must refuse a write");
        };
        assert_eq!(remembered.status, code::CLOSED);
    }

    #[test]
    fn the_handle_reports_the_file_it_resolved() {
        let tmp = TempDir::new("path");
        let path = tmp.path("mem.plugmem");
        let db = open_blocking(Some(path.clone()), None).expect("open");
        assert_eq!(db.path(), path);
    }

    #[test]
    fn missing_config_path_throws() {
        let tmp = TempDir::new("missing");
        let Err(err) = open_blocking(
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
