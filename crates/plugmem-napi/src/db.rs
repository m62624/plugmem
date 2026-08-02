//! The `Plugmem` class: a 1:1 napi mirror of plugmem-host's [`Database`] (and,
//! in read-only mode, [`ReadOnlyDatabase`]).
//!
//! Every method wraps the identically-named host verb — the engine logic is
//! 100% `plugmem-host`; this layer only marshals arguments and results across
//! the Node boundary. Inputs are typed `#[napi(object)]` structs so napi emits
//! precise TypeScript interfaces (autocomplete in a TS host like Pi); results
//! come back as the typed mirrors in [`crate::types`]. A [`HostError`] becomes a
//! thrown JS `Error`; the clock is the system clock, read per call (the engine
//! keeps none), exactly as the MCP server does.
//!
//! Opened `readOnly`, the instance observes another process's writer over a
//! shared snapshot: the read verbs answer, the write verbs throw, and the two
//! freshness verbs (`generation`/`refresh`) become available. Async offloading
//! lands in the next milestone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Error, Result, Task};
use napi_derive::napi;
use plugmem_host::{
    Database, FactId, HostError, LinkInput, ReadOnlyDatabase, RecallQuery, RememberInput, Settings,
    SettingsError, UnlinkInput,
};

use crate::types::{
    self, ExportedFact, FactSnapshot, MaintainReport, RecallResult, RememberOutcome, Stats,
};

/// Options for [`Plugmem::new`].
#[napi(object)]
#[derive(Default)]
pub struct OpenOptions {
    /// Embedding dimension (`Config::dim`). Omit or 0 to keep vectors off. When
    /// the config file configures an `[embedder]`, that embedder's dimension is
    /// authoritative and this — if given — must agree with it.
    pub dim: Option<u32>,
    /// Open read-only over another process's writer (requires a checkpointed
    /// database). The write verbs then throw; `generation`/`refresh` appear. A
    /// read-only handle never auto-embeds a text query — pass a vector instead.
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
/// observer of another process's writer. `Reader` is boxed — a
/// [`ReadOnlyDatabase`] owns a memory map and is far larger than the writer's
/// `Arc` handle, so boxing keeps the two variants close in size.
enum Handle {
    Writer(Database),
    Reader(Box<ReadOnlyDatabase>),
}

/// A memory over one plugmem file — the napi mirror of [`plugmem_host::Database`]
/// (writer) or [`plugmem_host::ReadOnlyDatabase`] (with `{ readOnly: true }`).
/// Construct it, call the verbs, and `close()` it to release the file when done.
#[napi]
pub struct Plugmem {
    /// `None` once `close()`d — every verb then throws "memory is closed".
    handle: Option<Handle>,
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
            Settings::load(options.config.as_deref().map(Path::new)).map_err(settings_to_napi)?;

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
                Error::from_reason(format!(
                    "cannot create database directory {}: {e}",
                    parent.display()
                ))
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
                return Err(Error::from_reason(format!(
                    "dim option {dim} disagrees with the configured embedder dimension {}",
                    e.dim()
                )));
            }
            settings.config.dim = dim;
        }

        let handle = if options.read_only.unwrap_or(false) {
            // A read-only handle observes another writer's snapshot and never
            // auto-embeds; drop the embedder and open over `settings.config`.
            let ro = Database::open_readonly(&path, settings.config).map_err(to_napi_err)?;
            Handle::Reader(Box::new(ro))
        } else {
            // The writer takes the embedder and maintenance policy from settings.
            let db = settings.open(&path).map_err(to_napi_err)?;
            Handle::Writer(db)
        };
        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Stores a fact; returns its id plus similar/conflicting live facts.
    /// @throws in read-only mode.
    #[napi]
    pub fn remember(&self, args: RememberArgs) -> Result<RememberOutcome> {
        do_remember(self.writer()?, &args, None)
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
    pub fn recall(&self, args: Option<RecallArgs>) -> Result<RecallResult> {
        let args = args.unwrap_or_default();
        let tags = str_refs(&args.tags);
        let entities = str_refs(&args.entities);
        let range = args
            .range
            .as_ref()
            .and_then(|r| Some((*r.first()? as u64, *r.get(1)? as u64)));
        let q = RecallQuery {
            now: now_ms(),
            text: args.query.as_deref(),
            vector: None,
            tags: &tags,
            entities: &entities,
            as_of: args.as_of.map(|f| f as u64),
            range,
            k: args.k.unwrap_or(0) as usize,
            token_budget: None,
            include_closed: args.closed.unwrap_or(false),
            ef: None,
        };
        let res = match self.handle()? {
            Handle::Writer(db) => db.recall(q),
            Handle::Reader(db) => db.recall(q),
        }
        .map_err(to_napi_err)?;
        types::to_typed(&res)
    }

    /// Tombstones fact `id` (physically purged at the next `maintain`). Returns
    /// whether it was a live fact. @throws in read-only mode.
    #[napi]
    pub fn forget(&self, id: u32) -> Result<bool> {
        self.writer()?
            .forget(now_ms(), FactId(id))
            .map_err(to_napi_err)
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
            .map_err(to_napi_err)
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
            .map_err(to_napi_err)
    }

    /// One fact's full card by `id`, or `null` if unknown/tombstoned.
    #[napi]
    pub fn get(&self, id: u32) -> Result<Option<FactSnapshot>> {
        let snap = match self.handle()? {
            Handle::Writer(db) => db.get(FactId(id)),
            Handle::Reader(db) => db.get(FactId(id)),
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
            Handle::Reader(db) => db.stats(),
        };
        types::to_typed(&stats)
    }

    /// Every currently-open fact, as an array (id-free, import-ready).
    #[napi]
    pub fn export(&self) -> Result<Vec<ExportedFact>> {
        let facts = match self.handle()? {
            Handle::Writer(db) => db.export(),
            Handle::Reader(db) => db.export(),
        };
        types::to_typed(&facts)
    }

    /// Content-integrity check; throws on the first inconsistency found.
    #[napi]
    pub fn verify(&self) -> Result<()> {
        match self.handle()? {
            Handle::Writer(db) => db.verify(),
            Handle::Reader(db) => db.verify(),
        }
        .map_err(to_napi_err)
    }

    /// Runs policy-driven maintenance; resolves with the before/after report.
    /// **Async** (returns a `Promise`): the pass may do disk I/O (compaction,
    /// HNSW work), so it runs on a libuv worker thread and never blocks the
    /// event loop. @throws synchronously in read-only mode.
    #[napi(ts_return_type = "Promise<MaintainReport>")]
    pub fn maintain(&self) -> Result<AsyncTask<MaintainTask>> {
        Ok(AsyncTask::new(MaintainTask {
            db: self.writer()?.clone(),
            now: now_ms(),
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
            Handle::Reader(db) => db.refresh().map_err(to_napi_err),
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
            Handle::Reader(_) => Err(read_only_error()),
        }
    }

    /// The read-only handle, or a writer / closed error.
    fn reader(&self) -> Result<&ReadOnlyDatabase> {
        match self.handle()? {
            Handle::Reader(db) => Ok(&**db),
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
}

impl Task for MaintainTask {
    type Output = plugmem_host::MaintainReport;
    type JsValue = MaintainReport;

    fn compute(&mut self) -> Result<Self::Output> {
        self.db.maintain(self.now).map_err(to_napi_err)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        types::to_typed(&output)
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

    fn compute(&mut self) -> Result<Self::Output> {
        self.db.checkpoint(self.now).map_err(to_napi_err)
    }

    fn resolve(&mut self, _env: Env, (): Self::Output) -> Result<Self::JsValue> {
        Ok(())
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
        valid_from: args.valid_from.map(|f| f as u64),
        ..RememberInput::text(now_ms(), &args.text)
    };
    let outcome = match revise {
        Some(target) => db.revise(target, input),
        None => db.remember(input),
    }
    .map_err(to_napi_err)?;
    types::to_typed(&outcome)
}

/// Borrow an optional `Vec<String>` as `&[&str]` (empty when absent).
fn str_refs(v: &Option<Vec<String>>) -> Vec<&str> {
    v.as_deref()
        .unwrap_or(&[])
        .iter()
        .map(String::as_str)
        .collect()
}

/// A host error as a thrown JS `Error` (the message is the host's own text).
fn to_napi_err(e: HostError) -> Error {
    Error::from_reason(e.to_string())
}

/// A config-resolution error (bad `config.toml`, or an `[embedder]` missing a
/// required field) as a thrown JS `Error`.
fn settings_to_napi(e: SettingsError) -> Error {
    Error::from_reason(e.to_string())
}

/// The error a write verb throws on a read-only handle.
fn read_only_error() -> Error {
    Error::from_reason("this memory is open read-only; writes are refused")
}

/// The error a read-only-only verb (`generation`/`refresh`) throws on a writer.
fn writer_only_error(verb: &str) -> Error {
    Error::from_reason(format!("`{verb}` is only available in read-only mode"))
}

/// The error every verb throws after `close()`.
fn closed_error() -> Error {
    Error::from_reason("this memory is closed")
}

/// Wall-clock now in unix milliseconds (the engine keeps no clock).
fn now_ms() -> u64 {
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
