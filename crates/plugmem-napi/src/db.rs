//! The `Plugmem` class: a 1:1 napi mirror of plugmem-host's [`Database`].
//!
//! Every method wraps the identically-named host verb — the engine logic is
//! 100% `plugmem-host`; this layer only marshals arguments and results across
//! the Node boundary. Inputs are typed `#[napi(object)]` structs so napi emits
//! precise TypeScript interfaces (autocomplete in a TS host like Pi); results
//! come back as native JS objects (host result types serialize through serde).
//! A [`HostError`] becomes a thrown JS `Error`; the clock is the system clock,
//! read per call (the engine keeps none), exactly as the MCP server does.
//!
//! N1 covers the writer verbs. Read-only mode and async offloading land in the
//! next milestones.

use napi::{Error, Result};
use napi_derive::napi;
use plugmem_host::{Config, Database, FactId, HostError, LinkInput, RecallQuery, RememberInput};
use serde::Serialize;
use serde_json::Value;

/// Options for [`Plugmem::new`]. `dim` is the embedding width (0 = vectors off,
/// the default); the embedder and the rest of the engine config arrive with the
/// shared settings loader in a later milestone.
#[napi(object)]
#[derive(Default)]
pub struct OpenOptions {
    /// Embedding dimension (`Config::dim`). Omit or 0 to keep vectors off.
    pub dim: Option<u32>,
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

/// A read-write memory over one plugmem file — the napi mirror of
/// [`plugmem_host::Database`]. Construct it, then call the verbs.
#[napi]
pub struct Plugmem {
    db: Database,
}

#[napi]
impl Plugmem {
    /// Opens (or creates) the memory at `path`.
    ///
    /// @throws if the file is locked by another writer, or on a config/IO error.
    #[napi(constructor)]
    pub fn new(path: String, options: Option<OpenOptions>) -> Result<Self> {
        let mut cfg = Config::default();
        if let Some(dim) = options.and_then(|o| o.dim) {
            cfg.dim = dim as usize;
        }
        let (db, _report) = Database::open(path, cfg).map_err(to_napi_err)?;
        Ok(Self { db })
    }

    /// Stores a fact; returns its id plus similar/conflicting live facts.
    #[napi]
    pub fn remember(&self, args: RememberArgs) -> Result<Value> {
        do_remember(&self.db, &args, None)
    }

    /// Closes fact `id` and records `args` as its successor; returns the outcome.
    #[napi]
    pub fn revise(&self, id: u32, args: RememberArgs) -> Result<Value> {
        do_remember(&self.db, &args, Some(FactId(id)))
    }

    /// Ranked, fused recall. Returns the structured result (its `rendered` field
    /// is the prompt-ready block; `facts`/`edges` are the structured hits).
    #[napi]
    pub fn recall(&self, args: Option<RecallArgs>) -> Result<Value> {
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
        to_value(&self.db.recall(q).map_err(to_napi_err)?)
    }

    /// Tombstones fact `id` (physically purged at the next `maintain`). Returns
    /// whether it was a live fact.
    #[napi]
    pub fn forget(&self, id: u32) -> Result<bool> {
        self.db.forget(now_ms(), FactId(id)).map_err(to_napi_err)
    }

    /// Upserts a typed edge `src -rel-> dst`.
    #[napi]
    pub fn link(&self, args: LinkArgs) -> Result<()> {
        self.db
            .link(LinkInput {
                now: now_ms(),
                src: &args.src,
                rel: &args.rel,
                dst: &args.dst,
                provenance: None,
            })
            .map_err(to_napi_err)
    }

    /// One fact's full card by `id`, or `null` if unknown/tombstoned.
    #[napi]
    pub fn get(&self, id: u32) -> Result<Option<Value>> {
        match self.db.get(FactId(id)) {
            Some(snap) => Ok(Some(to_value(&snap)?)),
            None => Ok(None),
        }
    }

    /// Engine size counters.
    #[napi]
    pub fn stats(&self) -> Result<Value> {
        to_value(&self.db.stats())
    }

    /// Every currently-open fact, as an array (id-free, import-ready).
    #[napi]
    pub fn export(&self) -> Result<Value> {
        to_value(&self.db.export())
    }

    /// Purges tombstones, compacts, and builds the vector index; returns the
    /// before/after report.
    #[napi]
    pub fn maintain(&self) -> Result<Value> {
        to_value(&self.db.maintain(now_ms()).map_err(to_napi_err)?)
    }

    /// Flushes the journal into a fresh snapshot.
    #[napi]
    pub fn checkpoint(&self) -> Result<()> {
        self.db.checkpoint(now_ms()).map_err(to_napi_err)
    }

    /// Content-integrity check; throws on the first inconsistency found.
    #[napi]
    pub fn verify(&self) -> Result<()> {
        self.db.verify().map_err(to_napi_err)
    }
}

/// `remember`/`revise` body (revise passes `Some(target)`): build the borrowed
/// [`RememberInput`] from the owned args and dispatch. The host embeds the text
/// outside its lock, so the wrapper passes only text.
fn do_remember(db: &Database, args: &RememberArgs, revise: Option<FactId>) -> Result<Value> {
    let tags = str_refs(&args.tags);
    let links: Vec<(&str, &str)> = args
        .links
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|l| (l.rel.as_str(), l.entity.as_str()))
        .collect();
    let input = RememberInput {
        entity: args.entity.as_deref(),
        tags: &tags,
        links: &links,
        valid_from: args.valid_from.map(|f| f as u64),
        ..RememberInput::text(now_ms(), &args.text)
    };
    let outcome = match revise {
        Some(target) => db.revise(target, input),
        None => db.remember(input),
    }
    .map_err(to_napi_err)?;
    to_value(&outcome)
}

/// Borrow an optional `Vec<String>` as `&[&str]` (empty when absent).
fn str_refs(v: &Option<Vec<String>>) -> Vec<&str> {
    v.as_deref()
        .unwrap_or(&[])
        .iter()
        .map(String::as_str)
        .collect()
}

/// Serialize a host result to a native JS value via serde.
fn to_value(v: &impl Serialize) -> Result<Value> {
    serde_json::to_value(v).map_err(|e| Error::from_reason(format!("serialization error: {e}")))
}

/// A host error as a thrown JS `Error` (the message is the host's own text).
fn to_napi_err(e: HostError) -> Error {
    Error::from_reason(e.to_string())
}

/// Wall-clock now in unix milliseconds (the engine keeps no clock).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
