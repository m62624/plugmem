//! The tools the server exposes — their schema definitions and the `tools/call`
//! executor. Each name is `plugmem_*` so it never collides with another MCP
//! server's tools. Descriptions come from [`crate::messages`]; result types
//! serialize straight to JSON via serde (the host `serde` feature); envelopes
//! from [`crate::rpc`].
//!
//! Input types (`RememberInput`/`RecallQuery`/`LinkInput`) borrow `&str`, so the
//! handlers own the parsed `String`/`Vec` in scope and lend slices into the
//! call. A [`HostError`] becomes a tool-level error (`isError`) so the model
//! reads it; missing `params` is the JSON-RPC `-32602`.

use std::sync::{Mutex, RwLock};

use plugmem_host::{
    Database, DbEntry, DbName, Embedder, FactId, HostError, IfMissing, LinkInput, MaintenanceMode,
    MaintenanceOptions, ReadOnlyDatabase, RecallQuery, RememberInput, UnlinkInput, Workspace,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{messages, rpc};

/// Read-only server state, shared across worker threads (behind an `Arc`): a
/// shared-mmap snapshot of another process's writer, plus the embedder (if
/// configured) used to embed a `recall` query — the read-only `recall` does not
/// auto-embed, so the server does it, mirroring the CLI's read-only path.
///
/// The snapshot is behind a `RwLock`: read verbs take the read guard and run
/// concurrently; `refresh` (rare) takes the write guard to re-map. The embedder
/// is behind its own `Mutex`, so embeds serialize (as in the host) without
/// holding the snapshot lock during the HTTP call.
pub struct ReaderShared {
    db: RwLock<ReadOnlyDatabase>,
    embedder: Mutex<Option<Box<dyn Embedder>>>,
}

impl ReaderShared {
    /// Wrap a read-only handle and its optional embedder for sharing.
    pub fn new(db: ReadOnlyDatabase, embedder: Option<Box<dyn Embedder>>) -> Self {
        Self {
            db: RwLock::new(db),
            embedder: Mutex::new(embedder),
        }
    }
}

const GENERATION: &str = "plugmem_generation";
const REFRESH: &str = "plugmem_refresh";

/// The tool names, shared by each definition and the `tools/call` dispatcher so
/// the advertised name and the routed name can never drift apart.
const REMEMBER: &str = "plugmem_remember";
const RECALL: &str = "plugmem_recall";
const REVISE: &str = "plugmem_revise";
const FORGET: &str = "plugmem_forget";
const LINK: &str = "plugmem_link";
const UNLINK: &str = "plugmem_unlink";
const SHOW: &str = "plugmem_show";
const STATS: &str = "plugmem_stats";
const EXPORT: &str = "plugmem_export";
const MAINTAIN: &str = "plugmem_maintain";
const CHECKPOINT: &str = "plugmem_checkpoint";
const VERIFY: &str = "plugmem_verify";
const VERSION: &str = "plugmem_version";
const ABOUT: &str = "plugmem_about";
const SETTINGS_HELP: &str = "plugmem_settings_help";

/// Every tool definition, in the order `tools/list` advertises them: the write
/// verbs, then the read verbs, then the operational verbs, then meta.
///
/// There is deliberately no `import` tool: bulk-loading from a `backup.jsonl`
/// reads a file on the server's disk, which a sandboxed/remote server can't see
/// — restoring from a file is `plugmem-cli import` (it has the disk and streams
/// in batches). `export` needs no file (facts ride in the reply). See the README.
pub fn definitions() -> Vec<Value> {
    vec![
        remember_def(),
        recall_def(),
        revise_def(),
        forget_def(),
        link_def(),
        unlink_def(),
        show_def(),
        stats_def(),
        export_def(),
        maintain_def(),
        checkpoint_def(),
        verify_def(),
        version_def(),
        about_def(),
        settings_help_def(),
    ]
}

/// Tools that answer without a database: the version, the blurb, the settings
/// catalogue. Named as a set because a workspace server must not make a caller
/// nominate a memory to ask what version it is running.
const STATELESS_TOOLS: &[&str] = &[VERSION, ABOUT, SETTINGS_HELP];

/// Tools that act on one database.
const DATABASE_TOOLS: &[&str] = &[
    REMEMBER, RECALL, REVISE, FORGET, LINK, UNLINK, SHOW, STATS, EXPORT, MAINTAIN, CHECKPOINT,
    VERIFY,
];

/// Answers the tools that need no database, or `None` for anything else.
fn stateless(name: &str, id: &Value, args: Option<&Value>) -> Option<Value> {
    let id = id.clone();
    match name {
        VERSION => Some(rpc::tool_result(
            id,
            format!("plugmem {}", env!("CARGO_PKG_VERSION")),
            false,
        )),
        ABOUT => Some(rpc::tool_result(
            id,
            messages::ABOUT_TOOL.to_string(),
            false,
        )),
        SETTINGS_HELP => Some(rpc::tool_result(
            id,
            render(&settings_help_value(), format_arg(args)),
            false,
        )),
        _ => None,
    }
}

/// Execute a `tools/call`: route by tool name, then hand off. A missing `params`
/// is a JSON-RPC error; an unknown tool is a tool-level error (`isError`).
pub fn call(db: &Database, id: Value, params: Option<&Value>) -> Value {
    let Some(params) = params else {
        return rpc::error(id, -32602, "missing params");
    };
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments");

    if let Some(reply) = stateless(name, &id, args) {
        return reply;
    }
    match name {
        REMEMBER => remember(db, id, args, None),
        RECALL => recall(db, id, args),
        REVISE => revise(db, id, args),
        FORGET => forget(db, id, args),
        LINK => link(db, id, args),
        UNLINK => unlink(db, id, args),
        SHOW => show(db, id, args),
        STATS => rpc::tool_result(id, render(&db.stats(), format_arg(args)), false),
        EXPORT => rpc::tool_result(id, render(&db.export(), format_arg(args)), false),
        MAINTAIN => match maintenance_options(args) {
            Ok(options) => match db.maintain_with_options(now_ms(), options) {
                Ok(report) => rpc::tool_result(id, render(&report, format_arg(args)), false),
                Err(e) => tool_error(id, &e),
            },
            Err(message) => rpc::tool_result(id, message, true),
        },
        CHECKPOINT => match db.checkpoint(now_ms()) {
            Ok(()) => rpc::tool_result(id, render(&json!({ "ok": true }), format_arg(args)), false),
            Err(e) => tool_error(id, &e),
        },
        VERIFY => match db.verify() {
            Ok(()) => rpc::tool_result(id, render(&json!({ "ok": true }), format_arg(args)), false),
            Err(e) => tool_error(id, &e),
        },
        other => rpc::tool_result(id, format!("unknown tool: {other}"), true),
    }
}

// ── Read-only mode ────────────────────────────────────────────────────────

/// Tool definitions a read-only server advertises: the read verbs, the two
/// freshness verbs, and meta. No write verbs.
pub fn definitions_ro() -> Vec<Value> {
    vec![
        recall_def(),
        show_def(),
        stats_def(),
        export_def(),
        verify_def(),
        format_only_def(GENERATION, messages::GENERATION_TOOL),
        format_only_def(REFRESH, messages::REFRESH_TOOL),
        version_def(),
        about_def(),
        settings_help_def(),
    ]
}

/// Execute a `tools/call` against a read-only server: read verbs + freshness;
/// a write verb is refused with a tool-level error. Read verbs take the snapshot
/// read guard (concurrent); `refresh` takes the write guard.
pub fn call_ro(reader: &ReaderShared, id: Value, params: Option<&Value>) -> Value {
    let Some(params) = params else {
        return rpc::error(id, -32602, "missing params");
    };
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments");

    if let Some(reply) = stateless(name, &id, args) {
        return reply;
    }
    match name {
        RECALL => recall_ro(reader, id, args),
        SHOW => {
            let db = reader.db.read().expect("snapshot lock");
            match db.get(FactId(id_arg(args))) {
                Some(snap) => rpc::tool_result(id, render(&snap, format_arg(args)), false),
                None => rpc::tool_result(id, format!("fact {} does not exist", id_arg(args)), true),
            }
        }
        STATS => {
            let stats = reader.db.read().expect("snapshot lock").stats();
            rpc::tool_result(id, render(&stats, format_arg(args)), false)
        }
        EXPORT => {
            let facts = reader.db.read().expect("snapshot lock").export();
            rpc::tool_result(id, render(&facts, format_arg(args)), false)
        }
        VERIFY => match reader.db.read().expect("snapshot lock").verify() {
            Ok(()) => rpc::tool_result(id, render(&json!({ "ok": true }), format_arg(args)), false),
            Err(e) => tool_error(id, &e),
        },
        GENERATION => {
            let generation = reader.db.read().expect("snapshot lock").generation();
            rpc::tool_result(
                id,
                render(&json!({ "generation": generation }), format_arg(args)),
                false,
            )
        }
        REFRESH => {
            let mut db = reader.db.write().expect("snapshot lock");
            match db.refresh() {
                Ok(moved) => {
                    let generation = db.generation();
                    rpc::tool_result(
                        id,
                        render(
                            &json!({ "refreshed": moved, "generation": generation }),
                            format_arg(args),
                        ),
                        false,
                    )
                }
                Err(e) => tool_error(id, &e),
            }
        }
        // A known write verb, or anything else: refused in read-only mode.
        REMEMBER | REVISE | FORGET | LINK | MAINTAIN | CHECKPOINT => {
            rpc::tool_result(id, messages::READ_ONLY_REFUSAL.into(), true)
        }
        SETTINGS_HELP => {
            rpc::tool_result(id, render(&settings_help_value(), format_arg(args)), false)
        }
        other => rpc::tool_result(id, format!("unknown tool: {other}"), true),
    }
}

// ── Workspace mode ────────────────────────────────────────────────────────

/// Argument naming which database a call is for. Present in the schema **only**
/// in workspace mode — see [`definitions_ws`].
const DB_ARG: &str = "db";

const WORKSPACE_LIST: &str = "plugmem_workspace_list";
const WORKSPACE_FIND: &str = "plugmem_workspace_find";

/// Tools that may bring a database into existence.
///
/// A write to a name nobody has used yet is how a new chat gets a memory —
/// that is the whole plug-and-play story. A *read* of an unknown name is
/// something else: almost always a typo or a hallucinated name, and answering
/// it with an empty result would hide the mistake. So creation is a property of
/// the verb, and this list is what says which.
const CREATING_TOOLS: &[&str] = &[REMEMBER, REVISE, FORGET, LINK, UNLINK, MAINTAIN, CHECKPOINT];

/// Workspace-mode server state, shared across worker threads.
///
/// There is deliberately **no "current database"**. With a worker pool, a
/// switch verb would make it shared mutable state: worker A switches to X, B
/// switches to Y, A reads and gets Y — a race that reproduces once in a
/// hundred calls and corrupts the wrong memory when it does. Every call carries
/// its own `db` instead, so the workers share only the pool, which has its own
/// lock.
pub struct WorkspaceShared {
    workspace: Workspace,
    /// The database used when a call omits `db`. `None` makes `db` required.
    default: Option<DbName>,
    /// Names this server will serve, or empty for "any name in the workspace".
    allowed: Vec<DbName>,
    /// Whether a write to an unknown name creates it.
    create: bool,
}

impl WorkspaceShared {
    /// Wraps a workspace for serving.
    pub fn new(
        workspace: Workspace,
        default: Option<DbName>,
        allowed: Vec<DbName>,
        create: bool,
    ) -> Self {
        Self {
            workspace,
            default,
            allowed,
            create,
        }
    }

    /// The workspace itself — the janitor thread sweeps idle handles through it.
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The database served when a call omits `db`, if there is one. `None` is
    /// what makes `db` a required argument in the advertised schema.
    pub fn default_db(&self) -> Option<&DbName> {
        self.default.as_ref()
    }

    /// Resolves the `db` argument of one call to an open database.
    fn resolve(&self, tool: &str, args: Option<&Value>) -> Result<Database, String> {
        let name = match arg_str(args, DB_ARG) {
            Some(given) => DbName::parse(given).map_err(|e| e.to_string())?,
            None => self
                .default
                .clone()
                .ok_or_else(|| messages::WORKSPACE_DB_REQUIRED.to_string())?,
        };
        if !self.allowed.is_empty() && !self.allowed.contains(&name) {
            return Err(format!(
                "{name} is not one of the memories this server was started with"
            ));
        }
        let missing = if self.create && CREATING_TOOLS.contains(&tool) {
            IfMissing::Create
        } else {
            IfMissing::Fail
        };
        self.workspace
            .get(&name, now_ms(), missing)
            .map_err(|e| e.to_string())
    }
}

/// Tool definitions a workspace server advertises: the ordinary set with a `db`
/// argument threaded through, plus the two verbs for finding a name.
///
/// The `db` property is **injected** rather than written into fifteen schemas,
/// so a tool's own definition stays the single description of that tool, and
/// the three startup modes cannot drift apart:
///
/// | started with | `db` |
/// |---|---|
/// | `--db FILE` | not in the schema at all (this function is not called) |
/// | `--workspace DIR --db NAME` | present, optional, defaulted |
/// | `--workspace DIR` | present, required |
///
/// That the field *disappears* in the single-database case is the point. In
/// MCP the model fills tool arguments, so a `db` field is a decision the model
/// makes on every call — while the caller that spawned the server usually knew
/// the answer for certain. Where the answer is already known, the question
/// should not be asked: it cannot be got wrong, and it costs no tokens.
pub fn definitions_ws(default: Option<&DbName>) -> Vec<Value> {
    let mut out = definitions();
    for tool in &mut out {
        // The tools that answer without a database keep their schema: nobody
        // should have to name a memory to ask what version is running.
        if tool
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|n| STATELESS_TOOLS.contains(&n))
        {
            continue;
        }
        let Some(schema) = tool.get_mut("inputSchema") else {
            continue;
        };
        schema["properties"][DB_ARG] = match default {
            Some(name) => json!({
                "type": "string",
                "default": name.as_str(),
                "description": messages::ARG_DB_OPTIONAL,
            }),
            None => json!({ "type": "string", "description": messages::ARG_DB }),
        };
        if default.is_none() {
            let required = schema
                .get_mut("required")
                .and_then(Value::as_array_mut)
                .map(std::mem::take)
                .unwrap_or_default();
            // `db` first: it is the one argument every one of these tools now
            // shares, and reading it first is reading which memory this is about.
            let mut with_db = vec![json!(DB_ARG)];
            with_db.extend(required);
            schema["required"] = Value::Array(with_db);
        }
    }
    out.push(workspace_list_def());
    out.push(workspace_find_def());
    out
}

/// Execute a `tools/call` against a workspace server: resolve `db`, then hand
/// the whole call to the ordinary single-database executor.
///
/// Resolution is the only thing this adds. Every verb behaves exactly as it
/// does against one database, because it *is* the same code — there is no
/// second implementation of `remember` to drift.
pub fn call_ws(shared: &WorkspaceShared, id: Value, params: Option<&Value>) -> Value {
    let Some(params) = params else {
        return rpc::error(id, -32602, "missing params");
    };
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments");

    if let Some(reply) = stateless(name, &id, args) {
        return reply;
    }
    match name {
        WORKSPACE_LIST => match shared.workspace.entries() {
            Ok(entries) => {
                rpc::tool_result(id, render(&entries_json(&entries), format_arg(args)), false)
            }
            Err(e) => rpc::tool_result(id, e.to_string(), true),
        },
        WORKSPACE_FIND => {
            let query = arg_str(args, "query").unwrap_or_default();
            let k = arg_u64(args, "k").unwrap_or(8).min(64) as usize;
            match shared.workspace.find(query, k, now_ms()) {
                Ok(entries) => {
                    rpc::tool_result(id, render(&entries_json(&entries), format_arg(args)), false)
                }
                Err(e) => rpc::tool_result(id, e.to_string(), true),
            }
        }
        // Resolution runs only for a tool that will actually use a database:
        // making an unrecognized name fail with "no such memory" would name the
        // wrong problem.
        name if DATABASE_TOOLS.contains(&name) => match shared.resolve(name, args) {
            Ok(db) => call(&db, id, Some(params)),
            Err(message) => rpc::tool_result(id, message, true),
        },
        other => rpc::tool_result(id, format!("unknown tool: {other}"), true),
    }
}

/// Registry entries as JSON. `DbEntry` is a host type without `serde`, and
/// adding a serialization dependency to the host for one wrapper's convenience
/// would be the wrong trade.
fn entries_json(entries: &[DbEntry]) -> Value {
    Value::Array(
        entries
            .iter()
            .map(|e| {
                json!({
                    "db": e.name.as_str(),
                    "description": e.description,
                    "tags": e.tags,
                    "owner": e.owner,
                    "archived": e.is_archived(),
                })
            })
            .collect(),
    )
}

fn workspace_list_def() -> Value {
    simple_def(WORKSPACE_LIST, messages::WORKSPACE_LIST_TOOL)
}

fn workspace_find_def() -> Value {
    json!({
        "name": WORKSPACE_FIND,
        "description": messages::WORKSPACE_FIND_TOOL,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": messages::ARG_WORKSPACE_QUERY },
                "k": { "type": "integer", "minimum": 0, "description": messages::ARG_K },
                "format": format_prop()
            },
            "required": ["query"]
        }
    })
}

/// Read-only `recall`: embed the query text with the server's embedder (if any)
/// before searching — the read-only handle carries no embedder and does not
/// auto-embed. Lexical/graph/time still answer without one. The embed (its
/// slow HTTP) runs under the embedder `Mutex`, not the snapshot lock; the
/// search then takes the snapshot read guard.
fn recall_ro(reader: &ReaderShared, id: Value, args: Option<&Value>) -> Value {
    let format = format_arg(args);
    let query = arg_str(args, "query").map(String::from);
    let explicit = match arg_vector(args) {
        Ok(vector) => vector,
        Err(message) => return rpc::tool_result(id, message, true),
    };
    // An explicit `vector` replaces the embedder outright; otherwise embed the
    // query text into an owned vector, if we can (embedder Mutex only).
    let vector = match (explicit, query.as_deref()) {
        (Some(vector), _) => Some(vector),
        (None, Some(text)) => {
            let mut embedder = reader.embedder.lock().expect("embedder lock");
            match embedder.as_mut() {
                Some(e) => match e.embed(&[text]) {
                    Ok(mut v) => v.pop(),
                    Err(e) => return tool_error(id, &e),
                },
                None => None,
            }
        }
        (None, None) => None,
    };
    let tags = arg_str_vec(args, "tags");
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let entities = arg_str_vec(args, "entities");
    let ent_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
    let (range, as_of) = match (arg_range(args), arg_ms(args, "as_of")) {
        (Ok(range), Ok(as_of)) => (range, as_of),
        (Err(message), _) | (_, Err(message)) => return rpc::tool_result(id, message, true),
    };
    let q = RecallQuery {
        now: now_ms(),
        text: query.as_deref(),
        vector: vector.as_deref(),
        tags: &tag_refs,
        entities: &ent_refs,
        as_of,
        range,
        k: arg_u64(args, "k").unwrap_or(0) as usize,
        token_budget: None,
        include_closed: arg_bool(args, "closed"),
        ef: None,
    };
    let db = reader.db.read().expect("snapshot lock");
    match db.recall(q) {
        Ok(res) if format == "human" => rpc::tool_result(id, res.rendered, false),
        Ok(res) => rpc::tool_result(id, render(&res, "json"), false),
        Err(e) => tool_error(id, &e),
    }
}

/// The `id` argument as a `u32` fact id (0 when absent — a `show`/`get` of a
/// non-existent fact is a clean "does not exist").
fn id_arg(args: Option<&Value>) -> u32 {
    arg_u64(args, "id").unwrap_or(0) as u32
}

// ── Write verbs ───────────────────────────────────────────────────────────

/// `plugmem_remember` / `plugmem_revise` body (revise passes `Some(target)`):
/// build the borrowed [`RememberInput`] from owned args and dispatch. The host
/// embeds the text outside its lock, so the tool only passes text.
fn remember(db: &Database, id: Value, args: Option<&Value>, revise: Option<FactId>) -> Value {
    let format = format_arg(args);
    let Some(text) = arg_str(args, "text") else {
        return rpc::tool_result(id, "missing required `text`".into(), true);
    };
    let tags = arg_str_vec(args, "tags");
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let links = arg_links(args);
    let link_refs: Vec<(&str, &str)> = links
        .iter()
        .map(|(r, e)| (r.as_str(), e.as_str()))
        .collect();
    let meta = arg_meta(args);
    let meta_refs: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let (valid_from, vector) = match (arg_ms(args, "valid_from"), arg_vector(args)) {
        (Ok(valid_from), Ok(vector)) => (valid_from, vector),
        (Err(message), _) | (_, Err(message)) => return rpc::tool_result(id, message, true),
    };
    let input = RememberInput {
        entity: arg_str(args, "entity"),
        tags: &tag_refs,
        links: &link_refs,
        metadata: (!meta_refs.is_empty()).then_some(meta_refs.as_slice()),
        valid_from,
        // Authoritative when given: the host embeds only when this is `None`,
        // so a caller's own vector skips the provider. The engine checks its
        // length against `dim`.
        vector: vector.as_deref(),
        ..RememberInput::text(now_ms(), text)
    };
    let res = match revise {
        Some(target) => db.revise(target, input),
        None => db.remember(input),
    };
    match res {
        Ok(outcome) => rpc::tool_result(id, render(&outcome, format), false),
        Err(e) => tool_error(id, &e),
    }
}

fn revise(db: &Database, id: Value, args: Option<&Value>) -> Value {
    let Some(target) = arg_u64(args, "id") else {
        return rpc::tool_result(id, "missing required `id`".into(), true);
    };
    remember(db, id, args, Some(FactId(target as u32)))
}

fn forget(db: &Database, id: Value, args: Option<&Value>) -> Value {
    let Some(fid) = arg_u64(args, "id") else {
        return rpc::tool_result(id, "missing required `id`".into(), true);
    };
    match db.forget(now_ms(), FactId(fid as u32)) {
        Ok(fresh) => rpc::tool_result(
            id,
            render(&json!({ "id": fid, "forgotten": fresh }), format_arg(args)),
            false,
        ),
        Err(e) => tool_error(id, &e),
    }
}

fn link(db: &Database, id: Value, args: Option<&Value>) -> Value {
    let (Some(src), Some(rel), Some(dst)) = (
        arg_str(args, "src"),
        arg_str(args, "rel"),
        arg_str(args, "dst"),
    ) else {
        return rpc::tool_result(id, "link needs `src`, `rel` and `dst`".into(), true);
    };
    match db.link(LinkInput {
        now: now_ms(),
        src,
        rel,
        dst,
        provenance: None,
    }) {
        Ok(()) => rpc::tool_result(
            id,
            render(
                &json!({ "src": src, "rel": rel, "dst": dst }),
                format_arg(args),
            ),
            false,
        ),
        Err(e) => tool_error(id, &e),
    }
}

fn unlink(db: &Database, id: Value, args: Option<&Value>) -> Value {
    let (Some(src), Some(rel), Some(dst)) = (
        arg_str(args, "src"),
        arg_str(args, "rel"),
        arg_str(args, "dst"),
    ) else {
        return rpc::tool_result(id, "unlink needs `src`, `rel` and `dst`".into(), true);
    };
    match db.unlink(UnlinkInput {
        now: now_ms(),
        src,
        rel,
        dst,
    }) {
        Ok(unlinked) => rpc::tool_result(
            id,
            render(
                &json!({ "src": src, "rel": rel, "dst": dst, "unlinked": unlinked }),
                format_arg(args),
            ),
            false,
        ),
        Err(e) => tool_error(id, &e),
    }
}

// ── Read verbs ────────────────────────────────────────────────────────────

/// `plugmem_recall`: `format:"human"` returns the engine's prompt-ready block;
/// `"json"` (default) returns the structured facts + edges. The host embeds the
/// text query inside `recall` (outside the lock), so the tool passes no vector.
fn recall(db: &Database, id: Value, args: Option<&Value>) -> Value {
    let format = format_arg(args);
    let tags = arg_str_vec(args, "tags");
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let entities = arg_str_vec(args, "entities");
    let ent_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
    let (range, as_of) = match (arg_range(args), arg_ms(args, "as_of")) {
        (Ok(range), Ok(as_of)) => (range, as_of),
        (Err(message), _) | (_, Err(message)) => return rpc::tool_result(id, message, true),
    };
    // Given, this replaces the embedder; left `None`, the host embeds the text
    // inside `recall` (outside its lock) exactly as before.
    let vector = match arg_vector(args) {
        Ok(vector) => vector,
        Err(message) => return rpc::tool_result(id, message, true),
    };
    let q = RecallQuery {
        now: now_ms(),
        text: arg_str(args, "query"),
        vector: vector.as_deref(),
        tags: &tag_refs,
        entities: &ent_refs,
        as_of,
        range,
        k: arg_u64(args, "k").unwrap_or(0) as usize,
        token_budget: None,
        include_closed: arg_bool(args, "closed"),
        ef: None,
    };
    match db.recall(q) {
        // Human = the rendered block; json = the structured result.
        Ok(res) if format == "human" => rpc::tool_result(id, res.rendered, false),
        Ok(res) => rpc::tool_result(id, render(&res, "json"), false),
        Err(e) => tool_error(id, &e),
    }
}

fn show(db: &Database, id: Value, args: Option<&Value>) -> Value {
    let Some(fid) = arg_u64(args, "id") else {
        return rpc::tool_result(id, "missing required `id`".into(), true);
    };
    match db.get(FactId(fid as u32)) {
        Some(snap) => rpc::tool_result(id, render(&snap, format_arg(args)), false),
        None => rpc::tool_result(id, format!("fact {fid} does not exist"), true),
    }
}

// ── Definitions ───────────────────────────────────────────────────────────

fn remember_def() -> Value {
    remember_like(REMEMBER, messages::REMEMBER_TOOL, false)
}

fn revise_def() -> Value {
    remember_like(REVISE, messages::REVISE_TOOL, true)
}

/// The shared remember/revise schema; `with_id` adds the required `id`.
fn remember_like(name: &str, description: &str, with_id: bool) -> Value {
    let mut props = json!({
        "text": { "type": "string", "description": messages::ARG_TEXT },
        "entity": { "type": "string", "description": messages::ARG_ENTITY },
        "tags": { "type": "array", "items": { "type": "string" }, "description": messages::ARG_TAGS },
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rel": { "type": "string" },
                    "entity": { "type": "string" }
                },
                "required": ["rel", "entity"]
            },
            "description": messages::ARG_LINKS
        },
        "metadata": {
            "type": "object",
            "additionalProperties": { "type": "string" },
            "description": messages::ARG_METADATA
        },
        "valid_from": { "type": "integer", "minimum": 0, "description": messages::ARG_VALID_FROM },
        "vector": vector_prop(),
        "format": format_prop()
    });
    let mut required = vec![json!("text")];
    if with_id {
        props["id"] = json!({ "type": "integer", "minimum": 0, "description": messages::ARG_ID });
        required.push(json!("id"));
    }
    json!({
        "name": name,
        "description": description,
        "inputSchema": { "type": "object", "properties": props, "required": required }
    })
}

/// The `vector` property, shared by every tool that accepts a precomputed
/// embedding.
fn vector_prop() -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "description": messages::ARG_VECTOR
    })
}

fn recall_def() -> Value {
    json!({
        "name": RECALL,
        "description": messages::RECALL_TOOL,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": messages::ARG_QUERY },
                "tags": { "type": "array", "items": { "type": "string" }, "description": messages::ARG_TAGS },
                "entities": { "type": "array", "items": { "type": "string" }, "description": messages::ARG_ENTITIES },
                "as_of": { "type": "integer", "minimum": 0, "description": messages::ARG_AS_OF },
                "range": {
                    "type": "array",
                    "items": { "type": "integer", "minimum": 0 },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": messages::ARG_RANGE
                },
                "k": { "type": "integer", "minimum": 0, "description": messages::ARG_K },
                "closed": { "type": "boolean", "description": messages::ARG_CLOSED },
                "vector": vector_prop(),
                "format": format_prop()
            }
        }
    })
}

fn forget_def() -> Value {
    id_only_def(FORGET, messages::FORGET_TOOL)
}

fn show_def() -> Value {
    id_only_def(SHOW, messages::SHOW_TOOL)
}

/// A tool whose only argument is a required `id` (+ the shared `format`).
fn id_only_def(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": { "type": "integer", "minimum": 0, "description": messages::ARG_ID },
                "format": format_prop()
            },
            "required": ["id"]
        }
    })
}

fn link_def() -> Value {
    edge_def(LINK, messages::LINK_TOOL)
}

fn unlink_def() -> Value {
    edge_def(UNLINK, messages::UNLINK_TOOL)
}

fn edge_def(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "src": { "type": "string", "description": messages::ARG_SRC },
                "rel": { "type": "string", "description": messages::ARG_REL },
                "dst": { "type": "string", "description": messages::ARG_DST },
                "format": format_prop()
            },
            "required": ["src", "rel", "dst"]
        }
    })
}

fn stats_def() -> Value {
    format_only_def(STATS, messages::STATS_TOOL)
}

fn export_def() -> Value {
    format_only_def(EXPORT, messages::EXPORT_TOOL)
}

fn maintain_def() -> Value {
    json!({
        "name": MAINTAIN,
        "description": messages::MAINTAIN_TOOL,
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": MAINTAIN_MODES,
                    "description": messages::ARG_MAINTAIN_MODE
                },
                "format": format_prop()
            }
        }
    })
}

/// The `mode` values `plugmem_maintain` accepts, in the order the schema
/// advertises them.
const MAINTAIN_MODES: [&str; 5] = [
    "auto",
    "compact",
    "reindex-text",
    "optimize-vectors",
    "full",
];

/// Reads the optional `mode` argument. Absent means `auto`, which is what the
/// tool did before the argument existed.
fn maintenance_options(args: Option<&Value>) -> Result<MaintenanceOptions, String> {
    let Some(mode) = arg_str(args, "mode") else {
        return Ok(MaintenanceOptions::auto());
    };
    match mode {
        "auto" => Ok(MaintenanceOptions::auto()),
        "full" => Ok(MaintenanceOptions::full()),
        "compact" => Ok(explicit(MaintenanceMode::Compact)),
        "reindex-text" => Ok(explicit(MaintenanceMode::ReindexText)),
        "optimize-vectors" => Ok(explicit(MaintenanceMode::OptimizeVectors)),
        other => Err(format!(
            "unknown maintenance mode `{other}`; expected one of {}",
            MAINTAIN_MODES.join(", ")
        )),
    }
}

/// An explicitly named mode, keeping `auto`'s bounded vector budget.
fn explicit(mode: MaintenanceMode) -> MaintenanceOptions {
    MaintenanceOptions {
        mode,
        ..MaintenanceOptions::auto()
    }
}

fn checkpoint_def() -> Value {
    format_only_def(CHECKPOINT, messages::CHECKPOINT_TOOL)
}

fn verify_def() -> Value {
    format_only_def(VERIFY, messages::VERIFY_TOOL)
}

fn version_def() -> Value {
    simple_def(VERSION, messages::VERSION_TOOL)
}

fn about_def() -> Value {
    simple_def(ABOUT, messages::ABOUT_TOOL)
}

fn settings_help_def() -> Value {
    format_only_def(SETTINGS_HELP, messages::SETTINGS_HELP_TOOL)
}

fn settings_help_value() -> Value {
    let help = plugmem_host::settings_help();
    let settings: Vec<_> = help
        .docs()
        .iter()
        .map(|doc| {
            json!({
                "section": doc.section,
                "key": doc.key,
                "type": doc.value_type,
                "default": doc.default,
                "description": doc.description,
                "scope": doc.scope.as_str(),
            })
        })
        .collect();
    json!({
        "topic": "settings",
        "config_path_precedence": help.config_path_precedence(),
        "default_config_path": plugmem_host::default_config_path()
            .map(|path| path.display().to_string()),
        "settings": settings,
    })
}

/// A no-argument tool definition (an empty input schema).
fn simple_def(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": { "type": "object", "properties": {} }
    })
}

/// A tool definition whose only argument is the shared `format`.
fn format_only_def(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": { "type": "object", "properties": { "format": format_prop() } }
    })
}

/// The shared `format` argument schema.
fn format_prop() -> Value {
    json!({
        "type": "string",
        "enum": ["human", "json"],
        "description": messages::ARG_FORMAT
    })
}

// ── Argument extraction ───────────────────────────────────────────────────

fn arg_str<'a>(args: Option<&'a Value>, key: &str) -> Option<&'a str> {
    args.and_then(|a| a.get(key)).and_then(Value::as_str)
}

fn arg_u64(args: Option<&Value>, key: &str) -> Option<u64> {
    args.and_then(|a| a.get(key)).and_then(Value::as_u64)
}

fn arg_bool(args: Option<&Value>, key: &str) -> bool {
    args.and_then(|a| a.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// A string array argument (non-string entries skipped), or empty.
fn arg_str_vec(args: Option<&Value>, key: &str) -> Vec<String> {
    args.and_then(|a| a.get(key))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// The `links` argument: `[{ "rel", "entity" }]` → owned `(rel, entity)` pairs
/// (entries missing either field are skipped).
fn arg_links(args: Option<&Value>) -> Vec<(String, String)> {
    args.and_then(|a| a.get("links"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let rel = v.get("rel").and_then(Value::as_str)?;
                    let entity = v.get("entity").and_then(Value::as_str)?;
                    Some((rel.to_string(), entity.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The `metadata` argument: a flat object of string values → owned key→value
/// pairs, sorted and deduped (a `BTreeMap`), non-string values skipped. The
/// engine canonicalizes regardless; sorting here keeps the borrowed pairs clean.
fn arg_meta(args: Option<&Value>) -> Vec<(String, String)> {
    args.and_then(|a| a.get("metadata"))
        .and_then(Value::as_object)
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default()
}

/// The `range` argument as `[from, to)`, accepting a `[from, to]` array.
/// A precomputed embedding argument.
///
/// Absent is fine; present but not an array of finite numbers is an error, for
/// the same reason as [`arg_range`] — a vector the server could not read used
/// to mean "embed the text instead", so the caller's own model was silently
/// swapped for the configured one.
fn arg_vector(args: Option<&Value>) -> Result<Option<Vec<f32>>, String> {
    let Some(value) = args.and_then(|a| a.get("vector")) else {
        return Ok(None);
    };
    let malformed = || Err("vector must be an array of numbers".to_string());
    let Some(items) = value.as_array() else {
        return malformed();
    };
    if items.is_empty() {
        return Err("vector must not be empty (omit it to use the embedder)".to_string());
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Some(number) = item.as_f64() else {
            return malformed();
        };
        if !number.is_finite() {
            return Err("vector must contain only finite numbers".to_string());
        }
        out.push(number as f32);
    }
    Ok(Some(out))
}

/// A unix-millisecond argument that is allowed to be absent but not malformed.
///
/// Same reasoning as [`arg_range`]: `as_of` and `valid_from` change *which*
/// facts come back, so a present-but-unusable value has to be reported.
/// `as_of` is the visibility filter (`recorded_at <= as_of`), so dropping it
/// answers about the present while the caller believes they asked about a past
/// instant — an entirely plausible wrong answer.
fn arg_ms(args: Option<&Value>, key: &str) -> Result<Option<u64>, String> {
    let Some(value) = args.and_then(|a| a.get(key)) else {
        return Ok(None);
    };
    value
        .as_u64()
        .map(Some)
        .ok_or_else(|| format!("{key} must be a whole, non-negative number of unix milliseconds"))
}

/// The `range` argument as `[from, to)` in unix milliseconds.
///
/// `Err` names what was wrong instead of dropping the window: a `range` that is
/// not two whole non-negative numbers used to fall through to `None`, taking
/// the engine's temporal source with it, so the call answered as if no window
/// had been asked for. A malformed argument has to be visible — a plausible
/// wrong answer is worse than an error.
fn arg_range(args: Option<&Value>) -> Result<Option<(u64, u64)>, String> {
    let Some(value) = args.and_then(|a| a.get("range")) else {
        return Ok(None);
    };
    let malformed = || Err("range must be exactly [from, to] in unix milliseconds".to_string());
    let Some(arr) = value.as_array() else {
        return malformed();
    };
    let [from, to] = arr.as_slice() else {
        return malformed();
    };
    let (Some(from), Some(to)) = (from.as_u64(), to.as_u64()) else {
        return malformed();
    };
    if from > to {
        return Err(format!("range start {from} is after its end {to}"));
    }
    Ok(Some((from, to)))
}

// ── Rendering ─────────────────────────────────────────────────────────────

/// The `format` argument of a tool call, defaulting to `"json"`.
fn format_arg(args: Option<&Value>) -> &str {
    args.and_then(|a| a.get("format"))
        .and_then(Value::as_str)
        .unwrap_or("json")
}

/// Serialize a result to text: compact JSON for `"json"` (an agent's default),
/// pretty-printed for `"human"`. serde does the work — no per-type renderer.
fn render<T: Serialize>(value: &T, format: &str) -> String {
    let out = if format == "human" {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    };
    out.unwrap_or_else(|e| format!("serialization error: {e}"))
}

/// A host error as a tool-level error result (the model reads and reacts).
///
/// Carries the host's follow-up line when there is one. A model that reads
/// "pool would grow past 2147483648 bytes" and nothing else has no idea the
/// number is a setting; with the hint it can say so to the person running it.
fn tool_error(id: Value, e: &HostError) -> Value {
    let text = match e.capacity_hint() {
        Some(hint) => format!("{e}\n{hint}"),
        None => e.to_string(),
    };
    rpc::tool_result(id, text, true)
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
    use plugmem_host::{Config, Database};

    /// A unique temp directory; removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-mcp-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn db(&self) -> std::path::PathBuf {
            self.0.join("m.plugmem")
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A tool-call `params` object.
    pub(super) fn params(name: &str, args: Value) -> Value {
        json!({ "name": name, "arguments": args })
    }

    /// The `text` field of a tool-call result envelope.
    pub(super) fn text(v: &Value) -> String {
        v["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    pub(super) fn is_error(v: &Value) -> bool {
        v["result"]["isError"].as_bool().unwrap()
    }

    #[test]
    fn arg_extractors_read_their_shapes() {
        let a = json!({
            "s": "hi", "n": 7, "b": true,
            "list": ["x", 1, "y"],
            "links": [{"rel": "r", "entity": "e"}, {"rel": "only"}],
            "metadata": {"uri": "s3://b/x", "n2": 5, "mime": "pdf"},
            "range": [10, 20]
        });
        let a = Some(&a);
        assert_eq!(arg_str(a, "s"), Some("hi"));
        assert_eq!(arg_str(a, "missing"), None);
        assert_eq!(arg_u64(a, "n"), Some(7));
        assert!(arg_bool(a, "b"));
        assert!(!arg_bool(a, "missing"));
        assert_eq!(arg_str_vec(a, "list"), vec!["x", "y"]); // non-strings skipped
        assert_eq!(arg_links(a), vec![("r".to_string(), "e".to_string())]); // partial skipped
        // metadata: sorted, non-string values skipped.
        assert_eq!(
            arg_meta(a),
            vec![
                ("mime".to_string(), "pdf".to_string()),
                ("uri".to_string(), "s3://b/x".to_string()),
            ]
        );
        assert_eq!(arg_range(a), Ok(Some((10, 20))));
        assert_eq!(arg_range(None), Ok(None));
        assert_eq!(format_arg(a), "json");
        assert_eq!(format_arg(Some(&json!({"format": "human"}))), "human");
        assert_eq!(id_arg(Some(&json!({"id": 3}))), 3);
        assert_eq!(id_arg(None), 0);
    }

    #[test]
    fn a_narrowing_argument_is_never_dropped_for_being_malformed() {
        // The failure this guards: a `range` or an `as_of` the extractor could
        // not read used to fall through to `None`, so the tool answered without
        // the temporal source (`range`) or as of now instead of the instant
        // asked for (`as_of`). Every shape below must be an error.
        for bad in [
            json!({ "range": [10] }),
            json!({ "range": [10, 20, 30] }),
            json!({ "range": [20, 10] }),
            json!({ "range": [-1, 20] }),
            json!({ "range": [1.5, 20] }),
            json!({ "range": "10..20" }),
        ] {
            assert!(arg_range(Some(&bad)).is_err(), "{bad}");
        }

        for bad in [
            json!({ "as_of": -1 }),
            json!({ "as_of": 1.5 }),
            json!({ "as_of": "now" }),
        ] {
            assert!(arg_ms(Some(&bad), "as_of").is_err(), "{bad}");
        }

        // Absent stays absent — an omitted filter is not a malformed one.
        assert_eq!(arg_ms(Some(&json!({})), "as_of"), Ok(None));
        assert_eq!(arg_ms(None, "as_of"), Ok(None));
        assert_eq!(arg_ms(Some(&json!({ "as_of": 7 })), "as_of"), Ok(Some(7)));
    }

    #[test]
    fn render_json_vs_human() {
        let v = json!({ "a": 1 });
        assert_eq!(render(&v, "json"), "{\"a\":1}");
        assert!(render(&v, "human").contains('\n'));
    }

    #[test]
    fn definitions_cover_both_modes() {
        let names = |defs: Vec<Value>| -> Vec<String> {
            defs.iter()
                .map(|d| d["name"].as_str().unwrap().to_owned())
                .collect()
        };
        let w = names(definitions());
        assert_eq!(w[0], REMEMBER);
        assert!(w.iter().any(|n| n == UNLINK));
        assert!(w.iter().any(|n| n == MAINTAIN) && w.iter().any(|n| n == CHECKPOINT));
        let ro = names(definitions_ro());
        assert!(ro.iter().any(|n| n == REFRESH) && ro.iter().any(|n| n == GENERATION));
        assert!(!ro.iter().any(|n| n == REMEMBER)); // no write verbs
    }

    #[test]
    fn settings_help_is_explicit_and_contains_shared_database_path() {
        let tmp = TempDir::new("settings-help");
        let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();
        let response = call(
            &db,
            json!(1),
            Some(&params(SETTINGS_HELP, json!({ "format": "json" }))),
        );
        let value: Value = serde_json::from_str(&text(&response)).unwrap();
        assert_eq!(value["topic"], "settings");
        assert!(
            value["settings"]
                .as_array()
                .unwrap()
                .iter()
                .any(|setting| { setting["section"] == "database" && setting["key"] == "path" })
        );
    }

    #[test]
    fn writer_call_dispatches_every_verb() {
        let tmp = TempDir::new("call");
        let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();

        // missing params → JSON-RPC error; unknown tool → tool error.
        assert_eq!(call(&db, json!(1), None)["error"]["code"], -32602);
        assert!(is_error(&call(
            &db,
            json!(1),
            Some(&params("plugmem_nope", json!({})))
        )));

        // remember → id 0.
        let r = call(
            &db,
            json!(1),
            Some(&params(
                "plugmem_remember",
                json!({"text": "prefers tokio", "entity": "user", "tags": ["pref"], "links": [{"rel":"at","entity":"acme"}]}),
            )),
        );
        assert!(!is_error(&r));
        let outcome: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(outcome["id"], 0);
        // remember without text → tool error.
        assert!(is_error(&call(
            &db,
            json!(1),
            Some(&params("plugmem_remember", json!({})))
        )));

        // recall json + human.
        let rj = call(
            &db,
            json!(2),
            Some(&params("plugmem_recall", json!({"query": "tokio"}))),
        );
        assert!(serde_json::from_str::<Value>(&text(&rj)).unwrap()["facts"].is_array());
        let rh = call(
            &db,
            json!(2),
            Some(&params(
                "plugmem_recall",
                json!({"query": "tokio", "format": "human"}),
            )),
        );
        assert!(text(&rh).contains("[f0]"));

        // show existing / missing.
        assert!(
            text(&call(
                &db,
                json!(3),
                Some(&params("plugmem_show", json!({"id": 0})))
            ))
            .contains("prefers tokio")
        );
        assert!(is_error(&call(
            &db,
            json!(3),
            Some(&params("plugmem_show", json!({"id": 999})))
        )));
        assert!(is_error(&call(
            &db,
            json!(3),
            Some(&params("plugmem_show", json!({})))
        ))); // no id

        // revise (id 0 → successor 1) / revise without id.
        let rv = call(
            &db,
            json!(4),
            Some(&params(
                "plugmem_revise",
                json!({"id": 0, "text": "prefers async-std"}),
            )),
        );
        assert_eq!(serde_json::from_str::<Value>(&text(&rv)).unwrap()["id"], 1);
        assert!(is_error(&call(
            &db,
            json!(4),
            Some(&params("plugmem_revise", json!({"text": "x"})))
        )));

        // link (ok / missing field).
        assert!(!is_error(&call(
            &db,
            json!(5),
            Some(&params(
                "plugmem_link",
                json!({"src": "user", "rel": "works_at", "dst": "acme"})
            ))
        )));
        assert!(is_error(&call(
            &db,
            json!(5),
            Some(&params("plugmem_link", json!({"src": "user"})))
        )));
        assert!(!is_error(&call(
            &db,
            json!(5),
            Some(&params(
                "plugmem_unlink",
                json!({"src": "user", "rel": "works_at", "dst": "acme"})
            ))
        )));
        assert!(is_error(&call(
            &db,
            json!(5),
            Some(&params("plugmem_unlink", json!({"src": "user"})))
        )));

        // stats / export / maintain / checkpoint / verify.
        assert!(
            serde_json::from_str::<Value>(&text(&call(
                &db,
                json!(6),
                Some(&params("plugmem_stats", json!({})))
            )))
            .unwrap()["facts"]
                .is_number()
        );
        assert!(
            serde_json::from_str::<Value>(&text(&call(
                &db,
                json!(7),
                Some(&params("plugmem_export", json!({})))
            )))
            .unwrap()
            .is_array()
        );
        assert!(!is_error(&call(
            &db,
            json!(8),
            Some(&params("plugmem_maintain", json!({})))
        )));
        assert!(!is_error(&call(
            &db,
            json!(9),
            Some(&params("plugmem_checkpoint", json!({})))
        )));
        assert!(!is_error(&call(
            &db,
            json!(10),
            Some(&params("plugmem_verify", json!({})))
        )));

        // forget the live successor.
        let f = call(
            &db,
            json!(11),
            Some(&params("plugmem_forget", json!({"id": 1}))),
        );
        assert_eq!(
            serde_json::from_str::<Value>(&text(&f)).unwrap()["forgotten"],
            true
        );
        assert!(is_error(&call(
            &db,
            json!(11),
            Some(&params("plugmem_forget", json!({})))
        ))); // no id

        // meta.
        assert!(
            text(&call(
                &db,
                json!(12),
                Some(&params("plugmem_version", json!({})))
            ))
            .contains("plugmem")
        );
        assert!(
            text(&call(
                &db,
                json!(13),
                Some(&params("plugmem_about", json!({})))
            ))
            .contains("skill")
        );
    }

    #[test]
    fn remember_accepts_metadata_and_show_returns_it_sorted() {
        let tmp = TempDir::new("meta");
        let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();

        // The schema advertises a `metadata` object of string values.
        let schema = &remember_def()["inputSchema"]["properties"]["metadata"];
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"]["type"], "string");

        // Remember with metadata whose keys arrive out of order.
        let r = call(
            &db,
            json!(1),
            Some(&params(
                "plugmem_remember",
                json!({"text": "a scan", "metadata": {"uri": "s3://b/x", "mime": "pdf"}}),
            )),
        );
        assert!(!is_error(&r));

        // show serializes the fact with its metadata, sorted by key.
        let card: Value = serde_json::from_str(&text(&call(
            &db,
            json!(2),
            Some(&params("plugmem_show", json!({"id": 0}))),
        )))
        .unwrap();
        assert_eq!(card["metadata"]["uri"], "s3://b/x");
        assert_eq!(card["metadata"]["mime"], "pdf");
    }

    #[test]
    fn read_only_call_dispatches_and_refuses_writes() {
        let tmp = TempDir::new("call-ro");
        // A writer stores + checkpoints, then is dropped so a read-only open sees
        // a published snapshot.
        {
            let (db, _) = Database::open(tmp.db(), Config::default()).unwrap();
            call(
                &db,
                json!(1),
                Some(&params(
                    "plugmem_remember",
                    json!({"text": "the sky is blue", "entity": "sky"}),
                )),
            );
            db.checkpoint(now_ms()).unwrap();
        }
        let ro = Database::open_readonly(tmp.db(), Config::default()).unwrap();
        let reader = ReaderShared::new(ro, None);

        // missing params / unknown tool.
        assert_eq!(call_ro(&reader, json!(1), None)["error"]["code"], -32602);
        assert!(is_error(&call_ro(
            &reader,
            json!(1),
            Some(&params("plugmem_nope", json!({})))
        )));

        // read verbs.
        assert!(
            serde_json::from_str::<Value>(&text(&call_ro(
                &reader,
                json!(2),
                Some(&params("plugmem_recall", json!({"query": "sky"})))
            )))
            .unwrap()["facts"]
                .is_array()
        );
        assert!(
            text(&call_ro(
                &reader,
                json!(3),
                Some(&params("plugmem_show", json!({"id": 0})))
            ))
            .contains("sky")
        );
        assert!(is_error(&call_ro(
            &reader,
            json!(3),
            Some(&params("plugmem_show", json!({"id": 999})))
        ))); // missing
        assert_eq!(
            serde_json::from_str::<Value>(&text(&call_ro(
                &reader,
                json!(4),
                Some(&params("plugmem_stats", json!({})))
            )))
            .unwrap()["facts"],
            1
        );
        assert!(
            serde_json::from_str::<Value>(&text(&call_ro(
                &reader,
                json!(5),
                Some(&params("plugmem_export", json!({})))
            )))
            .unwrap()
            .is_array()
        );
        assert!(!is_error(&call_ro(
            &reader,
            json!(6),
            Some(&params("plugmem_verify", json!({})))
        )));

        // freshness meta.
        assert!(
            serde_json::from_str::<Value>(&text(&call_ro(
                &reader,
                json!(7),
                Some(&params("plugmem_generation", json!({})))
            )))
            .unwrap()["generation"]
                .is_number()
        );
        let rf = call_ro(
            &reader,
            json!(8),
            Some(&params("plugmem_refresh", json!({}))),
        );
        assert_eq!(
            serde_json::from_str::<Value>(&text(&rf)).unwrap()["refreshed"],
            false
        );

        // meta + every write verb refused.
        assert!(
            text(&call_ro(
                &reader,
                json!(9),
                Some(&params("plugmem_version", json!({})))
            ))
            .contains("plugmem")
        );
        assert!(
            text(&call_ro(
                &reader,
                json!(10),
                Some(&params("plugmem_about", json!({})))
            ))
            .contains("skill")
        );
        for verb in [
            "plugmem_remember",
            "plugmem_revise",
            "plugmem_forget",
            "plugmem_link",
            "plugmem_unlink",
            "plugmem_maintain",
            "plugmem_checkpoint",
        ] {
            assert!(
                is_error(&call_ro(&reader, json!(11), Some(&params(verb, json!({}))))),
                "{verb} must be refused"
            );
        }
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::tests::{is_error, params, text};
    use super::*;
    use plugmem_host::{Config, Settings};

    /// A unique temp directory; removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-mcp-ws-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn name(s: &str) -> DbName {
        DbName::parse(s).unwrap()
    }

    fn shared(
        tmp: &TempDir,
        default: Option<&str>,
        allow: &[&str],
        create: bool,
    ) -> WorkspaceShared {
        let workspace = Settings::from_table(None)
            .unwrap()
            .open_workspace(&tmp.0)
            .unwrap();
        WorkspaceShared::new(
            workspace,
            default.map(name),
            allow.iter().copied().map(name).collect(),
            create,
        )
    }

    /// Every tool that takes a `db` argument, with its schema.
    fn db_prop(defs: &[Value], tool: &str) -> Option<Value> {
        defs.iter()
            .find(|d| d["name"] == tool)
            .and_then(|d| d["inputSchema"]["properties"].get(DB_ARG).cloned())
    }

    fn required(defs: &[Value], tool: &str) -> Vec<String> {
        defs.iter()
            .find(|d| d["name"] == tool)
            .and_then(|d| d["inputSchema"]["required"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect()
    }

    #[test]
    fn the_db_argument_appears_only_where_it_is_a_real_question() {
        // One database: the field does not exist. This is the default mode and
        // the one the model should never have to think about.
        let plain = definitions();
        assert_eq!(db_prop(&plain, REMEMBER), None);
        assert_eq!(required(&plain, REMEMBER), ["text"]);
        assert!(!plain.iter().any(|d| d["name"] == WORKSPACE_FIND));

        // A workspace with a default: the field exists, is optional, and says
        // what omitting it means.
        let defaulted = definitions_ws(Some(&name("chat-42")));
        let prop = db_prop(&defaulted, REMEMBER).unwrap();
        assert_eq!(prop["default"], "chat-42");
        assert_eq!(required(&defaulted, REMEMBER), ["text"]);

        // A workspace with no default: required, and listed first — the first
        // thing to settle is which memory this is about.
        let bare = definitions_ws(None);
        assert!(db_prop(&bare, REMEMBER).unwrap().get("default").is_none());
        assert_eq!(required(&bare, REMEMBER), [DB_ARG, "text"]);
        // Even a tool that required nothing before now requires `db`.
        assert_eq!(required(&bare, STATS), [DB_ARG]);

        // Both workspace modes advertise the two verbs for finding a name.
        for defs in [&defaulted, &bare] {
            assert!(defs.iter().any(|d| d["name"] == WORKSPACE_LIST));
            assert!(defs.iter().any(|d| d["name"] == WORKSPACE_FIND));
        }
    }

    #[test]
    fn every_tool_is_either_database_scoped_or_deliberately_not() {
        // The two sets together are exactly what the server advertises, so a
        // tool added later cannot quietly land in neither — which would leave
        // it unroutable in workspace mode.
        let mut advertised: Vec<String> = definitions()
            .iter()
            .map(|d| d["name"].as_str().unwrap().to_string())
            .collect();
        advertised.sort();
        let mut known: Vec<String> = DATABASE_TOOLS
            .iter()
            .chain(STATELESS_TOOLS)
            .map(|s| (*s).to_string())
            .collect();
        known.sort();
        assert_eq!(advertised, known);

        // Database-scoped tools gained `db`; the others deliberately did not.
        let bare = definitions_ws(None);
        for tool in DATABASE_TOOLS {
            assert!(db_prop(&bare, tool).is_some(), "{tool} has no {DB_ARG}");
        }
        for tool in STATELESS_TOOLS {
            assert_eq!(
                db_prop(&bare, tool),
                None,
                "{tool} should not take {DB_ARG}"
            );
        }
    }

    #[test]
    fn asking_what_version_is_running_needs_no_memory() {
        let tmp = TempDir::new("stateless");
        let ws = shared(&tmp, None, &[], true);
        for tool in STATELESS_TOOLS {
            let out = call_ws(&ws, json!(1), Some(&params(tool, json!({}))));
            assert!(!is_error(&out), "{tool}: {}", text(&out));
        }
    }

    #[test]
    fn a_call_reaches_the_named_database_and_only_that_one() {
        let tmp = TempDir::new("route");
        let ws = shared(&tmp, None, &[], true);

        for (db, fact) in [
            ("chat-42", "the sky is blue"),
            ("chat-43", "the sky is red"),
        ] {
            let out = call_ws(
                &ws,
                json!(1),
                Some(&params(REMEMBER, json!({ "db": db, "text": fact }))),
            );
            assert!(!is_error(&out), "{}", text(&out));
        }

        let out = call_ws(
            &ws,
            json!(2),
            Some(&params(RECALL, json!({ "db": "chat-42", "query": "sky" }))),
        );
        assert!(text(&out).contains("the sky is blue"), "{}", text(&out));
        assert!(!text(&out).contains("the sky is red"), "{}", text(&out));
    }

    #[test]
    fn without_a_default_the_db_argument_is_required_at_call_time_too() {
        let tmp = TempDir::new("required");
        let ws = shared(&tmp, None, &[], true);

        // A schema says required; a server must enforce it, because a model
        // that omits it must be told, not silently served someone else's memory.
        let out = call_ws(&ws, json!(1), Some(&params(STATS, json!({}))));
        assert!(is_error(&out));
        assert!(
            text(&out).contains("plugmem_workspace_find"),
            "{}",
            text(&out)
        );

        // With a default, the same call works and means the default.
        let ws = shared(&tmp, Some("chat-42"), &[], true);
        call_ws(
            &ws,
            json!(2),
            Some(&params(REMEMBER, json!({ "text": "a fact" }))),
        );
        let out = call_ws(&ws, json!(3), Some(&params(STATS, json!({}))));
        assert!(!is_error(&out));
        assert!(text(&out).contains("\"facts\":1"), "{}", text(&out));
    }

    #[test]
    fn creation_follows_the_verb_not_the_name() {
        let tmp = TempDir::new("create");
        let ws = shared(&tmp, None, &[], true);

        // Reading a name nobody has used is a typo far more often than it is a
        // new memory, so it is refused rather than answered with nothing.
        let out = call_ws(
            &ws,
            json!(1),
            Some(&params(RECALL, json!({ "db": "typo", "query": "x" }))),
        );
        assert!(is_error(&out));
        assert!(text(&out).contains("typo"));
        assert!(!ws.workspace().layout().exists(&name("typo")));

        // Writing to one creates it: that is how a new conversation gets a
        // memory without a registration step.
        let out = call_ws(
            &ws,
            json!(2),
            Some(&params(
                REMEMBER,
                json!({ "db": "chat-99", "text": "hello" }),
            )),
        );
        assert!(!is_error(&out), "{}", text(&out));
        assert!(ws.workspace().layout().exists(&name("chat-99")));

        // --no-create turns even that off.
        let strict = shared(&tmp, None, &[], false);
        let out = call_ws(
            &strict,
            json!(3),
            Some(&params(
                REMEMBER,
                json!({ "db": "chat-100", "text": "hello" }),
            )),
        );
        assert!(is_error(&out));
        assert!(!ws.workspace().layout().exists(&name("chat-100")));
    }

    #[test]
    fn a_name_outside_the_alphabet_or_the_allow_set_is_refused() {
        let tmp = TempDir::new("refuse");
        let ws = shared(&tmp, None, &["chat-42"], true);

        // Not a name at all — refused by the type, before any path is built.
        let out = call_ws(
            &ws,
            json!(1),
            Some(&params(STATS, json!({ "db": "../etc/passwd" }))),
        );
        assert!(is_error(&out));
        assert!(text(&out).contains("not a usable database name"));

        // A perfectly good name this server was not started for.
        let out = call_ws(
            &ws,
            json!(2),
            Some(&params(STATS, json!({ "db": "other" }))),
        );
        assert!(is_error(&out));
        assert!(text(&out).contains("not one of the memories"));

        // The allowed one works.
        call_ws(
            &ws,
            json!(3),
            Some(&params(REMEMBER, json!({ "db": "chat-42", "text": "x" }))),
        );
        let out = call_ws(
            &ws,
            json!(4),
            Some(&params(STATS, json!({ "db": "chat-42" }))),
        );
        assert!(!is_error(&out));
    }

    #[test]
    fn the_workspace_verbs_return_names_to_use_as_db() {
        let tmp = TempDir::new("find");
        let ws = shared(&tmp, None, &[], true);
        ws.workspace()
            .describe(
                &name("chat-42"),
                1_000,
                plugmem_host::Description {
                    text: "release planning and performance work",
                    tags: &["kind:chat"],
                    owner: Some("ann"),
                },
            )
            .unwrap();

        let listed = call_ws(&ws, json!(1), Some(&params(WORKSPACE_LIST, json!({}))));
        assert!(
            text(&listed).contains("\"db\":\"chat-42\""),
            "{}",
            text(&listed)
        );
        assert!(text(&listed).contains("\"owner\":\"ann\""));
        assert!(text(&listed).contains("\"archived\":false"));

        let found = call_ws(
            &ws,
            json!(2),
            Some(&params(
                WORKSPACE_FIND,
                json!({ "query": "release planning" }),
            )),
        );
        assert!(
            text(&found).contains("\"db\":\"chat-42\""),
            "{}",
            text(&found)
        );

        // And the name that came back is usable as `db` — the whole point.
        let out = call_ws(
            &ws,
            json!(3),
            Some(&params(STATS, json!({ "db": "chat-42" }))),
        );
        assert!(!is_error(&out));
    }

    #[test]
    fn concurrent_calls_to_different_databases_do_not_cross() {
        let tmp = TempDir::new("concurrent");
        let ws = std::sync::Arc::new(shared(&tmp, None, &[], true));
        let dbs = ["a", "b", "c", "d"];
        for db in dbs {
            call_ws(
                &ws,
                json!(0),
                Some(&params(
                    REMEMBER,
                    json!({ "db": db, "text": format!("i am {db}") }),
                )),
            );
        }

        // The direct test for the race a "switch database" verb would have
        // created: many workers, one shared server, each asking for a different
        // memory. Every answer must be its own.
        let mut handles = Vec::new();
        for db in dbs {
            for _ in 0..8 {
                let ws = std::sync::Arc::clone(&ws);
                handles.push(std::thread::spawn(move || {
                    let out = call_ws(
                        &ws,
                        json!(1),
                        Some(&params(RECALL, json!({ "db": db, "query": "i am" }))),
                    );
                    assert!(
                        text(&out).contains(&format!("i am {db}")),
                        "{db} got {}",
                        text(&out)
                    );
                    for other in dbs.iter().filter(|o| **o != db) {
                        assert!(!text(&out).contains(&format!("i am {other}")));
                    }
                }));
            }
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn an_unknown_tool_and_missing_params_behave_as_they_do_elsewhere() {
        let tmp = TempDir::new("misc");
        let ws = shared(&tmp, Some("chat-42"), &[], true);

        assert_eq!(call_ws(&ws, json!(1), None)["error"]["code"], -32602);
        let out = call_ws(&ws, json!(2), Some(&params("plugmem_nope", json!({}))));
        assert!(is_error(&out));
        assert!(text(&out).contains("unknown tool"));
    }

    #[test]
    fn a_registry_failure_is_a_tool_error_not_a_panic() {
        let tmp = TempDir::new("registry-busy");
        let ws = shared(&tmp, None, &[], true);
        // Hold the registry from "another process" so opening it fails.
        std::fs::create_dir_all(&tmp.0).unwrap();
        let _held = Database::open(ws.workspace().layout().registry_path(), Config::default())
            .unwrap()
            .0;

        for tool in [WORKSPACE_LIST, WORKSPACE_FIND] {
            let out = call_ws(&ws, json!(1), Some(&params(tool, json!({ "query": "x" }))));
            assert!(is_error(&out), "{tool}");
        }
    }
}
