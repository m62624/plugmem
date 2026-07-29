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
    Database, Embedder, FactId, HostError, LinkInput, ReadOnlyDatabase, RecallQuery, RememberInput,
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
const SHOW: &str = "plugmem_show";
const STATS: &str = "plugmem_stats";
const EXPORT: &str = "plugmem_export";
const MAINTAIN: &str = "plugmem_maintain";
const CHECKPOINT: &str = "plugmem_checkpoint";
const VERIFY: &str = "plugmem_verify";
const VERSION: &str = "plugmem_version";
const ABOUT: &str = "plugmem_about";

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
        show_def(),
        stats_def(),
        export_def(),
        maintain_def(),
        checkpoint_def(),
        verify_def(),
        version_def(),
        about_def(),
    ]
}

/// Execute a `tools/call`: route by tool name, then hand off. A missing `params`
/// is a JSON-RPC error; an unknown tool is a tool-level error (`isError`).
pub fn call(db: &Database, id: Value, params: Option<&Value>) -> Value {
    let Some(params) = params else {
        return rpc::error(id, -32602, "missing params");
    };
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments");

    match name {
        REMEMBER => remember(db, id, args, None),
        RECALL => recall(db, id, args),
        REVISE => revise(db, id, args),
        FORGET => forget(db, id, args),
        LINK => link(db, id, args),
        SHOW => show(db, id, args),
        STATS => rpc::tool_result(id, render(&db.stats(), format_arg(args)), false),
        EXPORT => rpc::tool_result(id, render(&db.export(), format_arg(args)), false),
        MAINTAIN => match db.maintain(now_ms()) {
            Ok(report) => rpc::tool_result(id, render(&report, format_arg(args)), false),
            Err(e) => tool_error(id, &e),
        },
        CHECKPOINT => match db.checkpoint(now_ms()) {
            Ok(()) => rpc::tool_result(id, render(&json!({ "ok": true }), format_arg(args)), false),
            Err(e) => tool_error(id, &e),
        },
        VERIFY => match db.verify() {
            Ok(()) => rpc::tool_result(id, render(&json!({ "ok": true }), format_arg(args)), false),
            Err(e) => tool_error(id, &e),
        },
        VERSION => rpc::tool_result(id, format!("plugmem {}", env!("CARGO_PKG_VERSION")), false),
        ABOUT => rpc::tool_result(id, messages::ABOUT_TOOL.to_string(), false),
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
        VERSION => rpc::tool_result(id, format!("plugmem {}", env!("CARGO_PKG_VERSION")), false),
        ABOUT => rpc::tool_result(id, messages::ABOUT_TOOL.to_string(), false),
        // A known write verb, or anything else: refused in read-only mode.
        REMEMBER | REVISE | FORGET | LINK | MAINTAIN | CHECKPOINT => {
            rpc::tool_result(id, messages::READ_ONLY_REFUSAL.into(), true)
        }
        other => rpc::tool_result(id, format!("unknown tool: {other}"), true),
    }
}

/// Read-only `recall`: embed the query text with the server's embedder (if any)
/// before searching — the read-only handle carries no embedder and does not
/// auto-embed. Lexical/graph/time still answer without one. The embed (its
/// slow HTTP) runs under the embedder `Mutex`, not the snapshot lock; the
/// search then takes the snapshot read guard.
fn recall_ro(reader: &ReaderShared, id: Value, args: Option<&Value>) -> Value {
    let format = format_arg(args);
    let query = arg_str(args, "query").map(String::from);
    // Embed the query text into an owned vector, if we can (embedder Mutex only).
    let vector = match query.as_deref() {
        Some(text) => {
            let mut embedder = reader.embedder.lock().expect("embedder lock");
            match embedder.as_mut() {
                Some(e) => match e.embed(&[text]) {
                    Ok(mut v) => v.pop(),
                    Err(e) => return tool_error(id, &e),
                },
                None => None,
            }
        }
        None => None,
    };
    let tags = arg_str_vec(args, "tags");
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let entities = arg_str_vec(args, "entities");
    let ent_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
    let q = RecallQuery {
        now: now_ms(),
        text: query.as_deref(),
        vector: vector.as_deref(),
        tags: &tag_refs,
        entities: &ent_refs,
        as_of: arg_u64(args, "as_of"),
        range: arg_range(args),
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
    let input = RememberInput {
        entity: arg_str(args, "entity"),
        tags: &tag_refs,
        links: &link_refs,
        metadata: (!meta_refs.is_empty()).then_some(meta_refs.as_slice()),
        valid_from: arg_u64(args, "valid_from"),
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
    let q = RecallQuery {
        now: now_ms(),
        text: arg_str(args, "query"),
        vector: None,
        tags: &tag_refs,
        entities: &ent_refs,
        as_of: arg_u64(args, "as_of"),
        range: arg_range(args),
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
    json!({
        "name": LINK,
        "description": messages::LINK_TOOL,
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
    format_only_def(MAINTAIN, messages::MAINTAIN_TOOL)
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
fn arg_range(args: Option<&Value>) -> Option<(u64, u64)> {
    let arr = args
        .and_then(|a| a.get("range"))
        .and_then(Value::as_array)?;
    Some((arr.first()?.as_u64()?, arr.get(1)?.as_u64()?))
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
fn tool_error(id: Value, e: &HostError) -> Value {
    rpc::tool_result(id, e.to_string(), true)
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
    fn params(name: &str, args: Value) -> Value {
        json!({ "name": name, "arguments": args })
    }

    /// The `text` field of a tool-call result envelope.
    fn text(v: &Value) -> String {
        v["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn is_error(v: &Value) -> bool {
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
        assert_eq!(arg_range(a), Some((10, 20)));
        assert_eq!(arg_range(None), None);
        assert_eq!(format_arg(a), "json");
        assert_eq!(format_arg(Some(&json!({"format": "human"}))), "human");
        assert_eq!(id_arg(Some(&json!({"id": 3}))), 3);
        assert_eq!(id_arg(None), 0);
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
        assert!(w.iter().any(|n| n == MAINTAIN) && w.iter().any(|n| n == CHECKPOINT));
        let ro = names(definitions_ro());
        assert!(ro.iter().any(|n| n == REFRESH) && ro.iter().any(|n| n == GENERATION));
        assert!(!ro.iter().any(|n| n == REMEMBER)); // no write verbs
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
