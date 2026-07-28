//! The tools the server exposes — their schema definitions and the `tools/call`
//! executor. Each name is `plugmem_*` so it never collides with another MCP
//! server's tools. Descriptions come from [`crate::messages`]; result types
//! serialize straight to JSON via serde (the host `serde` feature); envelopes
//! from [`crate::rpc`].

use plugmem_host::Database;
use serde::Serialize;
use serde_json::{Value, json};

use crate::{messages, rpc};

/// The tool names, shared by each definition and the `tools/call` dispatcher so
/// the advertised name and the routed name can never drift apart.
const STATS: &str = "plugmem_stats";
const VERSION: &str = "plugmem_version";
const ABOUT: &str = "plugmem_about";

/// Every tool definition, in the order `tools/list` advertises them.
pub fn definitions() -> Vec<Value> {
    vec![stats_def(), version_def(), about_def()]
}

/// A no-argument tool definition: a name, a description, and an empty input
/// schema. Shared by the tools that take no parameters.
fn simple_def(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": { "type": "object", "properties": {} }
    })
}

/// A read-tool definition that takes only the shared `format` argument.
fn format_def(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["human", "json"],
                    "description": messages::ARG_FORMAT
                }
            }
        }
    })
}

fn stats_def() -> Value {
    format_def(STATS, messages::STATS_TOOL)
}

/// `plugmem_version` — the MCP analog of `plugmem-cli --version`, so a model can
/// read the running version (it cannot see `initialize`'s `serverInfo.version`)
/// and compare it to the version its skill targets.
fn version_def() -> Value {
    simple_def(VERSION, messages::VERSION_TOOL)
}

/// `plugmem_about` — a pointer to the companion skill for agents that reached
/// this server without it. No version here; that is `plugmem_version`.
fn about_def() -> Value {
    simple_def(ABOUT, messages::ABOUT_TOOL)
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
        VERSION => rpc::tool_result(id, format!("plugmem {}", env!("CARGO_PKG_VERSION")), false),
        ABOUT => rpc::tool_result(id, messages::ABOUT_TOOL.to_string(), false),
        STATS => rpc::tool_result(id, render(&db.stats(), format_arg(args)), false),
        other => rpc::tool_result(id, format!("unknown tool: {other}"), true),
    }
}

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
