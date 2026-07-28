//! Minimal JSON-RPC 2.0 over stdio (newline-delimited, one message per line):
//! the read/write loop, the method dispatcher, and the three envelope
//! constructors. The tool logic itself lives in [`crate::tools`].
//!
//! This loop is sequential; the worker pool (one read thread, an `mpsc`
//! channel, N workers, a `Mutex<Stdout>`) lands in a later milestone. The
//! envelope shape and dispatch are the stable part.

use std::io::{self, BufRead, Write};

use plugmem_host::Database;
use serde_json::{Value, json};

use crate::{messages, tools};

/// Read newline-delimited requests from stdin and write replies to stdout,
/// flushing each. Blank and non-JSON lines are skipped; notifications (no `id`)
/// get no reply. Runs until stdin closes.
pub fn serve(db: &Database) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            continue; // ignore non-JSON lines
        };
        if let Some(response) = handle(db, &req) {
            let _ = writeln!(out, "{response}");
            let _ = out.flush();
        }
    }
}

/// Dispatch one JSON-RPC request. Returns `None` for notifications (no `id`) and
/// for anything that should not produce a reply.
fn handle(db: &Database, req: &Value) -> Option<Value> {
    let method = req.get("method")?.as_str()?;
    let id = req.get("id").cloned(); // absent ⇒ notification ⇒ no reply

    match method {
        "initialize" => id.map(|id| {
            result(
                id,
                json!({
                    "protocolVersion": messages::PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": messages::SERVER_NAME, "version": env!("CARGO_PKG_VERSION") }
                }),
            )
        }),
        "notifications/initialized" => None,
        "ping" => id.map(|id| result(id, json!({}))),
        "tools/list" => id.map(|id| result(id, json!({ "tools": tools::definitions() }))),
        "tools/call" => id.map(|id| tools::call(db, id, req.get("params"))),
        // Unknown method: error only for requests (notifications are ignored).
        _ => id.map(|id| error(id, -32601, "method not found")),
    }
}

/// A successful JSON-RPC response envelope.
pub fn result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// A JSON-RPC error response envelope.
pub fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// A tool result carrying text. `is_error` follows the MCP convention: tool-level
/// failures are reported in the result (not as a JSON-RPC error) so the model can
/// read and react to them.
pub fn tool_result(id: Value, text: String, is_error: bool) -> Value {
    result(
        id,
        json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }),
    )
}
