//! Minimal JSON-RPC 2.0 over stdio (newline-delimited, one message per line):
//! the read loop, the worker pool, the method dispatcher, and the three
//! envelope constructors. The tool logic itself lives in [`crate::tools`].
//!
//! Concurrency (no tokio — the engine is CPU-bound and the one I/O wait, the
//! embedder HTTP call, is blocking and outside the engine lock): one read
//! thread pulls stdin lines into an `mpsc` channel; a pool of worker threads
//! drains it, dispatches, and writes replies under a `Mutex<Stdout>`. Each
//! worker holds a cheap [`Shared`] clone — a writer clones the `Database`
//! (`Arc` + engine `RwLock`), a reader clones an `Arc<ReaderShared>` — so
//! independent requests (e.g. two recalls, embedding outside the lock) overlap.
//! Replies carry their `id`, so out-of-order completion is correct JSON-RPC.

use std::io::{self, BufRead, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use plugmem_host::Database;
use serde_json::{Value, json};

use crate::tools::ReaderShared;
use crate::{messages, tools};

/// A per-worker handle to the backend, cloned into every worker thread. A
/// writer clones the `Database` (an `Arc` around the engine's `RwLock`); a
/// reader shares one `ReaderShared` (its own `RwLock`/`Mutex` inside) via
/// `Arc`. The mode picks the advertised tool set and the dispatch.
#[derive(Clone)]
pub enum Shared {
    /// Read-write: the full verb surface.
    Writer(Database),
    /// Read-only: read verbs + `refresh`/`generation`; write verbs are refused.
    Reader(Arc<ReaderShared>),
}

/// Run the server: a read thread feeding an `mpsc` channel, `workers` worker
/// threads draining it. Returns when stdin closes and every queued request has
/// been answered.
pub fn serve(shared: Shared, workers: usize) {
    let (tx, rx) = mpsc::channel::<String>();
    let rx = Arc::new(Mutex::new(rx));
    let out = Arc::new(Mutex::new(io::stdout()));

    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers.max(1) {
        let rx = Arc::clone(&rx);
        let out = Arc::clone(&out);
        let shared = shared.clone();
        handles.push(thread::spawn(move || worker(&shared, &rx, &out)));
    }

    // Read thread: push non-empty stdin lines into the channel. Dropping `tx`
    // (on EOF) closes the channel, so workers drain the rest and exit.
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        if tx.send(line).is_err() {
            break; // all workers gone
        }
    }
    drop(tx);
    for h in handles {
        let _ = h.join();
    }
}

/// One worker: pull a line, dispatch, write the reply (if any) under the shared
/// stdout lock. The receiver is behind a `Mutex`; a worker locks only to `recv`
/// (releasing before it processes), so the others keep pulling.
fn worker(shared: &Shared, rx: &Mutex<mpsc::Receiver<String>>, out: &Mutex<io::Stdout>) {
    loop {
        let line = {
            let rx = rx.lock().expect("receiver lock");
            match rx.recv() {
                Ok(line) => line,
                Err(_) => break, // channel closed and drained
            }
        };
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            continue; // ignore non-JSON lines
        };
        if let Some(response) = handle(shared, &req) {
            let mut out = out.lock().expect("stdout lock");
            let _ = writeln!(out, "{response}");
            let _ = out.flush();
        }
    }
}

/// Dispatch one JSON-RPC request. Returns `None` for notifications (no `id`) and
/// for anything that should not produce a reply.
fn handle(shared: &Shared, req: &Value) -> Option<Value> {
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
        "tools/list" => id.map(|id| {
            let tools = match shared {
                Shared::Writer(_) => tools::definitions(),
                Shared::Reader(_) => tools::definitions_ro(),
            };
            result(id, json!({ "tools": tools }))
        }),
        "tools/call" => id.map(|id| {
            let params = req.get("params");
            match shared {
                Shared::Writer(db) => tools::call(db, id, params),
                Shared::Reader(reader) => tools::call_ro(reader, id, params),
            }
        }),
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
