//! Black-box tests of the MCP server: drive it over stdio JSON-RPC and check
//! replies. Each test opens a fresh memory file under `CARGO_TARGET_TMPDIR`.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::Value;

/// A unique memory path under the cargo target tmp dir (no embedder needed —
/// the default config runs lexical/graph/time recall).
fn temp_db(tag: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "mcp-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("m.plugmem")
}

/// Feed each request line to a server opened on `db`, return the parsed replies.
fn roundtrip(db: &PathBuf, requests: &[&str]) -> Vec<Value> {
    roundtrip_args(db, &[], requests)
}

/// Like [`roundtrip`], with extra binary arguments (e.g. `--read-only`).
///
/// **Synchronous**: send one request, await its reply, then the next — the
/// behavior of a real MCP client. This keeps a causal chain (remember → recall)
/// deterministic no matter how many workers the pool has (only one request is
/// ever in flight). A notification (no `id`) gets no reply, so none is awaited.
fn roundtrip_args(db: &PathBuf, extra: &[&str], requests: &[&str]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_plugmem-mcp"))
        .arg("--db")
        .arg(db)
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn plugmem-mcp");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut replies = Vec::new();
    for r in requests {
        writeln!(stdin, "{r}").unwrap();
        stdin.flush().unwrap();
        // A request carries an `id`; a notification does not (no reply to await).
        let is_request = serde_json::from_str::<Value>(r)
            .map(|v| v.get("id").is_some())
            .unwrap_or(false);
        if is_request {
            let mut line = String::new();
            stdout.read_line(&mut line).unwrap();
            replies.push(serde_json::from_str(line.trim()).unwrap());
        }
    }
    drop(stdin); // EOF → server exits
    let _ = child.wait();
    replies
}

/// Send every request at once (pipelined), then read all replies and index them
/// by `id`. Exercises the worker pool: independent requests may complete in any
/// order, so a positional read would be wrong — reply correlation is by `id`.
fn pipelined_by_id(
    db: &PathBuf,
    extra: &[&str],
    requests: &[&str],
) -> std::collections::HashMap<u64, Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_plugmem-mcp"))
        .arg("--db")
        .arg(db)
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn plugmem-mcp");
    {
        let stdin = child.stdin.as_mut().unwrap();
        for r in requests {
            writeln!(stdin, "{r}").unwrap();
        }
    } // drop stdin → EOF → server exits
    let output = child.wait_with_output().unwrap();
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .map(|v| (v["id"].as_u64().unwrap(), v))
        .collect()
}

#[test]
fn initialize_list_and_stats() {
    let db = temp_db("init");
    let resps = roundtrip(
        &db,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#, // notification → no reply
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"plugmem_stats","arguments":{}}}"#,
        ],
    );

    assert_eq!(resps.len(), 3, "the notification must not get a reply");
    assert_eq!(resps[0]["result"]["serverInfo"]["name"], "plugmem");
    assert_eq!(resps[0]["result"]["protocolVersion"], "2024-11-05");
    // The verb surface is advertised: write verbs first, meta last.
    let tools: Vec<&str> = resps[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(tools[0], "plugmem_remember");
    assert_eq!(tools.last(), Some(&"plugmem_about"));
    for expected in ["plugmem_recall", "plugmem_stats", "plugmem_version"] {
        assert!(
            tools.contains(&expected),
            "missing tool {expected} in {tools:?}"
        );
    }

    // stats returns machine JSON with the size counters; a fresh db has 0 facts.
    let text = resps[2]["result"]["content"][0]["text"].as_str().unwrap();
    let stats: Value = serde_json::from_str(text).unwrap();
    assert_eq!(stats["facts"], 0);
    assert_eq!(resps[2]["result"]["isError"], false);
}

#[test]
fn version_and_about_are_listed_and_callable() {
    let db = temp_db("meta");
    let resps = roundtrip(
        &db,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_version","arguments":{}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"plugmem_about","arguments":{}}}"#,
        ],
    );

    // plugmem_version returns the running version, matching serverInfo.
    let version = resps[0]["result"]["serverInfo"]["version"]
        .as_str()
        .unwrap();
    let vtext = resps[1]["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(resps[1]["result"]["isError"], false);
    assert!(
        vtext.contains(version),
        "version tool `{vtext}` should contain {version}"
    );

    // about points at the skill and the project.
    let atext = resps[2]["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(resps[2]["result"]["isError"], false);
    assert!(
        atext.contains("skill"),
        "about should mention the skill: {atext}"
    );
    assert!(
        atext.contains("github.com/m62624/plugmem"),
        "about should link the project: {atext}"
    );
}

#[test]
fn unknown_method_is_a_jsonrpc_error_and_unknown_tool_is_a_tool_error() {
    let db = temp_db("errors");
    let resps = roundtrip(
        &db,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"does/not/exist"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_nope","arguments":{}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call"}"#,
        ],
    );
    // Unknown method → JSON-RPC error -32601.
    assert_eq!(resps[0]["error"]["code"], -32601);
    // Unknown tool → tool-level error (in the result, not a protocol error).
    assert_eq!(resps[1]["result"]["isError"], true);
    // Missing params → JSON-RPC error -32602.
    assert_eq!(resps[2]["error"]["code"], -32602);
}

#[test]
fn writer_verbs_round_trip() {
    let db = temp_db("writer");
    let resps = roundtrip(
        &db,
        &[
            // remember a fact with an entity and a tag → id 0
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"plugmem_remember","arguments":{"text":"prefers tokio","entity":"user","tags":["pref"]}}}"#,
            // recall it (json) — should surface the fact
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_recall","arguments":{"query":"runtime tokio"}}}"#,
            // show fact 0
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"plugmem_show","arguments":{"id":0}}}"#,
            // revise fact 0
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"plugmem_revise","arguments":{"id":0,"text":"prefers async-std","entity":"user"}}}"#,
            // link two entities
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"plugmem_link","arguments":{"src":"user","rel":"works_at","dst":"acme"}}}"#,
            // export the open facts
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"plugmem_export","arguments":{}}}"#,
            // operational verbs
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"plugmem_maintain","arguments":{}}}"#,
            r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"plugmem_checkpoint","arguments":{}}}"#,
            r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"plugmem_verify","arguments":{}}}"#,
            // forget the (revised) successor fact 1
            r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"plugmem_forget","arguments":{"id":1}}}"#,
        ],
    );

    // remember → id 0, no error.
    let remembered: Value =
        serde_json::from_str(resps[0]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(remembered["id"], 0);
    assert_eq!(resps[0]["result"]["isError"], false);

    // recall → structured result carrying fact 0.
    let recalled: Value =
        serde_json::from_str(resps[1]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(
        recalled["facts"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "recall should surface the fact: {recalled}"
    );

    // show fact 0 → its text.
    let shown = resps[2]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(shown.contains("prefers tokio"), "show: {shown}");

    // revise → the successor id (1), no error.
    let revised: Value =
        serde_json::from_str(resps[3]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(revised["id"], 1);

    // link ok.
    assert_eq!(resps[4]["result"]["isError"], false);

    // export → a JSON array of the open facts (>=1).
    let exported: Value =
        serde_json::from_str(resps[5]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(exported.as_array().map(|a| !a.is_empty()).unwrap_or(false));

    // maintain/checkpoint/verify all succeed.
    assert_eq!(resps[6]["result"]["isError"], false);
    assert_eq!(resps[7]["result"]["isError"], false);
    assert_eq!(resps[8]["result"]["isError"], false);

    // forget the live successor → forgotten: true.
    let forgotten: Value =
        serde_json::from_str(resps[9]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(forgotten["forgotten"], true);
}

#[test]
fn recall_human_format_is_the_prompt_block() {
    let db = temp_db("recall-human");
    let resps = roundtrip(
        &db,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"plugmem_remember","arguments":{"text":"the sky is blue","entity":"sky"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_recall","arguments":{"query":"sky colour","format":"human"}}}"#,
        ],
    );
    // The human block carries the fact id marker `[f0]` (the prompt-ready text),
    // not a JSON object.
    let block = resps[1]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        block.contains("[f0]"),
        "human recall should be the block: {block}"
    );
}

#[test]
fn missing_required_argument_is_a_tool_error() {
    let db = temp_db("missing-arg");
    let resps = roundtrip(
        &db,
        &[
            // remember without text
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"plugmem_remember","arguments":{"entity":"x"}}}"#,
            // show a non-existent fact
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_show","arguments":{"id":999}}}"#,
        ],
    );
    assert_eq!(resps[0]["result"]["isError"], true);
    assert_eq!(resps[1]["result"]["isError"], true);
}

#[test]
fn read_only_serves_reads_and_refuses_writes() {
    let db = temp_db("ro");
    // A writer process stores a fact and checkpoints (so a read-only open has a
    // published snapshot), then exits when its stdin closes.
    let w = roundtrip(
        &db,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"plugmem_remember","arguments":{"text":"prefers tokio","entity":"user"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_checkpoint","arguments":{}}}"#,
        ],
    );
    assert_eq!(
        w[1]["result"]["isError"], false,
        "checkpoint should succeed"
    );

    // A separate read-only process observes that snapshot.
    let r = roundtrip_args(
        &db,
        &["--read-only"],
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugmem_stats","arguments":{}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"plugmem_recall","arguments":{"query":"tokio"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"plugmem_generation","arguments":{}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"plugmem_refresh","arguments":{}}}"#,
            // a write verb must be refused
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"plugmem_remember","arguments":{"text":"nope"}}}"#,
        ],
    );

    // The advertised set is read-only: refresh is offered, remember is not.
    let tools: Vec<&str> = r[0]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(tools.contains(&"plugmem_refresh"), "ro tools: {tools:?}");
    assert!(tools.contains(&"plugmem_generation"), "ro tools: {tools:?}");
    assert!(
        !tools.contains(&"plugmem_remember"),
        "ro must not offer writes: {tools:?}"
    );

    // stats sees the checkpointed fact.
    let stats: Value =
        serde_json::from_str(r[1]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(stats["facts"], 1);

    // recall (lexical, no embedder) surfaces it.
    let recalled: Value =
        serde_json::from_str(r[2]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(
        recalled["facts"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "ro recall should find the fact: {recalled}"
    );

    // generation is a number; refresh reports current generation, nothing newer.
    let generation: Value =
        serde_json::from_str(r[3]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(generation["generation"].is_number());
    let refreshed: Value =
        serde_json::from_str(r[4]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        refreshed["refreshed"], false,
        "nothing published since open"
    );

    // the write verb is refused as a tool-level error.
    assert_eq!(r[5]["result"]["isError"], true);
}

#[test]
fn worker_pool_answers_every_pipelined_request() {
    let db = temp_db("pool");
    // Fire many independent read-only requests at once with several workers; the
    // pool may answer in any order, but every id must get exactly one correct
    // reply and no line may be interleaved/garbled (Mutex<Stdout>).
    let mut reqs: Vec<String> = Vec::new();
    for i in 1..=20u64 {
        let verb = if i % 2 == 0 {
            "plugmem_stats"
        } else {
            "plugmem_version"
        };
        reqs.push(format!(
            r#"{{"jsonrpc":"2.0","id":{i},"method":"tools/call","params":{{"name":"{verb}","arguments":{{}}}}}}"#
        ));
    }
    let req_refs: Vec<&str> = reqs.iter().map(String::as_str).collect();
    let replies = pipelined_by_id(&db, &["--workers", "4"], &req_refs);

    assert_eq!(
        replies.len(),
        20,
        "every request must be answered exactly once"
    );
    for i in 1..=20u64 {
        let reply = replies
            .get(&i)
            .unwrap_or_else(|| panic!("no reply for id {i}"));
        assert_eq!(reply["result"]["isError"], false, "id {i}: {reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        if i % 2 == 0 {
            // stats → valid JSON with the counters.
            assert!(
                serde_json::from_str::<Value>(text).is_ok(),
                "id {i} stats: {text}"
            );
        } else {
            assert!(text.contains("plugmem"), "id {i} version: {text}");
        }
    }
}

#[test]
fn human_format_pretty_prints() {
    let db = temp_db("human");
    let resps = roundtrip(
        &db,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"plugmem_stats","arguments":{"format":"human"}}}"#,
        ],
    );
    let text = resps[0]["result"]["content"][0]["text"].as_str().unwrap();
    // Pretty JSON is multi-line and indented; compact JSON is not.
    assert!(text.contains('\n'), "human format should be pretty: {text}");
}
