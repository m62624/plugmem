//! Black-box tests of the MCP server: drive it over stdio JSON-RPC and check
//! replies. Each test opens a fresh memory file under `CARGO_TARGET_TMPDIR`.

use std::io::Write;
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
    let mut child = Command::new(env!("CARGO_BIN_EXE_plugmem-mcp"))
        .arg("--db")
        .arg(db)
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
        .map(|l| serde_json::from_str(l).unwrap())
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
    // The three v1 tools are advertised, in order.
    assert_eq!(resps[1]["result"]["tools"][0]["name"], "plugmem_stats");
    assert_eq!(resps[1]["result"]["tools"][1]["name"], "plugmem_version");
    assert_eq!(resps[1]["result"]["tools"][2]["name"], "plugmem_about");

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
