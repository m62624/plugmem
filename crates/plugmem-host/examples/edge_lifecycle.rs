//! Typed relationship lifecycle: `link`, `unlink`, and historical `as_of`
//! graph recall.
//!
//! Run with:
//!
//! ```text
//! cargo run -p plugmem-host --example edge_lifecycle
//! ```

use plugmem_host::{
    Config, Database, HostError, LinkInput, RecallQuery, RememberInput, UnlinkInput,
};

fn main() -> Result<(), HostError> {
    let dir = std::env::temp_dir().join(format!("plugmem-edge-lifecycle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("memory.plugmem");

    let (db, _) = Database::open(&path, Config::default())?;

    let ada = db.remember(RememberInput {
        now: 1_700_000_000_000,
        text: "Ada works on the storage engine.",
        entity: Some("ada"),
        tags: &["people"],
        links: &[],
        vector: None,
        valid_from: None,
        metadata: None,
    })?;
    db.remember(RememberInput {
        now: 1_700_000_000_010,
        text: "Acme builds embedded database tooling.",
        entity: Some("acme"),
        tags: &["orgs"],
        links: &[],
        vector: None,
        valid_from: None,
        metadata: None,
    })?;

    db.link(LinkInput {
        now: 1_700_000_000_100,
        src: "ada",
        rel: "works_at",
        dst: "acme",
        provenance: Some(ada.id),
    })?;
    println!("after link: current edges = {}", db.stats().edges);

    let entities = ["ada"];
    let current = db.recall(graph_query(1_700_000_000_200, None, &entities))?;
    assert_eq!(current.edges.len(), 1);

    let closed = db.unlink(UnlinkInput {
        now: 1_700_000_000_300,
        src: "ada",
        rel: "works_at",
        dst: "acme",
    })?;
    assert!(closed);
    println!(
        "after unlink: current edges = {}, historical edge versions = {}",
        db.stats().edges,
        db.stats().edge_versions
    );

    let now = db.recall(graph_query(1_700_000_000_400, None, &entities))?;
    let then = db.recall(graph_query(
        1_700_000_000_400,
        Some(1_700_000_000_200),
        &entities,
    ))?;
    assert_eq!(now.edges.len(), 0);
    assert_eq!(then.edges.len(), 1);
    println!("current recall sees {} edge(s)", now.edges.len());
    println!("as_of recall sees {} historical edge(s)", then.edges.len());

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

fn graph_query<'a>(now: u64, as_of: Option<u64>, entities: &'a [&'a str]) -> RecallQuery<'a> {
    let mut query = RecallQuery::text(now, "");
    query.text = None;
    query.entities = entities;
    query.as_of = as_of;
    query.k = 8;
    query
}
