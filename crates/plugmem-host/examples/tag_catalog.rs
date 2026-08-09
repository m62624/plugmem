//! Bounded tag discovery and history-preserving global tag removal.
//!
//! Run with a throwaway database path:
//!
//! ```text
//! cargo run -p plugmem-host --example tag_catalog -- /tmp/tag-catalog.plugmem
//! ```

use std::path::PathBuf;

use plugmem_host::{Config, Database, HostError, RecallQuery, RememberInput, TagQuery};

fn main() -> Result<(), HostError> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("plugmem-tag-catalog.plugmem"));
    let (db, _) = Database::open(&path, Config::default())?;

    db.remember(RememberInput {
        tags: &["project:plugmem", "personal"],
        ..RememberInput::text(1_700_000_000_000, "plugmem is an embedded memory engine")
    })?;
    let old = db.remember(RememberInput {
        tags: &["project:plugmem", "project:old"],
        ..RememberInput::text(1_700_000_000_100, "the old project label is temporary")
    })?;

    let page = db.list_tags(TagQuery {
        prefix: Some("project:"),
        limit: 64,
        ..TagQuery::default()
    })?;
    for tag in &page.items {
        println!("before: {} ({})", tag.name, tag.count);
    }
    assert!(page.next_cursor.is_none());

    let report = db.remove_tag(1_700_000_001_000, "project:old")?;
    println!("removed from {} current fact(s)", report.affected);
    assert_eq!(report.affected, 1);

    let current = db.list_tags(TagQuery::default())?;
    assert!(current.items.iter().all(|tag| tag.name != "project:old"));

    // The predecessor is closed, not deleted: an as-of query still sees it
    // with its historical tag classification.
    assert!(db.tags_of(old.id).iter().any(|tag| tag == "project:old"));
    let tags = ["project:old"];
    let historical = db.recall(RecallQuery {
        tags: &tags,
        as_of: Some(1_700_000_000_500),
        include_closed: true,
        ..RecallQuery::text(1_700_000_002_000, "old project label")
    })?;
    assert_eq!(historical.facts.len(), 1);
    println!("historical fact {} still has project:old", old.id.0);

    // Persist the derived catalogue. Reopening a legacy v1 image without this
    // section also rebuilds it automatically; its next checkpoint writes it.
    db.checkpoint(1_700_000_003_000)?;
    drop(db);
    let (reopened, _) = Database::open(&path, Config::default())?;
    let page = reopened.list_tags(TagQuery::default())?;
    println!("after reopen: {} current tag(s)", page.items.len());
    Ok(())
}
