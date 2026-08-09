//! Explicitly move a database between incompatible embedding spaces.
//!
//! This uses deterministic local embedders so the example needs no server:
//!
//! ```text
//! cargo run -p plugmem-host --example reembed
//! ```

use plugmem_host::{Config, Database, Embedder, HostError, RecallQuery, RememberInput, TagQuery};

struct DemoEmbedder {
    space: &'static str,
    dim: usize,
    phase: f32,
}

impl Embedder for DemoEmbedder {
    fn space_id(&self) -> &str {
        self.space
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        Ok(texts
            .iter()
            .map(|text| {
                (0..self.dim)
                    .map(|i| (text.len() as f32 + i as f32 + self.phase).sin())
                    .collect()
            })
            .collect())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("plugmem-reembed-example-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("memory.plugmem");

    let mut config = Config::default();
    config.dim = 4;
    let (db, _) = Database::builder(config)
        .embedder(Box::new(DemoEmbedder {
            space: "demo-v1",
            dim: 4,
            phase: 0.0,
        }))
        .open(&path)?;
    db.remember(RememberInput {
        tags: &["example", "kept"],
        ..RememberInput::text(1, "plugmem keeps temporal history")
    })?;

    let report = db.reembed_with(
        2,
        Box::new(DemoEmbedder {
            space: "demo-v2",
            dim: 6,
            phase: 1.0,
        }),
        32,
    )?;
    assert_eq!(report.previous_space.as_deref(), Some("demo-v1"));
    assert_eq!(report.new_space, "demo-v2");
    assert_eq!((report.previous_dim, report.new_dim), (4, 6));
    assert_eq!(db.list_tags(TagQuery::default())?.items.len(), 2);
    assert_eq!(db.recall(RecallQuery::text(3, "history"))?.facts.len(), 1);

    println!(
        "reembedded {} retained fact(s): {}D {} -> {}D {}",
        report.embedded,
        report.previous_dim,
        report.previous_space.as_deref().unwrap_or("untracked"),
        report.new_dim,
        report.new_space,
    );
    drop(db);
    std::fs::remove_dir_all(dir)?;
    Ok(())
}
