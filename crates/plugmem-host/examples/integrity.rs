//! On-demand integrity and recovery.
//!
//! Run with: `cargo run --example integrity -p plugmem-host`
//!
//! Shows the three integrity layers the host exposes over one file:
//!   * `scrub()`   — resumable byte-level container check (bitrot detector)
//!   * `verify()`  — content consistency (text UTF-8, vector bijection)
//!   * `recover()` — salvage a content-corrupt database into a clean copy
//!
//! The default open trusts the file and stays sparse, so integrity is a
//! deliberate, on-demand step — never paid at open time.

use plugmem_host::{Config, Database, HostError, RememberInput};

fn main() -> Result<(), HostError> {
    // A throwaway database under the OS temp directory.
    let dir = std::env::temp_dir().join(format!("plugmem-integrity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("agent.plugmem");
    let dst = dir.join("agent.recovered.plugmem");

    // Build a small memory and checkpoint it (a read-only open and a scrub both
    // need a checkpointed, empty-journal file).
    {
        let (db, _) = Database::open(&src, Config::default())?;
        for i in 0..10u64 {
            db.remember(RememberInput::text(i + 1, "a durable fact worth keeping"))?;
        }
        db.checkpoint(1_784_000_000_000)?;
        println!("wrote {} facts", db.stats().facts);
    }

    // scrub(): walk the container byte for byte, a slice at a time. You pace it
    // — here we run it to completion and print the final progress.
    {
        let ro = Database::open_readonly(&src, Config::default())?;
        let mut last = None;
        for step in ro.scrub()? {
            last = Some(step?); // an Err here would name the first damaged section
        }
        if let Some(p) = last {
            println!(
                "scrub ok: {}/{} bytes verified",
                p.done_bytes, p.total_bytes
            );
        }

        // verify(): content consistency (a heavier, one-shot check).
        ro.verify()?;
        println!("verify ok");
    }

    // recover(): open the source, drop any content-corrupt facts, rebuild, and
    // write a clean copy to `dst` — the source is left untouched. On this clean
    // database nothing is dropped; on a corrupt one, the report would count the
    // salvaged casualties.
    let report = Database::recover(&src, &dst, Config::default(), 1_784_000_100_000)?;
    println!(
        "recover: kept {}, dropped {} text + {} vector (source preserved)",
        report.kept, report.dropped_text, report.dropped_vector
    );

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
