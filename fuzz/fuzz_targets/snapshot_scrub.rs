//! The snapshot container parser and its checksum walk, on arbitrary bytes.
//!
//! `Snapshot::parse` is the outermost layer — it reads the header, the section
//! table and the offsets every later loader indexes with. `scrub` then walks
//! the file in budgeted slices to verify per-section and whole-file checksums,
//! which means it turns attacker-controlled offsets and lengths into slice
//! ranges. Both run before anything has been validated, so they are the first
//! code a corrupt or hostile file reaches.
//!
//! Kept apart from `snapshot_open` on purpose: this target reaches the parser
//! on inputs that the engine loader would reject early, so the fuzzer can
//! explore the container independently of the engine's own structures.

#![no_main]

use libfuzzer_sys::fuzz_target;
use plugmem_core::snapshot::Snapshot;

fuzz_target!(|data: &[u8]| {
    let Ok(snap) = Snapshot::parse(data) else {
        return;
    };

    // Metadata the parser vouched for.
    let _ = snap.config();
    let _ = snap.engine_ver();
    // Section lookup over the table, including kinds that are not present.
    for kind in 0..64u16 {
        let _ = snap.section(kind);
    }
    let _ = snap.section(u16::MAX);

    // The checksum walk, driven to completion in small slices so the cursor's
    // own bookkeeping (section boundaries, the final whole-file digest) is
    // stepped through rather than done in one pass.
    let mut cursor = snap.scrub_with_budget(64);
    let mut steps = 0u32;
    for progress in &mut cursor {
        if progress.is_err() {
            break;
        }
        steps += 1;
        // A malformed table must not make the walk unbounded.
        if steps > 100_000 {
            break;
        }
    }
});
