//! Replaying an arbitrary journal must never panic.
//!
//! The journal is the other untrusted file: an append-only log that a crash
//! can truncate mid-record and that lives next to the snapshot with no
//! checksum of its own. Replay decodes it into the *same* `apply_*` path live
//! writes use, so a decoder that trusts a length prefix would hand malformed
//! ids straight to the write side.
//!
//! Splitting the input gives the fuzzer both halves at once: a prefix it can
//! shape into a snapshot and a suffix replayed on top of it. Most inputs
//! exercise replay against an empty engine, which is the case that matters —
//! a torn tail on a fresh database.

#![no_main]

use libfuzzer_sys::fuzz_target;
use plugmem_core::{Config, Memory};

fuzz_target!(|data: &[u8]| {
    let cfg = Config::default();

    // Replay alone, over an empty engine.
    if let Ok((mem, _)) = Memory::from_bytes(None, data, cfg.clone()) {
        let _ = mem.verify();
        let _ = mem.stats();
        // A replayed engine must re-emit; the ids replay assigned are the ones
        // the snapshot writer will trust.
        let _ = mem.snapshot_bytes(0);
    }

    // Snapshot + journal tail: the pair a real open sees. The first byte picks
    // the split so the fuzzer can move the boundary.
    if let Some((&pick, rest)) = data.split_first() {
        let at = (pick as usize * rest.len()) / 256;
        let (snapshot, journal) = rest.split_at(at);
        if let Ok((mem, _)) = Memory::from_bytes(Some(snapshot), journal, cfg) {
            let _ = mem.verify();
            let _ = mem.snapshot_bytes(0);
        }
    }
});
