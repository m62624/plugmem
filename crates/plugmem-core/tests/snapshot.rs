//! Snapshot container tests (specs/03 test plan): canonical roundtrip,
//! build determinism, the full bitflip matrix (every byte, two bits — caught
//! by the structural parse or the on-demand scrub, never a panic, never a
//! silent wrong read), the truncation sweep, scrub semantics, and the Config
//! codec.

use plugmem_core::config::{ENCODED_LEN, RESERVED_AT};
use plugmem_core::snapshot::{FLAG_VECTORS, Snapshot, SnapshotWriter};
use plugmem_core::{Config, Error};

/// Runs a full container-checksum scrub over already-parsed bytes, returning
/// the first mismatch (or `Ok` when every section and the file hash verify).
fn scrub(bytes: &[u8]) -> Result<(), Error> {
    let snap = Snapshot::parse(bytes)?;
    for step in snap.scrub() {
        step?;
    }
    Ok(())
}

/// A small three-section file (empty section included) used throughout.
fn sample() -> Vec<u8> {
    let mut cfg_bytes = Vec::new();
    Config::default().encode(&mut cfg_bytes);
    let mut w = SnapshotWriter::new();
    w.section(1, b"facts-meta".to_vec()).unwrap();
    w.section(2, vec![0xC7; 200]).unwrap();
    w.section(9, Vec::new()).unwrap();
    w.finish(&cfg_bytes, 0, 1_784_000_000_000, "0.1.0")
}

#[test]
fn roundtrip_and_field_access() {
    let bytes = sample();
    assert_eq!(bytes.len() % 64, 0);
    let snap = Snapshot::parse(&bytes).unwrap();
    assert_eq!(snap.flags, 0);
    assert_eq!(snap.created_at, 1_784_000_000_000);
    assert_eq!(snap.engine_ver(), "0.1.0");
    assert_eq!(snap.section(1), Some(&b"facts-meta"[..]));
    assert_eq!(snap.section(2).unwrap().len(), 200);
    assert_eq!(snap.section(9), Some(&[][..]));
    assert_eq!(snap.section(3), None);
    let decoded = Config::decode(snap.config()).unwrap();
    assert_eq!(decoded, Config::default());
}

#[test]
fn build_is_deterministic() {
    assert_eq!(sample(), sample());
}

#[test]
fn writer_rejects_duplicate_kinds() {
    let mut w = SnapshotWriter::new();
    w.section(4, Vec::new()).unwrap();
    assert_eq!(
        w.section(4, b"again".to_vec()),
        Err(Error::Corrupt("duplicate section kind"))
    );
}

#[test]
fn vector_flag_roundtrips_and_unknown_flags_fail() {
    let w = SnapshotWriter::new();
    let bytes = w.finish(b"", FLAG_VECTORS, 7, "x");
    assert_eq!(Snapshot::parse(&bytes).unwrap().flags, FLAG_VECTORS);
    let w = SnapshotWriter::new();
    let bytes = w.finish(b"", 0b100, 7, "x");
    assert_eq!(
        Snapshot::parse(&bytes).unwrap_err(),
        Error::Corrupt("unknown flag bits set")
    );
}

#[test]
fn every_bitflip_is_caught_by_parse_or_scrub() {
    let bytes = sample();
    for at in 0..bytes.len() {
        for bit in [0x01u8, 0x80] {
            let mut b = bytes.clone();
            b[at] ^= bit;
            // No flip passes unnoticed: the structural parse rejects layout
            // damage, and the on-demand scrub catches every content/checksum
            // flip via the per-section and whole-file xxh3 (specs/16 §9).
            match Snapshot::parse(&b) {
                Err(_) => {}
                Ok(snap) => {
                    let caught = snap.scrub().any(|s| s.is_err());
                    assert!(
                        caught,
                        "flip at {at}/bit {bit:02x} slipped past parse and scrub"
                    );
                }
            }
        }
    }
}

#[test]
fn truncation_is_a_typed_error() {
    let bytes = sample();
    for cut in 0..bytes.len() {
        assert!(
            Snapshot::parse(&bytes[..cut]).is_err(),
            "prefix of {cut} bytes accepted"
        );
    }
}

#[test]
fn structural_gates_reachable_only_by_crafting() {
    // Trailing 64-byte block of zeros: aligned, structurally silent —
    // caught by the trailing-bytes rule (a structural check at parse, not a
    // checksum one).
    let mut bytes = sample();
    bytes.extend_from_slice(&[0u8; 64]);
    assert_eq!(
        Snapshot::parse(&bytes).unwrap_err(),
        Error::Corrupt("trailing bytes after the last section")
    );

    // Two table entries claiming the same kind (rewrite entry 2's kind to
    // entry 1's): a parse-side duplicate, distinct from the writer gate.
    let bytes = sample();
    let table_start =
        (64 + u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize).next_multiple_of(64);
    let mut b = bytes.clone();
    let kind = b[table_start..table_start + 2].to_vec();
    b[table_start + 32..table_start + 34].copy_from_slice(&kind);
    assert_eq!(
        Snapshot::parse(&b).unwrap_err(),
        Error::Corrupt("duplicate section kind")
    );

    // Nonzero padding right after a section's payload.
    let snap = Snapshot::parse(&bytes).unwrap();
    let payload = snap.section(1).unwrap();
    let pad_at = payload.as_ptr() as usize - bytes.as_ptr() as usize + payload.len();
    drop(snap);
    let mut b = bytes.clone();
    b[pad_at] = 1;
    assert_eq!(
        Snapshot::parse(&b).unwrap_err(),
        Error::Corrupt("nonzero padding after a section")
    );

    // A wrong (nonzero) file hash with all section checksums intact: the
    // structural parse accepts it (checksums are not verified at parse), and
    // the scrub reports it (specs/16 §9).
    let mut b = bytes.clone();
    b[20..28].copy_from_slice(&1u64.to_le_bytes());
    Snapshot::parse(&b).unwrap();
    assert_eq!(
        scrub(&b).unwrap_err(),
        Error::Corrupt("file checksum mismatch")
    );
}

#[test]
fn version_and_magic_gates() {
    let mut bytes = sample();
    bytes[4] = 2;
    // A future format version is UnsupportedVersion, not Corrupt — the
    // caller can suggest a migration.
    assert_eq!(
        Snapshot::parse(&bytes).unwrap_err(),
        Error::UnsupportedVersion(2)
    );
    let mut bytes = sample();
    bytes[0] = b'X';
    assert_eq!(
        Snapshot::parse(&bytes).unwrap_err(),
        Error::Corrupt("bad magic")
    );
}

#[test]
fn scrub_catches_content_flips_that_parse_accepts() {
    let bytes = sample();
    // A byte inside section 2's payload (0xC7 filler): parse accepts it (no
    // checksum at parse), the scrub reports the section mismatch (specs/16 §9).
    let at = bytes.iter().position(|&b| b == 0xC7).unwrap();
    let mut b = bytes.clone();
    b[at] ^= 0xFF;
    Snapshot::parse(&b).unwrap();
    assert_eq!(
        scrub(&b).unwrap_err(),
        Error::Corrupt("section checksum mismatch")
    );

    // Structural damage is still rejected at parse, before any scrub.
    let mut b = bytes.clone();
    b[10] = 1; // reserved header byte
    assert!(Snapshot::parse(&b).is_err());
    assert!(Snapshot::parse(&bytes[..64]).is_err());
}

#[test]
fn scrub_accepts_a_clean_image_and_a_zero_file_hash() {
    // A clean image scrubs Ok, ending at total == file length.
    let bytes = sample();
    let last = Snapshot::parse(&bytes)
        .unwrap()
        .scrub()
        .last()
        .unwrap()
        .unwrap();
    assert_eq!(last.done_bytes, last.total_bytes);
    assert_eq!(last.total_bytes, bytes.len() as u64);
    assert!(scrub(&bytes).is_ok());

    // The spec allows omitting the file hash (0 = absent); section checksums
    // still verify, so the scrub passes.
    let mut bytes = sample();
    bytes[20..28].copy_from_slice(&0u64.to_le_bytes());
    assert!(scrub(&bytes).is_ok());
}

#[test]
fn long_engine_version_is_truncated() {
    let w = SnapshotWriter::new();
    let bytes = w.finish(b"", 0, 0, "0.1.0-very-long-prerelease-tag");
    let snap = Snapshot::parse(&bytes).unwrap();
    assert_eq!(snap.engine_ver(), "0.1.0-very-long-prerelea");
    assert_eq!(snap.engine_ver().len(), 24);
}

#[test]
fn config_codec_roundtrip() {
    let mut cfg = Config::default();
    cfg.dim = 384;
    cfg.max_text = 2048;
    cfg.bm25_b = 0.5;
    cfg.flat_to_hnsw = 30_000;
    cfg.db_uuid = 0x0123_4567_89AB_CDEF_0011_2233_4455_6677;
    let mut bytes = Vec::new();
    cfg.encode(&mut bytes);
    assert_eq!(bytes.len(), ENCODED_LEN);
    assert_eq!(Config::decode(&bytes).unwrap(), cfg);
}

#[test]
fn config_codec_rejects_bad_input() {
    let mut bytes = Vec::new();
    Config::default().encode(&mut bytes);
    assert_eq!(
        Config::decode(&bytes[..ENCODED_LEN - 1]).unwrap_err(),
        Error::Corrupt("config block length mismatch")
    );
    let mut b = bytes.clone();
    b[RESERVED_AT] = 1;
    assert_eq!(
        Config::decode(&b).unwrap_err(),
        Error::Corrupt("reserved config bytes must be zero")
    );
    // A decoded config passes through the same range validation as a
    // hand-built one: dim = 5000 is rejected.
    let mut b = bytes.clone();
    b[0..8].copy_from_slice(&5000u64.to_le_bytes());
    assert_eq!(
        Config::decode(&b).unwrap_err(),
        Error::ConfigMismatch("dim must be <= 4096")
    );
    // Every byte flip in the block is an error or decodes to a config
    // that still passes validation — never a panic.
    for at in 0..bytes.len() {
        let mut b = bytes.clone();
        b[at] ^= 0x80;
        if let Ok(cfg) = Config::decode(&b) {
            cfg.validate().unwrap();
        }
    }
}

#[test]
fn scrub_slices_by_budget_and_is_resumable() {
    let bytes = sample();
    let total = bytes.len() as u64;

    // A tiny budget forces many slices; progress is monotonic and ends exactly
    // at the file length.
    let snap = Snapshot::parse(&bytes).unwrap();
    let mut steps = 0;
    let mut prev = 0u64;
    let mut last = None;
    for step in snap.scrub_with_budget(8) {
        let p = step.unwrap();
        assert!(p.done_bytes >= prev, "progress went backwards");
        assert!(p.done_bytes <= total);
        prev = p.done_bytes;
        last = Some(p);
        steps += 1;
    }
    assert!(
        steps > 1,
        "a tiny budget should take many slices, got {steps}"
    );
    assert_eq!(last.unwrap().done_bytes, total);

    // `.by_ref()` runs part of the scan, then the rest resumes from where it
    // paused — the cursor carries its state.
    let snap = Snapshot::parse(&bytes).unwrap();
    let mut cur = snap.scrub_with_budget(8);
    let first: Vec<_> = cur.by_ref().take(2).collect::<Result<_, _>>().unwrap();
    assert_eq!(first.len(), 2);
    let resumed = cur.last().unwrap().unwrap();
    assert_eq!(resumed.done_bytes, total, "the tail resumed to completion");
}

#[test]
fn scrub_is_fused_after_an_error() {
    // Flip a byte inside a section body so a section checksum mismatches.
    let bytes = sample();
    let at = bytes.iter().position(|&b| b == 0xC7).unwrap();
    let mut b = bytes.clone();
    b[at] ^= 0xFF;

    let snap = Snapshot::parse(&b).unwrap();
    let mut cur = snap.scrub_with_budget(8);
    // Some Ok slices may precede the mismatch; eventually one Err, then None.
    let mut saw_err = false;
    for step in cur.by_ref() {
        if step.is_err() {
            saw_err = true;
            break;
        }
    }
    assert!(saw_err, "the section flip must be reported");
    assert!(cur.next().is_none(), "the cursor is fused after an error");
}
