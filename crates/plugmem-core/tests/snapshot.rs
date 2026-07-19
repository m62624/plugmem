//! Snapshot container tests (specs/03 test plan): canonical roundtrip,
//! build determinism, the full bitflip matrix (every byte, two bits —
//! always a typed error, never a panic, never a silent wrong read), the
//! truncation sweep, fast_load semantics, and the Config codec.

use plugmem_core::config::{ENCODED_LEN, FAST_LOAD_AT, RESERVED_AT};
use plugmem_core::snapshot::{FLAG_VECTORS, Snapshot, SnapshotWriter};
use plugmem_core::{Config, Error};

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
    let snap = Snapshot::parse(&bytes, false).unwrap();
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
    assert_eq!(Snapshot::parse(&bytes, false).unwrap().flags, FLAG_VECTORS);
    let w = SnapshotWriter::new();
    let bytes = w.finish(b"", 0b100, 7, "x");
    assert_eq!(
        Snapshot::parse(&bytes, false).unwrap_err(),
        Error::Corrupt("unknown flag bits set")
    );
}

#[test]
fn every_bitflip_is_a_typed_error() {
    let bytes = sample();
    let baseline = Snapshot::parse(&bytes, false).unwrap();
    let sections: Vec<_> = [1u16, 2, 9]
        .iter()
        .map(|&k| baseline.section(k).unwrap().to_vec())
        .collect();
    drop(baseline);
    for at in 0..bytes.len() {
        for bit in [0x01u8, 0x80] {
            let mut b = bytes.clone();
            b[at] ^= bit;
            match Snapshot::parse(&b, false) {
                Err(_) => {}
                Ok(snap) => {
                    // No flip may pass unnoticed: if parsing still
                    // succeeds, the observable content must be intact
                    // (this would only be legal for a bit the format
                    // ignores — there is none, so fail loudly).
                    let same = [1u16, 2, 9]
                        .iter()
                        .zip(&sections)
                        .all(|(&k, want)| snap.section(k) == Some(&want[..]));
                    panic!("flip at {at}/bit {bit:02x} accepted (content intact: {same})");
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
            Snapshot::parse(&bytes[..cut], false).is_err(),
            "prefix of {cut} bytes accepted"
        );
    }
}

#[test]
fn structural_gates_reachable_only_by_crafting() {
    // Trailing 64-byte block of zeros: aligned, structurally silent —
    // caught by the trailing-bytes rule (under fast_load, to show it is a
    // structural check, not a checksum one).
    let mut bytes = sample();
    bytes.extend_from_slice(&[0u8; 64]);
    assert_eq!(
        Snapshot::parse(&bytes, true).unwrap_err(),
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
        Snapshot::parse(&b, true).unwrap_err(),
        Error::Corrupt("duplicate section kind")
    );

    // Nonzero padding right after a section's payload.
    let snap = Snapshot::parse(&bytes, true).unwrap();
    let payload = snap.section(1).unwrap();
    let pad_at = payload.as_ptr() as usize - bytes.as_ptr() as usize + payload.len();
    drop(snap);
    let mut b = bytes.clone();
    b[pad_at] = 1;
    assert_eq!(
        Snapshot::parse(&b, true).unwrap_err(),
        Error::Corrupt("nonzero padding after a section")
    );

    // A wrong (nonzero) file hash with all section checksums intact.
    let mut b = bytes.clone();
    b[20..28].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(
        Snapshot::parse(&b, false).unwrap_err(),
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
        Snapshot::parse(&bytes, false).unwrap_err(),
        Error::UnsupportedVersion(2)
    );
    let mut bytes = sample();
    bytes[0] = b'X';
    assert_eq!(
        Snapshot::parse(&bytes, false).unwrap_err(),
        Error::Corrupt("bad magic")
    );
}

#[test]
fn fast_load_skips_only_checksums() {
    let bytes = sample();
    // Find a byte inside section 2's payload (0xC7 filler).
    let at = bytes.iter().position(|&b| b == 0xC7).unwrap();
    let mut b = bytes.clone();
    b[at] ^= 0xFF;
    // Checksummed parse catches it; fast_load accepts the flip (that is
    // the documented trade for trusted local files).
    assert_eq!(
        Snapshot::parse(&b, false).unwrap_err(),
        Error::Corrupt("section checksum mismatch")
    );
    assert!(Snapshot::parse(&b, true).is_ok());
    // Structural rules still apply under fast_load.
    let mut b = bytes.clone();
    b[10] = 1; // reserved header byte
    assert!(Snapshot::parse(&b, true).is_err());
    assert!(Snapshot::parse(&bytes[..64], true).is_err());
}

#[test]
fn zero_file_hash_means_no_file_hash() {
    // The spec allows omitting the file hash (0 = absent); section
    // checksums still verify.
    let mut bytes = sample();
    bytes[20..28].copy_from_slice(&0u64.to_le_bytes());
    assert!(Snapshot::parse(&bytes, false).is_ok());
}

#[test]
fn long_engine_version_is_truncated() {
    let w = SnapshotWriter::new();
    let bytes = w.finish(b"", 0, 0, "0.1.0-very-long-prerelease-tag");
    let snap = Snapshot::parse(&bytes, false).unwrap();
    assert_eq!(snap.engine_ver(), "0.1.0-very-long-prerelea");
    assert_eq!(snap.engine_ver().len(), 24);
}

#[test]
fn config_codec_roundtrip() {
    let mut cfg = Config::default();
    cfg.dim = 384;
    cfg.max_text = 2048;
    cfg.bm25_b = 0.5;
    cfg.fast_load = true;
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
    b[FAST_LOAD_AT] = 2;
    assert_eq!(
        Config::decode(&b).unwrap_err(),
        Error::Corrupt("config fast_load byte must be 0 or 1")
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
