//! Journal op-codec tests: roundtrip of every op shape, a truncation and
//! bitflip sweep over decode (typed errors, never a panic), replay
//! rejection of semantically corrupt journals, and the storage-failure
//! path.

use plugmem_core::journal::{Op, scan};
use plugmem_core::{Config, Error, FactId, MemStorage, Memory, RememberInput, Storage};

fn cfg() -> Config {
    Config::default()
}

fn sample_ops() -> Vec<Op<'static>> {
    vec![
        Op::Remember {
            now: 100,
            valid_from: 100,
            entity: Some("user"),
            text: "prefers tokio",
            tags: vec!["pref", "rust"],
            links: vec![("works_on", "plugmem")],
            vector: vec![],
            metadata: vec![("mime", "text/plain"), ("uri", "s3://b/x")],
            revises: FactId::NONE,
            assigned: FactId(0),
        },
        Op::Remember {
            now: 200,
            valid_from: 150,
            entity: None,
            text: "",
            tags: vec![],
            links: vec![],
            vector: vec![],
            metadata: vec![],
            revises: FactId(0),
            assigned: FactId(1),
        },
        Op::Forget {
            now: 300,
            fact: FactId(1),
        },
        Op::Link {
            now: 400,
            src: "user",
            rel: "works_on",
            dst: "plugmem",
            provenance: FactId::NONE,
        },
        Op::Unlink {
            now: 450,
            src: "user",
            rel: "works_on",
            dst: "plugmem",
        },
        Op::Maintain {
            now: 500,
            mode: 0,
            max_hnsw_inserts: u32::MAX,
        },
    ]
}

#[test]
fn every_op_shape_roundtrips() {
    let ops = sample_ops();
    let mut buf = Vec::new();
    for op in &ops {
        op.encode(&mut buf);
    }
    let scanned = scan(&buf).unwrap();
    assert_eq!(scanned.entries.len(), ops.len());
    for (entry, want) in scanned.entries.iter().zip(&ops) {
        let got = Op::decode(entry.op, entry.payload).unwrap();
        assert_eq!(&got, want);
    }
}

#[test]
fn decode_survives_any_truncation_or_bitflip() {
    for op in sample_ops() {
        let mut buf = Vec::new();
        op.encode(&mut buf);
        let scanned = scan(&buf).unwrap();
        let (code, payload) = (scanned.entries[0].op, scanned.entries[0].payload);
        // Every prefix must be a typed error (or, for a prefix that is
        // itself a complete valid payload, a successful decode — the
        // trailing-bytes rule forbids that here by construction).
        for cut in 0..payload.len() {
            let got = Op::decode(code, &payload[..cut]);
            if code == 5 && cut == 8 {
                assert!(got.is_ok(), "old maintain marker payload remains valid");
            } else {
                assert!(got.is_err(), "op {code}: prefix of {cut} accepted");
            }
        }
        // Bitflips either fail typed or decode into some other valid op —
        // never panic. (Content integrity is the checksum layer's job.)
        for at in 0..payload.len() {
            let mut b = payload.to_vec();
            b[at] ^= 0x80;
            let _ = Op::decode(code, &b);
        }
        // Unknown op codes are corrupt; so are trailing bytes.
        assert_eq!(
            Op::decode(9, payload).unwrap_err(),
            Error::Corrupt("unknown journal op")
        );
        let mut extended = payload.to_vec();
        extended.push(0);
        assert_eq!(
            Op::decode(code, &extended).unwrap_err(),
            Error::Corrupt("journal record has trailing bytes")
        );
    }
}

#[test]
fn decode_rejects_op_revises_disagreement() {
    // An op-1 record whose revises field is set (and vice versa) is
    // structurally corrupt.
    let mut buf = Vec::new();
    Op::Remember {
        now: 1,
        valid_from: 1,
        entity: None,
        text: "x",
        tags: vec![],
        links: vec![],
        vector: vec![],
        metadata: vec![],
        revises: FactId(5),
        assigned: FactId(6),
    }
    .encode(&mut buf);
    let scanned = scan(&buf).unwrap();
    assert_eq!(scanned.entries[0].op, 2, "revises set encodes as op 2");
    assert_eq!(
        Op::decode(1, scanned.entries[0].payload).unwrap_err(),
        Error::Corrupt("journal revises field disagrees with op")
    );
}

/// Builds a journal from raw ops and opens an engine over it.
fn open_with(ops: &[Op<'_>]) -> Result<Memory<'static>, Error> {
    let mut journal = Vec::new();
    for op in ops {
        op.encode(&mut journal);
    }
    Memory::from_bytes(None, &journal, cfg()).map(|(m, _)| m)
}

#[test]
fn replay_rejects_semantically_corrupt_journals() {
    // Non-contiguous fact ids.
    let err = open_with(&[Op::Remember {
        now: 1,
        valid_from: 1,
        entity: None,
        text: "x",
        tags: vec![],
        links: vec![],
        vector: vec![],
        metadata: vec![],
        revises: FactId::NONE,
        assigned: FactId(7),
    }])
    .unwrap_err();
    assert_eq!(err, Error::Corrupt("journal fact ids are not contiguous"));

    // Revising a fact that does not exist.
    let err = open_with(&[Op::Remember {
        now: 1,
        valid_from: 1,
        entity: None,
        text: "x",
        tags: vec![],
        links: vec![],
        vector: vec![],
        metadata: vec![],
        revises: FactId(3),
        assigned: FactId(0),
    }])
    .unwrap_err();
    assert_eq!(err, Error::Corrupt("journal revises an unrevisable fact"));

    // Forgetting a fact that never existed.
    let err = open_with(&[Op::Forget {
        now: 1,
        fact: FactId(2),
    }])
    .unwrap_err();
    assert_eq!(err, Error::Corrupt("journal forgets an unknown fact"));

    // A maintain marker alone replays as a no-op.
    let mem = open_with(&[Op::Maintain {
        now: 1,
        mode: 0,
        max_hnsw_inserts: u32::MAX,
    }])
    .unwrap();
    assert_eq!(mem.facts_len(), 0);

    // A duplicated tail (double-applied journal segment) is skipped
    // idempotently, not re-executed.
    let rec = Op::Remember {
        now: 1,
        valid_from: 1,
        entity: None,
        text: "once",
        tags: vec![],
        links: vec![],
        vector: vec![],
        metadata: vec![],
        revises: FactId::NONE,
        assigned: FactId(0),
    };
    let mem = open_with(&[rec.clone(), rec]).unwrap();
    assert_eq!(mem.facts_len(), 1);
}

#[test]
fn garbage_snapshot_bytes_are_container_errors() {
    assert_eq!(
        Memory::from_bytes(Some(b"PLGM"), &[], cfg()).unwrap_err(),
        Error::Corrupt("snapshot shorter than its header")
    );
}

/// A storage whose journal always fails — exercises the Storage error
/// mapping on every mutating verb.
#[derive(Default)]
struct BrokenStorage;

impl Storage for BrokenStorage {
    type Error = &'static str;
    fn read_snapshot(&mut self) -> Result<Option<Vec<u8>>, &'static str> {
        Err("io down")
    }
    fn write_snapshot(&mut self, _: &[u8]) -> Result<(), &'static str> {
        Err("io down")
    }
    fn read_journal(&mut self) -> Result<Vec<u8>, &'static str> {
        Err("io down")
    }
    fn append_journal(&mut self, _: &[u8]) -> Result<(), &'static str> {
        Err("io down")
    }
    fn clear_journal(&mut self) -> Result<(), &'static str> {
        Err("io down")
    }
}

#[test]
fn storage_failures_surface_typed() {
    let mut broken = BrokenStorage;
    assert_eq!(
        Memory::open(&mut broken, cfg()).unwrap_err(),
        Error::Storage("\"io down\"".into())
    );
    let mut mem = Memory::new(cfg()).unwrap();
    let err = mem
        .remember(&mut broken, RememberInput::text(1, "x"))
        .unwrap_err();
    assert!(matches!(err, Error::Storage(_)));
    let err = mem.forget(&mut broken, 2, FactId(0)).unwrap_err();
    assert!(matches!(err, Error::Storage(_)));
    // Engine Debug prints a summary, not the contents.
    let dump = format!("{mem:?}");
    assert!(dump.contains("facts"));
    assert!(!dump.contains('x'));
}

#[test]
fn journal_and_direct_state_survive_mixed_reopen_cycles() {
    // Two reopen generations: state must be identical to continuous use.
    let mut store = MemStorage::new();
    let (mut mem, _) = Memory::open(&mut store, cfg()).unwrap();
    mem.remember(&mut store, RememberInput::text(1, "first"))
        .unwrap();
    let (mut mem, _) = Memory::open(&mut store, cfg()).unwrap();
    mem.remember(&mut store, RememberInput::text(2, "second"))
        .unwrap();
    let (mem, report) = Memory::open(&mut store, cfg()).unwrap();
    assert_eq!(report.replayed, 2);
    assert_eq!(mem.facts_len(), 2);
    assert_eq!(mem.get(FactId(0)).unwrap().text, "first");
    assert_eq!(mem.get(FactId(1)).unwrap().text, "second");
}
