//! Journal framing tests (specs/03 test plan): roundtrip, crash-truncation
//! at every byte of the tail record, mid-stream corruption, and the
//! Storage contract over `MemStorage`.

use plugmem_core::journal::{JournalEntry, JournalScan, encode_entry, scan};
use plugmem_core::{Error, MemStorage, Storage};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::*;

/// Encodes a sequence of (op, payload) records into one buffer.
fn encode_all(records: &[(u8, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(op, payload) in records {
        encode_entry(&mut out, op, payload);
    }
    out
}

#[test]
fn empty_journal_scans_to_nothing() {
    let got = scan(&[]).unwrap();
    assert!(got.entries.is_empty());
    assert!(!got.truncated_tail);
}

#[test]
fn multi_record_roundtrip() {
    let records: [(u8, &[u8]); 4] = [
        (1, b"remember: user likes tokio"),
        (3, &42u32.to_le_bytes()),
        (5, b""), // maintain marker: op with empty payload
        (2, &[0xFF; 300]),
    ];
    let buf = encode_all(&records);
    let got = scan(&buf).unwrap();
    assert!(!got.truncated_tail);
    let want: Vec<JournalEntry> = records
        .iter()
        .map(|&(op, payload)| JournalEntry { op, payload })
        .collect();
    assert_eq!(got.entries, want);
}

#[test]
fn torn_tail_at_every_byte_recovers_the_prefix() {
    let records: [(u8, &[u8]); 3] = [(1, b"first"), (4, b"second-edge"), (1, b"the torn one")];
    let buf = encode_all(&records);
    let intact = encode_all(&records[..2]);
    // Cut the buffer at every length that leaves the first two records
    // whole and the third torn: the scan must succeed, return exactly
    // those two, and flag the truncation. (A cut at exactly intact.len()
    // is a clean journal with two records — checked after the loop.)
    for cut in intact.len() + 1..buf.len() {
        let got = scan(&buf[..cut]).unwrap_or_else(|e| panic!("cut at {cut}: {e}"));
        assert_eq!(got.entries.len(), 2, "cut at {cut}");
        assert!(got.truncated_tail, "cut at {cut}");
        assert_eq!(got.entries[1].payload, b"second-edge");
    }
    let clean = scan(&intact).unwrap();
    assert_eq!(clean.entries.len(), 2);
    assert!(!clean.truncated_tail);
    assert!(!scan(&buf).unwrap().truncated_tail);
}

#[test]
fn torn_first_record_yields_empty_scan() {
    let buf = encode_all(&[(1, b"only")]);
    for cut in 0..buf.len() {
        let got = scan(&buf[..cut]).unwrap();
        assert!(got.entries.is_empty(), "cut at {cut}");
        assert_eq!(got.truncated_tail, cut > 0, "cut at {cut}");
    }
}

#[test]
fn mid_stream_bitflip_is_corrupt_tail_bitflip_is_torn() {
    let records: [(u8, &[u8]); 3] = [(1, b"aaaa"), (2, b"bbbb"), (3, b"cccc")];
    let buf = encode_all(&records);
    let one = encode_all(&records[..1]).len();
    let two = encode_all(&records[..2]).len();

    // Flip a payload byte of the middle record: checksum mismatch that is
    // NOT at the buffer tail -> hard corruption.
    let mut bad = buf.clone();
    bad[two - 1] ^= 0x01;
    assert_eq!(
        scan(&bad),
        Err(Error::Corrupt("journal checksum mismatch mid-stream"))
    );

    // Flip a payload byte of the LAST record: indistinguishable from a
    // torn write inside it -> recovered prefix + truncation flag.
    let mut torn = buf.clone();
    let last = torn.len() - 1;
    torn[last] ^= 0x01;
    let got = scan(&torn).unwrap();
    assert_eq!(got.entries.len(), 2);
    assert!(got.truncated_tail);

    // A zero length field can never be a torn prefix (every valid record
    // has len >= 1) -> hard corruption, wherever it sits.
    let mut zero = buf;
    zero[one..one + 4].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        scan(&zero),
        Err(Error::Corrupt("journal record with zero length"))
    );
}

#[test]
fn oversized_len_field_reads_as_torn_tail() {
    // A corrupted len that points past the end cannot be told apart from
    // an interrupted append of a genuinely long record; the scan keeps
    // the valid prefix (crash semantics, documented in the module).
    let mut buf = encode_all(&[(1, b"kept")]);
    let keep = buf.len();
    encode_entry(&mut buf, 2, b"soon-huge");
    buf[keep..keep + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let got = scan(&buf).unwrap();
    assert_eq!(got.entries.len(), 1);
    assert_eq!(got.entries[0].payload, b"kept");
    assert!(got.truncated_tail);
}

/// The Storage contract scenarios (specs/03), generic over the
/// implementation — `plugmem-host` will run the same body for
/// `FileStorage` when it lands (the suite moves to a shared home then).
fn storage_contract<S: Storage>(store: &mut S) {
    // First run: no snapshot, empty journal.
    assert_eq!(store.read_snapshot().unwrap(), None);
    assert_eq!(store.read_journal().unwrap(), Vec::<u8>::new());

    // Journal appends accumulate in order; framed entries survive a
    // read-back scan.
    let mut e1 = Vec::new();
    encode_entry(&mut e1, 1, b"alpha");
    let mut e2 = Vec::new();
    encode_entry(&mut e2, 3, b"beta");
    store.append_journal(&e1).unwrap();
    store.append_journal(&e2).unwrap();
    let journal = store.read_journal().unwrap();
    let JournalScan {
        entries,
        truncated_tail,
    } = scan(&journal).unwrap();
    assert!(!truncated_tail);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].payload, b"alpha");
    assert_eq!(entries[1].op, 3);

    // Snapshot write + journal clear = the post-snapshot state.
    store.write_snapshot(b"image-v1").unwrap();
    store.clear_journal().unwrap();
    assert_eq!(
        store.read_snapshot().unwrap().as_deref(),
        Some(&b"image-v1"[..])
    );
    assert_eq!(store.read_journal().unwrap(), Vec::<u8>::new());

    // A snapshot replace is total, not additive.
    store.write_snapshot(b"image-v2").unwrap();
    assert_eq!(
        store.read_snapshot().unwrap().as_deref(),
        Some(&b"image-v2"[..])
    );
}

#[test]
fn mem_storage_satisfies_the_contract() {
    storage_contract(&mut MemStorage::new());
}

#[test]
fn mem_storage_default_and_clone() {
    let mut a = MemStorage::default();
    assert_eq!(a, MemStorage::new());
    a.write_snapshot(b"x").unwrap();
    let b = a.clone();
    assert_eq!(a, b);
}

#[cfg(not(target_family = "wasm"))]
proptest! {
    // Any encoded sequence scans back exactly; any cut of the buffer
    // yields a prefix of the records and never an error.
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn roundtrip_and_prefix_stability(
        records in proptest::collection::vec(
            (any::<u8>(), proptest::collection::vec(any::<u8>(), 0..200)),
            0..12,
        ),
        cut_seed in any::<usize>(),
    ) {
        let refs: Vec<(u8, &[u8])> =
            records.iter().map(|(op, p)| (*op, p.as_slice())).collect();
        let buf = encode_all(&refs);
        let full = scan(&buf).unwrap();
        prop_assert!(!full.truncated_tail);
        prop_assert_eq!(full.entries.len(), records.len());
        for (got, (op, payload)) in full.entries.iter().zip(&records) {
            prop_assert_eq!(got.op, *op);
            prop_assert_eq!(got.payload, payload.as_slice());
        }

        // A random cut is always a clean prefix (possibly flagged torn).
        if !buf.is_empty() {
            let cut = cut_seed % buf.len();
            let partial = scan(&buf[..cut]).unwrap();
            prop_assert!(partial.entries.len() <= records.len());
            for (got, (op, payload)) in partial.entries.iter().zip(&records) {
                prop_assert_eq!(got.op, *op);
                prop_assert_eq!(got.payload, payload.as_slice());
            }
        }
    }
}
