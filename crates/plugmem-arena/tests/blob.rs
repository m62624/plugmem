//! Boundary tests for `BlobHeap` (specs/01 test plan) plus a property model
//! against `Vec<Vec<u8>>`.

use plugmem_arena::{BlobHeap, BlobHeapCfg, BlobId, Error};
use proptest::prelude::*;

#[test]
fn empty_heap() {
    let heap = BlobHeap::new(BlobHeapCfg::new());
    assert_eq!(heap.len(), 0);
    assert!(heap.is_empty());
    assert_eq!(heap.pool_bytes(), 0);
    assert_eq!(heap.iter().count(), 0);
}

#[test]
fn push_get_roundtrip_ids_are_dense() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new());
    let a = heap.push(b"alpha").unwrap();
    let b = heap.push(b"be").unwrap();
    let c = heap.push(b"gamma-gamma").unwrap();
    assert_eq!((a, b, c), (BlobId(0), BlobId(1), BlobId(2)));
    assert_eq!(heap.get(a), b"alpha");
    assert_eq!(heap.get(b), b"be");
    assert_eq!(heap.get(c), b"gamma-gamma");
    assert_eq!(heap.len(), 3);
    assert!(!heap.is_empty());
    assert_eq!(heap.pool_bytes(), 5 + 2 + 11);
}

#[test]
fn zero_length_blob_is_valid() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new());
    let empty = heap.push(b"").unwrap();
    let after = heap.push(b"x").unwrap();
    assert_eq!(heap.get(empty), b"");
    assert_eq!(heap.get(after), b"x");
    assert_eq!(heap.len(), 2);
    assert_eq!(heap.pool_bytes(), 1);
}

#[test]
fn max_blob_boundary() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new().with_max_blob(4));
    assert!(heap.push(&[7u8; 4]).is_ok());
    assert_eq!(
        heap.push(&[7u8; 5]),
        Err(Error::BlobTooLarge {
            len: 5,
            max_blob: 4
        })
    );
    // The failed push left the heap unchanged.
    assert_eq!(heap.len(), 1);
    assert_eq!(heap.pool_bytes(), 4);
}

#[test]
fn max_bytes_boundary() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new().with_max_bytes(10));
    assert!(heap.push(&[1u8; 6]).is_ok());
    assert!(heap.push(&[2u8; 4]).is_ok()); // exactly at the ceiling
    assert_eq!(
        heap.push(&[3u8; 1]),
        Err(Error::CapacityExceeded { max_bytes: 10 })
    );
    assert_eq!(heap.len(), 2);
    // Zero-length blobs still fit: they do not grow the pool.
    assert!(heap.push(b"").is_ok());
}

#[test]
#[should_panic]
fn get_with_dangling_id_panics() {
    let heap = BlobHeap::new(BlobHeapCfg::new());
    let _ = heap.get(BlobId(0));
}

#[test]
fn iter_yields_ids_in_order_with_contents() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new());
    let blobs: [&[u8]; 3] = [b"one", b"", b"three"];
    for blob in blobs {
        heap.push(blob).unwrap();
    }
    let got: Vec<(BlobId, &[u8])> = heap.iter().collect();
    assert_eq!(
        got,
        vec![
            (BlobId(0), b"one".as_slice()),
            (BlobId(1), b"".as_slice()),
            (BlobId(2), b"three".as_slice()),
        ]
    );
}

#[test]
fn clone_and_eq_compare_contents() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new());
    heap.push(b"data").unwrap();
    let copy = heap.clone();
    assert_eq!(heap, copy);
    let mut other = BlobHeap::new(BlobHeapCfg::new());
    other.push(b"ohter").unwrap();
    assert_ne!(heap, other);
}

#[test]
fn debug_is_a_summary() {
    let mut heap = BlobHeap::new(BlobHeapCfg::new());
    heap.push(b"secret-content").unwrap();
    let dump = format!("{heap:?}");
    assert!(dump.contains("blobs: 1"));
    assert!(dump.contains("pool_bytes: 14"));
    assert!(!dump.contains("secret"));
}

#[test]
fn cfg_default_and_builders() {
    assert_eq!(BlobHeapCfg::default(), BlobHeapCfg::new());
    let cfg = BlobHeapCfg::new().with_max_bytes(100).with_max_blob(10);
    assert_eq!(cfg.max_bytes, 100);
    assert_eq!(cfg.max_blob, 10);
}

proptest! {
    /// The heap must behave exactly like a `Vec<Vec<u8>>` under arbitrary
    /// push sequences: same ids, same contents, same iteration.
    #[test]
    // proptest's harness calls into the OS (cwd for failure
    // persistence), which miri's isolation forbids; UB-paths are covered
    // by the boundary tests.
    #[cfg_attr(miri, ignore)]
    fn behaves_like_vec_of_vecs(blobs in proptest::collection::vec(
        proptest::collection::vec(any::<u8>(), 0..200),
        1..64,
    )) {
        let mut heap = BlobHeap::new(BlobHeapCfg::new());
        let mut model: Vec<Vec<u8>> = Vec::new();
        for blob in &blobs {
            let id = heap.push(blob).unwrap();
            prop_assert_eq!(id.0 as usize, model.len());
            model.push(blob.clone());
        }
        prop_assert_eq!(heap.len(), model.len());
        for (i, blob) in model.iter().enumerate() {
            prop_assert_eq!(heap.get(BlobId(i as u32)), blob.as_slice());
        }
        let iterated: Vec<Vec<u8>> = heap.iter().map(|(_, b)| b.to_vec()).collect();
        prop_assert_eq!(iterated, model);
    }
}
