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

/// Dumps a heap's two sections into owned buffers (the on-disk form a
/// borrowed/overlay open reads back).
fn dump(heap: &BlobHeap) -> (Vec<u8>, Vec<u8>) {
    let (mut index, mut pool) = (Vec::new(), Vec::new());
    heap.dump_index(&mut index);
    heap.dump_pool(&mut pool);
    (index, pool)
}

#[test]
fn overlay_open_appends_to_tail_without_touching_base() {
    // Base: two blobs, serialized as if from a snapshot / mmap.
    let mut owned = BlobHeap::new(BlobHeapCfg::new());
    owned.push(b"alpha").unwrap();
    owned.push(b"beta").unwrap();
    let (index, pool) = dump(&owned);
    let base_snapshot = pool.clone();

    // Open over the borrowed base, then append: the new blob lands in the
    // owned tail, the base is never cloned or mutated.
    let mut heap = BlobHeap::load_borrowed(BlobHeapCfg::new(), &index, &pool).unwrap();
    assert_eq!(heap.len(), 2);
    let c = heap.push(b"gamma").unwrap();
    assert_eq!(c, BlobId(2));

    // Reads dispatch across the base/tail boundary correctly.
    assert_eq!(heap.get(BlobId(0)), b"alpha"); // base
    assert_eq!(heap.get(BlobId(1)), b"beta"); // base
    assert_eq!(heap.get(BlobId(2)), b"gamma"); // tail
    assert_eq!(heap.len(), 3);
    assert_eq!(heap.pool_bytes(), 5 + 4 + 5);

    // The borrowed base buffer is byte-for-byte unchanged by the append.
    assert_eq!(pool, base_snapshot);
}

#[test]
fn overlay_dump_is_byte_identical_to_owned_and_compares_equal() {
    // Fully-owned heap with three blobs.
    let mut owned = BlobHeap::new(BlobHeapCfg::new());
    for b in [b"a".as_slice(), b"bb", b"ccc"] {
        owned.push(b).unwrap();
    }
    // Same blobs, but the first is a borrowed base and the rest are appended
    // through the overlay alias — a different base/tail split.
    let mut seed = BlobHeap::new(BlobHeapCfg::new());
    seed.push(b"a").unwrap();
    let (index, pool) = dump(&seed);
    let mut overlay = BlobHeap::load_overlay(BlobHeapCfg::new(), &index, &pool).unwrap();
    overlay.push(b"bb").unwrap();
    overlay.push(b"ccc").unwrap();

    // Canonical dumps: byte-identical regardless of the split.
    assert_eq!(dump(&owned), dump(&overlay));
    // And logical equality holds across the two representations.
    assert_eq!(owned, overlay);
}

#[test]
fn overlay_iter_spans_base_and_tail() {
    let mut seed = BlobHeap::new(BlobHeapCfg::new());
    seed.push(b"x").unwrap();
    let (index, pool) = dump(&seed);
    let mut heap = BlobHeap::load_overlay(BlobHeapCfg::new(), &index, &pool).unwrap();
    heap.push(b"yy").unwrap();
    let got: Vec<Vec<u8>> = heap.iter().map(|(_, b)| b.to_vec()).collect();
    assert_eq!(got, vec![b"x".to_vec(), b"yy".to_vec()]);
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
