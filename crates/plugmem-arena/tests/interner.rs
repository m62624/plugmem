//! Boundary tests for `Interner` (test plan) plus a property model
//! against `HashMap<String, u32>`.

use std::collections::HashMap;

use plugmem_arena::{BlobHeapCfg, Error, Interner, TermId};
use proptest::prelude::*;

#[test]
fn lookup_never_creates() {
    let mut it = Interner::new(BlobHeapCfg::new());
    assert_eq!(it.lookup("ghost"), None);
    assert_eq!(it.len(), 0, "lookup must not intern");
    let id = it.intern("real").unwrap();
    assert_eq!(it.lookup("real"), Some(id));
    assert_eq!(it.lookup("ghost"), None);
    assert_eq!(it.len(), 1);
}

#[test]
fn empty_interner() {
    let terms = Interner::new(BlobHeapCfg::new());
    assert_eq!(terms.len(), 0);
    assert!(terms.is_empty());
    assert_eq!(terms.pool_bytes(), 0);
}

#[test]
fn pool_bytes_tracks_content() {
    let mut terms = Interner::new(BlobHeapCfg::new());
    terms.intern("alpha").unwrap();
    terms.intern("beta").unwrap();
    terms.intern("alpha").unwrap(); // duplicate: no new bytes
    assert_eq!(terms.pool_bytes(), "alpha".len() + "beta".len());
}

#[test]
fn intern_is_idempotent_and_ids_are_dense() {
    let mut terms = Interner::new(BlobHeapCfg::new());
    let a = terms.intern("alpha").unwrap();
    let b = terms.intern("beta").unwrap();
    assert_eq!((a, b), (TermId(0), TermId(1)));
    assert_eq!(terms.intern("alpha").unwrap(), a);
    assert_eq!(terms.intern("beta").unwrap(), b);
    assert_eq!(terms.len(), 2);
    assert!(!terms.is_empty());
    assert_eq!(terms.resolve(a), "alpha");
    assert_eq!(terms.resolve(b), "beta");
}

#[test]
fn empty_and_unicode_strings() {
    let mut terms = Interner::new(BlobHeapCfg::new());
    let empty = terms.intern("").unwrap();
    let snow = terms.intern("сне\u{0301}г ❄").unwrap();
    assert_ne!(empty, snow);
    assert_eq!(terms.resolve(empty), "");
    assert_eq!(terms.resolve(snow), "сне\u{0301}г ❄");
    assert_eq!(terms.intern("").unwrap(), empty);
}

#[test]
fn survives_rehash_growth() {
    // Initial table is 16 slots with load factor 0.7 -> several rehashes on
    // the way to 1000 terms; every earlier id must stay resolvable and
    // stable.
    let mut terms = Interner::new(BlobHeapCfg::new());
    let ids: Vec<TermId> = (0..1000)
        .map(|i| terms.intern(&format!("term-{i}")).unwrap())
        .collect();
    assert_eq!(terms.len(), 1000);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(terms.resolve(*id), format!("term-{i}"));
        assert_eq!(terms.intern(&format!("term-{i}")).unwrap(), *id);
    }
}

#[test]
fn heap_limits_propagate() {
    let mut terms = Interner::new(BlobHeapCfg::new().with_max_blob(4).with_max_bytes(6));
    assert_eq!(
        terms.intern("looong"),
        Err(Error::BlobTooLarge {
            len: 6,
            max_blob: 4
        })
    );
    assert!(terms.intern("abcd").is_ok());
    assert!(terms.intern("ab").is_ok()); // exactly at max_bytes
    assert_eq!(
        terms.intern("x"),
        Err(Error::CapacityExceeded { max_bytes: 6 })
    );
    // A failed intern leaves the interner unchanged: the existing terms
    // still resolve, and the failed one was not half-registered.
    assert_eq!(terms.len(), 2);
    assert_eq!(terms.intern("abcd").unwrap(), TermId(0));
}

#[test]
#[should_panic]
fn resolve_with_dangling_id_panics() {
    let terms = Interner::new(BlobHeapCfg::new());
    let _ = terms.resolve(TermId(7));
}

#[test]
fn debug_is_a_summary() {
    let mut terms = Interner::new(BlobHeapCfg::new());
    terms.intern("secret-word").unwrap();
    let dump = format!("{terms:?}");
    assert!(dump.contains("terms: 1"));
    assert!(dump.contains("table_slots: 16"));
    assert!(!dump.contains("secret"));
}

#[cfg(feature = "counters")]
#[test]
fn probe_counter_moves_and_resets() {
    let mut terms = Interner::new(BlobHeapCfg::new());
    terms.intern("a").unwrap();
    assert!(terms.probes() >= 1);
    terms.reset_probes();
    assert_eq!(terms.probes(), 0);
    // A hit probes at least the slot it lands on.
    terms.intern("a").unwrap();
    assert!(terms.probes() >= 1);
}

proptest! {
    /// The interner must be a bijection `string <-> id` equivalent to a
    /// `HashMap<String, u32>` with dense first-seen numbering. A small
    /// alphabet forces plenty of repeats (hits) and hash collisions.
    #[test]
    // proptest's harness calls into the OS (cwd for failure
    // persistence), which miri's isolation forbids; UB-paths are covered
    // by the boundary tests.
    #[cfg_attr(miri, ignore)]
    fn behaves_like_hashmap(words in proptest::collection::vec("[ab]{0,6}", 1..300)) {
        let mut terms = Interner::new(BlobHeapCfg::new());
        let mut model: HashMap<String, u32> = HashMap::new();
        for word in &words {
            let id = terms.intern(word).unwrap();
            let want = *model
                .entry(word.clone())
                .or_insert_with(|| (terms.len() - 1) as u32);
            prop_assert_eq!(id.0, want);
            prop_assert_eq!(terms.resolve(id), word.as_str());
        }
        prop_assert_eq!(terms.len(), model.len());
    }
}
