//! Id-sentinel plumbing and the Display contract of every error variant.

use plugmem_core::{EntityId, Error, FactId, NONE_U32};

#[test]
fn id_sentinel_plumbing() {
    assert_eq!(NONE_U32, u32::MAX);
    assert!(FactId::NONE.is_none());
    assert!(!FactId(0).is_none(), "0 is a valid id");
    assert_eq!(FactId(7).some(), Some(FactId(7)));
    assert_eq!(FactId::NONE.some(), None);
    assert_eq!(FactId::from_opt(Some(FactId(7))), FactId(7));
    assert_eq!(FactId::from_opt(None), FactId::NONE);

    assert!(EntityId::NONE.is_none());
    assert_eq!(EntityId(3).some(), Some(EntityId(3)));
    assert_eq!(EntityId::from_opt(None), EntityId::NONE);

    // Ids order and hash as their raw values (usable as map keys and in
    // BE-encoded arena keys alike).
    assert!(FactId(1) < FactId(2));
}

#[test]
fn error_display_contract() {
    let cases: &[(Error, &str)] = &[
        (
            Error::CapacityExceeded { what: "facts" },
            "capacity exceeded: facts",
        ),
        (
            Error::TooLarge {
                what: "text",
                len: 5000,
                max: 4096,
            },
            "text too large: 5000 bytes (max 4096)",
        ),
        (
            Error::DimMismatch {
                got: 512,
                want: 384,
            },
            "vector dimension mismatch: got 512, want 384",
        ),
        (Error::NotFound(FactId(42)), "fact 42 not found"),
        (Error::AlreadyClosed(FactId(7)), "fact 7 is already closed"),
        (
            Error::ConfigMismatch("dim must be <= 4096"),
            "config mismatch: dim must be <= 4096",
        ),
        (Error::Corrupt("bad header"), "corrupt input: bad header"),
        (
            Error::UnsupportedVersion(9),
            "unsupported snapshot format version 9",
        ),
    ];
    for (err, want) in cases {
        assert_eq!(&err.to_string(), want);
    }
}

#[test]
fn arena_errors_bubble_up_with_context() {
    let arena_err = plugmem_arena::Error::BadShardCount { got: 3 };
    let err: Error = arena_err.into();
    assert_eq!(err, Error::Arena(arena_err));
    assert!(err.to_string().starts_with("arena: "));
}
