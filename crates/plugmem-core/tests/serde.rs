//! `serde` feature: the public data types serialize (and, where they are
//! owned, round-trip) through JSON. Active only under `--features serde`.
//!
//! Two tiers, matching the derive policy:
//! - **owned** types get Serialize + Deserialize and must round-trip;
//! - **borrowed / `'static`** types (inputs, errors) are Serialize-only —
//!   they hold `&str`/`&[&str]` that cannot be deserialized owned.

#![cfg(feature = "serde")]

use plugmem_core::{
    BlobId, Config, EntityId, Error, FactId, FactRecord, RecallQuery, RecallResult, RememberInput,
    VALID_TO_OPEN,
};

/// Owned types round-trip byte-for-byte (Serialize + Deserialize).
#[test]
fn owned_types_round_trip() {
    // Dense ids are newtypes → bare numbers.
    assert_eq!(serde_json::to_string(&FactId(9)).unwrap(), "9");
    assert_eq!(serde_json::from_str::<EntityId>("3").unwrap(), EntityId(3));

    let cfg = Config::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<Config>(&json).unwrap(), cfg);

    // A raw record (arena-layer ids like BlobId inside) round-trips.
    let rec = FactRecord {
        id: FactId(123),
        entity: EntityId(42),
        flags: 0,
        kind: 0,
        text: BlobId(9),
        vector: 7,
        revises: FactId::NONE,
        recorded_at: 1_784_000_000_000,
        valid_from: 1_784_000_000_000,
        valid_to: VALID_TO_OPEN,
    };
    let json = serde_json::to_string(&rec).unwrap();
    assert_eq!(serde_json::from_str::<FactRecord>(&json).unwrap(), rec);

    // A composite owned result (Vec + String inside) round-trips too.
    let res = RecallResult::default();
    let json = serde_json::to_string(&res).unwrap();
    assert_eq!(
        serde_json::from_str::<RecallResult>(&json)
            .unwrap()
            .facts
            .len(),
        0
    );
}

/// Borrowed and `'static`-bearing types serialize (no Deserialize by design).
#[test]
fn borrowed_and_error_types_serialize() {
    // An input carrying &str/&[&str] serializes to a JSON object.
    let input = RememberInput {
        entity: Some("user"),
        tags: &["pref", "rust"],
        ..RememberInput::text(1_784_000_000_000, "prefers tokio")
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(
        json.contains("prefers tokio") && json.contains("pref"),
        "{json}"
    );

    let q = RecallQuery::text(1_784_000_100_000, "which runtime");
    assert!(serde_json::to_string(&q).unwrap().contains("which runtime"));

    // Error is Serialize-only (it holds &'static str messages).
    let err = Error::ConfigMismatch("dim must be <= 4096");
    assert!(
        serde_json::to_string(&err).unwrap().contains("dim must be"),
        "error serializes its message"
    );
}
