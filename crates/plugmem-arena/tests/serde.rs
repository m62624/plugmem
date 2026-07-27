//! `serde` feature: the public data types (ids and configs) round-trip through
//! JSON. Active only under `--features serde`; without it this file is empty.

#![cfg(feature = "serde")]

use plugmem_arena::{ArenaCfg, BlobHeapCfg, BlobId, ChunkPoolCfg, ShardMode, TermId};

/// A dense id is a newtype, so it serializes as a bare number (not an object)
/// and round-trips.
#[test]
fn ids_serialize_as_bare_numbers() {
    assert_eq!(serde_json::to_string(&BlobId(42)).unwrap(), "42");
    assert_eq!(serde_json::to_string(&TermId(7)).unwrap(), "7");
    assert_eq!(serde_json::from_str::<BlobId>("42").unwrap(), BlobId(42));
    assert_eq!(serde_json::from_str::<TermId>("7").unwrap(), TermId(7));
}

/// The config structs round-trip byte-for-byte through JSON.
#[test]
fn configs_round_trip() {
    let blob = BlobHeapCfg::new().with_max_bytes(1_000).with_max_blob(64);
    let json = serde_json::to_string(&blob).unwrap();
    assert_eq!(serde_json::from_str::<BlobHeapCfg>(&json).unwrap(), blob);

    let chunk = ChunkPoolCfg::new();
    let json = serde_json::to_string(&chunk).unwrap();
    assert_eq!(serde_json::from_str::<ChunkPoolCfg>(&json).unwrap(), chunk);

    let arena = ArenaCfg::new(1_024, ShardMode::Ordered).with_max_bytes(4_096);
    let json = serde_json::to_string(&arena).unwrap();
    let back: ArenaCfg = serde_json::from_str(&json).unwrap();
    assert_eq!(back, arena);
    assert_eq!(back.mode, ShardMode::Ordered);
}
