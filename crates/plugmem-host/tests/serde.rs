//! `serde` feature: the host's public data types round-trip through JSON.
//! Active only under `--features serde`. HostError is intentionally not covered
//! (it wraps `std::io::Error`, which is not serializable).

#![cfg(feature = "serde")]

use plugmem_host::{ExportedFact, FsyncPolicy, RecoverReport};

#[test]
fn host_data_types_round_trip() {
    let policy = FsyncPolicy::OnSnapshot;
    let json = serde_json::to_string(&policy).unwrap();
    assert_eq!(serde_json::from_str::<FsyncPolicy>(&json).unwrap(), policy);

    let report = RecoverReport {
        kept: 100,
        dropped_text: 2,
        dropped_vector: 1,
        dropped_metadata: 3,
    };
    let json = serde_json::to_string(&report).unwrap();
    assert_eq!(
        serde_json::from_str::<RecoverReport>(&json).unwrap(),
        report
    );

    let fact = ExportedFact {
        text: "prefers tokio".to_string(),
        entity: Some("user".to_string()),
        tags: vec!["pref".to_string()],
        metadata: std::collections::BTreeMap::from([("uri".to_string(), "s3://b/x".to_string())]),
        recorded_at: 1_784_000_000_000,
        valid_from: 1_784_000_000_000,
    };
    let json = serde_json::to_string(&fact).unwrap();
    assert_eq!(serde_json::from_str::<ExportedFact>(&json).unwrap(), fact);
}
