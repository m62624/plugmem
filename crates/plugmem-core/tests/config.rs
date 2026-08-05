//! Config validation tests: the defaults pass, and every documented range
//! check fires with its own message (each `validate` branch is reachable).

use plugmem_core::{Config, Error, MAX_HNSW_DEGREE, MAX_SHARDS, ShardLayout};

/// Asserts that mutating one field trips `validate` with `msg`.
fn rejects(mutate: impl FnOnce(&mut Config), msg: &str) {
    let mut cfg = Config::default();
    mutate(&mut cfg);
    match cfg.validate() {
        Err(Error::ConfigMismatch(got)) => assert_eq!(got, msg),
        other => panic!("expected ConfigMismatch({msg:?}), got {other:?}"),
    }
}

#[test]
fn defaults_are_valid() {
    let cfg = Config::default();
    cfg.validate().unwrap();
    // Spot-check the table.
    assert_eq!(cfg.dim, 0);
    assert_eq!(cfg.max_bytes, 2 * 1024 * 1024 * 1024);
    assert_eq!(cfg.rrf_k, 60);
    assert_eq!(cfg.flat_to_hnsw, 24_000);
    // A fresh database is empty, and an empty database belongs on the shard
    // floor. It grows from here through `maintain`, so these are a starting
    // point rather than a tuning choice — see `Config::shards_facts`.
    let floor = ShardLayout::default();
    assert_eq!(cfg.shards_facts, floor.facts);
    assert_eq!(cfg.shards_entities, floor.entities);
    assert_eq!(cfg.shards_edges, floor.edges);
    assert_eq!(cfg.shards_temporal, floor.temporal);
    assert_eq!(cfg.shards_postings, floor.postings);
}

#[test]
fn a_realistic_vector_config_is_valid() {
    let mut cfg = Config::default();
    cfg.dim = 384;
    cfg.validate().unwrap();
    cfg.dim = 4096;
    cfg.validate().unwrap();
}

#[test]
fn every_range_check_fires() {
    rejects(|c| c.dim = 4097, "dim must be <= 4096");
    rejects(
        |c| c.shards_facts = 3,
        "shards_facts must be a power of two",
    );
    rejects(
        |c| c.shards_entities = 0,
        "shards_entities must be a power of two",
    );
    rejects(
        |c| c.shards_edges = 100,
        "shards_edges must be a power of two",
    );
    rejects(
        |c| c.shards_temporal = 7,
        "shards_temporal must be a power of two",
    );
    rejects(
        |c| c.shards_postings = 6,
        "shards_postings must be a power of two",
    );
    rejects(
        |c| c.shards_facts = MAX_SHARDS * 2,
        "shards_facts exceeds MAX_SHARDS",
    );
    rejects(
        |c| c.shards_entities = MAX_SHARDS * 2,
        "shards_entities exceeds MAX_SHARDS",
    );
    rejects(
        |c| c.shards_edges = MAX_SHARDS * 2,
        "shards_edges exceeds MAX_SHARDS",
    );
    rejects(
        |c| c.shards_temporal = MAX_SHARDS * 2,
        "shards_temporal exceeds MAX_SHARDS",
    );
    rejects(
        |c| c.shards_postings = MAX_SHARDS * 2,
        "shards_postings exceeds MAX_SHARDS",
    );
    rejects(|c| c.max_text = 0, "max_text must be in 1..=max_blob");
    rejects(
        |c| {
            c.max_text = 128 * 1024;
        },
        "max_text must be in 1..=max_blob",
    );
    rejects(
        |c| {
            c.max_bytes = 1024;
        },
        "max_blob must be <= max_bytes",
    );
    rejects(|c| c.bm25_k1 = 0.0, "bm25_k1 must be positive");
    rejects(|c| c.bm25_k1 = f32::NAN, "bm25_k1 must be positive");
    rejects(|c| c.bm25_b = 1.5, "bm25_b must be in [0, 1]");
    rejects(|c| c.rrf_k = 0, "rrf_k must be >= 1");
    rejects(|c| c.w_bm25 = -1.0, "w_bm25 must be finite and >= 0");
    rejects(|c| c.w_vec = f32::INFINITY, "w_vec must be finite and >= 0");
    rejects(|c| c.w_graph = f32::NAN, "w_graph must be finite and >= 0");
    rejects(|c| c.w_time = -0.1, "w_time must be finite and >= 0");
    rejects(|c| c.w_recency = -2.0, "w_recency must be finite and >= 0");
    rejects(|c| c.half_life_days = 0, "half_life_days must be >= 1");
    rejects(|c| c.graph_decay = 0.0, "graph_decay must be in (0, 1]");
    rejects(|c| c.graph_decay = 1.1, "graph_decay must be in (0, 1]");
    rejects(|c| c.similar_cos = -0.5, "similar_cos must be in [0, 1]");
    rejects(
        |c| c.similar_jaccard = 2.0,
        "similar_jaccard must be in [0, 1]",
    );
    rejects(|c| c.hnsw_m = 1, "hnsw_m must be >= 2");
    rejects(|c| c.hnsw_m0 = 8, "hnsw_m0 must be >= hnsw_m");
    rejects(
        |c| {
            c.hnsw_m = MAX_HNSW_DEGREE + 1;
            c.hnsw_m0 = MAX_HNSW_DEGREE + 1;
            c.hnsw_ef_construction = MAX_HNSW_DEGREE + 1;
        },
        "hnsw_m exceeds MAX_HNSW_DEGREE",
    );
    rejects(
        |c| c.hnsw_m0 = MAX_HNSW_DEGREE + 1,
        "hnsw_m0 exceeds MAX_HNSW_DEGREE",
    );
    rejects(
        |c| c.hnsw_ef_construction = 4,
        "hnsw_ef_construction must be >= hnsw_m",
    );
    rejects(|c| c.hnsw_ef_search = 0, "hnsw_ef_search must be >= 1");
    rejects(|c| c.flat_to_hnsw = 0, "flat_to_hnsw must be >= 1");
}

#[test]
fn zero_weights_and_boundary_values_are_valid() {
    let mut cfg = Config::default();
    cfg.w_bm25 = 0.0;
    cfg.w_vec = 0.0;
    cfg.w_graph = 0.0;
    cfg.w_time = 0.0;
    cfg.w_recency = 0.0;
    cfg.bm25_b = 0.0;
    cfg.graph_decay = 1.0;
    cfg.similar_cos = 1.0;
    cfg.similar_jaccard = 0.0;
    cfg.graph_depth = 0;
    cfg.validate().unwrap();

    // Depth has no ceiling: what a graph walk costs is held by the entity and
    // edge caps in the recall path, so a hop count only decides how far a
    // *sparse* walk may go — and there the hops are the cheap part.
    cfg.graph_depth = u32::MAX;
    cfg.validate().unwrap();
}

#[test]
fn a_64bit_class_database_is_refused_on_32bit_hosts_with_a_typed_error() {
    // Field order in the encoded block: `dim` is the first u64, `max_bytes`
    // the second (see `Config::encode`); every size field is a fixed-width
    // u64, so the block itself is identical on 32-bit and 64-bit builds.
    const U64_BYTES: usize = 8;
    const DIM_AT: usize = 0;
    const MAX_BYTES_AT: usize = DIM_AT + U64_BYTES;
    const EIGHT_GIB: u64 = 8 * 1024 * 1024 * 1024;

    let mut bytes = Vec::new();
    Config::default().encode(&mut bytes);
    bytes[MAX_BYTES_AT..MAX_BYTES_AT + U64_BYTES].copy_from_slice(&EIGHT_GIB.to_le_bytes());

    let decoded = Config::decode(&bytes);
    if usize::try_from(EIGHT_GIB).is_ok() {
        // 64-bit host (native, wasm64): the limit is representable and adopted.
        assert_eq!(decoded.unwrap().max_bytes as u64, EIGHT_GIB);
    } else {
        // 32-bit host (wasm32): a typed refusal, not a "corrupt file" lie —
        // the database needs more address space than this build can offer.
        match decoded {
            Err(Error::ConfigMismatch(msg)) => {
                assert_eq!(msg, "database requires a 64-bit address space");
            }
            other => panic!("expected ConfigMismatch, got {other:?}"),
        }
    }
}
