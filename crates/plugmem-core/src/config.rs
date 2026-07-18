//! Engine configuration (specs/05).
//!
//! Every knob that changes how bytes are interpreted lives here, because
//! the config is persisted inside the snapshot: opening an existing
//! database with an incompatible config (different `dim`, different shard
//! counts) is a typed error, not a silent reinterpretation.

use crate::error::Error;

/// Full engine configuration with the specs/05 defaults.
///
/// Plain data: construct with [`Config::default`], override fields, then
/// let the engine call [`Config::validate`] (it is also callable directly —
/// useful for surfacing config errors early in wrappers).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Config {
    /// Vector dimension; `0` disables the vector layer entirely. Max 4096.
    pub dim: usize,
    /// Total ceiling for all byte pools (the wasm32 passport: ≤ 2 GiB).
    pub max_bytes: usize,
    /// Maximum fact text length in bytes.
    pub max_text: usize,
    /// Maximum single blob length in bytes.
    pub max_blob: usize,
    /// Shard count of the facts arena (power of two).
    pub shards_facts: usize,
    /// Shard count of the entities arena (power of two).
    pub shards_entities: usize,
    /// Shard count of each edge arena (power of two).
    pub shards_edges: usize,
    /// Shard count of the temporal arena (power of two).
    pub shards_temporal: usize,
    /// Shard count of the postings arena (power of two).
    pub shards_postings: usize,
    /// BM25 `k1` (term-frequency saturation).
    pub bm25_k1: f32,
    /// BM25 `b` (length normalization), in `[0, 1]`.
    pub bm25_b: f32,
    /// The RRF rank constant (`score += w / (rrf_k + rank)`).
    pub rrf_k: u32,
    /// RRF weight of the lexical (BM25) source.
    pub w_bm25: f32,
    /// RRF weight of the vector source.
    pub w_vec: f32,
    /// RRF weight of the graph source.
    pub w_graph: f32,
    /// RRF weight of the temporal-range source.
    pub w_time: f32,
    /// Strength of the recency boost (`0` disables it).
    pub w_recency: f32,
    /// Recency half-life in days.
    pub half_life_days: u32,
    /// Graph expansion depth limit.
    pub graph_depth: u32,
    /// Per-hop weight decay of graph candidates, in `(0, 1]`.
    pub graph_decay: f32,
    /// Cosine threshold for vector-based similar-detection, in `[0, 1]`.
    pub similar_cos: f32,
    /// Jaccard threshold for lexical similar-detection, in `[0, 1]`.
    pub similar_jaccard: f32,
    /// HNSW: neighbors per node on upper levels.
    pub hnsw_m: usize,
    /// HNSW: neighbors per node on level 0.
    pub hnsw_m0: usize,
    /// HNSW: beam width during construction.
    pub hnsw_ef_construction: usize,
    /// HNSW: default beam width during search (per-query override exists).
    pub hnsw_ef_search: usize,
    /// Vector count at which `maintain` switches Flat → HNSW.
    pub flat_to_hnsw: usize,
    /// Skip per-section xxh3 checks when loading a trusted snapshot.
    pub fast_load: bool,
}

impl Default for Config {
    /// The specs/05 defaults table.
    fn default() -> Self {
        Self {
            dim: 0,
            max_bytes: 2 * 1024 * 1024 * 1024,
            max_text: 4096,
            max_blob: 64 * 1024,
            shards_facts: 1024,
            shards_entities: 256,
            shards_edges: 512,
            shards_temporal: 512,
            shards_postings: 2048,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            rrf_k: 60,
            w_bm25: 1.0,
            w_vec: 1.0,
            w_graph: 1.0,
            w_time: 1.0,
            w_recency: 0.25,
            half_life_days: 180,
            graph_depth: 2,
            graph_decay: 0.5,
            similar_cos: 0.85,
            similar_jaccard: 0.5,
            hnsw_m: 16,
            hnsw_m0: 32,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
            flat_to_hnsw: 24_000,
            fast_load: false,
        }
    }
}

/// One weight-range check: finite and non-negative.
fn check_weight(v: f32, what: &'static str) -> Result<(), Error> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(Error::ConfigMismatch(what))
    }
}

/// One unit-interval check: finite and inside `[0, 1]`.
fn check_unit(v: f32, what: &'static str) -> Result<(), Error> {
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(Error::ConfigMismatch(what))
    }
}

impl Config {
    /// Checks every field against its documented range.
    ///
    /// Returns [`Error::ConfigMismatch`] naming the offending field. The
    /// engine calls this on every construction path; wrappers may call it
    /// earlier to fail fast.
    pub fn validate(&self) -> Result<(), Error> {
        if self.dim > 4096 {
            return Err(Error::ConfigMismatch("dim must be <= 4096"));
        }
        for (shards, what) in [
            (self.shards_facts, "shards_facts must be a power of two"),
            (
                self.shards_entities,
                "shards_entities must be a power of two",
            ),
            (self.shards_edges, "shards_edges must be a power of two"),
            (
                self.shards_temporal,
                "shards_temporal must be a power of two",
            ),
            (
                self.shards_postings,
                "shards_postings must be a power of two",
            ),
        ] {
            if !shards.is_power_of_two() {
                return Err(Error::ConfigMismatch(what));
            }
        }
        if self.max_text == 0 || self.max_text > self.max_blob {
            return Err(Error::ConfigMismatch("max_text must be in 1..=max_blob"));
        }
        if self.max_blob > self.max_bytes {
            return Err(Error::ConfigMismatch("max_blob must be <= max_bytes"));
        }
        if !(self.bm25_k1.is_finite() && self.bm25_k1 > 0.0) {
            return Err(Error::ConfigMismatch("bm25_k1 must be positive"));
        }
        check_unit(self.bm25_b, "bm25_b must be in [0, 1]")?;
        if self.rrf_k == 0 {
            return Err(Error::ConfigMismatch("rrf_k must be >= 1"));
        }
        check_weight(self.w_bm25, "w_bm25 must be finite and >= 0")?;
        check_weight(self.w_vec, "w_vec must be finite and >= 0")?;
        check_weight(self.w_graph, "w_graph must be finite and >= 0")?;
        check_weight(self.w_time, "w_time must be finite and >= 0")?;
        check_weight(self.w_recency, "w_recency must be finite and >= 0")?;
        if self.half_life_days == 0 {
            return Err(Error::ConfigMismatch("half_life_days must be >= 1"));
        }
        if self.graph_depth > 4 {
            return Err(Error::ConfigMismatch("graph_depth must be <= 4"));
        }
        if !(self.graph_decay.is_finite() && self.graph_decay > 0.0 && self.graph_decay <= 1.0) {
            return Err(Error::ConfigMismatch("graph_decay must be in (0, 1]"));
        }
        check_unit(self.similar_cos, "similar_cos must be in [0, 1]")?;
        check_unit(self.similar_jaccard, "similar_jaccard must be in [0, 1]")?;
        if self.hnsw_m < 2 {
            return Err(Error::ConfigMismatch("hnsw_m must be >= 2"));
        }
        if self.hnsw_m0 < self.hnsw_m {
            return Err(Error::ConfigMismatch("hnsw_m0 must be >= hnsw_m"));
        }
        if self.hnsw_ef_construction < self.hnsw_m {
            return Err(Error::ConfigMismatch(
                "hnsw_ef_construction must be >= hnsw_m",
            ));
        }
        if self.hnsw_ef_search == 0 {
            return Err(Error::ConfigMismatch("hnsw_ef_search must be >= 1"));
        }
        if self.flat_to_hnsw == 0 {
            return Err(Error::ConfigMismatch("flat_to_hnsw must be >= 1"));
        }
        Ok(())
    }
}
