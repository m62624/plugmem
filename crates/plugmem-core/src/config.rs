//! Engine configuration (specs/05).
//!
//! Every knob that changes how bytes are interpreted lives here, because
//! the config is persisted inside the snapshot: opening an existing
//! database with an incompatible config (different `dim`, different shard
//! counts) is a typed error, not a silent reinterpretation.

use alloc::vec::Vec;

use crate::error::Error;

/// Serialized width of one `u64`-encoded size field.
const U64_BYTES: usize = core::mem::size_of::<u64>();
/// Serialized width of one `f32` field.
const F32_BYTES: usize = core::mem::size_of::<f32>();
/// Serialized width of one `u32` field.
const U32_BYTES: usize = core::mem::size_of::<u32>();
/// Serialized width of the `db_uuid` field (`u128`).
const UUID_BYTES: usize = core::mem::size_of::<u128>();

/// Number of `usize` fields in the encoded block (stored as `u64`).
const USIZE_FIELDS: usize = 14;
/// Number of `f32` fields in the encoded block.
const F32_FIELDS: usize = 10;
/// Number of `u32` fields in the encoded block.
const U32_FIELDS: usize = 3;
/// Byte offset of the `f32` field group.
const F32S_AT: usize = USIZE_FIELDS * U64_BYTES;
/// Byte offset of the `u32` field group.
const U32S_AT: usize = F32S_AT + F32_FIELDS * F32_BYTES;
/// Byte offset of the `db_uuid` field.
const DB_UUID_AT: usize = U32S_AT + U32_FIELDS * U32_BYTES;
/// Byte offset of the reserved zero tail (directly after `db_uuid`).
pub const RESERVED_AT: usize = DB_UUID_AT + UUID_BYTES;
/// Length of the reserved zero tail.
const RESERVED_LEN: usize = 8;
/// Exact byte length of the encoded config block (see [`Config::encode`]).
pub const ENCODED_LEN: usize = RESERVED_AT + RESERVED_LEN;

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
    /// Database lineage identity (specs/12 §6). Minted **once** by the
    /// host at creation (the `no_std` core has no RNG) and persisted in
    /// every snapshot; it survives `maintain` and re-saves, so external
    /// holders of ids can tell "same database" from "a different one".
    /// `0` means an unnamed (ephemeral/test) database. On open, `0` here
    /// adopts whatever the snapshot stores; a nonzero value must match
    /// the stored one or the open fails with `ConfigMismatch`.
    pub db_uuid: u128,
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
            db_uuid: 0,
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

    /// Appends the fixed binary form of the config to `out` — the config
    /// block of the snapshot (specs/03). Layout, all little-endian, in
    /// field-declaration order: `usize` fields as `u64`, `f32` fields as
    /// their IEEE 754 bits, then `rrf_k`/`half_life_days`/`graph_depth` as
    /// `u32`, `db_uuid` as a `u128`, then 8 reserved zero bytes; exactly
    /// [`ENCODED_LEN`] bytes. Encoding is lossless and canonical (float bits
    /// round-trip exactly).
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.reserve(ENCODED_LEN);
        for v in [
            self.dim,
            self.max_bytes,
            self.max_text,
            self.max_blob,
            self.shards_facts,
            self.shards_entities,
            self.shards_edges,
            self.shards_temporal,
            self.shards_postings,
            self.hnsw_m,
            self.hnsw_m0,
            self.hnsw_ef_construction,
            self.hnsw_ef_search,
            self.flat_to_hnsw,
        ] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for v in [
            self.bm25_k1,
            self.bm25_b,
            self.w_bm25,
            self.w_vec,
            self.w_graph,
            self.w_time,
            self.w_recency,
            self.graph_decay,
            self.similar_cos,
            self.similar_jaccard,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in [self.rrf_k, self.half_life_days, self.graph_depth] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.db_uuid.to_le_bytes());
        out.extend_from_slice(&[0u8; RESERVED_LEN]);
    }

    /// Decodes a config block written by [`Config::encode`] and runs
    /// [`Config::validate`] on the result.
    ///
    /// The input is untrusted: a wrong length or nonzero reserved bytes are
    /// [`Error::Corrupt`]; out-of-range field values surface as the same
    /// [`Error::ConfigMismatch`] a hand-built config would get.
    ///
    /// All size fields are stored as fixed-width `u64`, so the block is
    /// identical on 32-bit and 64-bit builds of the engine. A value that
    /// overflows this platform's `usize` means the database was created
    /// with limits only a 64-bit address space can hold (e.g. `max_bytes`
    /// beyond 4 GiB on a wasm64 or native host) — the file is not corrupt,
    /// this host is too small for it, hence [`Error::ConfigMismatch`].
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != ENCODED_LEN {
            return Err(Error::Corrupt("config block length mismatch"));
        }
        let mut at = 0usize;
        let mut take_usize = || -> Result<usize, Error> {
            let v = u64::from_le_bytes(bytes[at..at + U64_BYTES].try_into().unwrap());
            at += U64_BYTES;
            usize::try_from(v)
                .map_err(|_| Error::ConfigMismatch("database requires a 64-bit address space"))
        };
        let dim = take_usize()?;
        let max_bytes = take_usize()?;
        let max_text = take_usize()?;
        let max_blob = take_usize()?;
        let shards_facts = take_usize()?;
        let shards_entities = take_usize()?;
        let shards_edges = take_usize()?;
        let shards_temporal = take_usize()?;
        let shards_postings = take_usize()?;
        let hnsw_m = take_usize()?;
        let hnsw_m0 = take_usize()?;
        let hnsw_ef_construction = take_usize()?;
        let hnsw_ef_search = take_usize()?;
        let flat_to_hnsw = take_usize()?;
        let mut at = F32S_AT;
        let mut take_f32 = || {
            let v = f32::from_le_bytes(bytes[at..at + F32_BYTES].try_into().unwrap());
            at += F32_BYTES;
            v
        };
        let bm25_k1 = take_f32();
        let bm25_b = take_f32();
        let w_bm25 = take_f32();
        let w_vec = take_f32();
        let w_graph = take_f32();
        let w_time = take_f32();
        let w_recency = take_f32();
        let graph_decay = take_f32();
        let similar_cos = take_f32();
        let similar_jaccard = take_f32();
        let mut at = U32S_AT;
        let mut take_u32 = || {
            let v = u32::from_le_bytes(bytes[at..at + U32_BYTES].try_into().unwrap());
            at += U32_BYTES;
            v
        };
        let rrf_k = take_u32();
        let half_life_days = take_u32();
        let graph_depth = take_u32();
        let db_uuid = u128::from_le_bytes(bytes[DB_UUID_AT..RESERVED_AT].try_into().unwrap());
        if bytes[RESERVED_AT..ENCODED_LEN] != [0u8; RESERVED_LEN] {
            return Err(Error::Corrupt("reserved config bytes must be zero"));
        }
        let cfg = Self {
            dim,
            max_bytes,
            max_text,
            max_blob,
            shards_facts,
            shards_entities,
            shards_edges,
            shards_temporal,
            shards_postings,
            bm25_k1,
            bm25_b,
            rrf_k,
            w_bm25,
            w_vec,
            w_graph,
            w_time,
            w_recency,
            half_life_days,
            graph_depth,
            graph_decay,
            similar_cos,
            similar_jaccard,
            hnsw_m,
            hnsw_m0,
            hnsw_ef_construction,
            hnsw_ef_search,
            flat_to_hnsw,
            db_uuid,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
