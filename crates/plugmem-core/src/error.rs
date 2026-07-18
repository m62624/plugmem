//! Engine error type (specs/05).
//!
//! A panic inside the engine is a bug by definition; every failure mode is
//! a typed variant here. Variants that only become reachable with later
//! stages (storage, snapshot loading) are added together with those stages.

use crate::id::FactId;

/// Every way an engine call can fail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A pool or table would grow past its configured ceiling.
    #[error("capacity exceeded: {what}")]
    CapacityExceeded {
        /// Which limit was hit (e.g. `"facts"`, `"tags per fact"`).
        what: &'static str,
    },

    /// A single input value is larger than its configured maximum.
    #[error("{what} too large: {len} bytes (max {max})")]
    TooLarge {
        /// Which input was oversized (e.g. `"text"`).
        what: &'static str,
        /// Actual length in bytes (or elements, per `what`).
        len: usize,
        /// The configured ceiling.
        max: usize,
    },

    /// An input vector's dimension does not match `Config::dim`.
    #[error("vector dimension mismatch: got {got}, want {want}")]
    DimMismatch {
        /// Dimension of the provided vector.
        got: usize,
        /// Dimension the database was configured with.
        want: usize,
    },

    /// The referenced fact does not exist (or was purged).
    #[error("fact {} not found", (.0).0)]
    NotFound(FactId),

    /// `revise` targeted a fact whose validity interval is already closed.
    #[error("fact {} is already closed", (.0).0)]
    AlreadyClosed(FactId),

    /// The supplied `Config` is invalid, or incompatible with the config
    /// stored in an existing database (changing `dim` or shard counts
    /// requires a reindex, not an open).
    #[error("config mismatch: {0}")]
    ConfigMismatch(&'static str),

    /// A snapshot or journal failed validation.
    #[error("corrupt input: {0}")]
    Corrupt(&'static str),

    /// The snapshot was written by an unknown format version.
    #[error("unsupported snapshot format version {0}")]
    UnsupportedVersion(u16),

    /// An underlying storage-structure error (bubbled up from the arena
    /// layer with its context intact).
    #[error("arena: {0}")]
    Arena(#[from] plugmem_arena::Error),
}
