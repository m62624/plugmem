//! Engine error type.
//!
//! A panic inside the engine is a bug by definition; every failure mode is
//! a typed variant here. Variants that only become reachable with later
//! stages (storage, snapshot loading) are added together with those stages.

use crate::id::FactId;

extern crate alloc;

/// Every way an engine call can fail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

    /// A paginated read continued after the underlying catalog changed.
    #[error("tag cursor is stale; restart without a cursor")]
    StaleCursor,

    /// A database contains vectors written before embedding-space identities
    /// were persisted. Their model cannot be inferred safely.
    #[error("vector space is untracked; run an explicit reembed")]
    UntrackedVectorSpace,

    /// The configured embedder belongs to a different semantic vector space
    /// than the one recorded in the database.
    #[error(
        "vector space mismatch: stored {stored:?}, requested {requested:?}; run an explicit reembed"
    )]
    VectorSpaceMismatch {
        /// Identity persisted with the current vectors.
        stored: alloc::string::String,
        /// Identity supplied by the configured embedder.
        requested: alloc::string::String,
    },

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

    /// An input violates a structural rule that is not a size limit
    /// (a link without a subject entity, an empty tag, a name with no
    /// indexable characters).
    #[error("invalid input: {0}")]
    Invalid(&'static str),

    /// The [`Storage`](crate::storage::Storage) implementation failed;
    /// carries the implementation error's debug rendering (the engine
    /// stays generic and cloneable, the wrapper logs the original).
    #[error("storage: {0}")]
    Storage(alloc::string::String),

    /// An underlying storage-structure error (bubbled up from the arena
    /// layer with its context intact).
    #[error("arena: {0}")]
    Arena(#[from] plugmem_arena::Error),
}
