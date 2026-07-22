//! Host-layer errors (specs/13 §5).

use std::path::PathBuf;

/// Every way the host layer can fail. Engine failures pass through as
/// [`HostError::Engine`]; everything filesystem- or network-shaped is
/// typed here.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HostError {
    /// The database file is exclusively locked by another process (or
    /// another handle in this process). One file has one owner — open a
    /// different file, or drop the other handle (specs/13 §1).
    #[error("database at {} is locked by another process", path.display())]
    Locked {
        /// The database base path.
        path: PathBuf,
    },

    /// A filesystem operation failed.
    #[error("i/o on {}: {source}", path.display())]
    Io {
        /// The file the operation touched.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A read-only open ([`crate::Database::open_readonly`]) found a
    /// non-empty journal. Replaying it would mutate the engine — copying
    /// whole arenas up from the mapped bytes (copy-on-write) — which
    /// defeats the zero-copy intent. Open the database read-write once to
    /// checkpoint it (fold the journal into the snapshot), then retry
    /// (specs/16 §3).
    #[error("database at {} needs a checkpoint before a read-only open (non-empty journal)", path.display())]
    NeedsCheckpoint {
        /// The database base path.
        path: PathBuf,
    },

    /// [`crate::Database::recover`] refused a salvage because the snapshot is
    /// larger than the rebuild budget. Recovery rebuilds the surviving image
    /// in RAM (a maintenance pass materializes owned pools ≈ the image size),
    /// so a database well past available memory cannot be recovered this way —
    /// restore from a backup instead (Tier 0), or pass a higher limit if the
    /// memory is there (specs/16 §9).
    #[error(
        "snapshot at {} is {image_bytes} bytes, over the {limit}-byte rebuild budget — restore from backup",
        path.display()
    )]
    TooLargeToRecover {
        /// The source database base path.
        path: PathBuf,
        /// The snapshot image size.
        image_bytes: u64,
        /// The rebuild budget it exceeded.
        limit: u64,
    },

    /// The engine returned a typed error.
    #[error(transparent)]
    Engine(#[from] plugmem_core::Error),

    /// The embedder transport or response was unusable (the message
    /// names what exactly: status, dimension mismatch, malformed JSON).
    #[error("embedder: {0}")]
    Embed(String),
}

impl HostError {
    /// Shorthand for wrapping an I/O error with its path.
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}
