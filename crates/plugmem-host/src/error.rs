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
