//! Host-layer errors.

use std::path::PathBuf;

/// Every way the host layer can fail. Engine failures pass through as
/// [`HostError::Engine`]; everything filesystem- or network-shaped is
/// typed here.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HostError {
    /// The database file is exclusively locked by another process (or
    /// another handle in this process). One local database has one owner — open a
    /// different file, or drop the other handle.
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

    /// There is no published snapshot generation to map: the database has
    /// never been checkpointed, so a read-only open ([`crate::Database::
    /// open_readonly`]) or a [`crate::Database::scrub`] has nothing to point
    /// at. Open it read-write once and checkpoint, then retry.
    ///
    /// A *dirty journal* is not this error. A reader maps the published
    /// generation and never reads the journal, which is snapshot isolation:
    /// it answers as of the last checkpoint rather than refusing until you
    /// take one.
    #[error("database at {} has no published snapshot yet: checkpoint it once first", path.display())]
    NeedsCheckpoint {
        /// The database base path.
        path: PathBuf,
    },

    /// The engine returned a typed error.
    #[error(transparent)]
    Engine(#[from] plugmem_core::Error),

    /// The embedder transport or response was unusable (the message
    /// names what exactly: status, dimension mismatch, malformed JSON).
    #[error("embedder: {0}")]
    Embed(String),

    /// A long explicit reembed is already staging a replacement generation.
    /// Writes fail immediately rather than waiting behind model/network work.
    #[error("database is being reembedded; retry the write after it completes")]
    ReembedBusy,
}

/// What to tell someone whose pool hit its ceiling.
///
/// Kept as one string in one place so the CLI, the MCP server and the Node
/// binding say the same thing. The engine cannot say it: `plugmem-core` knows
/// nothing about config files, and its message is therefore a bare byte count
/// — true, and useless on its own.
pub const MAX_BYTES_HINT: &str = "that ceiling is `max_bytes` (`[engine] max_bytes` in config.toml), \
and it applies to each pool separately rather than to their sum. Its default is not a capacity \
judgement — it is the figure that keeps every pool addressable where `usize` is 32 bits, so a \
database written anywhere opens anywhere. Raising it costs exactly that portability: a 32-bit \
host then refuses the file with a typed error instead of misreading it.";

impl HostError {
    /// The follow-up line for a pool that ran out of room, or `None` when this
    /// error is something else.
    ///
    /// The number in the message is a setting; the setting has a name and one
    /// specific trade-off. Callers that talk to a person should print this
    /// after the error itself.
    pub fn capacity_hint(&self) -> Option<&'static str> {
        let Self::Engine(engine) = self else {
            return None;
        };
        matches!(
            engine,
            plugmem_core::Error::CapacityExceeded { .. }
                | plugmem_core::Error::Arena(plugmem_core::ArenaError::CapacityExceeded { .. })
        )
        .then_some(MAX_BYTES_HINT)
    }

    /// Shorthand for wrapping an I/O error with its path.
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pool_ceiling_carries_its_follow_up_and_nothing_else_does() {
        // The engine's own message for this is a bare byte count — true, and
        // useless to somebody who does not know the number is a setting. Every
        // surface prints this line after it, so it has to actually attach.
        for engine in [
            plugmem_core::Error::CapacityExceeded { what: "vectors" },
            plugmem_core::Error::Arena(plugmem_core::ArenaError::CapacityExceeded {
                max_bytes: 65_536,
            }),
        ] {
            let hint = HostError::Engine(engine).capacity_hint();
            assert_eq!(hint, Some(MAX_BYTES_HINT));
            assert!(hint.unwrap().contains("max_bytes"));
        }

        // Anything else must not: a lock conflict followed by a lecture about
        // pool sizing is worse than no follow-up at all.
        assert_eq!(
            HostError::Engine(plugmem_core::Error::TooLarge {
                what: "text",
                len: 9_000,
                max: 4_096,
            })
            .capacity_hint(),
            None
        );
        assert_eq!(HostError::Embed("no provider".into()).capacity_hint(), None);
        assert_eq!(
            HostError::Locked {
                path: PathBuf::from("/tmp/m.plugmem"),
            }
            .capacity_hint(),
            None
        );
    }
}
