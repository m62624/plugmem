//! The crate-wide error type.
//!
//! All four storage structures share one error enum: capacity problems are
//! always typed errors, never panics, and a caller embedding several
//! structures (the plugmem engine does) handles one type.

use crate::arena::PAGE_BYTES;
use crate::chunk::CHUNK_PAYLOAD;

/// Errors returned by storage operations. No operation in this crate panics
/// on resource exhaustion — capacity problems are always typed errors.
/// (Contract violations — a wrong key length, an out-of-range id — panic,
/// because they are caller bugs, not runtime conditions; each method
/// documents its panics.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Error {
    /// Growing a byte pool would exceed the structure's configured
    /// `max_bytes` ceiling.
    #[error("capacity exceeded: pool would grow past {max_bytes} bytes")]
    CapacityExceeded {
        /// The configured ceiling that would have been crossed.
        max_bytes: usize,
    },
    /// The [`Slot`](crate::Slot) layout constants are invalid.
    #[error(
        "invalid slot layout: size {size}, key_len {key_len} (require 1 <= key_len <= size <= {PAGE_BYTES})"
    )]
    BadSlot {
        /// Declared [`Slot::SIZE`](crate::Slot::SIZE).
        size: usize,
        /// Declared [`Slot::KEY_LEN`](crate::Slot::KEY_LEN).
        key_len: usize,
    },
    /// [`ArenaCfg::shards`](crate::ArenaCfg::shards) is not a non-zero power
    /// of two.
    #[error("shard count must be a non-zero power of two, got {got}")]
    BadShardCount {
        /// The rejected shard count.
        got: usize,
    },
    /// A blob passed to [`BlobHeap::push`](crate::BlobHeap::push) is longer
    /// than the configured [`max_blob`](crate::BlobHeapCfg::max_blob).
    #[error("blob of {len} bytes exceeds the configured max_blob of {max_blob} bytes")]
    BlobTooLarge {
        /// Length of the rejected blob.
        len: usize,
        /// The configured per-blob ceiling.
        max_blob: usize,
    },
    /// A value passed to [`ChunkPool::push`](crate::ChunkPool::push) is
    /// longer than a chunk's payload ([`CHUNK_PAYLOAD`] bytes) and can never
    /// fit without crossing a chunk boundary.
    #[error("value of {len} bytes exceeds the chunk payload of {CHUNK_PAYLOAD} bytes")]
    ValueTooLarge {
        /// Length of the rejected value.
        len: usize,
    },
    /// A serialized image failed load-time validation (`load` methods).
    ///
    /// The message names the violated invariant. Loading never panics on
    /// arbitrary bytes — every inconsistency maps to this variant.
    #[error("corrupt image: {0}")]
    Corrupt(&'static str),
}
