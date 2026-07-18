//! plugmem engine core.
//!
//! The `no_std + alloc`, single-threaded memory engine on top of
//! [`plugmem_arena`]: the data model (facts, entities, edges,
//! bitemporality), the indexes (BM25, temporal, graph, vectors), hybrid
//! recall, and the snapshot/journal machinery. The core owns no I/O, no
//! clock and no threads — timestamps arrive as parameters, bytes leave
//! through the `Storage` trait.
//!
//! Design: `specs/02..05`. Implementation lands in stages 2-4; modules
//! appear here in dependency order as those stages complete.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod config;
pub mod error;
pub mod id;
pub mod journal;
pub mod model;
pub mod snapshot;
pub mod storage;
pub mod tokenizer;

pub use config::Config;
pub use error::Error;
pub use id::{EntityId, FactId, NONE_U32};
pub use model::{
    EdgeSlot, EntityByName, EntityRecord, FactAux, FactRecord, TemporalSlot, VALID_TO_OPEN,
    fact_flags,
};
pub use storage::{MemStorage, Storage};
// The arena-layer ids that appear in model records are part of this
// crate's public surface too.
pub use plugmem_arena::{BlobId, TermId};
