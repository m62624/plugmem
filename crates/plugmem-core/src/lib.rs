//! plugmem engine core.
//!
//! The `no_std + alloc`, single-threaded memory engine on top of
//! [`plugmem_arena`]: the data model (facts, entities, edges,
//! bitemporality), the indexes (BM25, temporal, graph, vectors), hybrid
//! recall, and the snapshot/journal machinery. The core owns no I/O, no
//! clock and no threads — timestamps arrive as parameters, bytes leave
//! through the `Storage` trait.
//!
//! Design: `..05`. Implementation lands in stages 2-4; modules
//! appear here in dependency order as those stages complete.

#![no_std]
// The engine holds zero `unsafe`: every byte-reinterpretation lives in
// `plugmem_arena`, behind its single audited `unsafe`. Forbidding it here turns
// "the core is UB-free" into a compile-time guarantee — which is why the MIRI
// audit covers only the arena, not this crate's (necessarily safe) code.
#![forbid(unsafe_code)]

extern crate alloc;

pub mod config;
pub mod error;
pub mod id;
pub mod index;
pub mod journal;
pub mod memory;
pub(crate) mod metadata;
pub mod model;
pub mod snapshot;
pub mod storage;
pub mod tokenizer;

pub use config::{Config, MAX_SHARDS};
pub use error::Error;
pub use id::{EdgeId, EntityId, FactId, NONE_U32};
pub use memory::{
    FactFault, FactView, LinkInput, MaintainReport, MaintenanceMode, MaintenanceOptions, Memory,
    OpenReport, RecallQuery, RecallResult, RecallScratch, RecalledEdge, RecalledFact,
    RememberInput, RememberOutcome, Similar, SimilarReason, Stats, UnlinkInput,
};
pub use model::{
    EdgeSlot, EntityByName, EntityRecord, FactAux, FactRecord, TemporalSlot, VALID_TO_OPEN,
    fact_flags,
};
pub use storage::{MemScratch, MemStorage, Scratch, Storage};
// The arena-layer ids that appear in model records are part of this
// crate's public surface too.
pub use plugmem_arena::{BlobId, TermId};
