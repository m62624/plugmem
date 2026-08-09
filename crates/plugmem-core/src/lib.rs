#![doc = include_str!("../README.md")]
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

pub use config::{Config, MAX_HNSW_DEGREE, MAX_SHARDS};
pub use error::Error;
pub use id::{EdgeId, EntityId, FactId, NONE_U32};
pub use memory::{
    DEFAULT_TAG_PAGE_LIMIT, FactFault, FactView, GuardedRememberOutcome, LinkInput,
    MAX_TAG_PAGE_LIMIT, MAX_VECTOR_SPACE_ID_BYTES, MaintainReport, MaintenanceMode,
    MaintenanceOptions, Memory, OpenReport, RecallQuery, RecallResult, RecallScratch, RecalledEdge,
    RecalledFact, ReembedError, ReembedReport, RememberInput, RememberOutcome, RemoveTagReport,
    ShardLayout, Similar, SimilarReason, Stats, TagPage, TagQuery, TagSummary, UnlinkInput,
};
pub use model::{
    EdgeSlot, EntityByName, EntityRecord, FactAux, FactRecord, TemporalSlot, VALID_TO_OPEN,
    fact_flags,
};
pub use storage::{MemScratch, MemStorage, Scratch, Storage};
// The arena-layer ids that appear in model records are part of this
// crate's public surface too.
pub use plugmem_arena::{BlobId, TermId};
// The arena's own error appears inside `Error::Arena`, so anything matching on
// it needs the type without depending on the arena crate directly.
pub use plugmem_arena::Error as ArenaError;
