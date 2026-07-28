//! Native host layer for the plugmem engine: file-backed storage with
//! exclusive locking, a thread-safe database handle with a maintenance
//! policy, and embedding providers (specs/13).
//!
//! This crate is the "point it at a file and go" Rust experience:
//!
//! ```no_run
//! use plugmem_host::{Config, Database, RecallQuery, RememberInput};
//!
//! let (db, _report) = Database::open("agent.plugmem", Config::default())?;
//! db.remember(RememberInput::text(1_784_000_000_000, "prefers tokio"))?;
//! let out = db.recall(RecallQuery::text(1_784_000_100_000, "runtime?"))?;
//! println!("{}", out.rendered);
//! # Ok::<(), plugmem_host::HostError>(())
//! ```
//!
//! Concurrency model (specs/13 §1): one file has one owning process —
//! a second open is refused with [`HostError::Locked`]; within the
//! process, clone the [`Database`] handle across threads and agents;
//! different files are fully independent. Embedding calls run outside
//! the database lock.

mod db;
mod embedder;
mod error;
mod readonly;
#[cfg(feature = "config")]
mod settings;
mod storage;

pub use db::{Database, DatabaseBuilder, ExportedFact, FactSnapshot, RecoverReport};
pub use embedder::{Embedder, NullEmbedder, OpenAiCompatEmbedder};
pub use error::HostError;
pub use readonly::{ReadOnlyDatabase, Scrub};
#[cfg(feature = "config")]
pub use settings::{Settings, SettingsError, read_config};
pub use storage::{FileScratch, FileStorage, FsyncPolicy};

// The engine types a host caller works with, re-exported so simple
// embedders need only this crate.
pub use plugmem_core::snapshot::{DEFAULT_SCRUB_BUDGET, ScrubProgress};
pub use plugmem_core::{
    Config, EntityId, Error, FactId, FactRecord, LinkInput, MaintainReport, OpenReport,
    RecallQuery, RecallResult, RecallScratch, RecalledEdge, RecalledFact, RememberInput,
    RememberOutcome, Similar, SimilarReason, Stats, VALID_TO_OPEN, fact_flags,
};
pub use plugmem_core::{MemScratch, Scratch};
