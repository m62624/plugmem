#![doc = include_str!("../README.md")]
//! Native host layer for the plugmem engine: file-backed storage with
//! exclusive locking, a thread-safe database handle with a maintenance
//! policy, and embedding providers.
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
//! Concurrency model: one local database has one owning process —
//! a second open is refused with [`HostError::Locked`]; within the
//! process, clone the [`Database`] handle across threads and agents;
//! different files are fully independent. Embedding calls run outside
//! the database lock.
//!
//! With the default `config` feature, the host also accepts the shared
//! `config.toml` format through [`Settings::load`] and [`read_config`]. The
//! `SETTINGS.md` file is the reference documentation for that format, not an
//! input file. Disable the feature with `default-features = false` when a
//! caller wants only programmatic [`Config`] construction.

mod db;
mod embedder;
mod error;
mod paths;
mod readonly;
#[cfg(feature = "config")]
mod settings;
#[cfg(feature = "config")]
mod settings_help;
mod storage;
mod workspace;

pub use db::{
    DEFAULT_REEMBED_BATCH_SIZE, Database, DatabaseBuilder, ExportPage, ExportedFact, FactSnapshot,
    RecoverReport,
};
pub use embedder::{Embedder, NullEmbedder, OpenAiCompatEmbedder, SharedEmbedder};
pub use error::HostError;
pub use paths::{default_config_dir, default_config_path, default_data_dir, default_database_path};
pub use readonly::{ReadOnlyDatabase, Scrub};
#[cfg(feature = "config")]
pub use settings::{Settings, SettingsError, WorkspaceSettings, read_config};
#[cfg(feature = "config")]
pub use settings_help::{SettingDoc, SettingScope, SettingWarning, SettingsHelp, settings_help};
pub use storage::{FileScratch, FileStorage, FsyncPolicy};
pub use workspace::{
    ARCHIVED_TAG, DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_MAX_OPEN, DbEntry, DbName, Description,
    ENTRY_TAG, IfMissing, MAX_DB_NAME, MAX_OPEN_CEILING, NameProblem, Opener, ReindexReport,
    SELF_ENTITY, Workspace, WorkspaceError, WorkspaceIssue, WorkspaceLayout, WorkspaceLease,
    WorkspaceLimits,
};

// The engine types a host caller works with, re-exported so simple
// embedders need only this crate.
pub use plugmem_core::snapshot::{DEFAULT_SCRUB_BUDGET, ScrubProgress};
pub use plugmem_core::{
    Config, DEFAULT_TAG_PAGE_LIMIT, EdgeId, EntityId, Error, FactId, FactRecord, LinkInput,
    MAX_TAG_PAGE_LIMIT, MaintainReport, MaintenanceMode, MaintenanceOptions, OpenReport,
    RecallQuery, RecallResult, RecallScratch, RecalledEdge, RecalledFact, ReembedReport,
    RememberInput, RememberOutcome, RemoveTagReport, ShardLayout, Similar, SimilarReason, Stats,
    TagPage, TagQuery, TagSummary, UnlinkInput, VALID_TO_OPEN, fact_flags,
};
pub use plugmem_core::{MemScratch, Scratch};
