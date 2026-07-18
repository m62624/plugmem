//! The [`Storage`] trait and the in-memory reference implementation
//! (specs/03).
//!
//! The core never touches files or the network: every byte entering or
//! leaving the engine goes through this trait. Native wrappers implement
//! it over files (`plugmem-host::FileStorage`), the wasm bridge over host
//! callbacks; [`MemStorage`] backs tests and ephemeral databases.

use alloc::vec::Vec;

/// Where snapshot and journal bytes live.
///
/// The engine calls `append_journal` synchronously at the end of every
/// mutating operation — that call is the durability point. Fsync policy,
/// atomicity of `write_snapshot` (tmp + rename) and locking are the
/// implementation's business, not the core's.
pub trait Storage {
    /// Implementation-specific failure (I/O error, host callback error…).
    type Error: core::fmt::Debug;

    /// The full snapshot image, or `None` when no database exists yet
    /// (first run).
    fn read_snapshot(&mut self) -> Result<Option<Vec<u8>>, Self::Error>;

    /// Atomically replaces the snapshot image.
    fn write_snapshot(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;

    /// All journal bytes accumulated since the last snapshot (empty is
    /// normal).
    fn read_journal(&mut self) -> Result<Vec<u8>, Self::Error>;

    /// Appends one framed journal entry (see [`crate::journal`]).
    fn append_journal(&mut self, entry: &[u8]) -> Result<(), Self::Error>;

    /// Discards the journal (called after a successful snapshot).
    fn clear_journal(&mut self) -> Result<(), Self::Error>;
}

/// In-memory [`Storage`]: two byte buffers, no failure modes.
///
/// The reference implementation for tests and for ephemeral databases
/// (an agent's scratch memory that intentionally dies with the process).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MemStorage {
    snapshot: Option<Vec<u8>>,
    journal: Vec<u8>,
}

impl MemStorage {
    /// A storage with no snapshot and an empty journal (first run).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for MemStorage {
    type Error = core::convert::Infallible;

    fn read_snapshot(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.snapshot.clone())
    }

    fn write_snapshot(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.snapshot = Some(bytes.to_vec());
        Ok(())
    }

    fn read_journal(&mut self) -> Result<Vec<u8>, Self::Error> {
        Ok(self.journal.clone())
    }

    fn append_journal(&mut self, entry: &[u8]) -> Result<(), Self::Error> {
        self.journal.extend_from_slice(entry);
        Ok(())
    }

    fn clear_journal(&mut self) -> Result<(), Self::Error> {
        self.journal.clear();
        Ok(())
    }
}
