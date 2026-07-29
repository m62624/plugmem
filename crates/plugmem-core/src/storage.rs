//! The [`Storage`] trait and the in-memory reference implementation
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

/// An append-only staging area: build it sequentially with
/// [`write`](Scratch::write), then [`freeze`](Scratch::freeze) it into a
/// readable byte view for random and sequential reads. Dropping it discards the
/// storage.
///
/// It is the write-side sibling of [`Storage`]: `no_std` core owns an algorithm
/// that needs to spill more bytes than fit in RAM and read them back, and the
/// host injects the actual backing — typically a temp file that is memory-mapped
/// on `freeze` (see `plugmem-host`'s `FileScratch`). The reference
/// [`MemScratch`] backs tests with a plain in-RAM buffer.
///
/// # Contract
///
/// - [`write`](Scratch::write) appends; call it any number of times, in the
///   order the bytes should land.
/// - [`freeze`](Scratch::freeze) ends the build and returns **all** written
///   bytes as one contiguous slice; the borrow keeps the region readable. No
///   `write` after a `freeze`.
/// - What `freeze` hands back must equal the concatenation of every `write`.
///
/// plugmem uses it for the disk-first `maintain`/`recover` rebuild, which streams
/// the large pools (vectors, text) through a `Scratch` instead of holding them
/// in RAM — but the trait itself knows nothing of that.
pub trait Scratch {
    /// Implementation-specific failure (I/O error, host callback error…).
    type Error: core::fmt::Debug;

    /// Appends `bytes` to the staging area. Not to be called after
    /// [`freeze`](Scratch::freeze).
    fn write(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Bytes appended so far.
    fn len(&self) -> u64;

    /// Whether nothing has been written yet.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ends the build and borrows everything written as one contiguous slice,
    /// for random and sequential reads (a file backing typically maps itself
    /// here). The borrow keeps the region alive, which is also what forbids a
    /// later `write`.
    fn freeze(&mut self) -> Result<&[u8], Self::Error>;
}

/// In-memory [`Scratch`]: a growable buffer that "freezes" to a borrow of
/// itself. The reference implementation — it backs tests, and because it
/// drives the streaming path with no files it is the vehicle for the
/// `disk_first == in_memory` property test. It stages in RAM, so
/// it does not *save* RAM; a host `Scratch` over a temp file is what bounds RAM
/// in practice.
#[derive(Debug, Default, Clone)]
pub struct MemScratch {
    buf: Vec<u8>,
}

impl MemScratch {
    /// An empty staging buffer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Scratch for MemScratch {
    type Error = core::convert::Infallible;

    fn write(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    fn len(&self) -> u64 {
        self.buf.len() as u64
    }

    fn freeze(&mut self) -> Result<&[u8], Self::Error> {
        Ok(&self.buf)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_scratch_builds_and_freezes_to_the_concatenation() {
        let mut s = MemScratch::new();
        assert!(s.is_empty());
        s.write(b"hello ").unwrap();
        s.write(b"world").unwrap();
        assert_eq!(s.len(), 11);
        assert!(!s.is_empty());
        // freeze returns every written byte, in order, as one slice.
        assert_eq!(s.freeze().unwrap(), b"hello world");
        // random read into the frozen view.
        assert_eq!(&s.freeze().unwrap()[6..], b"world");
    }
}
