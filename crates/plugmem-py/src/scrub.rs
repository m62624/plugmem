//! The `Scrub` class: a resumable, byte-level check of the snapshot container.
//!
//! `verify` and `scrub` answer different questions and neither replaces the
//! other. `verify` is *content*-level — the text is valid UTF-8, the fact↔slot
//! vector mapping agrees with itself, the two edge mirrors match. `scrub` is
//! *byte*-level: it recomputes each section's stored xxh3 and the whole-file
//! hash, which is what catches a flipped bit that the structure happily accepts.
//!
//! It is paced by the caller rather than run to completion, because it is meant
//! to be affordable on a live database — the ZFS model, on demand instead of at
//! open. Each step hashes at most a budget's worth of bytes and stops.
//!
//! Each step releases the GIL, and not because hashing is slow. Over an mmap the
//! bytes are **paged in from disk** as they are read (they stay reclaimable
//! afterwards, which is why a scrub never residents the whole file), so a step's
//! real cost is a page fault whose duration is however long that storage takes.
//! Holding the interpreter for that would stall every other thread in the
//! process for an interval nobody can predict.
//!
//! Both `next()` and the iterator protocol are provided, and they are the same
//! step: `next()` is the surface `plugmem-napi` has, `__next__` is how a Python
//! caller expects to walk something. Neither is a second implementation.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::error::{self, Result};
use crate::types::ScrubProgress;

/// A resumable byte-level check of one snapshot generation.
///
/// **Holding this object holds a lock.** It pins the generation it is scanning
/// with a shared lock for its whole life, so the writer's garbage collection
/// cannot reclaim that generation under it. Finish the scan, or `close()` it —
/// do not park one indefinitely. It is also a context manager, so
/// `with db.scrub() as scan:` releases the pin however the block ends.
///
/// One-shot: once it has returned `None`, or raised, it is done. Ask the
/// database for another to scan again.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "plugmem._plugmem")]
pub struct Scrub {
    /// `None` once exhausted, failed, or closed — dropping the host cursor is
    /// what releases the pinned generation, so it is dropped eagerly rather
    /// than left for whenever the Python object is collected.
    cursor: Mutex<Option<plugmem_host::Scrub>>,
}

impl Scrub {
    /// Wrap a host cursor. Not a Python constructor: a scrub is obtained from
    /// the database it scans, which is what ties it to a real generation.
    pub(crate) fn new(inner: plugmem_host::Scrub) -> Self {
        Self {
            cursor: Mutex::new(Some(inner)),
        }
    }
}

/// The cursor guard. A panicked step cannot leave a cursor half-advanced (the
/// host iterator is fused and owns its own position), so a poisoned lock is
/// recovered — the same rule the engine lock follows.
fn locked(
    cursor: &Mutex<Option<plugmem_host::Scrub>>,
) -> MutexGuard<'_, Option<plugmem_host::Scrub>> {
    cursor
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[gen_stub_pymethods]
#[pymethods]
impl Scrub {
    /// Hash the next budget's worth of bytes and report progress, or return
    /// `None` when the generation has been checked end to end.
    ///
    /// Raises `EngineError` on a mismatch, naming the section — that is the
    /// scrub finding what it was looking for.
    fn next(&self, py: Python<'_>) -> Result<Option<ScrubProgress>> {
        py.detach(|| {
            let mut guard = locked(&self.cursor);
            let Some(cursor) = guard.as_mut() else {
                return Ok(None);
            };
            match cursor.next() {
                Some(Ok(progress)) => Ok(Some(ScrubProgress::from(progress))),
                // Exhausted or failed: drop the cursor either way, because the
                // pinned generation should not outlive the reason it was held.
                Some(Err(e)) => {
                    *guard = None;
                    Err(error::engine(e))
                }
                None => {
                    *guard = None;
                    Ok(None)
                }
            }
        })
    }

    /// Release the pinned generation now. Calling it twice is fine.
    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            let mut guard = locked(&self.cursor);
            let cursor = guard.take();
            drop(guard);
            drop(cursor);
        });
    }

    /// Whether the scan is still holding its generation and has steps left.
    fn active(&self) -> bool {
        locked(&self.cursor).is_some()
    }

    /// `for progress in db.scrub():` — the iterator is the scrub itself.
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// One step, in the iterator protocol's terms. The same step `next()`
    /// takes; exhaustion is `StopIteration` instead of `None`.
    fn __next__(&self, py: Python<'_>) -> Result<ScrubProgress> {
        match self.next(py)? {
            Some(progress) => Ok(progress),
            None => Err(PyStopIteration::new_err(())),
        }
    }

    /// `with db.scrub() as scan:` — returns the scrub itself.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Releases the pinned generation when the `with` block ends.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> bool {
        self.close(py);
        // False: an exception inside the block propagates.
        false
    }

    fn __repr__(&self) -> String {
        let state = if self.active() { "active" } else { "finished" };
        format!("Scrub({state})")
    }
}
