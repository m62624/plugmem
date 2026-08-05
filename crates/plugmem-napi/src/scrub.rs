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
//! # Why every step is a Promise
//!
//! Not because hashing is slow. Over an mmap the bytes are **paged in from
//! disk** as they are read (they stay reclaimable afterwards, which is why a
//! scrub never residents the whole file), so a step's real cost is a page
//! fault, and its duration is however long that storage takes. Blocking I/O,
//! uninterruptible, of unknown length: exactly what must not happen on the one
//! thread that runs JavaScript. So a step is an `AsyncTask` on a libuv worker,
//! and `next()` returns a `Promise`.
//!
//! That also rules out the synchronous iterator protocol — napi's
//! `#[napi(iterator)]` — whose `next()` runs where it is called.

use std::sync::{Arc, Mutex, MutexGuard};

use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Task};
use napi_derive::napi;

use crate::db::ExportSource;
use crate::error::{self, Produced, Result};
use crate::types::ScrubProgress;

/// Options for `Plugmem#scrub`.
#[napi(object)]
#[derive(Default)]
pub struct ScrubOptions {
    /// Bytes to hash per step, at most. Default: the engine's own (1 MiB).
    ///
    /// This is the knob trading responsiveness for throughput: smaller steps
    /// return to JavaScript more often, larger ones spend less time crossing
    /// back and forth. It changes no answer, only the grain.
    pub budget: Option<f64>,
}

/// A resumable byte-level check of one snapshot generation.
///
/// **Holding this object holds a lock.** It pins the generation it is scanning
/// with a shared lock for its whole life, so the writer's garbage collection
/// cannot reclaim that generation under it. Finish the scan, or `close()` it —
/// do not park one indefinitely.
///
/// One-shot: once it has returned `null`, or thrown, it is done. Ask the
/// database for another to scan again.
#[napi]
pub struct Scrub {
    /// `None` once exhausted, failed, or `close()`d — dropping the host cursor
    /// is what releases the pinned generation, so it is dropped eagerly rather
    /// than left for whenever the JS object is collected.
    cursor: Arc<Mutex<Option<plugmem_host::Scrub>>>,
}

impl Scrub {
    /// Wraps a host cursor. Not a JS constructor: a scrub is obtained from the
    /// database it scans, which is what ties it to a real generation.
    pub(crate) fn new(inner: plugmem_host::Scrub) -> Self {
        Self {
            cursor: Arc::new(Mutex::new(Some(inner))),
        }
    }
}

/// The cursor guard. A panicked step cannot leave a cursor half-advanced (the
/// host iterator is fused and owns its own position), so a poisoned lock is
/// recovered — the same rule the engine lock follows.
fn locked(
    cursor: &Mutex<Option<plugmem_host::Scrub>>,
) -> MutexGuard<'_, Option<plugmem_host::Scrub>> {
    cursor.lock().unwrap_or_else(|e| e.into_inner())
}

#[napi]
impl Scrub {
    /// Hashes the next slice and resolves with the progress so far, or `null`
    /// when the scan is complete.
    ///
    /// **Async**: see the module note — a step is disk I/O, not arithmetic.
    ///
    /// Concurrent calls are serialized rather than refused: the cursor has one
    /// position, so two overlapping steps would be two different slices of the
    /// same scan, which is what a caller pacing it from two places asked for.
    ///
    /// @throws on the first mismatch, naming what failed its checksum. The
    /// scrub is finished at that point; a further call resolves `null`.
    #[napi(ts_return_type = "Promise<ScrubProgress | null>")]
    pub fn next(&self) -> AsyncTask<ScrubStepTask> {
        AsyncTask::new(ScrubStepTask {
            cursor: Arc::clone(&self.cursor),
        })
    }

    /// Releases the pinned generation now, abandoning an unfinished scan.
    ///
    /// Idempotent, and the same bargain as `Plugmem#close`: waiting for the
    /// garbage collector to do it means the writer cannot reclaim that
    /// generation until it happens.
    #[napi]
    pub fn close(&mut self) {
        *locked(&self.cursor) = None;
    }

    /// Whether this scrub still has work — `false` once it finished, failed or
    /// was closed. Cheap: it inspects the handle, not the file.
    #[napi]
    pub fn active(&self) -> bool {
        locked(&self.cursor).is_some()
    }
}

/// The libuv-thread body of [`Scrub::next`].
pub struct ScrubStepTask {
    cursor: Arc<Mutex<Option<plugmem_host::Scrub>>>,
}

impl Task for ScrubStepTask {
    type Output = Produced<Option<plugmem_host::ScrubProgress>>;
    type JsValue = Option<ScrubProgress>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let mut slot = locked(&self.cursor);
        let Some(cursor) = slot.as_mut() else {
            return Ok(Ok(None));
        };
        Ok(match cursor.next() {
            Some(Ok(progress)) => Ok(Some(progress)),
            // Exhausted or failed: either way this cursor is spent, and
            // dropping it here is what unpins the generation — the caller
            // should not have to remember to `close()` a scan that ended.
            Some(Err(e)) => {
                *slot = None;
                Err(error::engine(e))
            }
            None => {
                *slot = None;
                Ok(None)
            }
        })
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output
            .map_err(|e| error::to_js(env, e))?
            .map(ScrubProgress::from))
    }
}

/// The libuv-thread body of `Plugmem#scrub`: pinning the generation, mapping it
/// and parsing its section table are all file work.
pub struct ScrubOpenTask {
    source: ExportSource,
    budget: usize,
}

impl ScrubOpenTask {
    /// Checks the arguments and builds the not-yet-scheduled task.
    pub(crate) fn new(source: ExportSource, options: Option<ScrubOptions>) -> Result<Self> {
        let budget = match options.and_then(|o| o.budget) {
            None => plugmem_host::DEFAULT_SCRUB_BUDGET,
            Some(b) if b.is_finite() && b >= 1.0 => b as usize,
            Some(b) => {
                return Err(error::invalid_arg(format!(
                    "budget must be a finite number of bytes >= 1, got {b}"
                )));
            }
        };
        Ok(Self { source, budget })
    }
}

impl Task for ScrubOpenTask {
    type Output = Produced<plugmem_host::Scrub>;
    type JsValue = Scrub;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(match &self.source {
            ExportSource::Writer(db) => db.scrub_with_budget(self.budget),
            ExportSource::Reader(db) => db.scrub_with_budget(self.budget),
        }
        .map_err(error::open))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(Scrub::new(output.map_err(|e| error::to_js(env, e))?))
    }
}
