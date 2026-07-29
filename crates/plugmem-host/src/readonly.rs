//! [`ReadOnlyDatabase`]: a zero-copy read-only open over an mmap'd
//! snapshot (specs/16 §3).
//!
//! A normal [`Database`](crate::Database) open reads the whole snapshot
//! into RAM (every byte pool is copied into an arena). For a large,
//! read-mostly database that is wasteful: `open_readonly` maps the
//! snapshot file instead and lets the engine's byte pools *borrow* the
//! mapped pages, so the OS residents only the bytes `recall`/`get`
//! actually touch. An 8 GiB database opens in milliseconds with a few
//! pages resident, not 8 GiB.
//!
//! The handle is read-only by construction — it exposes `recall`/`get`/
//! `stats` and nothing that mutates. It requires a *checkpointed*
//! database (empty journal): replaying a journal would copy whole arenas
//! up (copy-on-write) and defeat the zero-copy intent, so a non-empty
//! journal is refused with [`HostError::NeedsCheckpoint`].
//!
//! Locking is a **shared** advisory lock held for the handle's whole life
//! (specs/13 §3): many read-only handles — in this process or others — map
//! the same file at once, so a read-mostly database serves concurrent
//! readers. A shared lock still excludes every exclusive (read-write)
//! owner, so no cooperating process writes or truncates the file while it
//! is mapped — which is exactly the safety argument for the mmap (see the
//! `unsafe` block in [`ReadOnlyDatabase::open`]).
//!
//! # When you actually need this — [`Database`] vs [`ReadOnlyDatabase`]
//!
//! Most callers do **not** need a read-only handle. The distinction is about
//! **who else has the file open**, where "who else" means a **separate OS
//! process** — a different running program (a different PID): a second copy of
//! the CLI, an MCP server, another service — *not* another thread or another
//! `Database` value inside your own program.
//!
//! - **One process reads and writes → just [`Database::open`](crate::Database::open).**
//!   A read-write handle keeps an *overlay* (the mapped snapshot plus the journal
//!   replayed in RAM), so `remember` is visible to the very next `recall` on that
//!   same handle, with no checkpoint and no second open. This is **read-your-writes**:
//!   an agent that stores a fact and immediately recalls it needs one handle and
//!   sees its own write instantly. Opening the same database *twice* from one
//!   process — once read-write, once read-only — is pointless and is **not** how
//!   you get freshness; it only costs you a stale second view.
//!
//! - **Another process must read the same file while a writer is live →
//!   [`Database::open_readonly`](crate::Database::open_readonly).** A separate
//!   program cannot share the writer's in-RAM overlay (it is another address
//!   space entirely), so it maps the last *published* generation instead. Such a
//!   handle is a **point-in-time snapshot**: it observes the database "as of the
//!   last checkpoint" and never moves on its own — the writer publishing a newer
//!   generation does not disturb the snapshot you are already reading. To advance
//!   to a freshly published generation, call [`ReadOnlyDatabase::refresh`], which
//!   is a cheap 24-byte manifest read that re-maps only when the writer has
//!   actually published something newer (see its docs).
//!
//! In short: `refresh`, `open_readonly`, and snapshot-isolation lag exist **only**
//! for a reader looking at *another process's* writer. Within a single process,
//! [`Database`] alone is always fresh.

use std::cell::RefCell;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(feature = "counters")]
use std::sync::Mutex;

use memmap2::Mmap;
use plugmem_core::snapshot::{DEFAULT_SCRUB_BUDGET, ScrubCursor, ScrubProgress, Snapshot};
use plugmem_core::{Config, FactId, Memory, RecallQuery, RecallResult, RecallScratch, Stats};

thread_local! {
    /// Per-thread recall scratch — the read-only analog of the one in
    /// [`crate::db`]. `recall` borrows the mapped engine shared (`&Memory`), so
    /// many threads recall one handle at once, each reusing its own scratch.
    static RECALL_SCRATCH: RefCell<RecallScratch> = RefCell::new(RecallScratch::new());
}

use crate::db::FactSnapshot;
use crate::error::HostError;
use crate::storage::{pin_current_generation, read_manifest};

self_cell::self_cell!(
    /// Owns the memory map and the [`Memory`] that borrows it. `self_cell`
    /// keeps the self-reference safe: the only `unsafe` on this path is
    /// the inherent mmap call, not the borrow.
    struct MappedMemory {
        owner: Mmap,
        #[covariant]
        dependent: BorrowedMemory,
    }
);

/// The dependent type constructor `self_cell` reborrows per access.
/// [`Memory`] is covariant in its lifetime (its byte pools are
/// `Cow<'a, [u8]>`), so borrowing the map is sound.
type BorrowedMemory<'a> = Memory<'a>;

/// A read-only database handle backed by a memory-mapped snapshot
/// (specs/16). See the module docs. `Send + Sync` — share it across
/// threads behind a reference or an `Arc`.
pub struct ReadOnlyDatabase {
    /// The map and the engine borrowing it. Normally no lock: every verb
    /// borrows it shared (`&Memory`) — `recall` keeps its mutable scratch
    /// per-thread — so many threads read one handle concurrently. Under
    /// `counters` the engine embeds the arena's non-`Sync` counter `Cells`, so
    /// it is wrapped in a `Mutex` to stay `Sync` (readers serialize — fine for
    /// that single-threaded perf build). Purely internal: the public API is the
    /// same under every feature.
    #[cfg(not(feature = "counters"))]
    mapped: MappedMemory,
    #[cfg(feature = "counters")]
    mapped: Mutex<MappedMemory>,
    /// Holds a **shared** lock on the mapped generation file for this handle's
    /// whole life — never read, but it *pins* the generation against the
    /// writer's GC (the writer's exclusive try-lock fails while we hold this),
    /// so the immutable snapshot we borrow can never be reclaimed under us.
    _pin: File,
    /// The database base (manifest) path.
    path: PathBuf,
    /// The generation number this handle is pinned to — the snapshot it maps.
    /// Compared against the manifest by [`ReadOnlyDatabase::refresh`] to tell
    /// whether the writer has published anything newer.
    generation: u64,
    /// Kept so [`ReadOnlyDatabase::refresh`] can rebuild the borrowed engine
    /// over a freshly mapped generation with the same configuration.
    cfg: Config,
}

impl ReadOnlyDatabase {
    /// Opens the database at `path` read-only over an mmap (specs/16 §3).
    ///
    /// # Errors
    ///
    /// [`HostError::NeedsCheckpoint`] when the database has no published
    /// snapshot generation yet (checkpoint it once, then retry); [`HostError::Io`]
    /// when the generation file cannot be mapped; [`HostError::Engine`] for a
    /// corrupt image or a config mismatch.
    pub(crate) fn open(path: impl Into<PathBuf>, cfg: Config) -> Result<Self, HostError> {
        let base: PathBuf = path.into();
        // Pin the current generation with a shared lock (no writer lock — a
        // reader coexists with the writer). The reader maps this immutable
        // generation and ignores the journal, which belongs to the *next*
        // generation the writer is building: this is the snapshot-isolation
        // reader, "as of the last published checkpoint".
        let Some((pin, genp, generation)) = pin_current_generation(&base)? else {
            // No published generation yet — checkpoint the database first.
            return Err(HostError::NeedsCheckpoint { path: base });
        };

        // SAFETY: mapping a file is inherently unsafe — a concurrent truncate or
        // overwrite would fault the process on the next page access. Our
        // argument (specs/16 §5): a generation file is **immutable** (a
        // checkpoint publishes a *new* generation, never rewrites this one), and
        // `pin` holds a shared lock on it for this handle's whole life, so the
        // writer's GC cannot reclaim it under us. A foreign `truncate`/`rm` is
        // out of contract — the same caveat as corrupting any live database file.
        let map = unsafe { Mmap::map(&pin) }.map_err(|e| HostError::io(&genp, e))?;
        let mapped = MappedMemory::try_new(map, |map| {
            Memory::from_bytes_borrowed(&map[..], &[], cfg.clone())
        })?;

        Ok(Self {
            #[cfg(not(feature = "counters"))]
            mapped,
            #[cfg(feature = "counters")]
            mapped: Mutex::new(mapped),
            _pin: pin,
            path: base,
            generation,
            cfg,
        })
    }

    /// Runs `f` over the mapped engine (`&Memory`). Normally a lock-free shared
    /// borrow (concurrent readers); under `counters` it takes the `Mutex` first.
    /// Private — the lock strategy never reaches the public API.
    #[cfg(not(feature = "counters"))]
    fn with_mem<R>(&self, f: impl FnOnce(&Memory<'_>) -> R) -> R {
        f(self.mapped.borrow_dependent())
    }

    #[cfg(feature = "counters")]
    fn with_mem<R>(&self, f: impl FnOnce(&Memory<'_>) -> R) -> R {
        let guard = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
        f(guard.borrow_dependent())
    }

    /// Runs a recall (specs/04). Same semantics as
    /// [`Database::recall`](crate::Database::recall) minus the embedder:
    /// a text-only query is not auto-embedded, so pass a vector for the
    /// vector source.
    pub fn recall(&self, q: RecallQuery<'_>) -> Result<RecallResult, HostError> {
        self.with_mem(|mem| {
            RECALL_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let mut out = RecallResult::default();
                mem.recall_into(q, &mut scratch, &mut out)?;
                Ok(out)
            })
        })
    }

    /// An owned copy of one fact, or `None` for unknown/tombstoned ids.
    pub fn get(&self, id: FactId) -> Option<FactSnapshot> {
        self.with_mem(|mem| {
            mem.get(id).map(|v| FactSnapshot {
                record: v.record,
                text: v.text.to_string(),
                metadata: crate::db::metadata_map(mem, id),
            })
        })
    }

    /// Engine size counters.
    pub fn stats(&self) -> Stats {
        self.with_mem(|mem| mem.stats())
    }

    /// Runs the on-demand integrity check (specs/16 §9) — the equivalent of
    /// SQLite's `integrity_check`. A read-only open validates only the metadata
    /// (the mapped text and vector pools stay non-resident); this sweeps them
    /// and reports any latent corruption. Reads the whole image, so it residents
    /// the pools it checks.
    ///
    /// # Errors
    ///
    /// [`HostError::Engine`] for the first inconsistency found.
    pub fn verify(&self) -> Result<(), HostError> {
        Ok(self.with_mem(|mem| mem.verify())?)
    }

    /// A resumable byte-level container scrub of the snapshot file, with the
    /// default slice budget (specs/16 §9 — the ZFS-scrub model). See
    /// [`Scrub`] and [`ReadOnlyDatabase::scrub_with_budget`].
    ///
    /// # Errors
    ///
    /// [`HostError::Locked`]/[`HostError::Io`]/[`HostError::Engine`] if the
    /// file cannot be locked, mapped, or structurally parsed for the scan.
    pub fn scrub(&self) -> Result<Scrub, HostError> {
        self.scrub_with_budget(DEFAULT_SCRUB_BUDGET)
    }

    /// A resumable container scrub hashing at most `budget` bytes per
    /// [`Iterator::next`] (specs/16 §9).
    ///
    /// The returned [`Scrub`] owns its own map and its own shared advisory
    /// lock over the same file, so it holds a reader's lock for its whole
    /// life (a writer is refused with [`HostError::Locked`] while any scrub
    /// or read-only handle lives) and can be moved to its own thread — the
    /// caller paces the scan (`next`, pause, resume, cancel) exactly like
    /// the core [`ScrubCursor`]. Dropping it releases the lock.
    ///
    /// It is independent of `self`: the scrub keeps running after this handle
    /// is dropped. A non-empty journal is not an obstacle — the scrub checks
    /// the on-disk snapshot container as-is.
    ///
    /// # Errors
    ///
    /// As [`ReadOnlyDatabase::scrub`].
    pub fn scrub_with_budget(&self, budget: usize) -> Result<Scrub, HostError> {
        Scrub::open(&self.path, budget)
    }

    /// Dumps the currently-open facts for a human-readable backup
    /// (specs/06). See [`ExportedFact`](crate::ExportedFact). Collects the whole
    /// set; for a large database prefer [`export_each`](Self::export_each).
    pub fn export(&self) -> Vec<crate::db::ExportedFact> {
        self.with_mem(crate::db::export_facts)
    }

    /// Streams the currently-open facts, calling `f` once per fact under the map
    /// — the whole dump is never materialized (the zero-copy analog of
    /// [`Database::export_each`](crate::Database::export_each)).
    pub fn export_each(&self, f: impl FnMut(crate::db::ExportedFact)) {
        self.with_mem(|mem| crate::db::export_facts_each(mem, f));
    }

    /// The database base path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The snapshot generation this handle is pinned to — the point in time it
    /// reads "as of". Monotonic: a writer's checkpoint publishes a strictly
    /// higher number. Compare it against a later call, or drive your own
    /// freshness policy around [`refresh`](Self::refresh) with it.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Advances this handle to the writer's latest published generation, if
    /// there is a newer one. Returns `true` when it re-mapped onto a newer
    /// snapshot (subsequent reads now observe it), `false` when nothing changed.
    ///
    /// This is the **only** way a read-only handle moves forward in time: an
    /// open handle is a point-in-time snapshot and never advances on its own
    /// (see the module docs). It exists for a reader watching **another
    /// process's** writer; a single process that reads and writes uses one
    /// [`Database`](crate::Database) handle and sees its own writes instantly,
    /// with no `refresh` at all.
    ///
    /// It is cheap to call speculatively — the freshness check is a read of the
    /// tiny fixed-size manifest (a handful of bytes), and the `mmap` re-map
    /// happens *only* when the writer has actually published a newer generation.
    /// In steady state (no new checkpoint) it does no mapping and returns `false`
    /// for the cost of that manifest read, so calling it before each read is a
    /// reasonable "always fresh" policy; batching (refresh every N reads, or on a
    /// timer) trades a bounded staleness for even fewer manifest reads. Re-mapping
    /// borrows the new generation's pages lazily — no whole-file copy, no journal
    /// replay, no index rebuild — and drops the old map, so RAM does not grow.
    ///
    /// The freshness policy is intentionally left to the caller: an autorefresh
    /// baked into every read would forfeit snapshot isolation for callers who
    /// need a *stable* view across a series of queries. Keep the reader stable by
    /// not calling this; advance it by calling it.
    ///
    /// # Errors
    ///
    /// [`HostError::Io`] if the newer generation cannot be mapped;
    /// [`HostError::Engine`] for a corrupt image. On any error the handle is
    /// left untouched on its current generation (the re-map is built before it
    /// replaces the live one).
    pub fn refresh(&mut self) -> Result<bool, HostError> {
        // Cheap detect: read the fixed-size manifest and bail unless the writer
        // has published a strictly newer generation.
        match read_manifest(&self.path)? {
            Some(latest) if latest > self.generation => {}
            _ => return Ok(false),
        }
        // Pin and map the current published generation. `pin_current_generation`
        // re-reads the manifest and retries the GC race, so the pinned number is
        // the freshest one on disk — which may even exceed the value we just
        // read. If it is not actually newer than ours (a checkpoint raced back,
        // impossible given monotonicity but cheap to guard), report no change.
        let Some((pin, genp, generation)) = pin_current_generation(&self.path)? else {
            return Ok(false);
        };
        if generation <= self.generation {
            return Ok(false);
        }
        // SAFETY: identical to `open` — a generation file is immutable (a
        // checkpoint publishes a *new* generation, never rewrites this one), and
        // `pin` holds a shared lock on it for as long as we keep it, so the
        // writer's GC cannot reclaim it under us. Built before we swap it in, so
        // a failure leaves the live map intact.
        let map = unsafe { Mmap::map(&pin) }.map_err(|e| HostError::io(&genp, e))?;
        let cfg = self.cfg.clone();
        let mapped =
            MappedMemory::try_new(map, |map| Memory::from_bytes_borrowed(&map[..], &[], cfg))?;
        // Commit: replace the map (dropping the old one and its pin) and record
        // the new generation. The old `_pin`'s shared lock releases here, letting
        // GC reclaim the generation we just left once nothing else pins it.
        #[cfg(not(feature = "counters"))]
        {
            self.mapped = mapped;
        }
        #[cfg(feature = "counters")]
        {
            self.mapped = Mutex::new(mapped);
        }
        self._pin = pin;
        self.generation = generation;
        Ok(true)
    }
}

impl std::fmt::Debug for ReadOnlyDatabase {
    /// Summary only — the contents are the user's memory.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ReadOnlyDatabase")
            .field("path", &self.path)
            .field("facts", &stats.facts)
            .field("entities", &stats.entities)
            .finish()
    }
}

self_cell::self_cell!(
    /// Owns the memory map and the [`ScrubCursor`] that borrows it. As with
    /// [`MappedMemory`], the only `unsafe` is the inherent mmap call, not the
    /// self-reference.
    struct MappedScrub {
        owner: Mmap,
        #[covariant]
        dependent: BorrowedScrub,
    }
);

/// The dependent type constructor. [`ScrubCursor`] is covariant in its
/// lifetime (it borrows the mapped bytes as `&'a [u8]` and owns the rest),
/// so borrowing the map is sound.
type BorrowedScrub<'a> = ScrubCursor<'a>;

/// A resumable, byte-level container scrub over a memory-mapped snapshot
/// (specs/16 §9 — the ZFS-scrub model). Obtained from
/// [`ReadOnlyDatabase::scrub`].
///
/// It implements [`Iterator`]: each [`Iterator::next`] hashes up to the slice
/// budget and yields `Ok(ScrubProgress)`, verifying each section's stored
/// xxh3 as its body completes and the whole-file hash at EOF; the first
/// mismatch yields `Err(HostError::Engine(Error::Corrupt(..)))` and then
/// `None` (fused). Because it only reads the mapped bytes linearly, the pages
/// fault in, get hashed and stay reclaimable — a scrub never residents the
/// whole file.
///
/// It pins its generation with a shared lock for its whole life (independent of
/// the handle it came from), so the writer's GC cannot reclaim it while it runs.
/// It is [`Send`] — pace it on its own thread. One-shot: obtain a new scrub to
/// scan again.
pub struct Scrub {
    mapped: MappedScrub,
    /// Holds the shared lock on the scrubbed generation for the scrub's whole
    /// life (never read — the pin is the point), independent of the handle.
    _pin: File,
}

impl Scrub {
    /// Pins and maps the current generation at `base`, then builds the cursor.
    /// See [`ReadOnlyDatabase::scrub_with_budget`].
    fn open(base: &Path, budget: usize) -> Result<Self, HostError> {
        // Pin the current generation with a shared lock (coexists with other
        // readers and the writer; blocks only the writer's GC of this one).
        let Some((pin, genp, _generation)) = pin_current_generation(base)? else {
            return Err(HostError::NeedsCheckpoint {
                path: base.to_path_buf(),
            });
        };

        // SAFETY: identical to `ReadOnlyDatabase::open` — a generation file is
        // immutable, and `pin` holds a shared lock on it for this scrub's whole
        // life, so GC cannot reclaim it under the map (specs/16 §5).
        let map = unsafe { Mmap::map(&pin) }.map_err(|e| HostError::io(&genp, e))?;

        let mapped = MappedScrub::try_new(map, |map| {
            Snapshot::parse(&map[..])
                .map(|snap| snap.scrub_with_budget(budget))
                .map_err(HostError::from)
        })?;

        Ok(Self { mapped, _pin: pin })
    }
}

impl Iterator for Scrub {
    type Item = Result<ScrubProgress, HostError>;

    /// Hashes the next slice, mapping a core [`Error`](plugmem_core::Error)
    /// mismatch into [`HostError::Engine`]. `None` once complete or fused.
    fn next(&mut self) -> Option<Self::Item> {
        self.mapped
            .with_dependent_mut(|_map, cur| cur.next())
            .map(|step| step.map_err(HostError::from))
    }
}

impl std::fmt::Debug for Scrub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scrub").finish_non_exhaustive()
    }
}
