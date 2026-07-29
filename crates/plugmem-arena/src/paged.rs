//! Page-addressed byte backing with per-page copy-on-write.
//!
//! The page-based structures ([`Arena`](crate::Arena),
//! [`ChunkPool`](crate::ChunkPool)) mutate bytes *in place* — a slot shift, a
//! page split, a chunk write — so they cannot use the append-only tail scheme
//! of [`BlobHeap`](crate::BlobHeap). This backing gives them the same
//! zero-copy open nonetheless: the bytes present at open are **borrowed** from
//! a longer-lived buffer (typically a memory-mapped file), and the first write
//! to a borrowed page copies just *that* page into owned storage (per-page
//! copy-on-write). Pages appended after open live in an owned tail. A pool
//! opened this way is mutated without ever cloning the whole borrowed base —
//! only the touched and freshly grown pages become resident.
//!
//! `PAGE` is the fixed page size in bytes (a const generic so one type serves
//! both the arena's 4 KiB pages and the chunk pool's 64-byte chunks).
//!
//! The overlay stays true to this crate's flat philosophy (`01`):
//! **no `Box`, no map, no tree** — the copy-on-write layer is two flat `Vec`s,
//! a dense `u32` redirect and one contiguous copy pool, exactly like the
//! arena's existing `next`/`counts`/`pool`. A copied page is an amortized push
//! into a single `Vec`, not a per-page allocation. The backing is
//! `no_std`/`alloc`-only and contains no `unsafe`: the owner's growth path
//! keeps its own (measured) allocation code.

use alloc::vec::Vec;

/// Redirect sentinel: a base page not yet copied into the overlay pool.
const NONE: u32 = u32::MAX;

/// A page-addressed byte pool that either owns its bytes or borrows a base and
/// overlays per-page owned copies for the pages that get written.
///
/// Page `idx` occupies the byte range `[idx * PAGE, idx * PAGE + PAGE)` of the
/// logical pool. Reads go through [`Paged::page`]; writes through
/// [`Paged::page_mut`] (which copies a borrowed base page up on first touch);
/// the owner grows the pool by appending to [`Paged::grown_tail_mut`].
pub(crate) struct Paged<'a, const PAGE: usize> {
    inner: Inner<'a, PAGE>,
}

enum Inner<'a, const PAGE: usize> {
    /// Fully owned pool — the writable default (`new`/`load`) and the only
    /// shape on wasm32. Page `idx` is a direct slice of the vector, exactly as
    /// before overlay existed, so this path is unchanged.
    Owned(Vec<u8>),
    /// Borrowed base with a flat copy-on-write overlay (the overlay open):
    /// - `base`: the pages present at open, read-only (an mmap'd image);
    /// - `grown`: pages appended after open (`idx >= base_pages`), addressed
    ///   directly by `idx - base_pages`;
    /// - `over_pool`: a contiguous pool of `PAGE`-sized copies of base pages
    ///   that have been written to;
    /// - `over_slot`: dense per-base-page redirect (`base_pages` long),
    ///   [`NONE`] while the page still lives in `base`, otherwise the page's
    ///   slot in `over_pool`.
    ///
    /// Both overlay `Vec`s are flat — no `Box`, no map — mirroring the arena's
    /// existing dense metadata.
    Cow {
        base: &'a [u8],
        base_pages: usize,
        grown: Vec<u8>,
        over_pool: Vec<u8>,
        over_slot: Vec<u32>,
    },
}

impl<'a, const PAGE: usize> Paged<'a, PAGE> {
    /// An empty owned pool.
    pub(crate) const fn owned_empty() -> Self {
        Self {
            inner: Inner::Owned(Vec::new()),
        }
    }

    /// An owned pool over an existing byte vector (`load`: a copied image).
    pub(crate) fn owned_from(bytes: Vec<u8>) -> Self {
        Self {
            inner: Inner::Owned(bytes),
        }
    }

    /// A pool that borrows `base` and overlays writes (`load_overlay`): no base
    /// byte is copied until a page is written. `base.len()` is a whole number
    /// of pages (the caller validates the dumped image first).
    pub(crate) fn borrowed(base: &'a [u8]) -> Self {
        let base_pages = base.len() / PAGE;
        Self {
            inner: Inner::Cow {
                base,
                base_pages,
                grown: Vec::new(),
                over_pool: Vec::new(),
                over_slot: alloc::vec![NONE; base_pages],
            },
        }
    }

    /// Total bytes of the logical pool (`base` + `grown`, or the owned vector).
    /// The overlay copies in `over_pool` shadow base pages and do not change
    /// it.
    pub(crate) fn len(&self) -> usize {
        match &self.inner {
            Inner::Owned(v) => v.len(),
            Inner::Cow { base, grown, .. } => base.len() + grown.len(),
        }
    }

    /// The `PAGE` bytes of page `idx`, resolving overlay copies and the grown
    /// tail. Bytes beyond the page's live prefix may be uninitialized only for
    /// freshly grown pages — the same contract the owner already upholds — so
    /// callers read no further than the page's occupancy.
    pub(crate) fn page(&self, idx: u32) -> &[u8] {
        match &self.inner {
            Inner::Owned(v) => {
                let start = idx as usize * PAGE;
                &v[start..start + PAGE]
            }
            Inner::Cow {
                base,
                base_pages,
                grown,
                over_pool,
                over_slot,
            } => {
                let idx = idx as usize;
                if idx >= *base_pages {
                    let start = (idx - *base_pages) * PAGE;
                    &grown[start..start + PAGE]
                } else if over_slot[idx] != NONE {
                    let start = over_slot[idx] as usize * PAGE;
                    &over_pool[start..start + PAGE]
                } else {
                    let start = idx * PAGE;
                    &base[start..start + PAGE]
                }
            }
        }
    }

    /// Mutable bytes of page `idx`. If the page is a borrowed base page not yet
    /// overlaid, it is copied into the flat overlay pool first (per-page
    /// copy-on-write) so the borrowed base is never mutated; grown and owned
    /// pages are returned directly.
    pub(crate) fn page_mut(&mut self, idx: u32) -> &mut [u8] {
        match &mut self.inner {
            Inner::Owned(v) => {
                let start = idx as usize * PAGE;
                &mut v[start..start + PAGE]
            }
            Inner::Cow {
                base,
                base_pages,
                grown,
                over_pool,
                over_slot,
            } => {
                let idx = idx as usize;
                if idx >= *base_pages {
                    let start = (idx - *base_pages) * PAGE;
                    &mut grown[start..start + PAGE]
                } else {
                    if over_slot[idx] == NONE {
                        // First write to this base page: append a copy to the
                        // flat overlay pool (amortized, not a per-page alloc)
                        // and record its slot.
                        let slot = over_pool.len() / PAGE;
                        let src = idx * PAGE;
                        over_pool.extend_from_slice(&base[src..src + PAGE]);
                        over_slot[idx] = slot as u32;
                    }
                    let start = over_slot[idx] as usize * PAGE;
                    &mut over_pool[start..start + PAGE]
                }
            }
        }
    }

    /// The owned vector the owner appends fresh pages to: the whole vector when
    /// owned, the grown tail when overlaying. The owner computes the new page
    /// index from [`Paged::len`] *before* growing, then extends this by one
    /// page (its own allocation code — the arena keeps its measured
    /// uninitialized-page `unsafe` here, the chunk pool grows zeroed).
    pub(crate) fn grown_tail_mut(&mut self) -> &mut Vec<u8> {
        match &mut self.inner {
            Inner::Owned(v) => v,
            Inner::Cow { grown, .. } => grown,
        }
    }
}
