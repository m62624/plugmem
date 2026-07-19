//! The recall indexes (specs/04): shared posting storage, the BM25
//! lexical index, and plain id-list indexes (tag → facts,
//! entity → facts) with sorted-list intersection.
//!
//! The temporal index needs no module of its own — it is an Ordered
//! [`Arena`](plugmem_arena::Arena) of
//! [`TemporalSlot`](crate::model::TemporalSlot)s queried with `range`,
//! and lives directly in the engine.

pub mod bm25;
pub mod hnsw;
pub mod postings;
pub mod varint;
pub mod vecpool;

use alloc::vec::Vec;

use crate::id::FactId;
use postings::PostingStore;

/// Key → sorted fact-id lists without term frequencies: the tag and
/// entity indexes.
pub type IdListIndex = PostingStore<false>;

/// Reusable scratch for [`intersect`] (two ping-pong buffers; after
/// warm-up an intersection allocates nothing).
#[derive(Debug, Default)]
pub struct IntersectScratch {
    a: Vec<u32>,
    b: Vec<u32>,
}

impl IntersectScratch {
    /// Empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Intersects the sorted lists of `keys` into `out` (ascending ids).
///
/// Starts from the shortest list (fewest candidates) and filters it
/// against the others by merge — cost is O(shortest + Σ walked prefixes).
/// An absent key means an empty list, so the intersection is empty; an
/// empty `keys` slice yields an empty result (the caller treats "no tag
/// filter" as "no allow-set", not as "allow everything").
pub fn intersect(
    index: &IdListIndex,
    keys: &[u32],
    scratch: &mut IntersectScratch,
    out: &mut Vec<FactId>,
) {
    out.clear();
    let Some((first, rest)) = keys
        .iter()
        .min_by_key(|&&k| index.count(k))
        .map(|&smallest| {
            (
                smallest,
                keys.iter().copied().filter(move |&k| k != smallest),
            )
        })
    else {
        return;
    };
    scratch.a.clear();
    scratch.a.extend(index.entries(first).map(|(id, _)| id.0));
    for key in rest {
        if scratch.a.is_empty() {
            break;
        }
        scratch.b.clear();
        let mut other = index.entries(key).map(|(id, _)| id.0).peekable();
        for &id in &scratch.a {
            while other.peek().is_some_and(|&o| o < id) {
                other.next();
            }
            if other.peek() == Some(&id) {
                scratch.b.push(id);
            }
        }
        core::mem::swap(&mut scratch.a, &mut scratch.b);
    }
    out.extend(scratch.a.iter().map(|&id| FactId(id)));
}
