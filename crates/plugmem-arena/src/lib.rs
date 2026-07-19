//! Flat byte-pool storage structures for plugmem.
//!
//! This crate is the storage foundation of the plugmem engine, but it is
//! deliberately generic: nothing here knows about facts, vectors or LLMs.
//! If you need a compact, allocation-frugal, `no_std` sorted container whose
//! in-memory representation *is* its serialized form, you can lift it into
//! your own project as-is — see `examples/` for self-contained walkthroughs.
//!
//! # Philosophy
//!
//! 1. **State is flat bytes.** A container is one contiguous byte pool plus a
//!    few small metadata arrays. No per-element allocations, no pointers, no
//!    `Box`/`Rc` graphs. Persisting a container is `memcpy`; loading it back
//!    is bounds-checking the metadata and adopting the bytes.
//! 2. **Costs are local and visible.** Every operation touches one 4 KiB
//!    page (one cache-friendly unit). Worst cases are small, fixed and
//!    measured — the optional [`counters`](#feature-flags) feature exposes
//!    deterministic work counters used as CI performance gates.
//! 3. **Keys are big-endian.** Byte-wise comparison of an encoded key must
//!    equal the numeric comparison of its source value, so binary search and
//!    ordered iteration work directly on raw bytes. Helpers live in [`key`].
//!
//! # The four structures
//!
//! | Structure | Shape | Typical use |
//! |---|---|---|
//! | [`Arena`] | sorted fixed-size records, sharded 4 KiB pages | primary record store, ordered indexes |
//! | [`BlobHeap`] | append-only variable-length blobs, dense ids | texts, names, raw vectors |
//! | [`ChunkPool`] | many small growable lists over 64-byte chunks | posting lists, adjacency lists |
//! | [`Interner`] | string -> dense `u32` (heap + flat hash table) | terms, tags, entity names |
//!
//! # Quick start
//!
//! ```
//! use plugmem_arena::{Arena, ArenaCfg, ShardMode, Slot, key};
//!
//! /// A tiny fixed-size record: 4-byte big-endian key + 1-byte payload.
//! #[derive(Debug, PartialEq)]
//! struct Rec {
//!     id: u32,
//!     level: u8,
//! }
//!
//! impl Slot for Rec {
//!     const SIZE: usize = 5;
//!     const KEY_LEN: usize = 4;
//!     fn write(&self, out: &mut [u8]) {
//!         key::write_u32(out, self.id);
//!         out[4] = self.level;
//!     }
//!     fn read(bytes: &[u8]) -> Self {
//!         Rec { id: key::read_u32(bytes), level: bytes[4] }
//!     }
//! }
//!
//! let mut arena = Arena::<Rec>::new(ArenaCfg::new(64, ShardMode::Ordered)).unwrap();
//! arena.insert(&Rec { id: 7, level: 3 }).unwrap();
//! arena.insert(&Rec { id: 1, level: 9 }).unwrap();
//!
//! let mut key_buf = [0u8; 4];
//! key::write_u32(&mut key_buf, 7);
//! assert_eq!(arena.get(&key_buf), Some(Rec { id: 7, level: 3 }));
//!
//! // Ordered mode: iteration yields ascending keys across all shards.
//! let ids: Vec<u32> = arena.iter().map(|r| r.id).collect();
//! assert_eq!(ids, [1, 7]);
//! ```
//!
//! # The one `unsafe`
//!
//! The single default `unsafe` in this crate is page allocation without
//! zeroing (`Vec::reserve` + `set_len`). It is kept because it was
//! *measured*, not assumed: on the wasm target (our primary portability
//! target) zeroing freshly grown pages made the allocation path **12x
//! slower** (wasmtime, 32k pages: 3889 us zeroed vs 316 us uninit), while on
//! native x86-64 the difference is noise. The safety invariant is simple and
//! local: *bytes of a page beyond `count * Slot::SIZE` are never read* —
//! every read is bounded by the per-shard element count, and a slot is fully
//! written before `count` is incremented. See `Arena::ensure_page` for the
//! full safety comment. A consequence worth knowing: `Arena` intentionally
//! implements neither `Clone` nor `PartialEq`, because a byte-wise clone or
//! comparison would read those uninitialized tails.
//!
//! Bounds-check elimination (`get_unchecked`) was measured on the same
//! harness and rejected: <= 1% on native, *slower* under wasm (the runtime
//! bounds-checks linear memory anyway). Safe indexing everywhere else.
//!
//! # Feature flags
//!
//! - `std` *(default)* — nothing yet beyond linking `std` for consumers'
//!   convenience; the crate is fully functional as `no_std + alloc`.
//! - `counters` — deterministic work counters (`Counters`) on every
//!   container: key comparisons, bytes shifted, pages allocated. Zero cost
//!   when disabled (the increments compile away).
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod key;

mod arena;
mod blob;
mod chunk;
mod error;
mod interner;
mod slot;

pub use arena::{Arena, ArenaCfg, Iter, PAGE_BYTES, ShardMode};
pub use blob::{BlobHeap, BlobHeapCfg, BlobId};
pub use chunk::{CHUNK_BYTES, CHUNK_PAYLOAD, ChunkIter, ChunkPool, ChunkPoolCfg, ListHandle};
pub use error::Error;
pub use interner::{Interner, TermId};
pub use slot::Slot;

#[cfg(feature = "counters")]
pub use arena::Counters;
