//! Flat byte-pool storage structures for plugmem.
//!
//! Design: `specs/01-arena.md`. Implementation lands in stage 1.
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;
