//! The [`Slot`] trait: type erasure of fixed-size records to raw bytes.

/// A fixed-size record that can be stored in an [`Arena`](crate::Arena).
///
/// The container itself only ever sees bytes — this trait is the *entire*
/// typed surface. Type erasure is deliberate: the generic container code is
/// instantiated once per slot **size**, not once per rich type, which keeps
/// monomorphization (and therefore wasm binary size) small, and lets binary
/// search and shifting operate on raw byte ranges.
///
/// # Layout contract
///
/// A slot is `SIZE` bytes: the first `KEY_LEN` bytes are the **key prefix**,
/// the rest is payload. Rules:
///
/// - `1 <= KEY_LEN <= SIZE <= PAGE_BYTES` (4096) — checked at
///   [`Arena::new`](crate::Arena::new), violations return
///   [`Error::BadSlot`](crate::Error::BadSlot).
/// - The key prefix must be **order-preserving**: byte-wise comparison of
///   two encoded keys must equal the intended ordering of the records.
///   Encode integers big-endian — use the [`key`](crate::key) helpers.
/// - `write` must fill all `SIZE` bytes (the container never zeroes slots
///   for you — see the crate-level notes on the uninitialized-page
///   optimization); `read` must be its inverse: `read(write(x)) == x`.
///
/// # Example
///
/// ```
/// use plugmem_arena::{key, Slot};
///
/// /// An edge record: `[src | dst]` key, 4-byte payload.
/// struct Edge { src: u32, dst: u32, weight: u32 }
///
/// impl Slot for Edge {
///     const SIZE: usize = 12;
///     const KEY_LEN: usize = 8;
///     fn write(&self, out: &mut [u8]) {
///         key::write_u32(out, self.src);
///         key::write_u32(&mut out[4..], self.dst);
///         key::write_u32(&mut out[8..], self.weight);
///     }
///     fn read(bytes: &[u8]) -> Self {
///         Edge {
///             src: key::read_u32(bytes),
///             dst: key::read_u32(&bytes[4..]),
///             weight: key::read_u32(&bytes[8..]),
///         }
///     }
/// }
/// ```
pub trait Slot {
    /// Total size of the record in bytes.
    const SIZE: usize;

    /// Length of the key prefix; sorting and lookups compare only these
    /// leading bytes. Must not exceed [`SIZE`](Self::SIZE).
    const KEY_LEN: usize;

    /// Serializes the record into `out` (`out.len() == SIZE`), filling every
    /// byte.
    fn write(&self, out: &mut [u8]);

    /// Reconstructs the record from `bytes` (`bytes.len() == SIZE`).
    fn read(bytes: &[u8]) -> Self;
}

/// Byte arrays are slots whose key is the whole value (set semantics) —
/// handy for id/hash sets, mirroring the first test version of this
/// structure which stored 20/32-byte identifiers.
impl<const N: usize> Slot for [u8; N] {
    const SIZE: usize = N;
    const KEY_LEN: usize = N;

    #[inline]
    fn write(&self, out: &mut [u8]) {
        out.copy_from_slice(self);
    }

    #[inline]
    fn read(bytes: &[u8]) -> Self {
        let mut arr = [0u8; N];
        arr.copy_from_slice(bytes);
        arr
    }
}
