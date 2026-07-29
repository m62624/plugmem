//! Big-endian key encoding helpers.
//!
//! Every sorted structure in this crate compares keys **byte-wise**
//! (`[u8]::cmp`). For that comparison to match the numeric order of the
//! encoded values, integers must be stored **big-endian**: the most
//! significant byte comes first, so lexicographic byte order equals numeric
//! order. (Little-endian encodings are only usable for equality lookups —
//! this is a classic pitfall.)
//!
//! Composite keys are built by concatenation: encode the most significant
//! component first. For example a temporal index key `(timestamp, id)` is
//! `write_u64(&mut buf[..8], ts)` followed by `write_u32(&mut buf[8..], id)`
//! — byte order then sorts by timestamp first, id second.

/// Encodes `v` big-endian into the first 4 bytes of `out`.
///
/// # Panics
///
/// Panics if `out.len() < 4`.
#[inline]
pub fn write_u32(out: &mut [u8], v: u32) {
    out[..4].copy_from_slice(&v.to_be_bytes());
}

/// Decodes a big-endian `u32` from the first 4 bytes of `bytes`.
///
/// # Panics
///
/// Panics if `bytes.len() < 4`.
#[inline]
pub fn read_u32(bytes: &[u8]) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[..4]);
    u32::from_be_bytes(b)
}

/// Encodes `v` big-endian into the first 8 bytes of `out`.
///
/// # Panics
///
/// Panics if `out.len() < 8`.
#[inline]
pub fn write_u64(out: &mut [u8], v: u64) {
    out[..8].copy_from_slice(&v.to_be_bytes());
}

/// Decodes a big-endian `u64` from the first 8 bytes of `bytes`.
///
/// # Panics
///
/// Panics if `bytes.len() < 8`.
#[inline]
pub fn read_u64(bytes: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[..8]);
    u64::from_be_bytes(b)
}

/// Encodes the composite key `(hi, lo)` big-endian into the first 12 bytes
/// of `out` (`hi` first — it is the more significant component).
///
/// This is the layout of e.g. the temporal index key
/// `[recorded_at: u64 | fact_id: u32]`.
///
/// # Panics
///
/// Panics if `out.len() < 12`.
#[inline]
pub fn write_pair(out: &mut [u8], hi: u64, lo: u32) {
    write_u64(out, hi);
    write_u32(&mut out[8..], lo);
}

/// Decodes a composite key previously encoded with [`write_pair`].
///
/// # Panics
///
/// Panics if `bytes.len() < 12`.
#[inline]
pub fn read_pair(bytes: &[u8]) -> (u64, u32) {
    (read_u64(bytes), read_u32(&bytes[8..]))
}
