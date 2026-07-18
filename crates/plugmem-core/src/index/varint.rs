//! LEB128 varint coding for posting lists (specs/04 §2).
//!
//! Posting entries store fact-id *deltas*; lists are sorted by
//! construction (fact ids are assigned monotonically), so deltas are
//! small and most entries take 1 byte instead of 4.

/// Maximum encoded size of a `u32` (5 × 7 bits ≥ 32 bits).
pub const MAX_VARINT: usize = 5;

/// Encodes `v` into `out`, returning the number of bytes written (1..=5).
pub fn encode_u32(v: u32, out: &mut [u8; MAX_VARINT]) -> usize {
    let mut v = v;
    let mut n = 0;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out[n] = byte;
            return n + 1;
        }
        out[n] = byte | 0x80;
        n += 1;
    }
}

/// Decodes one `u32` from the front of `bytes`, returning the value and
/// the number of bytes consumed. `None` on truncated input or a value
/// that does not fit 32 bits (a 5th byte with more than 4 payload bits,
/// or a continuation bit on the 5th byte).
pub fn decode_u32(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut v = 0u32;
    for (i, &byte) in bytes.iter().take(MAX_VARINT).enumerate() {
        let payload = u32::from(byte & 0x7F);
        if i == MAX_VARINT - 1 && (byte & 0x80 != 0 || payload > 0x0F) {
            return None;
        }
        v |= payload << (7 * i);
        if byte & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}
