//! Canonical encoding of a fact's optional metadata: a small key→value map of
//! UTF-8 strings, stored as one opaque blob per fact in the `metas` heap
//!
//! The engine never interprets metadata — it is a place for pointers and
//! attributes (a URI to rehydrate the real payload from another store, a mime
//! type, an external key). This module owns the **byte format contract**; every
//! higher layer (host, CLI, MCP, napi) works in key→value pairs and never sees
//! these bytes.
//!
//! # Layout
//!
//! ```text
//! [count: u32 LE]
//! repeated count times, keys STRICTLY ascending (raw UTF-8 bytes), no dups:
//!   [klen: u32 LE][key bytes UTF-8]
//!   [vlen: u32 LE][val bytes UTF-8]
//! ```
//!
//! # One canonical order
//!
//! Order is decided in exactly one place — [`encode`] sorts the keys ascending
//! and rejects duplicates. Every reader ([`decode`], and through it
//! [`Memory::metadata_of`](crate::Memory::metadata_of) and
//! [`Memory::verify`](crate::Memory::verify)) walks the blob in stored order, so
//! the pairs come back in that same ascending order at every layer. Nothing
//! sorts a second time, so core and host can never disagree on order or count.

use alloc::vec::Vec;

use crate::error::Error;

/// Width of a `u32` length prefix.
const U32: usize = 4;

/// Encodes key→value `pairs` into one canonical metadata blob: keys sorted
/// ascending (raw UTF-8 bytes), duplicates rejected. Callers pass pairs in any
/// order — this is the single place order is fixed.
///
/// # Errors
///
/// [`Error::Invalid`] if two pairs share a key.
pub(crate) fn encode(pairs: &[(&str, &str)]) -> Result<Vec<u8>, Error> {
    // Sort a view by key without disturbing the caller's slice.
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by(|&a, &b| pairs[a].0.as_bytes().cmp(pairs[b].0.as_bytes()));
    for w in order.windows(2) {
        if pairs[w[0]].0 == pairs[w[1]].0 {
            return Err(Error::Invalid("duplicate metadata key"));
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
    for &i in &order {
        let (k, v) = pairs[i];
        out.extend_from_slice(&(k.len() as u32).to_le_bytes());
        out.extend_from_slice(k.as_bytes());
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        out.extend_from_slice(v.as_bytes());
    }
    Ok(out)
}

/// Decodes a metadata blob into `out` (cleared first), validating the whole
/// image: bounds, UTF-8, strictly ascending unique keys, and no trailing bytes.
/// The pairs borrow `bytes`, so they come back in stored (ascending) order.
///
/// Untrusted input — a malformed blob is [`Error::Corrupt`], never a panic.
pub(crate) fn decode<'a>(bytes: &'a [u8], out: &mut Vec<(&'a str, &'a str)>) -> Result<(), Error> {
    out.clear();
    let mut at = 0usize;
    let count = take_u32(bytes, &mut at)?;
    let mut prev: Option<&str> = None;
    for _ in 0..count {
        let k = take_str(bytes, &mut at)?;
        let v = take_str(bytes, &mut at)?;
        match prev {
            Some(p) if k.as_bytes() <= p.as_bytes() => {
                return Err(Error::Corrupt("metadata keys are not strictly ascending"));
            }
            _ => {}
        }
        prev = Some(k);
        out.push((k, v));
    }
    if at != bytes.len() {
        return Err(Error::Corrupt("metadata blob has trailing bytes"));
    }
    Ok(())
}

/// Reads a `u32 LE`; advances `at`. Bounds-checked.
fn take_u32(bytes: &[u8], at: &mut usize) -> Result<u32, Error> {
    let end = at
        .checked_add(U32)
        .filter(|&e| e <= bytes.len())
        .ok_or(Error::Corrupt("metadata blob truncated inside a field"))?;
    let v = u32::from_le_bytes(bytes[*at..end].try_into().unwrap());
    *at = end;
    Ok(v)
}

/// Reads a length-prefixed UTF-8 string; advances `at`. Bounds- and
/// UTF-8-checked.
fn take_str<'a>(bytes: &'a [u8], at: &mut usize) -> Result<&'a str, Error> {
    let len = take_u32(bytes, at)? as usize;
    let end = at
        .checked_add(len)
        .filter(|&e| e <= bytes.len())
        .ok_or(Error::Corrupt("metadata string overruns its blob"))?;
    let s = core::str::from_utf8(&bytes[*at..end])
        .map_err(|_| Error::Corrupt("metadata string is not UTF-8"))?;
    *at = end;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn encode_sorts_keys_and_round_trips() {
        // Deliberately unsorted input: the encoder fixes the order.
        let pairs = [
            ("uri", "s3://b/x"),
            ("mime", "application/pdf"),
            ("page", "3"),
        ];
        let bytes = encode(&pairs).unwrap();
        let mut out = Vec::new();
        decode(&bytes, &mut out).unwrap();
        assert_eq!(
            out,
            vec![
                ("mime", "application/pdf"),
                ("page", "3"),
                ("uri", "s3://b/x"),
            ]
        );
    }

    #[test]
    fn empty_map_encodes_to_a_count_of_zero() {
        let bytes = encode(&[]).unwrap();
        assert_eq!(bytes, 0u32.to_le_bytes());
        let mut out = vec![("stale", "stale")];
        decode(&bytes, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let err = encode(&[("k", "a"), ("k", "b")]).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }

    #[test]
    fn decode_rejects_unsorted_or_duplicate_keys() {
        // Hand-built blob with descending keys must not decode.
        let mut bad = Vec::new();
        bad.extend_from_slice(&2u32.to_le_bytes());
        for (k, v) in [("b", "1"), ("a", "2")] {
            bad.extend_from_slice(&(k.len() as u32).to_le_bytes());
            bad.extend_from_slice(k.as_bytes());
            bad.extend_from_slice(&(v.len() as u32).to_le_bytes());
            bad.extend_from_slice(v.as_bytes());
        }
        let mut out = Vec::new();
        assert!(matches!(decode(&bad, &mut out), Err(Error::Corrupt(_))));
    }

    #[test]
    fn decode_rejects_truncation_and_trailing_bytes() {
        let bytes = encode(&[("k", "v")]).unwrap();
        let mut out = Vec::new();
        assert!(matches!(
            decode(&bytes[..bytes.len() - 1], &mut out),
            Err(Error::Corrupt(_))
        ));
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(matches!(decode(&extra, &mut out), Err(Error::Corrupt(_))));
    }

    #[test]
    fn decode_rejects_non_utf8_key() {
        let mut bad = Vec::new();
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.push(0xff); // not UTF-8
        bad.extend_from_slice(&0u32.to_le_bytes());
        let mut out = Vec::new();
        assert!(matches!(decode(&bad, &mut out), Err(Error::Corrupt(_))));
    }
}
