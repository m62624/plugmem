//! Journal record framing.
//!
//! The journal is an append-only sequence of framed records:
//!
//! ```text
//! [len u32 LE][check u32 LE][op u8][payload ...]     len = 1 + payload
//! ```
//!
//! `check` is the low 32 bits of xxh3-64 over `[op][payload]`. Framing is
//! op-agnostic — the op byte's meaning (Remember, Revise, …) belongs to
//! the engine's replay layer, which lands with the verbs.
//!
//! # Torn tails vs corruption
//!
//! There is exactly one writer and appends are sequential, so a crash can
//! only leave a *prefix* of the last record. That yields a clean rule for
//! [`scan`]:
//!
//! - a record whose frame extends past the end of the buffer is the torn
//!   tail: the scan succeeds, drops it, and reports `truncated_tail`;
//! - a complete frame with a bad checksum that ends exactly at the buffer
//!   end is also treated as a torn tail (a torn write inside the payload
//!   of the final record looks like this);
//! - any other inconsistency — a bad checksum mid-stream, a `len` of 0
//!   (no valid record has one, and a torn prefix of ≥ 4 bytes always
//!   carries a valid `len`) — is [`Error::Corrupt`].

use alloc::vec::Vec;

use xxhash_rust::xxh3::xxh3_64;

use crate::error::Error;

/// Serialized width of a `u32` field.
const U32_BYTES: usize = core::mem::size_of::<u32>();
/// Serialized width of a `u64` field.
const U64_BYTES: usize = core::mem::size_of::<u64>();
/// Serialized width of an `f32` field.
const F32_BYTES: usize = core::mem::size_of::<f32>();

/// Frame header size: `len` (u32) + `check` (u32).
const HEADER: usize = U32_BYTES + U32_BYTES;

/// One decoded journal record, borrowing the scanned buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct JournalEntry<'a> {
    /// Operation tag (engine-defined op table).
    pub op: u8,
    /// The operation's binary payload.
    pub payload: &'a [u8],
}

/// Result of scanning a journal buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalScan<'a> {
    /// All valid records, in append order.
    pub entries: Vec<JournalEntry<'a>>,
    /// `true` when a torn tail record was dropped (crash between appends —
    /// reported in the open report, not an error).
    pub truncated_tail: bool,
}

/// Checksum over the contiguous `[op][payload]` body slice: low 32 bits
/// of xxh3-64.
fn body_checksum(body: &[u8]) -> u32 {
    xxh3_64(body) as u32
}

/// Appends one framed record to `out` (the bytes handed to
/// [`Storage::append_journal`](crate::storage::Storage::append_journal)).
pub fn encode_entry(out: &mut Vec<u8>, op: u8, payload: &[u8]) {
    let len = 1 + payload.len();
    let len32 = u32::try_from(len).expect("journal payload fits u32 by construction");
    out.reserve(HEADER + len);
    out.extend_from_slice(&len32.to_le_bytes());
    // The checksum needs the contiguous body; build it in place and hash
    // the slice we just wrote.
    let check_pos = out.len();
    out.extend_from_slice(&[0u8; U32_BYTES]);
    out.push(op);
    out.extend_from_slice(payload);
    let check = body_checksum(&out[check_pos + U32_BYTES..]);
    out[check_pos..check_pos + U32_BYTES].copy_from_slice(&check.to_le_bytes());
}

/// One decoded engine operation (op table). `Revise` is
/// `Remember` with `revises` set — the two share a payload, only the op
/// byte differs.
///
/// Not `Eq`: the raw `f32` vector rides along so replay can re-quantize
/// it deterministically, and `f32` is only `PartialEq`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Op<'a> {
    /// Op 1/2: a new fact (op 2 additionally closes `revises`).
    Remember {
        /// Host timestamp of the operation.
        now: u64,
        /// Resolved validity start (the engine defaults it before
        /// journaling — replay never re-derives).
        valid_from: u64,
        /// Subject entity name, if any.
        entity: Option<&'a str>,
        /// Fact text.
        text: &'a str,
        /// Tags, verbatim.
        tags: Vec<&'a str>,
        /// `(rel, target_entity)` link pairs.
        links: Vec<(&'a str, &'a str)>,
        /// The raw embedding as remembered (empty = none). Stored
        /// pre-quantization so replay re-quantizes with the same pure
        /// function and reproduces every slot byte for byte.
        vector: Vec<f32>,
        /// Metadata key→value pairs as remembered (empty = none). Replay
        /// re-canonicalizes them (sorts, dedups) the same way `remember` did,
        /// so the stored blob is reproduced byte for byte.
        metadata: Vec<(&'a str, &'a str)>,
        /// Predecessor being revised ([`crate::id::FactId::NONE`] for op 1).
        revises: crate::id::FactId,
        /// The fact id assigned at execution time — authoritative on
        /// replay.
        assigned: crate::id::FactId,
    },
    /// Op 3: tombstone a fact.
    Forget {
        /// Host timestamp of the operation.
        now: u64,
        /// The fact being forgotten.
        fact: crate::id::FactId,
    },
    /// Op 4: upsert a typed edge between two entities.
    Link {
        /// Host timestamp of the operation.
        now: u64,
        /// Source entity name.
        src: &'a str,
        /// Relation term, verbatim.
        rel: &'a str,
        /// Destination entity name.
        dst: &'a str,
        /// Provenance fact, or [`crate::id::FactId::NONE`].
        provenance: crate::id::FactId,
    },
    /// Op 5: marker that a maintenance pass ran at this point.
    Maintain {
        /// Host timestamp of the operation.
        now: u64,
        /// Maintenance mode encoded by the engine.
        mode: u8,
        /// HNSW insertion budget, or `u32::MAX` for unlimited.
        max_hnsw_inserts: u32,
    },
    /// Op 6: close the current typed edge between two entities.
    Unlink {
        /// Host timestamp of the operation.
        now: u64,
        /// Source entity name.
        src: &'a str,
        /// Relation term, verbatim.
        rel: &'a str,
        /// Destination entity name.
        dst: &'a str,
    },
    /// Op 7: remove one tag from every current fact by creating revisions.
    RemoveTag {
        /// Host timestamp used as every successor's validity start.
        now: u64,
        /// Verbatim tag name.
        tag: &'a str,
    },
}

/// Appends a length-prefixed string (`u32 LE` + bytes).
fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Reads a length-prefixed string; advances `at`.
fn take_str<'a>(bytes: &'a [u8], at: &mut usize) -> Result<&'a str, Error> {
    let len = take_u32(bytes, at)? as usize;
    let end = at
        .checked_add(len)
        .filter(|&e| e <= bytes.len())
        .ok_or(Error::Corrupt("journal string overruns its record"))?;
    let s = core::str::from_utf8(&bytes[*at..end])
        .map_err(|_| Error::Corrupt("journal string is not UTF-8"))?;
    *at = end;
    Ok(s)
}

/// Reads a `u32 LE`; advances `at`.
fn take_u32(bytes: &[u8], at: &mut usize) -> Result<u32, Error> {
    let end = *at + U32_BYTES;
    if end > bytes.len() {
        return Err(Error::Corrupt("journal record truncated inside a field"));
    }
    let v = u32::from_le_bytes(bytes[*at..end].try_into().unwrap());
    *at = end;
    Ok(v)
}

/// Reads a `u64 LE`; advances `at`.
fn take_u64(bytes: &[u8], at: &mut usize) -> Result<u64, Error> {
    let end = *at + U64_BYTES;
    if end > bytes.len() {
        return Err(Error::Corrupt("journal record truncated inside a field"));
    }
    let v = u64::from_le_bytes(bytes[*at..end].try_into().unwrap());
    *at = end;
    Ok(v)
}

/// Reads a `u32 LE` count followed by that many `f32 LE`; advances `at`.
fn take_vec_f32(bytes: &[u8], at: &mut usize) -> Result<Vec<f32>, Error> {
    let count = take_u32(bytes, at)? as usize;
    // Bounds math in u64, like the snapshot container: on 32-bit targets
    // `count * F32_BYTES` in usize can wrap and slip a hostile count past the
    // check, and `with_capacity` on an unchecked count aborts a wasm32
    // process (caught by the wasm32-wasip1 test run). Only after the
    // check is the allocation known to be bounded by the input length.
    let end = *at as u64 + count as u64 * F32_BYTES as u64;
    if end > bytes.len() as u64 {
        return Err(Error::Corrupt("journal vector overruns its record"));
    }
    let end = end as usize;
    let mut v = Vec::with_capacity(count);
    let mut p = *at;
    while p < end {
        v.push(f32::from_le_bytes(
            bytes[p..p + F32_BYTES].try_into().unwrap(),
        ));
        p += F32_BYTES;
    }
    *at = end;
    Ok(v)
}

impl<'a> Op<'a> {
    /// Encodes the operation as one framed journal entry appended to
    /// `out` (via [`encode_entry`]).
    pub fn encode(&self, out: &mut Vec<u8>) {
        let mut payload = Vec::new();
        let op = match self {
            Op::Remember {
                now,
                valid_from,
                entity,
                text,
                tags,
                links,
                vector,
                metadata,
                revises,
                assigned,
            } => {
                payload.extend_from_slice(&now.to_le_bytes());
                payload.extend_from_slice(&valid_from.to_le_bytes());
                payload.extend_from_slice(&revises.0.to_le_bytes());
                payload.extend_from_slice(&assigned.0.to_le_bytes());
                match entity {
                    Some(name) => {
                        payload.push(1);
                        put_str(&mut payload, name);
                    }
                    None => payload.push(0),
                }
                put_str(&mut payload, text);
                payload.push(tags.len() as u8);
                for tag in tags {
                    put_str(&mut payload, tag);
                }
                payload.push(links.len() as u8);
                for (rel, dst) in links {
                    put_str(&mut payload, rel);
                    put_str(&mut payload, dst);
                }
                payload.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                for &x in vector {
                    payload.extend_from_slice(&x.to_le_bytes());
                }
                payload.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
                for (k, v) in metadata {
                    put_str(&mut payload, k);
                    put_str(&mut payload, v);
                }
                if revises.is_none() { 1 } else { 2 }
            }
            Op::Forget { now, fact } => {
                payload.extend_from_slice(&now.to_le_bytes());
                payload.extend_from_slice(&fact.0.to_le_bytes());
                3
            }
            Op::Link {
                now,
                src,
                rel,
                dst,
                provenance,
            } => {
                payload.extend_from_slice(&now.to_le_bytes());
                payload.extend_from_slice(&provenance.0.to_le_bytes());
                put_str(&mut payload, src);
                put_str(&mut payload, rel);
                put_str(&mut payload, dst);
                4
            }
            Op::Unlink { now, src, rel, dst } => {
                payload.extend_from_slice(&now.to_le_bytes());
                put_str(&mut payload, src);
                put_str(&mut payload, rel);
                put_str(&mut payload, dst);
                6
            }
            Op::Maintain {
                now,
                mode,
                max_hnsw_inserts,
            } => {
                payload.extend_from_slice(&now.to_le_bytes());
                payload.push(*mode);
                payload.extend_from_slice(&max_hnsw_inserts.to_le_bytes());
                5
            }
            Op::RemoveTag { now, tag } => {
                payload.extend_from_slice(&now.to_le_bytes());
                put_str(&mut payload, tag);
                7
            }
        };
        encode_entry(out, op, &payload);
    }

    /// Decodes one operation from a scanned entry. The payload is
    /// untrusted (the checksum guards transport integrity, not origin):
    /// every read is bounds-checked, malformed input is
    /// [`Error::Corrupt`], never a panic.
    pub fn decode(op: u8, payload: &'a [u8]) -> Result<Op<'a>, Error> {
        use crate::id::FactId;
        let at = &mut 0usize;
        let decoded = match op {
            1 | 2 => {
                let now = take_u64(payload, at)?;
                let valid_from = take_u64(payload, at)?;
                let revises = FactId(take_u32(payload, at)?);
                let assigned = FactId(take_u32(payload, at)?);
                if (op == 2) == revises.is_none() {
                    return Err(Error::Corrupt("journal revises field disagrees with op"));
                }
                let entity = match payload.get(*at) {
                    Some(0) => {
                        *at += 1;
                        None
                    }
                    Some(1) => {
                        *at += 1;
                        Some(take_str(payload, at)?)
                    }
                    _ => return Err(Error::Corrupt("journal entity flag is invalid")),
                };
                let text = take_str(payload, at)?;
                let tag_cnt = *payload
                    .get(*at)
                    .ok_or(Error::Corrupt("journal record truncated inside a field"))?;
                *at += 1;
                let mut tags = Vec::with_capacity(tag_cnt as usize);
                for _ in 0..tag_cnt {
                    tags.push(take_str(payload, at)?);
                }
                let link_cnt = *payload
                    .get(*at)
                    .ok_or(Error::Corrupt("journal record truncated inside a field"))?;
                *at += 1;
                let mut links = Vec::with_capacity(link_cnt as usize);
                for _ in 0..link_cnt {
                    let rel = take_str(payload, at)?;
                    let dst = take_str(payload, at)?;
                    links.push((rel, dst));
                }
                let vector = take_vec_f32(payload, at)?;
                let meta_cnt = take_u32(payload, at)?;
                let mut metadata = Vec::new();
                for _ in 0..meta_cnt {
                    let k = take_str(payload, at)?;
                    let v = take_str(payload, at)?;
                    metadata.push((k, v));
                }
                Op::Remember {
                    now,
                    valid_from,
                    entity,
                    text,
                    tags,
                    links,
                    vector,
                    metadata,
                    revises,
                    assigned,
                }
            }
            3 => Op::Forget {
                now: take_u64(payload, at)?,
                fact: FactId(take_u32(payload, at)?),
            },
            4 => {
                let now = take_u64(payload, at)?;
                let provenance = FactId(take_u32(payload, at)?);
                let src = take_str(payload, at)?;
                let rel = take_str(payload, at)?;
                let dst = take_str(payload, at)?;
                Op::Link {
                    now,
                    src,
                    rel,
                    dst,
                    provenance,
                }
            }
            5 => {
                let now = take_u64(payload, at)?;
                let (mode, max_hnsw_inserts) = if *at == payload.len() {
                    (0, u32::MAX)
                } else {
                    let mode = *payload
                        .get(*at)
                        .ok_or(Error::Corrupt("journal record truncated inside a field"))?;
                    *at += 1;
                    let max_hnsw_inserts = take_u32(payload, at)?;
                    (mode, max_hnsw_inserts)
                };
                Op::Maintain {
                    now,
                    mode,
                    max_hnsw_inserts,
                }
            }
            6 => {
                let now = take_u64(payload, at)?;
                let src = take_str(payload, at)?;
                let rel = take_str(payload, at)?;
                let dst = take_str(payload, at)?;
                Op::Unlink { now, src, rel, dst }
            }
            7 => {
                let now = take_u64(payload, at)?;
                let tag = take_str(payload, at)?;
                Op::RemoveTag { now, tag }
            }
            _ => return Err(Error::Corrupt("unknown journal op")),
        };
        if *at != payload.len() {
            return Err(Error::Corrupt("journal record has trailing bytes"));
        }
        Ok(decoded)
    }
}

/// Scans a whole journal buffer into records (validation + tail-recovery
/// rules in the module docs).
pub fn scan(journal: &[u8]) -> Result<JournalScan<'_>, Error> {
    let mut entries = Vec::new();
    let mut pos = 0usize;
    while pos < journal.len() {
        let rest = &journal[pos..];
        if rest.len() < HEADER {
            return Ok(JournalScan {
                entries,
                truncated_tail: true,
            });
        }
        let len = u32::from_le_bytes(rest[..U32_BYTES].try_into().unwrap()) as usize;
        if len == 0 {
            return Err(Error::Corrupt("journal record with zero length"));
        }
        // `HEADER + len` is computed with a checked add: on a 32-bit target
        // `len` can reach u32::MAX and the bare `HEADER + len` overflows
        // usize (a debug-build panic — the loader must never panic on any
        // bytes). An overflow means the record claims more than any buffer
        // can hold, so it is a torn tail like the `get` miss below.
        let Some(body) = HEADER
            .checked_add(len)
            .and_then(|end| rest.get(HEADER..end))
        else {
            return Ok(JournalScan {
                entries,
                truncated_tail: true,
            });
        };
        let want = u32::from_le_bytes(rest[U32_BYTES..HEADER].try_into().unwrap());
        if body_checksum(body) != want {
            if pos + HEADER + len == journal.len() {
                return Ok(JournalScan {
                    entries,
                    truncated_tail: true,
                });
            }
            return Err(Error::Corrupt("journal checksum mismatch mid-stream"));
        }
        entries.push(JournalEntry {
            op: body[0],
            payload: &body[1..],
        });
        pos += HEADER + len;
    }
    Ok(JournalScan {
        entries,
        truncated_tail: false,
    })
}
