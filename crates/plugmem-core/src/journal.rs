//! Journal record framing (specs/03).
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

/// Frame header size: `len` + `check`.
const HEADER: usize = 8;

/// One decoded journal record, borrowing the scanned buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalEntry<'a> {
    /// Operation tag (engine-defined; see specs/03 op table).
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
    out.extend_from_slice(&[0u8; 4]);
    out.push(op);
    out.extend_from_slice(payload);
    let check = body_checksum(&out[check_pos + 4..]);
    out[check_pos..check_pos + 4].copy_from_slice(&check.to_le_bytes());
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
        let len = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
        if len == 0 {
            return Err(Error::Corrupt("journal record with zero length"));
        }
        let Some(body) = rest.get(HEADER..HEADER + len) else {
            return Ok(JournalScan {
                entries,
                truncated_tail: true,
            });
        };
        let want = u32::from_le_bytes(rest[4..8].try_into().unwrap());
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
