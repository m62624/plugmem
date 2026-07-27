//! Snapshot container: the file format's header, section table and
//! checksums (specs/03).
//!
//! A snapshot is the engine's memory image: a 64-byte header, the binary
//! [`Config`](crate::Config) block, a section table, then the sections —
//! each one a structure dump from `plugmem-arena` (`dump_*`/`load`
//! methods there), everything 64-byte aligned:
//!
//! ```text
//! [header 64][config][pad][table: n × 32][pad][section][pad]...
//! ```
//!
//! All container fields are little-endian. Byte layout of the header:
//!
//! | off | size | field |
//! |---|---|---|
//! | 0 | 4 | magic `0x504C_474D` ("PLGM") |
//! | 4 | 2 | format version (this module: 1) |
//! | 6 | 2 | flags (bit 0: vector section present; rest reserved zero) |
//! | 8 | 2 | section count |
//! | 10 | 6 | reserved, zero |
//! | 16 | 4 | config block length |
//! | 20 | 8 | file xxh3 |
//! | 28 | 8 | created_at (informational, unix ms) |
//! | 36 | 24 | engine version, UTF-8, zero-padded (informational) |
//! | 60 | 4 | reserved, zero |
//!
//! and of one section-table entry:
//!
//! | off | size | field |
//! |---|---|---|
//! | 0 | 2 | kind |
//! | 2 | 2 | alignment (always 64) |
//! | 4 | 4 | reserved, zero |
//! | 8 | 8 | offset from file start |
//! | 16 | 8 | length |
//! | 24 | 8 | section xxh3 |
//!
//! The file hash covers the **whole file with the hash field zeroed**
//! (tightened relative to the original spec draft, which hashed only the
//! bytes after the field and left the header prefix — notably `flags` —
//! covered by nothing but field validation).
//!
//! # Trust model
//!
//! [`Snapshot::parse`] treats its input as untrusted: every offset and
//! length is bounds-checked with u64 arithmetic before any access, and the
//! layout must be *canonical* (sections contiguous in table order, all
//! padding zero, no trailing bytes) — arbitrary bytes can produce any
//! [`Error`] but never a panic. The container xxh3 checksums are **not**
//! verified at parse; that runs on demand, in slices, via [`Snapshot::scrub`]
//! (the ZFS-scrub model, specs/16 §9), keeping an open sparse on large files.

use alloc::vec::Vec;

use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::error::Error;

/// `"PLGM"` interpreted as a little-endian u32.
pub const MAGIC: u32 = 0x504C_474D;

/// Snapshot format version written and accepted by this build.
pub const FORMAT_VERSION: u16 = 1;

/// Flag bit: the image carries vector-layer sections.
pub const FLAG_VECTORS: u16 = 1;

/// Header size in bytes.
const HEADER: usize = 64;

/// Section-table entry size in bytes.
const ENTRY: usize = 32;

/// Section alignment (also the padding unit of every block).
const ALIGN: usize = 64;

/// Offset of the file-hash field inside the header.
const FILE_HASH_AT: usize = 20;

/// Rounds up to the next multiple of [`ALIGN`].
fn align_up(v: u64) -> u64 {
    v.div_ceil(ALIGN as u64) * ALIGN as u64
}

/// xxh3 of the whole file with the file-hash field zeroed.
fn file_hash(bytes: &[u8]) -> u64 {
    let mut h = Xxh3::new();
    h.update(&bytes[..FILE_HASH_AT]);
    h.update(&[0u8; 8]);
    h.update(&bytes[FILE_HASH_AT + 8..]);
    h.digest()
}

/// Builds a snapshot file from section dumps.
///
/// ```
/// use plugmem_core::snapshot::{Snapshot, SnapshotWriter};
///
/// let mut w = SnapshotWriter::new();
/// w.section(7, b"payload".to_vec()).unwrap();
/// let bytes = w.finish(b"config-bytes", 0, 1_700_000_000_000, "0.1.0");
/// let snap = Snapshot::parse(&bytes).unwrap();
/// assert_eq!(snap.config(), b"config-bytes");
/// assert_eq!(snap.section(7), Some(&b"payload"[..]));
/// assert_eq!(snap.section(8), None);
/// ```
#[derive(Debug, Default)]
pub struct SnapshotWriter {
    sections: Vec<(u16, Vec<u8>)>,
}

impl SnapshotWriter {
    /// An empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one section. Kinds must be unique within a file.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] when `kind` was already added (a composition
    /// bug surfaced as a typed error rather than a corrupt file).
    pub fn section(&mut self, kind: u16, bytes: Vec<u8>) -> Result<(), Error> {
        if self.sections.iter().any(|&(k, _)| k == kind) {
            return Err(Error::Corrupt("duplicate section kind"));
        }
        self.sections.push((kind, bytes));
        Ok(())
    }

    /// Assembles the final file: header, config block, table, aligned
    /// sections, checksums. Deterministic — identical inputs produce
    /// identical bytes.
    ///
    /// `engine_ver` is informational; its first 24 UTF-8 bytes are stored
    /// (callers pass an ASCII semver in practice).
    pub fn finish(self, config: &[u8], flags: u16, created_at: u64, engine_ver: &str) -> Vec<u8> {
        let config_end = HEADER as u64 + config.len() as u64;
        let table_start = align_up(config_end);
        let table_end = table_start + (self.sections.len() * ENTRY) as u64;

        // Lay out section offsets first.
        let mut offsets = Vec::with_capacity(self.sections.len());
        let mut cursor = align_up(table_end);
        for (_, bytes) in &self.sections {
            offsets.push(cursor);
            cursor = align_up(cursor + bytes.len() as u64);
        }
        let file_len = cursor as usize;

        let mut out = alloc::vec![0u8; file_len];
        out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out[6..8].copy_from_slice(&flags.to_le_bytes());
        out[8..10].copy_from_slice(&(self.sections.len() as u16).to_le_bytes());
        out[16..20].copy_from_slice(&(config.len() as u32).to_le_bytes());
        out[28..36].copy_from_slice(&created_at.to_le_bytes());
        let ver = engine_ver.as_bytes();
        let ver_len = ver.len().min(24);
        out[36..36 + ver_len].copy_from_slice(&ver[..ver_len]);
        out[HEADER..HEADER + config.len()].copy_from_slice(config);

        for (i, (kind, bytes)) in self.sections.iter().enumerate() {
            let at = table_start as usize + i * ENTRY;
            out[at..at + 2].copy_from_slice(&kind.to_le_bytes());
            out[at + 2..at + 4].copy_from_slice(&(ALIGN as u16).to_le_bytes());
            out[at + 8..at + 16].copy_from_slice(&offsets[i].to_le_bytes());
            out[at + 16..at + 24].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            out[at + 24..at + 32].copy_from_slice(&xxh3_64(bytes).to_le_bytes());
            let start = offsets[i] as usize;
            out[start..start + bytes.len()].copy_from_slice(bytes);
        }

        let hash = file_hash(&out);
        out[FILE_HASH_AT..FILE_HASH_AT + 8].copy_from_slice(&hash.to_le_bytes());
        out
    }
}

/// One section's size and checksum, computed in the streaming writer's
/// first pass (see [`build_prefix`]).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SectionMeta {
    /// Section kind tag (unique per file).
    pub kind: u16,
    /// Section body length in bytes (before alignment padding).
    pub len: u64,
    /// xxh3 of the section body.
    pub hash: u64,
}

/// The header + config + section-table region of a snapshot: everything
/// before the first section body. Small and bounded (kilobytes), so it is
/// materialized even when the bodies stream (specs/16 §9).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Prefix {
    /// The prefix bytes, ready to write. The file-hash field is left zero;
    /// the caller patches it via [`SnapshotSink::patch`] once the
    /// whole body has streamed through the running hash.
    pub bytes: Vec<u8>,
    /// Absolute byte offset of each section body, in `metas` order.
    pub offsets: Vec<u64>,
    /// Total file length (aligned end of the last section).
    pub file_len: u64,
}

/// A byte sink for streaming a snapshot without materializing the whole
/// image (specs/16 §9): `write` appends in file order, `patch` overwrites a
/// short run at an absolute offset — used once for the header file-hash
/// field, the only non-sequential write, done after the body has streamed.
pub trait SnapshotSink {
    /// Appends `bytes` at the current position.
    ///
    /// # Errors
    /// Whatever the underlying sink reports (e.g. an I/O error).
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error>;

    /// Overwrites `bytes` at absolute offset `at` (already-written region).
    ///
    /// # Errors
    /// Whatever the underlying sink reports.
    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error>;
}

impl SnapshotSink for &mut Vec<u8> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.extend_from_slice(bytes);
        Ok(())
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        let at = at as usize;
        self[at..at + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

/// Absolute offset of the header file-hash field — the one location the
/// streaming writer patches after the running hash is known.
pub const FILE_HASH_OFFSET: u64 = FILE_HASH_AT as u64;

/// Lays out the header, config block and section table from per-section
/// `(kind, len, hash)` metadata (the streaming writer's first pass), byte
/// for byte identical to [`SnapshotWriter::finish`]'s leading region. The
/// caller then streams each section body followed by
/// [`pad_len`] zero bytes, feeding a running [`Xxh3`] the prefix and every
/// body/pad byte, and finally calls [`SnapshotSink::patch`].
pub fn build_prefix(
    config: &[u8],
    flags: u16,
    created_at: u64,
    engine_ver: &str,
    metas: &[SectionMeta],
) -> Prefix {
    let count = metas.len();
    let config_end = HEADER as u64 + config.len() as u64;
    let table_start = align_up(config_end);
    let table_end = table_start + (count * ENTRY) as u64;

    let mut offsets = Vec::with_capacity(count);
    let mut cursor = align_up(table_end);
    for m in metas {
        offsets.push(cursor);
        cursor = align_up(cursor + m.len);
    }
    let file_len = cursor;

    let prefix_len = align_up(table_end) as usize;
    let mut out = alloc::vec![0u8; prefix_len];
    out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    out[6..8].copy_from_slice(&flags.to_le_bytes());
    out[8..10].copy_from_slice(&(count as u16).to_le_bytes());
    out[16..20].copy_from_slice(&(config.len() as u32).to_le_bytes());
    out[28..36].copy_from_slice(&created_at.to_le_bytes());
    let ver = engine_ver.as_bytes();
    let ver_len = ver.len().min(24);
    out[36..36 + ver_len].copy_from_slice(&ver[..ver_len]);
    out[HEADER..HEADER + config.len()].copy_from_slice(config);

    for (i, m) in metas.iter().enumerate() {
        let at = table_start as usize + i * ENTRY;
        out[at..at + 2].copy_from_slice(&m.kind.to_le_bytes());
        out[at + 2..at + 4].copy_from_slice(&(ALIGN as u16).to_le_bytes());
        out[at + 8..at + 16].copy_from_slice(&offsets[i].to_le_bytes());
        out[at + 16..at + 24].copy_from_slice(&m.len.to_le_bytes());
        out[at + 24..at + 32].copy_from_slice(&m.hash.to_le_bytes());
    }
    // The file-hash field (FILE_HASH_AT) stays zero; the caller patches it.
    Prefix {
        bytes: out,
        offsets,
        file_len,
    }
}

/// Alignment-padding length that follows a section body of `len` starting
/// at `offset` (zero bytes up to the next 64-byte boundary).
pub fn pad_len(offset: u64, len: u64) -> usize {
    (align_up(offset + len) - (offset + len)) as usize
}

/// A parsed, validated snapshot borrowing the input buffer.
#[derive(Debug)]
pub struct Snapshot<'a> {
    bytes: &'a [u8],
    /// Header flags (see [`FLAG_VECTORS`]).
    pub flags: u16,
    /// Creation timestamp as written (informational).
    pub created_at: u64,
    config_len: usize,
    /// `(kind, start, len, stored_hash)` per section, in file order. The
    /// stored xxh3 is retained (not verified at parse) so the on-demand
    /// [`ScrubCursor`] can check it in slices (specs/16 §9).
    sections: Vec<(u16, usize, usize, u64)>,
    engine_ver_len: usize,
}

impl<'a> Snapshot<'a> {
    /// Parses and structurally validates a snapshot file (trust model in the
    /// module docs): framing, canonical layout and bounds only. The container
    /// xxh3 checksums are **not** verified here — that is on demand, in slices,
    /// via [`Snapshot::scrub`] (specs/16 §9). Content pools (text, vectors)
    /// are validated lazily too; see [`Memory::verify`](crate::Memory::verify).
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedVersion`] for a foreign format version;
    /// [`Error::Corrupt`] for everything else that fails structural validation.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER {
            return Err(Error::Corrupt("snapshot shorter than its header"));
        }
        if !bytes.len().is_multiple_of(ALIGN) {
            return Err(Error::Corrupt("snapshot length is not 64-byte aligned"));
        }
        if u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != MAGIC {
            return Err(Error::Corrupt("bad magic"));
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let flags = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        if flags & !FLAG_VECTORS != 0 {
            return Err(Error::Corrupt("unknown flag bits set"));
        }
        let section_cnt = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        if bytes[10..16] != [0u8; 6] || bytes[60..64] != [0u8; 4] {
            return Err(Error::Corrupt("reserved header bytes must be zero"));
        }
        let config_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let created_at = u64::from_le_bytes(bytes[28..36].try_into().unwrap());
        let ver_bytes = &bytes[36..60];
        let engine_ver_len = ver_bytes.iter().position(|&b| b == 0).unwrap_or(24);
        if ver_bytes[engine_ver_len..].iter().any(|&b| b != 0) {
            return Err(Error::Corrupt("engine version is not zero-terminated"));
        }
        if core::str::from_utf8(&ver_bytes[..engine_ver_len]).is_err() {
            return Err(Error::Corrupt("engine version is not UTF-8"));
        }

        let file_len = bytes.len() as u64;
        let config_end = HEADER as u64 + config_len as u64;
        let table_start = align_up(config_end);
        let table_end = table_start + (section_cnt * ENTRY) as u64;
        if table_end > file_len {
            return Err(Error::Corrupt("section table out of bounds"));
        }
        if bytes[config_end as usize..table_start as usize]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(Error::Corrupt("nonzero padding after the config block"));
        }

        // The layout must be canonical: sections contiguous in table order.
        let mut sections = Vec::with_capacity(section_cnt);
        let mut expected = align_up(table_end);
        if bytes[table_end as usize..expected as usize]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(Error::Corrupt("nonzero padding after the section table"));
        }
        for i in 0..section_cnt {
            let at = table_start as usize + i * ENTRY;
            let kind = u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
            let align = u16::from_le_bytes(bytes[at + 2..at + 4].try_into().unwrap());
            if align as usize != ALIGN {
                return Err(Error::Corrupt("section alignment must be 64"));
            }
            if bytes[at + 4..at + 8] != [0u8; 4] {
                return Err(Error::Corrupt("reserved section bytes must be zero"));
            }
            let offset = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
            let len = u64::from_le_bytes(bytes[at + 16..at + 24].try_into().unwrap());
            let want = u64::from_le_bytes(bytes[at + 24..at + 32].try_into().unwrap());
            if offset != expected {
                return Err(Error::Corrupt("sections must be contiguous in table order"));
            }
            let end = offset
                .checked_add(len)
                .ok_or(Error::Corrupt("section length overflow"))?;
            if end > file_len {
                return Err(Error::Corrupt("section out of bounds"));
            }
            if sections.iter().any(|&(k, _, _, _)| k == kind) {
                return Err(Error::Corrupt("duplicate section kind"));
            }
            expected = align_up(end);
            if bytes[end as usize..expected.min(file_len) as usize]
                .iter()
                .any(|&b| b != 0)
            {
                return Err(Error::Corrupt("nonzero padding after a section"));
            }
            sections.push((kind, offset as usize, len as usize, want));
        }
        if expected != file_len {
            return Err(Error::Corrupt("trailing bytes after the last section"));
        }

        Ok(Self {
            bytes,
            flags,
            created_at,
            config_len,
            sections,
            engine_ver_len,
        })
    }

    /// The binary config block (decode with
    /// [`Config::decode`](crate::Config::decode)).
    pub fn config(&self) -> &'a [u8] {
        &self.bytes[HEADER..HEADER + self.config_len]
    }

    /// The bytes of one section, or `None` when the file has no section of
    /// that kind.
    pub fn section(&self, kind: u16) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|&&(k, _, _, _)| k == kind)
            .map(|&(_, start, len, _)| &self.bytes[start..start + len])
    }

    /// The engine version string the file was written by (informational).
    pub fn engine_ver(&self) -> &'a str {
        core::str::from_utf8(&self.bytes[36..36 + self.engine_ver_len])
            .expect("validated during parse")
    }

    /// A resumable container-checksum scan of this snapshot with the default
    /// slice budget ([`DEFAULT_SCRUB_BUDGET`]). See [`ScrubCursor`].
    pub fn scrub(&self) -> ScrubCursor<'a> {
        self.scrub_with_budget(DEFAULT_SCRUB_BUDGET)
    }

    /// A resumable container-checksum scan hashing at most `budget` bytes per
    /// [`Iterator::next`] (specs/16 §9). See [`ScrubCursor`].
    pub fn scrub_with_budget(&self, budget: usize) -> ScrubCursor<'a> {
        ScrubCursor {
            bytes: self.bytes,
            sections: self
                .sections
                .iter()
                .map(|&(_, s, l, w)| (s, l, w))
                .collect(),
            budget: budget.max(1),
            pos: 0,
            sec: 0,
            sec_hash: Xxh3::new(),
            file: Xxh3::new(),
            done: false,
        }
    }
}

/// Default number of bytes a [`ScrubCursor`] hashes per `next()` slice
/// (specs/16 §9). The `scrub` slice-size bench sweeps 64 KiB…64 MiB and finds
/// throughput essentially flat (the scan is xxh3/memory-bandwidth-bound, not
/// per-slice-overhead-bound), so the budget is a *pacing* quantum, not a
/// throughput knob: 1 MiB is ~sub-millisecond per slice — fine-grained enough
/// to pause, cancel or report progress smoothly, with negligible overhead.
pub const DEFAULT_SCRUB_BUDGET: usize = 1 << 20;

/// Progress of an in-flight [`ScrubCursor`] scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScrubProgress {
    /// Bytes checksummed so far (equals `total_bytes` on the final item).
    pub done_bytes: u64,
    /// The file length — the total to checksum.
    pub total_bytes: u64,
}

/// A resumable, budget-bounded container-checksum scan over a parsed snapshot
/// (specs/16 §9 — the ZFS-scrub model, run on demand instead of at open).
///
/// It implements [`Iterator`]: each `next()` hashes up to the slice budget,
/// verifying each section's stored xxh3 as its body completes and the
/// whole-file hash at EOF. It yields `Ok(ScrubProgress)` per slice, then
/// `None`; the first mismatch yields `Err(Error::Corrupt)` and then `None`
/// (fused). Because it only *reads* the bytes linearly, over an mmap the
/// pages fault in, get hashed and stay reclaimable — a scrub never residents
/// the whole file. It is one-shot: build a new cursor to scan again.
pub struct ScrubCursor<'a> {
    bytes: &'a [u8],
    /// `(start, len, stored_hash)` per section, in file order.
    sections: Vec<(usize, usize, u64)>,
    budget: usize,
    pos: usize,
    /// Index of the section currently being hashed / next to complete.
    sec: usize,
    /// Streaming hash of the current section body.
    sec_hash: Xxh3,
    /// Streaming whole-file hash (with the file-hash field fed as zero).
    file: Xxh3,
    done: bool,
}

impl ScrubCursor<'_> {
    /// Feeds `bytes[from..to]` into the whole-file hash, substituting zeros
    /// for the 8-byte file-hash field — matching [`file_hash`] exactly.
    fn feed_file(&mut self, from: usize, to: usize) {
        let before_end = to.min(FILE_HASH_AT);
        if from < before_end {
            self.file.update(&self.bytes[from..before_end]);
        }
        let z_start = from.max(FILE_HASH_AT);
        let z_end = to.min(FILE_HASH_AT + 8);
        if z_start < z_end {
            self.file.update(&[0u8; 8][..z_end - z_start]);
        }
        let after_start = from.max(FILE_HASH_AT + 8);
        if after_start < to {
            self.file.update(&self.bytes[after_start..to]);
        }
    }

    /// Compares the current section's streamed digest to its stored hash and
    /// advances to the next section (resetting the section hasher).
    fn complete_section(&mut self, want: u64) -> Result<(), Error> {
        if self.sec_hash.digest() != want {
            self.done = true;
            return Err(Error::Corrupt("section checksum mismatch"));
        }
        self.sec += 1;
        self.sec_hash = Xxh3::new();
        Ok(())
    }
}

impl Iterator for ScrubCursor<'_> {
    type Item = Result<ScrubProgress, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let n = self.sections.len();
        let file_len = self.bytes.len();
        let mut budget_left = self.budget;

        while budget_left > 0 && self.pos < file_len {
            // Flush any zero-length sections starting exactly here.
            while self.sec < n {
                let (start, len, want) = self.sections[self.sec];
                if self.pos == start && len == 0 {
                    if let Err(e) = self.complete_section(want) {
                        return Some(Err(e));
                    }
                } else {
                    break;
                }
            }

            let (in_body, boundary) = match self.sections.get(self.sec) {
                Some(&(start, len, _)) if self.pos >= start => (true, start + len),
                Some(&(start, _, _)) => (false, start),
                None => (false, file_len),
            };
            let end = (self.pos + budget_left).min(boundary);
            self.feed_file(self.pos, end);
            if in_body {
                self.sec_hash.update(&self.bytes[self.pos..end]);
            }
            budget_left -= end - self.pos;
            self.pos = end;

            if in_body && self.pos == boundary {
                let want = self.sections[self.sec].2;
                if let Err(e) = self.complete_section(want) {
                    return Some(Err(e));
                }
            }
        }

        if self.pos == file_len {
            // Any trailing zero-length sections sit exactly at EOF.
            while self.sec < n {
                let want = self.sections[self.sec].2;
                if let Err(e) = self.complete_section(want) {
                    return Some(Err(e));
                }
            }
            self.done = true;
            let stored = u64::from_le_bytes(
                self.bytes[FILE_HASH_AT..FILE_HASH_AT + 8]
                    .try_into()
                    .unwrap(),
            );
            // 0 means "no file hash" (the writer always emits one).
            if stored != 0 && self.file.digest() != stored {
                return Some(Err(Error::Corrupt("file checksum mismatch")));
            }
        }

        Some(Ok(ScrubProgress {
            done_bytes: self.pos as u64,
            total_bytes: file_len as u64,
        }))
    }
}
