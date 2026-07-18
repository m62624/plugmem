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
//! length is bounds-checked with u64 arithmetic before any access, the
//! layout must be *canonical* (sections contiguous in table order, all
//! padding zero, no trailing bytes), and checksums are verified —
//! arbitrary bytes can produce any [`Error`] but never a panic. Passing
//! `fast_load` skips only the checksum passes (`Config::fast_load`, for
//! trusted local files); every structural rule still applies.

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
/// let snap = Snapshot::parse(&bytes, false).unwrap();
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

/// A parsed, validated snapshot borrowing the input buffer.
#[derive(Debug)]
pub struct Snapshot<'a> {
    bytes: &'a [u8],
    /// Header flags (see [`FLAG_VECTORS`]).
    pub flags: u16,
    /// Creation timestamp as written (informational).
    pub created_at: u64,
    config_len: usize,
    /// `(kind, start, len)` per section, in file order.
    sections: Vec<(u16, usize, usize)>,
    engine_ver_len: usize,
}

impl<'a> Snapshot<'a> {
    /// Parses and validates a snapshot file (trust model in the module
    /// docs). `fast_load` skips the checksum passes only.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedVersion`] for a foreign format version;
    /// [`Error::Corrupt`] for everything else that fails validation.
    pub fn parse(bytes: &'a [u8], fast_load: bool) -> Result<Self, Error> {
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
            if offset != expected {
                return Err(Error::Corrupt("sections must be contiguous in table order"));
            }
            let end = offset
                .checked_add(len)
                .ok_or(Error::Corrupt("section length overflow"))?;
            if end > file_len {
                return Err(Error::Corrupt("section out of bounds"));
            }
            if sections.iter().any(|&(k, _, _)| k == kind) {
                return Err(Error::Corrupt("duplicate section kind"));
            }
            expected = align_up(end);
            if bytes[end as usize..expected.min(file_len) as usize]
                .iter()
                .any(|&b| b != 0)
            {
                return Err(Error::Corrupt("nonzero padding after a section"));
            }
            sections.push((kind, offset as usize, len as usize));
        }
        if expected != file_len {
            return Err(Error::Corrupt("trailing bytes after the last section"));
        }

        if !fast_load {
            for (i, &(_, start, len)) in sections.iter().enumerate() {
                let at = table_start as usize + i * ENTRY;
                let want = u64::from_le_bytes(bytes[at + 24..at + 32].try_into().unwrap());
                if xxh3_64(&bytes[start..start + len]) != want {
                    return Err(Error::Corrupt("section checksum mismatch"));
                }
            }
            let stored =
                u64::from_le_bytes(bytes[FILE_HASH_AT..FILE_HASH_AT + 8].try_into().unwrap());
            // 0 means "no file hash" (the spec allows omitting it; our
            // writer always emits one).
            if stored != 0 && file_hash(bytes) != stored {
                return Err(Error::Corrupt("file checksum mismatch"));
            }
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
            .find(|&&(k, _, _)| k == kind)
            .map(|&(_, start, len)| &self.bytes[start..start + len])
    }

    /// The engine version string the file was written by (informational).
    pub fn engine_ver(&self) -> &'a str {
        core::str::from_utf8(&self.bytes[36..36 + self.engine_ver_len])
            .expect("validated during parse")
    }
}
