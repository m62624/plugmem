//! The current tag catalog: snapshot base plus a small LSM-style overlay.
//!
//! Facts remain the source of truth. The base is a compact sorted list loaded
//! from the snapshot; writes place count overrides into small sorted runs. A
//! new tag therefore does not shift the whole catalog, while reads merge at
//! most one base and `O(log changed_tags)` runs. A checkpoint emits the merged
//! view and reopening clears the overlay.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::Ordering;

use plugmem_arena::{Interner, TermId};
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::error::Error;

/// Default number of tags returned by one page.
pub const DEFAULT_TAG_PAGE_LIMIT: usize = 64;
/// Hard public bound for one tag page.
pub const MAX_TAG_PAGE_LIMIT: usize = 256;

const RUN_BASE: usize = 64;
const ENTRY_BYTES: usize = 8;
const CURSOR_PREFIX: &str = "t1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    term: u32,
    count: u32,
}

/// One active tag and the number of current facts carrying it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TagSummary {
    /// Verbatim tag name.
    pub name: String,
    /// Current, non-tombstoned and non-closed facts carrying the tag.
    pub count: u32,
}

/// A bounded tag-catalog query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TagQuery<'a> {
    /// Optional exact, case-sensitive prefix.
    pub prefix: Option<&'a str>,
    /// Opaque cursor returned by the previous page.
    pub cursor: Option<&'a str>,
    /// Page size; `0` selects [`DEFAULT_TAG_PAGE_LIMIT`].
    pub limit: usize,
}

/// One bounded page of active tags.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TagPage {
    /// Tags in deterministic UTF-8 lexical order.
    pub items: Vec<TagSummary>,
    /// Pass to the next call; `None` when the scan is complete.
    pub next_cursor: Option<String>,
}

/// A derived current-tag index. Facts and their tag lists remain authoritative.
#[derive(Debug, Default)]
pub(super) struct TagCatalog {
    /// Fully merged state as of the mapped snapshot.
    base: Vec<Entry>,
    /// Sorted count overrides accumulated since the snapshot. Each tag occurs
    /// in at most one run or the buffer.
    runs: Vec<Option<Vec<Entry>>>,
    /// Small sorted level-zero run. Insertion shifts at most `RUN_BASE` slots.
    buffer: Vec<Entry>,
    /// Order-independent fingerprint of the current `(name, count)` set.
    fingerprint: u64,
}

impl TagCatalog {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn load(bytes: &[u8], terms: &Interner<'_>) -> Result<Self, Error> {
        if !bytes.len().is_multiple_of(ENTRY_BYTES) {
            return Err(Error::Corrupt("tag catalog is not an entry sequence"));
        }
        let mut base = Vec::with_capacity(bytes.len() / ENTRY_BYTES);
        let mut previous: Option<&str> = None;
        let mut fingerprint = 0u64;
        for raw in bytes.chunks_exact(ENTRY_BYTES) {
            let term = u32::from_le_bytes(raw[..4].try_into().unwrap());
            let count = u32::from_le_bytes(raw[4..].try_into().unwrap());
            if term as usize >= terms.len() || count == 0 {
                return Err(Error::Corrupt("tag catalog entry is invalid"));
            }
            let name = terms.resolve(TermId(term));
            if previous.is_some_and(|old| old >= name) {
                return Err(Error::Corrupt("tag catalog is not strictly sorted"));
            }
            previous = Some(name);
            fingerprint ^= entry_hash(name, count);
            base.push(Entry { term, count });
        }
        Ok(Self {
            base,
            runs: Vec::new(),
            buffer: Vec::new(),
            fingerprint,
        })
    }

    pub(super) fn from_counts(mut entries: Vec<(TermId, u32)>, terms: &Interner<'_>) -> Self {
        entries.retain(|(_, count)| *count != 0);
        entries.sort_unstable_by(|a, b| terms.resolve(a.0).cmp(terms.resolve(b.0)));
        let base: Vec<Entry> = entries
            .into_iter()
            .map(|(term, count)| Entry {
                term: term.0,
                count,
            })
            .collect();
        let fingerprint = base.iter().fold(0, |hash, entry| {
            hash ^ entry_hash(terms.resolve(TermId(entry.term)), entry.count)
        });
        Self {
            base,
            runs: Vec::new(),
            buffer: Vec::new(),
            fingerprint,
        }
    }

    pub(super) fn count(&self, terms: &Interner<'_>, term: TermId) -> u32 {
        self.overlay_entry(terms, term).map_or_else(
            || self.base_entry(terms, term).map_or(0, |e| e.count),
            |e| e.count,
        )
    }

    /// Applies a signed current-fact count change without allocating in the
    /// common case. The first change since a snapshot creates one override.
    pub(super) fn change(&mut self, terms: &Interner<'_>, term: TermId, delta: i32) {
        let old = self.count(terms, term);
        let new = if delta >= 0 {
            old.saturating_add(delta as u32)
        } else {
            old.saturating_sub(delta.unsigned_abs())
        };
        if old == new {
            return;
        }
        let name = terms.resolve(term);
        if old != 0 {
            self.fingerprint ^= entry_hash(name, old);
        }
        if new != 0 {
            self.fingerprint ^= entry_hash(name, new);
        }

        if let Ok(at) = search(&self.buffer, terms, name) {
            self.buffer[at].count = new;
            return;
        }
        for run in self.runs.iter_mut().filter_map(Option::as_mut) {
            if let Ok(at) = search(run, terms, name) {
                run[at].count = new;
                return;
            }
        }

        let at = search(&self.buffer, terms, name).unwrap_err();
        self.buffer.insert(
            at,
            Entry {
                term: term.0,
                count: new,
            },
        );
        if self.buffer.len() >= RUN_BASE {
            self.flush_buffer(terms);
        }
    }

    fn flush_buffer(&mut self, terms: &Interner<'_>) {
        if self.buffer.is_empty() {
            return;
        }
        let mut carry = core::mem::take(&mut self.buffer);
        let mut level = 0usize;
        loop {
            if level == self.runs.len() {
                self.runs.push(Some(carry));
                break;
            }
            match self.runs[level].take() {
                None => {
                    self.runs[level] = Some(carry);
                    break;
                }
                Some(existing) => {
                    carry = merge(existing, carry, terms);
                    level += 1;
                }
            }
        }
    }

    fn base_entry(&self, terms: &Interner<'_>, term: TermId) -> Option<&Entry> {
        let name = terms.resolve(term);
        search(&self.base, terms, name)
            .ok()
            .map(|at| &self.base[at])
    }

    fn overlay_entry(&self, terms: &Interner<'_>, term: TermId) -> Option<&Entry> {
        let name = terms.resolve(term);
        if let Ok(at) = search(&self.buffer, terms, name) {
            return Some(&self.buffer[at]);
        }
        self.runs
            .iter()
            .filter_map(Option::as_ref)
            .find_map(|run| search(run, terms, name).ok().map(|at| &run[at]))
    }

    fn is_overridden(&self, terms: &Interner<'_>, term: u32) -> bool {
        self.overlay_entry(terms, TermId(term)).is_some()
    }

    pub(super) fn page(
        &self,
        terms: &Interner<'_>,
        db_uuid: u128,
        query: TagQuery<'_>,
    ) -> Result<TagPage, Error> {
        let limit = if query.limit == 0 {
            DEFAULT_TAG_PAGE_LIMIT
        } else {
            query.limit
        };
        if limit > MAX_TAG_PAGE_LIMIT {
            return Err(Error::TooLarge {
                what: "tag page limit",
                len: limit,
                max: MAX_TAG_PAGE_LIMIT,
            });
        }
        let prefix = query.prefix.unwrap_or("");
        let prefix_hash = xxh3_64(prefix.as_bytes());
        let after = match query.cursor {
            None => None,
            Some(cursor) => {
                let decoded = decode_cursor(cursor)?;
                if decoded.fingerprint != self.fingerprint
                    || decoded.prefix_hash != prefix_hash
                    || decoded.db_uuid != db_uuid
                {
                    return Err(Error::StaleCursor);
                }
                if decoded.term as usize >= terms.len()
                    || xxh3_64(terms.resolve(TermId(decoded.term)).as_bytes()) != decoded.term_hash
                {
                    return Err(Error::StaleCursor);
                }
                Some(terms.resolve(TermId(decoded.term)))
            }
        };

        let mut items = Vec::with_capacity(limit.saturating_add(1));
        // One monotonically advancing cursor per sorted source. Re-running a
        // binary search for every emitted tag would be bounded by `limit`, but
        // could repeatedly cross a long stretch of overridden base entries.
        // These positions make the complete page one k-way merge instead.
        let mut positions: Vec<usize> = self
            .sources()
            .map(|(_, source)| first_after(source, terms, after, prefix))
            .collect();
        while items.len() <= limit {
            let mut best: Option<(usize, Entry)> = None;
            for (source_idx, (is_base, source)) in self.sources().enumerate() {
                let mut at = positions[source_idx];
                while let Some(entry) = source.get(at).copied() {
                    let name = terms.resolve(TermId(entry.term));
                    if !name.starts_with(prefix) {
                        at = source.len();
                        break;
                    }
                    if entry.count != 0 && !(is_base && self.is_overridden(terms, entry.term)) {
                        if best.is_none_or(|(_, old)| name < terms.resolve(TermId(old.term))) {
                            best = Some((source_idx, entry));
                        }
                        break;
                    }
                    at += 1;
                }
                positions[source_idx] = at;
            }
            let Some((source_idx, entry)) = best else {
                break;
            };
            positions[source_idx] += 1;
            let name = terms.resolve(TermId(entry.term));
            items.push(TagSummary {
                name: name.to_string(),
                count: entry.count,
            });
        }

        let has_more = items.len() > limit;
        if has_more {
            items.pop();
        }
        let next_cursor = if has_more {
            let last = items.last().expect("a page with more has a last item");
            let term = terms
                .lookup(&last.name)
                .expect("catalog names are interned")
                .0;
            Some(encode_cursor(Cursor {
                fingerprint: self.fingerprint,
                prefix_hash,
                db_uuid,
                term,
                term_hash: xxh3_64(last.name.as_bytes()),
            }))
        } else {
            None
        };
        Ok(TagPage { items, next_cursor })
    }

    fn sources(&self) -> impl Iterator<Item = (bool, &[Entry])> {
        core::iter::once((true, self.base.as_slice()))
            .chain(core::iter::once((false, self.buffer.as_slice())))
            .chain(
                self.runs
                    .iter()
                    .filter_map(Option::as_deref)
                    .map(|run| (false, run)),
            )
    }

    /// Canonical merged snapshot section. Zero-count overrides remove base
    /// entries; names remain in the shared interner as historical residue.
    pub(super) fn dump(&self, terms: &Interner<'_>) -> Vec<u8> {
        let mut entries: Vec<Entry> = self
            .base
            .iter()
            .copied()
            .filter(|entry| !self.is_overridden(terms, entry.term))
            .collect();
        entries.extend(self.buffer.iter().copied().filter(|e| e.count != 0));
        for run in self.runs.iter().filter_map(Option::as_ref) {
            entries.extend(run.iter().copied().filter(|e| e.count != 0));
        }
        entries.sort_unstable_by(|a, b| {
            terms
                .resolve(TermId(a.term))
                .cmp(terms.resolve(TermId(b.term)))
        });
        let mut out = Vec::with_capacity(entries.len() * ENTRY_BYTES);
        for entry in entries {
            out.extend_from_slice(&entry.term.to_le_bytes());
            out.extend_from_slice(&entry.count.to_le_bytes());
        }
        out
    }

    pub(super) fn pool_bytes(&self) -> usize {
        let entries = self.base.len()
            + self.buffer.len()
            + self
                .runs
                .iter()
                .filter_map(Option::as_ref)
                .map(Vec::len)
                .sum::<usize>();
        entries * core::mem::size_of::<Entry>()
    }
}

fn search(entries: &[Entry], terms: &Interner<'_>, name: &str) -> Result<usize, usize> {
    entries.binary_search_by(|entry| terms.resolve(TermId(entry.term)).cmp(name))
}

fn first_after(
    entries: &[Entry],
    terms: &Interner<'_>,
    after: Option<&str>,
    prefix: &str,
) -> usize {
    let floor = after.unwrap_or(prefix);
    entries.partition_point(|entry| {
        let name = terms.resolve(TermId(entry.term));
        match after {
            Some(_) => name <= floor,
            None => name < floor,
        }
    })
}

fn merge(a: Vec<Entry>, b: Vec<Entry>, terms: &Interner<'_>) -> Vec<Entry> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut ai, mut bi) = (0usize, 0usize);
    while ai < a.len() || bi < b.len() {
        let ordering = match (a.get(ai), b.get(bi)) {
            (Some(left), Some(right)) => terms
                .resolve(TermId(left.term))
                .cmp(terms.resolve(TermId(right.term))),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => break,
        };
        match ordering {
            Ordering::Less => {
                out.push(a[ai]);
                ai += 1;
            }
            Ordering::Greater => {
                out.push(b[bi]);
                bi += 1;
            }
            Ordering::Equal => {
                debug_assert_eq!(a[ai].term, b[bi].term);
                out.push(b[bi]);
                ai += 1;
                bi += 1;
            }
        }
    }
    out
}

fn entry_hash(name: &str, count: u32) -> u64 {
    let mut hash = Xxh3::new();
    hash.update(name.as_bytes());
    hash.update(&count.to_le_bytes());
    hash.digest()
}

#[derive(Clone, Copy)]
struct Cursor {
    fingerprint: u64,
    prefix_hash: u64,
    db_uuid: u128,
    term: u32,
    term_hash: u64,
}

fn encode_cursor(cursor: Cursor) -> String {
    format!(
        "{CURSOR_PREFIX}:{:016x}:{:016x}:{:032x}:{:08x}:{:016x}",
        cursor.fingerprint, cursor.prefix_hash, cursor.db_uuid, cursor.term, cursor.term_hash
    )
}

fn decode_cursor(raw: &str) -> Result<Cursor, Error> {
    let mut parts = raw.split(':');
    let valid = parts.next() == Some(CURSOR_PREFIX);
    let fingerprint = parts.next().and_then(|s| u64::from_str_radix(s, 16).ok());
    let prefix_hash = parts.next().and_then(|s| u64::from_str_radix(s, 16).ok());
    let db_uuid = parts.next().and_then(|s| u128::from_str_radix(s, 16).ok());
    let term = parts.next().and_then(|s| u32::from_str_radix(s, 16).ok());
    let term_hash = parts.next().and_then(|s| u64::from_str_radix(s, 16).ok());
    if !valid
        || parts.next().is_some()
        || fingerprint.is_none()
        || prefix_hash.is_none()
        || db_uuid.is_none()
        || term.is_none()
        || term_hash.is_none()
    {
        return Err(Error::Invalid("malformed tag cursor"));
    }
    Ok(Cursor {
        fingerprint: fingerprint.unwrap(),
        prefix_hash: prefix_hash.unwrap(),
        db_uuid: db_uuid.unwrap(),
        term: term.unwrap(),
        term_hash: term_hash.unwrap(),
    })
}
