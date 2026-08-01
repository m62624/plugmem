//! Static Unicode policy tables used by the tokenizer.
//!
//! The tables in this module are intentionally small and allocation-free. They
//! describe project policy (CJK bigram sources, ignorable format characters,
//! lexical joiners, and optional language folds), while Unicode's complete
//! alphabetic and numeric properties remain delegated to `char`.

/// Inclusive Unicode scalar-value ranges for dictionary-free CJK bigrams.
pub(crate) const CJK_UNIGRAM_RANGES: &[(u32, u32)] = &[
    (0x3400, 0x4DBF),   // CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   // CJK Unified Ideographs
    (0xF900, 0xFAFF),   // CJK Compatibility Ideographs
    (0x20000, 0x2FFFF), // CJK Extensions B–T
    (0x3041, 0x309F),   // Hiragana
];

/// Inclusive Unicode scalar-value ranges for default-ignorable format marks
/// that UAX #29 may keep inside a word segment.
pub(crate) const IGNORABLE_FORMAT_RANGES: &[(u32, u32)] = &[
    (0x00AD, 0x00AD), // SOFT HYPHEN
    (0x200B, 0x200F), // zero-width and bidi marks
    (0x202A, 0x202E), // bidi embedding/override marks
    (0x2060, 0x2064), // word joiner and invisible operators
    (0xFEFF, 0xFEFF), // ZERO WIDTH NO-BREAK SPACE / BOM
];

/// Punctuation allowed between two lexical characters.
pub(crate) const WORD_JOINERS: &[char] = &['\'', '\u{2019}', '.', ',', '_'];

/// Search-policy folds applied after Unicode lowercase. The table is kept
/// separate from the scanner so language-specific behavior is visible and
/// extensible instead of being hidden in a character branch.
pub(crate) const RUSSIAN_SEARCH_FOLDS: &[(char, char)] = &[('ё', 'е')];

/// Returns whether `c` belongs to one of the inclusive scalar-value ranges.
#[inline]
pub(crate) fn in_ranges(c: char, ranges: &[(u32, u32)]) -> bool {
    let value = c as u32;
    ranges
        .iter()
        .any(|&(start, end)| (start..=end).contains(&value))
}

/// Returns the mapped character for a small policy table, if present.
#[inline]
pub(crate) fn mapped_char(c: char, table: &[(char, char)]) -> Option<char> {
    table
        .iter()
        .find_map(|&(from, to)| (from == c).then_some(to))
}
