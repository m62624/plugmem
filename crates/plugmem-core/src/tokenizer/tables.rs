//! Static Unicode policy tables used by the tokenizer.
//!
//! The tables in this module are intentionally small and allocation-free. They
//! describe project policy (CJK bigram sources, ignorable format characters,
//! lexical joiners, and optional language folds), while Unicode's complete
//! alphabetic and numeric properties remain delegated to `char`.

/// Inclusive assigned Unicode scalar-value ranges for dictionary-free CJK
/// bigrams. These are the Unicode 17 ranges used by the Rust toolchain; the
/// small gaps matter because an unassigned scalar must not join two adjacent
/// ideographs into a false bigram.
pub(crate) const CJK_UNIGRAM_RANGES: &[(u32, u32)] = &[
    (0x3400, 0x4DBF), // CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF), // CJK Unified Ideographs
    (0xF900, 0xFA6D), // CJK Compatibility Ideographs
    (0xFA70, 0xFAD9),
    (0x20000, 0x2A6DF), // CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F), // Extension C
    (0x2B740, 0x2B81D), // Extension D
    (0x2B820, 0x2CEAD), // Extension E
    (0x2CEB0, 0x2EBE0), // Extension F
    (0x2EBF0, 0x2EE5D), // Extension I
    (0x30000, 0x3134A), // Extension G
    (0x31350, 0x323AF), // Extension H
    (0x323B0, 0x33479), // Extension J
    (0x2F800, 0x2FA1D), // CJK Compatibility Ideographs Supplement
    (0x3041, 0x3096),   // Hiragana
    (0x309D, 0x309F),   // Hiragana iteration marks
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

/// Returns whether punctuation may join two lexical characters.
#[inline]
pub(crate) fn is_word_joiner(c: char) -> bool {
    WORD_JOINERS.contains(&c)
}

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
