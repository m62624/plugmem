//! Static Unicode policy tables used by the tokenizer.
//!
//! The tables in this module are intentionally small and allocation-free. They
//! describe project policy (lexical joiners and optional language folds), while
//! Unicode's complete properties are delegated to ICU4X.

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

/// Returns the mapped character for a small policy table, if present.
#[inline]
pub(crate) fn mapped_char(c: char, table: &[(char, char)]) -> Option<char> {
    table
        .iter()
        .find_map(|&(from, to)| (from == c).then_some(to))
}
