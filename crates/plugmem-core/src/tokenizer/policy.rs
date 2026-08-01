//! Search policies for tokenizer folding.

/// Controls search-specific folding without changing the scanner or its
/// allocation behavior.
///
/// The default preserves the tokenizer's existing index contract: Latin
/// diacritics are folded for recall and Russian `ё` is treated as `е`. Use
/// [`TokenizerPolicy::unicode`] when callers need Unicode lowercase and
/// normalization without either language/search-specific equivalence.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TokenizerPolicy {
    /// Fold Latin precomposed diacritics to their ASCII base.
    pub fold_latin_diacritics: bool,
    /// Treat Russian small letter `ё` as `е` after lowercase.
    pub fold_russian_yo: bool,
}

impl TokenizerPolicy {
    /// A language-neutral policy: Unicode lowercase and normalization only.
    pub const fn unicode() -> Self {
        Self {
            fold_latin_diacritics: false,
            fold_russian_yo: false,
        }
    }

    /// The search policy used by [`crate::tokenizer::Tokenizer::new`] and
    /// existing indexes.
    pub const fn search() -> Self {
        Self {
            fold_latin_diacritics: true,
            fold_russian_yo: true,
        }
    }
}

impl Default for TokenizerPolicy {
    fn default() -> Self {
        Self::search()
    }
}
