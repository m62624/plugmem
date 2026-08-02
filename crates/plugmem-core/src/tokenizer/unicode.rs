//! ICU4X-backed Unicode primitives used by the production tokenizer.

use alloc::string::String;

use icu_casemap::CaseMapperBorrowed;
use icu_locale_core::LanguageIdentifier;
use icu_normalizer::ComposingNormalizerBorrowed;
use icu_properties::{
    CodePointMapData, CodePointSetData,
    props::{Alphabetic, DefaultIgnorableCodePoint, GeneralCategory, Ideographic, Script},
};
use icu_segmenter::{
    WordSegmenter, WordSegmenterBorrowed, iterators::WordBreakIterator,
    options::WordBreakInvariantOptions, scaffold::Utf8,
};
use writeable::Writeable;

/// Borrowed, compiled Unicode data shared by tokenizer instances.
///
/// The data itself is static. [`Tokenizer`](super::Tokenizer) owns only the
/// mutable scratch strings needed to materialize normalized text and lowercase
/// segments. ICU4X's automatic word segmenter includes dictionary/LSTM data for
/// scripts whose word boundaries cannot be recovered from UAX #29 rules alone.
#[derive(Debug)]
pub(super) struct UnicodeBackend {
    normalizer: ComposingNormalizerBorrowed<'static>,
    case_mapper: CaseMapperBorrowed<'static>,
    word_segmenter: WordSegmenterBorrowed<'static>,
    root_locale: LanguageIdentifier,
}

impl UnicodeBackend {
    /// Creates the production ICU4X backend with complex-script data enabled.
    pub(super) fn new() -> Self {
        Self {
            normalizer: ComposingNormalizerBorrowed::new_nfkc(),
            case_mapper: CaseMapperBorrowed::new(),
            word_segmenter: WordSegmenter::new_auto(WordBreakInvariantOptions::default()),
            root_locale: LanguageIdentifier::UNKNOWN,
        }
    }

    /// Replaces `output` with ICU4X NFKC normalization of `text`.
    #[inline]
    pub(super) fn normalize_into(&self, text: &str, output: &mut String) {
        output.clear();
        let _ = self.normalizer.normalize_to(text, output);
    }

    /// Replaces `output` with locale-neutral Unicode lowercase text.
    #[inline]
    pub(super) fn lowercase_into(&self, text: &str, output: &mut String) {
        output.clear();
        let _ = self
            .case_mapper
            .lowercase(text, &self.root_locale)
            .write_to(output);
    }

    /// Returns UAX #29 boundaries with ICU4X complex-script tailoring.
    #[inline]
    pub(super) fn word_boundaries<'text>(
        &self,
        text: &'text str,
    ) -> WordBreakIterator<'static, 'text, Utf8> {
        self.word_segmenter.segment_str(text)
    }

    /// Returns whether a character is Unicode default-ignorable format data.
    #[inline]
    pub(super) fn is_default_ignorable(c: char) -> bool {
        CodePointSetData::new::<DefaultIgnorableCodePoint>().contains(c)
    }

    /// Returns whether a character is Unicode-alphabetic, including marks
    /// such as Tibetan U+0F71 that are valid standalone lexical terms but are
    /// not considered alphanumeric by Rust's `char` predicate.
    #[inline]
    pub(super) fn is_alphabetic(c: char) -> bool {
        CodePointSetData::new::<Alphabetic>().contains(c)
    }

    /// Returns whether a character has one of Unicode's three `Mark`
    /// general categories. This is intentionally broader than canonical
    /// combining class: some script marks, such as Sinhala U+0D81, are
    /// `NonspacingMark` characters with combining class zero.
    #[inline]
    pub(super) fn is_mark(c: char) -> bool {
        matches!(
            CodePointMapData::<GeneralCategory>::new().get(c),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
    }

    /// Returns whether a character belongs to the CJK/Hiragana bigram policy.
    ///
    /// `Ideographic` covers assigned Han ideographs, including compatibility
    /// ideographs, while `Script::Hiragana` retains the project's existing
    /// Japanese Hiragana bigram behavior without a hand-maintained range list.
    /// Mark characters are excluded even if another Unicode property also
    /// classifies them as ideographic: they have no standalone base glyph for
    /// the CJK adjacency machine and would bypass canonical token emission.
    #[inline]
    pub(super) fn is_cjk_unigram(c: char) -> bool {
        !Self::is_mark(c)
            && (CodePointSetData::new::<Ideographic>().contains(c)
                || CodePointMapData::<Script>::new().get(c) == Script::Hiragana)
    }
}

impl Default for UnicodeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for UnicodeBackend {
    fn clone(&self) -> Self {
        Self::new()
    }
}
