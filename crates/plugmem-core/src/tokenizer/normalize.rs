//! Input normalization for the tokenizer.

use alloc::string::String;

use super::unicode::UnicodeBackend;

/// Replaces `out` with the NFKC form of `text`.
pub(super) fn normalize_into(backend: &UnicodeBackend, text: &str, out: &mut String) {
    // ICU4X performs the same NFKC identity fast path internally for already
    // normalized input while covering all Unicode versions represented by its
    // compiled data.
    backend.normalize_into(text, out);
}
