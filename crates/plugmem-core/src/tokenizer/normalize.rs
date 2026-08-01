//! Input normalization for the tokenizer.

use alloc::string::String;

use unicode_normalization::UnicodeNormalization;

/// Replaces `out` with the NFKC form of `text`.
pub(super) fn normalize_into(text: &str, out: &mut String) {
    out.clear();
    if text.is_ascii() {
        // NFKC is the identity on ASCII — avoid walking the normalization
        // tables on the common English and identifier path.
        out.push_str(text);
    } else {
        out.extend(text.nfkc());
    }
}
