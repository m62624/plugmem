//! Unicode folding of segmented lexical text.

use alloc::string::String;

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::{decompose_canonical, is_combining_mark};

use super::emit::emit_truncated;
use super::policy::TokenizerPolicy;
use super::tables;

/// Folds one UAX #29 segment into the reusable token scratch buffers.
pub(super) fn fold_segment(
    segment: &str,
    token: &mut String,
    canonical: &mut String,
    policy: TokenizerPolicy,
    sink: &mut dyn FnMut(&str),
) {
    token.clear();
    let mut needs_nfkc_again = false;
    for c in segment.chars() {
        for lowercase in c.to_lowercase() {
            if fold_into(lowercase, token, policy, &mut needs_nfkc_again) {
                emit_folded(token, canonical, needs_nfkc_again, policy, sink);
                token.clear();
                needs_nfkc_again = false;
            }
        }
    }
    emit_folded(token, canonical, needs_nfkc_again, policy, sink);
}

/// `true` for format characters that carry no lexical content in running
/// text. UAX #29 may keep these characters inside a word segment, so they
/// must be removed without turning them into a token boundary.
#[inline]
fn is_ignorable_format(c: char) -> bool {
    tables::in_ranges(c, tables::IGNORABLE_FORMAT_RANGES)
}

/// Pushes one lowercased character into the token, applying project folding
/// rules. Returns `true` when the character is a boundary after an assembled
/// token.
fn fold_into(
    c: char,
    out: &mut String,
    policy: TokenizerPolicy,
    needs_nfkc_again: &mut bool,
) -> bool {
    if policy.fold_russian_yo
        && let Some(mapped) = tables::mapped_char(c, tables::RUSSIAN_SEARCH_FOLDS)
    {
        out.push(mapped);
        return false;
    }
    if is_ignorable_format(c) {
        return false;
    }
    if is_combining_mark(c) {
        *needs_nfkc_again = true;
        if !out.ends_with(|previous: char| previous.is_ascii_alphanumeric()) {
            out.push(c);
        }
        return false;
    }
    if !c.is_alphanumeric() {
        if out.ends_with(char::is_alphanumeric) && tables::is_word_joiner(c) {
            out.push(c);
            return false;
        }
        return !out.is_empty();
    }
    if c.is_ascii() {
        out.push(c);
        return false;
    }

    // Canonical decompositions are short; keeping this fixed-size avoids a
    // temporary allocation in the hot folding path.
    let mut parts = [char::MAX; 8];
    let mut count = 0usize;
    decompose_canonical(c, |decomposed| {
        if count < parts.len() {
            parts[count] = decomposed;
        }
        count += 1;
    });
    if policy.fold_latin_diacritics
        && count <= parts.len()
        && count > 0
        && parts[0].is_ascii_alphanumeric()
    {
        for &decomposed in &parts[..count] {
            if !is_combining_mark(decomposed) {
                out.push(decomposed);
            }
        }
    } else {
        out.push(c);
    }
    false
}

/// Re-normalizes a folded token only when lowercasing exposed combining
/// marks. Lowercase can change a base character while leaving a mark sequence
/// whose canonical order is visible only after folding.
fn emit_folded(
    token: &mut String,
    canonical: &mut String,
    needs_nfkc_again: bool,
    policy: TokenizerPolicy,
    sink: &mut dyn FnMut(&str),
) {
    if needs_nfkc_again {
        canonical.clear();
        canonical.extend(token.nfkc());
        token.clear();
        let mut boundary = false;
        for c in canonical.chars() {
            for lowercase in c.to_lowercase() {
                if fold_into(lowercase, token, policy, &mut boundary) {
                    drop_leading_marks_before_base(token);
                    emit_truncated(token, sink);
                    token.clear();
                    boundary = false;
                }
            }
        }
        drop_leading_marks_before_base(token);
        emit_truncated(token, sink);
    } else {
        emit_truncated(token, sink);
    }
}

/// Removes combining marks that precede a real token character. A mark-only
/// token remains valid for compatibility with the existing tokenizer.
fn drop_leading_marks_before_base(token: &mut String) {
    let mut prefix = 0usize;
    for (offset, c) in token.char_indices() {
        if is_combining_mark(c) {
            prefix = offset + c.len_utf8();
        } else {
            break;
        }
    }
    if prefix == token.len() {
        if !token.chars().any(char::is_alphanumeric) {
            token.clear();
        }
        return;
    }
    if prefix < token.len() {
        token.replace_range(..prefix, "");
    }
}
