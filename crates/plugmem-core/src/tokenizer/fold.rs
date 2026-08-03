//! Unicode folding of segmented lexical text.

use alloc::string::String;

use unicode_normalization::char::decompose_canonical;

use super::emit::emit_truncated;
use super::policy::TokenizerPolicy;
use super::tables;
use super::unicode::UnicodeBackend;

/// Folds one UAX #29 segment into the reusable token scratch buffers.
pub(super) fn fold_segment(
    segment: &str,
    token: &mut String,
    canonical: &mut String,
    policy: TokenizerPolicy,
    unicode: &UnicodeBackend,
    sink: &mut dyn FnMut(&str),
) {
    token.clear();
    let mut needs_nfkc_again = false;
    for c in segment.chars() {
        if fold_into(c, token, policy, &mut needs_nfkc_again) {
            emit_folded(token, canonical, needs_nfkc_again, policy, unicode, sink);
            token.clear();
            needs_nfkc_again = false;
        }
    }
    emit_folded(token, canonical, needs_nfkc_again, policy, unicode, sink);
}

/// `true` for format characters that carry no lexical content in running
/// text. UAX #29 may keep these characters inside a word segment, so they
/// must be removed without turning them into a token boundary.
#[inline]
fn is_ignorable_format(c: char) -> bool {
    UnicodeBackend::is_default_ignorable(c)
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
    if UnicodeBackend::is_mark(c) {
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
            if !UnicodeBackend::is_mark(decomposed) {
                out.push(decomposed);
            }
        }
    } else {
        out.push(c);
    }
    false
}

/// Re-normalizes a folded token when folding exposed Unicode marks.
///
/// One pass is not always enough: a policy fold can expose a new composition,
/// and that composition can expose another foldable mark sequence. For
/// example, `ё + diaeresis + grave` becomes `е + grave`, which NFKC composes
/// to `ѐ` on the next pass. Keep applying the pass until the fold no longer
/// changes the normalized spelling. If the mark is not composable, the
/// normalized and folded buffers are already equal and the loop stops.
fn emit_folded(
    token: &mut String,
    canonical: &mut String,
    needs_nfkc_again: bool,
    policy: TokenizerPolicy,
    unicode: &UnicodeBackend,
    sink: &mut dyn FnMut(&str),
) {
    let mut needs_nfkc_again = needs_nfkc_again;
    while needs_nfkc_again {
        unicode.normalize_into(token, canonical);
        token.clear();
        needs_nfkc_again = false;
        for c in canonical.chars() {
            if fold_into(c, token, policy, &mut needs_nfkc_again) {
                emit_truncated(token, sink);
                token.clear();
                needs_nfkc_again = false;
            }
        }

        // A non-composable mark remains after NFKC. Folding it again would
        // produce the same token forever, so equality is the fixed-point
        // condition for the remaining mark path.
        if needs_nfkc_again && token == canonical {
            needs_nfkc_again = false;
        }
    }
    emit_truncated(token, sink);
}
