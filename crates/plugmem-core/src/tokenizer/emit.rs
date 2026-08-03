//! Canonical token emission and byte-budget handling.

use super::{tables::is_word_joiner, unicode::UnicodeBackend};
use unicode_normalization::char::canonical_combining_class;

/// Upper bound on an emitted token, in bytes.
pub const MAX_TOKEN_BYTES: usize = 64;

/// Sends the assembled token, truncated to [`MAX_TOKEN_BYTES`] at a char
/// boundary. Unicode marks and word joiners are meaningful only after a
/// lexical base, so leading marks and joiners are removed without allocating.
/// Keeping either would make re-tokenization context-sensitive (for example,
/// `\u{300}word` → `word`, `_word` → `word`, and `word.` → `word`). A mark-only
/// token is retained only when it is a single mark with Unicode Alphabetic
/// semantics; this keeps valid standalone script marks such as U+0F71
/// searchable without letting contextual mark/filler runs escape as tokens.
/// Empty and non-lexical tokens are guarded against rather than asserted.
pub(super) fn emit_truncated(token: &str, sink: &mut dyn FnMut(&str)) {
    let token = trim_leading_contextual_chars(token);
    // Some Unicode marks carry the `Alphabetic` property and are
    // valid standalone search tokens (for example U+0F71). Others, such as
    // U+0300, are only an Extend character: ICU will not tokenize them when
    // presented alone. Never emit a mark-only/non-alphabetic token, because
    // it cannot satisfy the tokenizer's fixed-point contract.
    if token.is_empty() || !has_emit_lexical_content(token) {
        return;
    }
    if let Some((start, end)) = invalid_apostrophe_joiner(token) {
        emit_truncated(&token[..start], sink);
        emit_truncated(&token[end..], sink);
        return;
    }
    let mut end = token.len().min(MAX_TOKEN_BYTES);
    while !token.is_char_boundary(end) {
        end -= 1;
    }

    // A mark adjacent to a trailing joiner is not lexical content of the
    // token: in isolation UAX #29 breaks it away after the joiner is trimmed.
    // Remove the whole contextual suffix, regardless of whether the mark or
    // joiner comes last (`word_◌`, `word◌_`, or `word_◌_`). Marks following a
    // real base without a trailing joiner are retained.
    let original_end = end;
    let mut saw_trailing_joiner = false;
    while end > 0 {
        let Some(c) = token[..end].chars().next_back() else {
            break;
        };
        if is_word_joiner(c) {
            saw_trailing_joiner = true;
            end -= c.len_utf8();
        } else if UnicodeBackend::is_mark(c) {
            end -= c.len_utf8();
        } else {
            if !saw_trailing_joiner {
                end = original_end;
            }
            break;
        }
    }
    if end == 0 && !saw_trailing_joiner {
        end = original_end;
    }
    if end == 0 {
        return;
    }
    if !has_emit_lexical_content(&token[..end]) {
        return;
    }
    sink(&token[..end]);
}

/// Removes Unicode marks that precede a lexical base without allocating.
///
/// A leading mark can be attached to a preceding segment by UAX #29, while
/// retokenizing the emitted string sees it as a standalone prefix. Removing
/// that prefix makes emission independent of the surrounding input. A token
/// made entirely of a Unicode-alphabetic mark is kept because it
/// is a valid standalone token for scripts that use such marks as letters.
fn trim_leading_marks(token: &str) -> &str {
    let mut prefix = 0usize;
    for (offset, c) in token.char_indices() {
        if UnicodeBackend::is_mark(c) {
            prefix = offset + c.len_utf8();
        } else {
            break;
        }
    }
    // Leave a mark-only token untouched. The lexical-content guard in the
    // caller decides whether that standalone mark is meaningful.
    if prefix == token.len() {
        token
    } else {
        &token[prefix..]
    }
}

/// Removes any sequence of leading marks and word joiners. The loop matters
/// for inputs such as `'.' + U+0300 + 'o'`: removing the joiner exposes a
/// leading mark that must be removed in the next pass.
fn trim_leading_contextual_chars(mut token: &str) -> &str {
    loop {
        let next = trim_leading_marks(token).trim_start_matches(is_word_joiner);
        if next.len() == token.len() {
            return token;
        }
        token = next;
    }
}

fn invalid_apostrophe_joiner(token: &str) -> Option<(usize, usize)> {
    let mut chars = token.char_indices().peekable();
    while let Some((offset, c)) = chars.next() {
        if c != '\'' && c != '\u{2019}' {
            continue;
        }
        let left = previous_non_mark(&token[..offset]).is_some_and(is_letter);
        let right = next_non_mark(chars.clone()).is_some_and(is_letter);
        if !left || !right {
            return Some((offset, offset + c.len_utf8()));
        }
    }
    None
}

fn previous_non_mark(text: &str) -> Option<char> {
    text.chars().rev().find(|&c| !UnicodeBackend::is_mark(c))
}

fn next_non_mark<'a>(chars: impl Iterator<Item = (usize, char)> + 'a) -> Option<char> {
    chars.map(|(_, c)| c).find(|&c| !UnicodeBackend::is_mark(c))
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic() || UnicodeBackend::is_alphabetic(c)
}

fn has_emit_lexical_content(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let mut only_one = true;
    let mut all_marks = UnicodeBackend::is_mark(first);
    if !all_marks && is_lexical_base(first) {
        return true;
    }
    for c in chars {
        only_one = false;
        let is_mark = UnicodeBackend::is_mark(c);
        all_marks &= is_mark;
        if !is_mark && is_lexical_base(c) {
            return true;
        }
    }
    all_marks && only_one && first.is_alphanumeric() && canonical_combining_class(first) != 0
}

fn is_lexical_base(c: char) -> bool {
    (c.is_alphanumeric() || UnicodeBackend::is_alphabetic(c)) && !UnicodeBackend::is_mark(c)
}
