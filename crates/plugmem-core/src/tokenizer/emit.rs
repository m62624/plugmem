//! Canonical token emission and byte-budget handling.

use super::tables::is_word_joiner;
use unicode_normalization::char::is_combining_mark;

/// Upper bound on an emitted token, in bytes.
pub const MAX_TOKEN_BYTES: usize = 64;

/// Sends the assembled token, truncated to [`MAX_TOKEN_BYTES`] at a char
/// boundary. Combining marks and word joiners are meaningful only after a
/// lexical base, so leading marks and joiners are removed without allocating.
/// Keeping either would make re-tokenization context-sensitive (for example,
/// `\u{300}word` → `word`, `_word` → `word`, and `word.` → `word`). A mark-only
/// token is retained when it has Unicode alphanumeric semantics; this keeps
/// valid standalone script marks such as U+0F71 searchable. Empty and
/// non-lexical tokens are guarded against rather than asserted.
pub(super) fn emit_truncated(token: &str, sink: &mut dyn FnMut(&str)) {
    let token = trim_leading_contextual_chars(token);
    // Some Unicode combining marks carry the `Alphabetic` property and are
    // valid standalone search tokens (for example U+0F71). Others, such as
    // U+0300, are only an Extend character: ICU will not tokenize them when
    // presented alone. Never emit a mark-only/non-alphanumeric token, because
    // it cannot satisfy the tokenizer's fixed-point contract.
    if token.is_empty() || !token.chars().any(char::is_alphanumeric) {
        return;
    }
    let mut end = token.len().min(MAX_TOKEN_BYTES);
    while !token.is_char_boundary(end) {
        end -= 1;
    }

    // A combining mark after a trailing joiner is not lexical content of the
    // token: in isolation UAX #29 breaks it away after the joiner is trimmed.
    // Remove that suffix so emission remains a fixed point. Marks following a
    // real base are retained, including mark-only compatibility cases.
    let before_marks = end;
    while end > 0 {
        let Some(c) = token[..end].chars().next_back() else {
            break;
        };
        if !is_combining_mark(c) {
            break;
        }
        end -= c.len_utf8();
    }
    if end > 0 {
        let Some(c) = token[..end].chars().next_back() else {
            return;
        };
        if !is_word_joiner(c) {
            end = before_marks;
        }
    } else {
        end = before_marks;
    }

    while end > 0 {
        let Some(c) = token[..end].chars().next_back() else {
            // `end > 0` and `&str`'s UTF-8 invariant make this unreachable;
            // keep the emission API total if that invariant ever changes.
            break;
        };
        if !is_word_joiner(c) {
            break;
        }
        end -= c.len_utf8();
    }
    if end == 0 {
        return;
    }
    sink(&token[..end]);
}

/// Removes combining marks that precede a lexical base without allocating.
///
/// A leading mark can be attached to a preceding segment by UAX #29, while
/// retokenizing the emitted string sees it as a standalone prefix. Removing
/// that prefix makes emission independent of the surrounding input. A token
/// made entirely of a Unicode-alphanumeric combining mark is kept because it
/// is a valid standalone token for scripts that use such marks as letters.
fn trim_leading_marks(token: &str) -> &str {
    let mut prefix = 0usize;
    for (offset, c) in token.char_indices() {
        if is_combining_mark(c) {
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
