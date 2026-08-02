//! Canonical token emission and byte-budget handling.

use super::tables::is_word_joiner;
use unicode_normalization::char::is_combining_mark;

/// Upper bound on an emitted token, in bytes.
pub const MAX_TOKEN_BYTES: usize = 64;

/// Sends the assembled token, truncated to [`MAX_TOKEN_BYTES`] at a char
/// boundary. Word joiners are meaningful only between lexical characters, so
/// leading and trailing joiners are removed without allocating. Keeping one
/// would make re-tokenization context-sensitive (for example, `_word` →
/// `word` and `word.` → `word`). Empty and non-lexical tokens are guarded
/// against rather than asserted.
pub(super) fn emit_truncated(token: &str, sink: &mut dyn FnMut(&str)) {
    let token = token.trim_start_matches(is_word_joiner);
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
