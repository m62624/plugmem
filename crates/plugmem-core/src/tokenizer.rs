//! The core tokenizer (specs/04 §1).
//!
//! Lives in the engine because indexing must work identically everywhere,
//! including wasm — so it uses nothing but `core`:
//!
//! - a token is a maximal run of [`char::is_alphanumeric`] characters;
//!   everything else separates. Inner `'` and `-` do **not** join words
//!   (v1: simplicity over morphology). Digits tokenize as-is.
//! - normalization is [`char::to_lowercase`] — the full Unicode simple
//!   case mapping whose tables are already inside every Rust binary.
//! - no stemming or lemmatization in v1.
//! - CJK ideographs become single-character tokens (unigrams) — honest but
//!   weak for v1; bigrams are an open question in the spec. Kana and
//!   Hangul currently tokenize as runs like any other alphabet.
//! - a token longer than [`MAX_TOKEN_BYTES`] is truncated at the last
//!   char boundary that fits (long tokens sharing a 64-byte prefix
//!   therefore collapse — accepted by spec).

use alloc::string::String;

/// Upper bound on an emitted token, in bytes (specs/04).
pub const MAX_TOKEN_BYTES: usize = 64;

/// `true` for CJK unified ideographs (BMP blocks, compatibility block and
/// the supplementary-plane extensions) — the characters emitted as
/// unigrams.
fn is_cjk_ideograph(c: char) -> bool {
    matches!(
        c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FFFF
    )
}

/// Emits the buffered run (truncated to [`MAX_TOKEN_BYTES`]) and clears
/// the buffer. Empty buffer = no-op.
fn flush(buf: &mut String, sink: &mut impl FnMut(&str)) {
    if buf.is_empty() {
        return;
    }
    let mut end = buf.len().min(MAX_TOKEN_BYTES);
    while !buf.is_char_boundary(end) {
        end -= 1;
    }
    sink(&buf[..end]);
    buf.clear();
}

/// Splits `text` into normalized tokens, calling `sink` for each one.
///
/// `buf` is the caller-owned scratch (reused across calls — after warm-up
/// tokenization allocates nothing, which the zero-alloc recall invariant
/// depends on). The emitted `&str` is only valid for the duration of one
/// `sink` call.
///
/// ```
/// use plugmem_core::tokenizer::tokenize;
///
/// let mut buf = String::new();
/// let mut tokens = Vec::new();
/// tokenize("Hello, МИР-42!", &mut buf, |t| tokens.push(t.to_owned()));
/// assert_eq!(tokens, ["hello", "мир", "42"]);
/// ```
pub fn tokenize(text: &str, buf: &mut String, mut sink: impl FnMut(&str)) {
    buf.clear();
    for c in text.chars() {
        if is_cjk_ideograph(c) {
            // Unigram: whatever run was open ends, the ideograph stands
            // alone (lowercase is the identity for ideographs).
            flush(buf, &mut sink);
            buf.push(c);
            flush(buf, &mut sink);
        } else if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                // The mapping can emit non-alphanumeric combining marks
                // (the one case in Unicode: 'İ' → "i\u{307}"); dropping
                // them keeps every token a fixed point of tokenization.
                if lc.is_alphanumeric() {
                    buf.push(lc);
                }
            }
        } else {
            flush(buf, &mut sink);
        }
    }
    flush(buf, &mut sink);
}
