//! The core tokenizer, v2 (specs/04 §1).
//!
//! The pipeline mirrors what the strongest lexical engines (Lucene's
//! ICU/Standard analyzers, SQLite FTS5 `unicode61`) converge on, built
//! from the pure-`core` unicode-rs table crates so it runs identically on
//! native and every wasm runtime:
//!
//! 1. **NFKC normalization** of the input (fullwidth `Ａ` → `A`,
//!    ligature `ﬁ` → `fi`, decomposed marks recomposed) — one pass into a
//!    reused scratch buffer.
//! 2. **UAX #29 word segmentation** (`unicode-segmentation`) — the same
//!    boundary standard ICU implements. Consequences worth knowing:
//!    `don't` and `o'clock` stay whole (apostrophe joins letters),
//!    `3.14` / `v1.2.3` / `example.com` stay whole (`.`/`,` join
//!    digits and letters), `snake_case` stays whole (`_` joins),
//!    `gpt-4o` splits on the hyphen.
//! 3. **Per-token folding**: full Unicode lowercase; Latin diacritics
//!    stripped (`café` → `cafe`, the FTS5 `remove_diacritics` behavior,
//!    applied only to Latin bases so Cyrillic `й` is untouched); the
//!    Russian-specific `ё` → `е` fold every major Russian search engine
//!    applies.
//! 4. **CJK**: Han ideographs and Hiragana come out of UAX #29 as
//!    single-character segments; adjacent ones are joined into
//!    overlapping **bigrams** (the Lucene `CJKBigramFilter` scheme — the
//!    standard dictionary-free CJK treatment), a lone character stays a
//!    unigram. Katakana and Hangul already segment into word runs and are
//!    kept as words.
//! 5. A token longer than [`MAX_TOKEN_BYTES`] is truncated at the last
//!    char boundary that fits (long tokens sharing a 64-byte prefix
//!    collapse — accepted by spec).
//!
//! No stemming or lemmatization in v1. Emitted tokens are canonical: a
//! fixed point of the tokenizer (fixed by a property test).

use alloc::string::String;

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::{decompose_canonical, is_combining_mark};
use unicode_segmentation::UnicodeSegmentation;

/// Upper bound on an emitted token, in bytes (specs/04).
pub const MAX_TOKEN_BYTES: usize = 64;

/// `true` for characters treated as CJK unigram sources: Han ideographs
/// (BMP blocks, the compatibility block, supplementary-plane extensions)
/// and Hiragana. Katakana and Hangul are excluded on purpose — UAX #29
/// already groups them into word runs.
fn is_cjk_unigram(c: char) -> bool {
    matches!(
        c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FFFF
            | 0x3041..=0x309F
    )
}

/// Streaming tokenizer with reusable scratch buffers.
///
/// One instance per engine (or per thread of a wrapper): after warm-up
/// [`Tokenizer::tokenize`] allocates nothing, which the zero-alloc recall
/// invariant depends on.
#[derive(Debug, Default, Clone)]
pub struct Tokenizer {
    /// NFKC-normalized copy of the input.
    norm: String,
    /// The token being assembled (folded word or CJK bigram).
    token: String,
}

impl Tokenizer {
    /// A tokenizer with empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Splits `text` into normalized tokens, calling `sink` for each one.
    ///
    /// The emitted `&str` is only valid for the duration of one `sink`
    /// call.
    ///
    /// ```
    /// use plugmem_core::tokenizer::Tokenizer;
    ///
    /// let mut tk = Tokenizer::new();
    /// let mut tokens = Vec::new();
    /// tk.tokenize("Hello, МИР-42! 東京タワー", &mut |t| tokens.push(t.to_owned()));
    /// assert_eq!(tokens, ["hello", "мир", "42", "東京", "タワー"]);
    /// ```
    pub fn tokenize(&mut self, text: &str, sink: &mut dyn FnMut(&str)) {
        self.norm.clear();
        if text.is_ascii() {
            // NFKC is the identity on ASCII — skip the table walk (the
            // common English/code case; ~3x on the ASCII benchmark).
            self.norm.push_str(text);
        } else {
            self.norm.extend(text.nfkc());
        }
        let token = &mut self.token;

        // The CJK adjacency machine: previous unigram char + run length.
        // At a run boundary the machine may still owe a token: a lone
        // char is a unigram; longer runs were already emitted as bigrams.
        let mut prev_cjk: Option<char> = None;
        let mut run_len = 0usize;
        fn flush_cjk(
            prev: &mut Option<char>,
            run_len: &mut usize,
            token: &mut String,
            sink: &mut dyn FnMut(&str),
        ) {
            if let Some(p) = prev.take()
                && *run_len == 1
            {
                token.clear();
                token.push(p);
                sink(token);
            }
            *run_len = 0;
        }

        for seg in self.norm.split_word_bounds() {
            let mut chars = seg.chars();
            let first = chars.next();
            let single = first.is_some() && chars.next().is_none();
            match first {
                Some(c) if single && is_cjk_unigram(c) => {
                    if let Some(p) = prev_cjk {
                        token.clear();
                        token.push(p);
                        token.push(c);
                        sink(token);
                    }
                    prev_cjk = Some(c);
                    run_len += 1;
                }
                _ if seg.chars().any(char::is_alphanumeric) => {
                    flush_cjk(&mut prev_cjk, &mut run_len, token, sink);
                    token.clear();
                    for c in seg.chars() {
                        for lc in c.to_lowercase() {
                            fold_into(lc, token);
                        }
                    }
                    emit_truncated(token, sink);
                }
                _ => flush_cjk(&mut prev_cjk, &mut run_len, token, sink),
            }
        }
        flush_cjk(&mut prev_cjk, &mut run_len, token, sink);
    }
}

/// Pushes one lowercased char into the token, applying the folding rules:
/// `ё` → `е`; Latin precomposed diacritics stripped to their ASCII base;
/// combining marks dropped after an ASCII base (this also absorbs the one
/// Unicode lowercase expansion that emits a mark, `İ` → `i` + U+0307).
/// Everything else — Cyrillic `й`, Greek, Kana — is kept precomposed.
fn fold_into(c: char, out: &mut String) {
    if c == 'ё' {
        out.push('е');
        return;
    }
    if c.is_ascii() {
        out.push(c);
        return;
    }
    if is_combining_mark(c) {
        if !out.ends_with(|p: char| p.is_ascii_alphanumeric()) {
            out.push(c);
        }
        return;
    }
    // Canonical decomposition into a tiny fixed buffer (canonical
    // decompositions are at most a few chars).
    let mut parts = [char::MAX; 8];
    let mut n = 0usize;
    decompose_canonical(c, |d| {
        if n < parts.len() {
            parts[n] = d;
        }
        n += 1;
    });
    if n <= parts.len() && n > 0 && parts[0].is_ascii_alphanumeric() {
        for &d in &parts[..n] {
            if !is_combining_mark(d) {
                out.push(d);
            }
        }
    } else {
        out.push(c);
    }
}

/// Sends the assembled token, truncated to [`MAX_TOKEN_BYTES`] at a char
/// boundary. Empty tokens (theoretically unreachable — a word segment
/// always folds to at least one char) are guarded against rather than
/// asserted.
fn emit_truncated(token: &str, sink: &mut dyn FnMut(&str)) {
    if token.is_empty() {
        return;
    }
    let mut end = token.len().min(MAX_TOKEN_BYTES);
    while !token.is_char_boundary(end) {
        end -= 1;
    }
    sink(&token[..end]);
}
