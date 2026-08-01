//! The core tokenizer, v2.
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

mod tables;
use self::tables as tokenizer_tables;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::{decompose_canonical, is_combining_mark};
use unicode_segmentation::UnicodeSegmentation;

/// Upper bound on an emitted token, in bytes.
pub const MAX_TOKEN_BYTES: usize = 64;

/// Controls search-specific folding without changing the scanner or its
/// allocation behavior.
///
/// The default preserves the tokenizer's existing index contract: Latin
/// diacritics are folded for recall and Russian `ё` is treated as `е`. Use
/// [`TokenizerPolicy::unicode`] when callers need Unicode lowercase and
/// normalization without either language/search-specific equivalence.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TokenizerPolicy {
    /// Fold Latin precomposed diacritics to their ASCII base.
    pub fold_latin_diacritics: bool,
    /// Treat Russian small letter `ё` as `е` after lowercase.
    pub fold_russian_yo: bool,
}

impl TokenizerPolicy {
    /// A language-neutral policy: Unicode lowercase and normalization only.
    pub const fn unicode() -> Self {
        Self {
            fold_latin_diacritics: false,
            fold_russian_yo: false,
        }
    }

    /// The search policy used by [`Tokenizer::new`] and existing indexes.
    pub const fn search() -> Self {
        Self {
            fold_latin_diacritics: true,
            fold_russian_yo: true,
        }
    }
}

impl Default for TokenizerPolicy {
    fn default() -> Self {
        Self::search()
    }
}

/// `true` for characters treated as CJK unigram sources: Han ideographs
/// (BMP blocks, the compatibility block, supplementary-plane extensions)
/// and Hiragana. Katakana and Hangul are excluded on purpose — UAX #29
/// already groups them into word runs.
fn is_cjk_unigram(c: char) -> bool {
    tokenizer_tables::in_ranges(c, tokenizer_tables::CJK_UNIGRAM_RANGES)
}

/// Streaming tokenizer with reusable scratch buffers.
///
/// One instance per engine (or per thread of a wrapper): after warm-up
/// [`Tokenizer::tokenize`] allocates nothing, which the zero-alloc recall
/// invariant depends on.
#[derive(Debug, Default, Clone)]
pub struct Tokenizer {
    /// Folding policy. This is copied into the hot loop and has no heap cost.
    policy: TokenizerPolicy,
    /// NFKC-normalized copy of the input.
    norm: String,
    /// The token being assembled (folded word or CJK bigram).
    token: String,
    /// Reused scratch for the rare post-fold NFKC pass. Lowercasing can
    /// expose a non-canonical combining-mark order even though `norm` was
    /// normalized before folding.
    canonical: String,
}

impl Tokenizer {
    /// A tokenizer with empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a tokenizer with an explicit folding policy.
    pub fn with_policy(policy: TokenizerPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    /// Returns the policy used by this tokenizer.
    pub const fn policy(&self) -> TokenizerPolicy {
        self.policy
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
        let policy = self.policy;
        let token = &mut self.token;
        let canonical = &mut self.canonical;

        // The CJK adjacency machine: previous unigram char + run length.
        // At a run boundary the machine may still owe a token: a lone
        // char is a unigram; longer runs were already emitted as bigrams.
        let mut cjk_run = CjkRun::default();

        for seg in self.norm.split_word_bounds() {
            let mut chars = seg.chars();
            let first = chars.next();
            let single = first.is_some() && chars.next().is_none();
            match first {
                Some(c) if single && is_cjk_unigram(c) => {
                    cjk_run.push(c, token, sink);
                }
                _ if seg.chars().any(char::is_alphanumeric) => {
                    cjk_run.flush(token, sink);
                    fold_segment(seg, token, canonical, policy, sink);
                }
                _ => cjk_run.flush(token, sink),
            }
        }
        cjk_run.flush(token, sink);
    }
}

/// Tracks adjacent Han/Hiragana segments and emits overlapping bigrams.
#[derive(Debug, Default)]
struct CjkRun {
    previous: Option<char>,
    length: usize,
}

impl CjkRun {
    fn push(&mut self, current: char, token: &mut String, sink: &mut dyn FnMut(&str)) {
        if let Some(previous) = self.previous {
            token.clear();
            token.push(previous);
            token.push(current);
            sink(token);
        }
        self.previous = Some(current);
        self.length += 1;
    }

    fn flush(&mut self, token: &mut String, sink: &mut dyn FnMut(&str)) {
        if let Some(previous) = self.previous.take()
            && self.length == 1
        {
            token.clear();
            token.push(previous);
            sink(token);
        }
        self.length = 0;
    }
}

/// Folds one UAX #29 segment into the reusable token scratch buffers.
fn fold_segment(
    segment: &str,
    token: &mut String,
    canonical: &mut String,
    policy: TokenizerPolicy,
    sink: &mut dyn FnMut(&str),
) {
    token.clear();
    let mut needs_nfkc = false;
    for c in segment.chars() {
        for lowercase in c.to_lowercase() {
            if fold_into(lowercase, token, policy, &mut needs_nfkc) {
                emit_folded(token, canonical, needs_nfkc, policy, sink);
                token.clear();
                needs_nfkc = false;
            }
        }
    }
    emit_folded(token, canonical, needs_nfkc, policy, sink);
}

/// `true` for the default-ignorable format characters that occur inside
/// running text (soft hyphen, zero-width space/joiners, bidi marks, word
/// joiner block, BOM). UAX #29 glues them into word segments (they are
/// `Format`/`Extend` for word breaking) but they carry no lexical content
/// and must never survive into a term.
fn is_ignorable_format(c: char) -> bool {
    tokenizer_tables::in_ranges(c, tokenizer_tables::IGNORABLE_FORMAT_RANGES)
}

/// Pushes one lowercased char into the token, applying the folding rules:
///
/// - ignorable format characters are dropped ([`is_ignorable_format`]);
/// - combining marks are dropped after an ASCII base (this also absorbs
///   the one Unicode lowercase expansion that emits a mark, `İ` → `i` +
///   U+0307) and kept otherwise; emission removes leading marks when a
///   later base character exists, while preserving mark-only terms;
/// - documented word-internal joiners survive only after an alphanumeric
///   one (`don't`, `3.14`, `snake_case`); other punctuation is a boundary;
///   trailing joiners are removed at emission so every token is a fixed
///   point when tokenized in isolation;
/// - Latin precomposed diacritics are stripped to their ASCII base;
///   everything else — Cyrillic `й`, Greek, Kana — is kept precomposed.
///
/// Returns `true` when `c` is a real boundary after an already assembled
/// token. Ignorable format characters are removed without splitting; this is
/// what keeps `co\u{AD}operate` together while preventing a dropped `_` or `:`
/// from gluing two otherwise separate lexical pieces.
fn fold_into(c: char, out: &mut String, policy: TokenizerPolicy, needs_nfkc: &mut bool) -> bool {
    if policy.fold_russian_yo
        && let Some(mapped) =
            tokenizer_tables::mapped_char(c, tokenizer_tables::RUSSIAN_SEARCH_FOLDS)
    {
        out.push(mapped);
        return false;
    }
    if is_ignorable_format(c) {
        return false;
    }
    if is_combining_mark(c) {
        *needs_nfkc = true;
        if !out.ends_with(|p: char| p.is_ascii_alphanumeric()) {
            out.push(c);
        }
        return false;
    }
    if !c.is_alphanumeric() {
        if out.ends_with(char::is_alphanumeric) && is_word_joiner(c) {
            out.push(c);
            return false;
        }
        return !out.is_empty();
    }
    if c.is_ascii() {
        out.push(c);
        return false;
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
    if policy.fold_latin_diacritics && n <= parts.len() && n > 0 && parts[0].is_ascii_alphanumeric()
    {
        for &d in &parts[..n] {
            if !is_combining_mark(d) {
                out.push(d);
            }
        }
    } else {
        out.push(c);
    }
    false
}

/// Returns whether punctuation is allowed inside a lexical token.
///
/// This is deliberately a small allow-list. UAX #29 decides which source
/// characters belong to a segment, but it does not by itself guarantee that
/// the folded segment is a fixed point when tokenized in isolation.
fn is_word_joiner(c: char) -> bool {
    tokenizer_tables::WORD_JOINERS.contains(&c)
}

/// Re-normalizes a folded token only when lowercase/folding saw combining
/// marks. NFKC runs before folding for the common path, but Unicode lowercase
/// can change the base character while leaving a mark sequence whose canonical
/// order is visible only after the fold. The folding pass is repeated after
/// NFKC because NFKC may recreate a project-specific form such as `ё`, which
/// must still be folded to `е`.
fn emit_folded(
    token: &mut String,
    canonical: &mut String,
    needs_nfkc: bool,
    policy: TokenizerPolicy,
    sink: &mut dyn FnMut(&str),
) {
    if needs_nfkc {
        canonical.clear();
        canonical.extend(token.nfkc());
        token.clear();
        let mut ignored = false;
        for c in canonical.chars() {
            for lc in c.to_lowercase() {
                if fold_into(lc, token, policy, &mut ignored) {
                    drop_leading_marks_before_base(token);
                    emit_truncated(token, sink);
                    token.clear();
                    ignored = false;
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
/// token remains valid and is kept for compatibility with the tokenizer's
/// existing behavior; a leading mark plus a base would otherwise disappear
/// when that emitted token is normalized in isolation.
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

/// Sends the assembled token, truncated to [`MAX_TOKEN_BYTES`] at a char
/// boundary. Word joiners are meaningful only between lexical characters, so
/// leading and trailing joiners are removed without allocating. Keeping one
/// would make re-tokenization context-sensitive (for example, `_word` →
/// `word` and `word.` → `word`). Empty tokens are guarded against rather than
/// asserted.
fn emit_truncated(token: &str, sink: &mut dyn FnMut(&str)) {
    let token = token.trim_start_matches(is_word_joiner);
    if token.is_empty() {
        return;
    }
    let mut end = token.len().min(MAX_TOKEN_BYTES);
    while !token.is_char_boundary(end) {
        end -= 1;
    }
    while end > 0 {
        let c = token[..end]
            .chars()
            .next_back()
            .expect("non-empty token has a last char");
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
