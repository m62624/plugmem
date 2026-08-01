//! The core tokenizer, v2.
//!
//! The tokenizer is deliberately split into four stages:
//!
//! 1. NFKC normalization of the input;
//! 2. UAX #29 word segmentation and CJK adjacency tracking;
//! 3. policy-driven Unicode folding;
//! 4. canonical, byte-budgeted token emission.
//!
//! The stages share caller-owned scratch buffers, so the ordinary Latin,
//! Cyrillic, and generic-script paths remain allocation-free after warm-up and
//! behave identically on native and WASM. ICU4X's dictionary/LSTM path for
//! complex scripts may allocate inside its iterator, but uses the same token
//! policy and canonical emission rules.
//! Emitted tokens are canonical fixed points of the tokenizer.

use alloc::string::String;

mod emit;
mod fold;
mod normalize;
mod policy;
mod segment;
mod tables;
mod unicode;

pub use self::emit::MAX_TOKEN_BYTES;
pub use self::policy::TokenizerPolicy;
use self::segment::CjkRun;
use self::unicode::UnicodeBackend;

/// Streaming tokenizer with reusable scratch buffers.
///
/// One instance should be reused by an engine or by one thread of a wrapper.
/// After warm-up, [`Tokenizer::tokenize`] performs no heap allocation on the
/// generic Unicode path. Complex scripts may use ICU4X dictionary/LSTM
/// scratch allocations to obtain language-aware word boundaries.
#[derive(Debug, Default, Clone)]
pub struct Tokenizer {
    /// Folding policy. This is copied into the hot loop and has no heap cost.
    policy: TokenizerPolicy,
    /// NFKC-normalized copy of the input.
    norm: String,
    /// The token being assembled (folded word or CJK bigram).
    token: String,
    /// Lowercase copy of the current ICU word segment.
    lower: String,
    /// Reused scratch for the rare post-fold NFKC pass.
    canonical: String,
    /// Compiled Unicode data and segmentation rules.
    unicode: UnicodeBackend,
}

impl Tokenizer {
    /// Creates a tokenizer with empty scratch buffers and the search policy.
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
    /// The emitted `&str` is only valid for the duration of one `sink` call.
    ///
    /// ```
    /// use plugmem_core::tokenizer::Tokenizer;
    ///
    /// let mut tokenizer = Tokenizer::new();
    /// let mut tokens = Vec::new();
    /// tokenizer.tokenize("Hello, МИР-42! 東京タワー", &mut |token| {
    ///     tokens.push(token.to_owned())
    /// });
    /// assert_eq!(tokens, ["hello", "мир", "42", "東京", "タワー"]);
    /// ```
    pub fn tokenize(&mut self, text: &str, sink: &mut dyn FnMut(&str)) {
        normalize::normalize_into(&self.unicode, text, &mut self.norm);

        let policy = self.policy;
        let token = &mut self.token;
        let lower = &mut self.lower;
        let canonical = &mut self.canonical;
        let unicode = &self.unicode;
        let mut cjk_run = CjkRun::default();
        let mut start = 0usize;

        for end in unicode.word_boundaries(&self.norm) {
            if end == start {
                continue;
            }
            let segment = &self.norm[start..end];
            if segment.chars().any(char::is_alphanumeric) {
                if segment.chars().all(UnicodeBackend::is_cjk_unigram) {
                    for character in segment.chars() {
                        cjk_run.push(character, token, sink);
                    }
                } else {
                    cjk_run.flush(token, sink);
                    unicode.lowercase_into(segment, lower);
                    fold::fold_segment(lower, token, canonical, policy, unicode, sink);
                }
            } else {
                cjk_run.flush(token, sink);
            }
            start = end;
        }
        cjk_run.flush(token, sink);
    }
}
