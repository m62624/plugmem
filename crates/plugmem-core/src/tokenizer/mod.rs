//! The core tokenizer, v2.
//!
//! The tokenizer is deliberately split into four stages:
//!
//! 1. NFKC normalization of the input;
//! 2. UAX #29 word segmentation and CJK adjacency tracking;
//! 3. policy-driven Unicode folding;
//! 4. canonical, byte-budgeted token emission.
//!
//! The stages share caller-owned scratch buffers, so the hot path remains
//! allocation-free after warm-up and behaves identically on native and WASM.
//! Emitted tokens are canonical fixed points of the tokenizer.

use alloc::string::String;

mod emit;
mod fold;
mod normalize;
mod policy;
mod segment;
mod tables;

pub use self::emit::MAX_TOKEN_BYTES;
pub use self::policy::TokenizerPolicy;
use self::segment::{CjkRun, is_cjk_unigram};
use unicode_segmentation::UnicodeSegmentation;

/// Streaming tokenizer with reusable scratch buffers.
///
/// One instance should be reused by an engine or by one thread of a wrapper.
/// After warm-up, [`Tokenizer::tokenize`] performs no heap allocation.
#[derive(Debug, Default, Clone)]
pub struct Tokenizer {
    /// Folding policy. This is copied into the hot loop and has no heap cost.
    policy: TokenizerPolicy,
    /// NFKC-normalized copy of the input.
    norm: String,
    /// The token being assembled (folded word or CJK bigram).
    token: String,
    /// Reused scratch for the rare post-fold NFKC pass.
    canonical: String,
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
        normalize::normalize_into(text, &mut self.norm);

        let policy = self.policy;
        let token = &mut self.token;
        let canonical = &mut self.canonical;
        let mut cjk_run = CjkRun::default();

        for segment in self.norm.split_word_bounds() {
            let mut chars = segment.chars();
            let first = chars.next();
            let single = first.is_some() && chars.next().is_none();
            match first {
                Some(character) if single && is_cjk_unigram(character) => {
                    cjk_run.push(character, token, sink);
                }
                _ if segment.chars().any(char::is_alphanumeric) => {
                    cjk_run.flush(token, sink);
                    fold::fold_segment(segment, token, canonical, policy, sink);
                }
                _ => cjk_run.flush(token, sink),
            }
        }
        cjk_run.flush(token, sink);
    }
}
