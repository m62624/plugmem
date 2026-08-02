//! Word-segment and CJK adjacency handling.

use alloc::string::String;

use super::{fold, policy::TokenizerPolicy, unicode::UnicodeBackend};

/// Tracks adjacent Han/Hiragana segments and emits overlapping bigrams.
#[derive(Debug, Default)]
pub(super) struct CjkRun {
    previous: Option<char>,
    length: usize,
}

impl CjkRun {
    pub(super) fn push(&mut self, current: char, token: &mut String, sink: &mut dyn FnMut(&str)) {
        if let Some(previous) = self.previous {
            token.clear();
            token.push(previous);
            token.push(current);
            sink(token);
        }
        self.previous = Some(current);
        self.length += 1;
    }

    pub(super) fn flush(&mut self, token: &mut String, sink: &mut dyn FnMut(&str)) {
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

/// Owns the reusable scratch references needed to process ICU word segments.
/// Keeping this state together makes the script-boundary policy explicit and
/// prevents the tokenizer's hot loop from growing a long argument list.
pub(super) struct SegmentProcessor<'a> {
    policy: TokenizerPolicy,
    unicode: &'a UnicodeBackend,
    cjk_run: &'a mut CjkRun,
    token: &'a mut String,
    lower: &'a mut String,
    canonical: &'a mut String,
}

impl<'a> SegmentProcessor<'a> {
    /// Creates a processor over the tokenizer's caller-owned scratch buffers.
    pub(super) fn new(
        policy: TokenizerPolicy,
        unicode: &'a UnicodeBackend,
        cjk_run: &'a mut CjkRun,
        token: &'a mut String,
        lower: &'a mut String,
        canonical: &'a mut String,
    ) -> Self {
        Self {
            policy,
            unicode,
            cjk_run,
            token,
            lower,
            canonical,
        }
    }

    /// Processes one ICU word segment while preserving lexical policies at
    /// script boundaries.
    ///
    /// ICU's word iterator is allowed to return a segment containing adjacent
    /// scripts. The CJK bigram policy is narrower than a generic word segment,
    /// so treating such a segment as one folded token makes the result depend
    /// on whether the token is later retokenized in isolation. Split the
    /// segment into maximal CJK and non-CJK runs before applying either
    /// policy. CJK runs remain stateful across ICU segment boundaries;
    /// non-CJK runs use the normal folding path unchanged.
    pub(super) fn process(&mut self, segment: &str, sink: &mut dyn FnMut(&str)) {
        let mut run_start = 0;
        let mut run_is_cjk = None;

        for (offset, character) in segment.char_indices() {
            let is_cjk = UnicodeBackend::is_cjk_unigram(character);
            if let Some(previous) = run_is_cjk
                && previous != is_cjk
            {
                self.process_run(&segment[run_start..offset], previous, sink);
                run_start = offset;
            }
            run_is_cjk = Some(is_cjk);
        }

        if let Some(is_cjk) = run_is_cjk {
            self.process_run(&segment[run_start..], is_cjk, sink);
        }
    }

    /// Flushes an adjacency run at a non-word boundary or at end of input.
    pub(super) fn flush(&mut self, sink: &mut dyn FnMut(&str)) {
        self.cjk_run.flush(self.token, sink);
    }

    fn process_run(&mut self, run: &str, is_cjk: bool, sink: &mut dyn FnMut(&str)) {
        if is_cjk {
            for character in run.chars() {
                self.cjk_run.push(character, self.token, sink);
            }
        } else {
            self.cjk_run.flush(self.token, sink);
            self.unicode.lowercase_into(run, self.lower);
            fold::fold_segment(
                self.lower,
                self.token,
                self.canonical,
                self.policy,
                self.unicode,
                sink,
            );
        }
    }
}
