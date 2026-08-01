//! Word-segment and CJK adjacency handling.

use alloc::string::String;

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
