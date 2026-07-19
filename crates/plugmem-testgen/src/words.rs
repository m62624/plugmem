//! Deterministic vocabularies (specs/07 §6, specs/11 §3): pronounceable
//! syllable words and a Zipf sampler over their ranks.
//!
//! Words are *derived from an index*, not sampled: `word_for(i)` is a
//! pure function encoding `i` in base-`SYLLABLES` (consonant + vowel
//! pairs), so every index maps to a unique, tokenizer-friendly word and
//! the whole vocabulary is reproducible without storing anything.
//! Distinct vocabularies (dictionary, tags, entity names) draw from
//! disjoint index ranges via their `salt` offsets.

use crate::rng::Rng;

/// Consonants of the syllable alphabet.
const CONSONANTS: &[u8] = b"bdfgklmnprstvz";
/// Vowels of the syllable alphabet.
const VOWELS: &[u8] = b"aeiou";
/// Number of distinct syllables.
const SYLLABLES: usize = CONSONANTS.len() * VOWELS.len();

/// Builds the unique word for an index: its base-`SYLLABLES` digits as
/// syllables, at least two syllables so even index 0 is a real word
/// ("baba", not "ba").
pub fn word_for(index: usize) -> String {
    let mut digits = [0usize; 12];
    let mut n = 0;
    let mut rest = index;
    loop {
        digits[n] = rest % SYLLABLES;
        n += 1;
        rest /= SYLLABLES;
        if rest == 0 {
            break;
        }
    }
    n = n.max(2); // leading zero-syllables pad short indexes
    let mut out = String::with_capacity(n * 2);
    for &digit in digits[..n].iter().rev() {
        out.push(CONSONANTS[digit / VOWELS.len()] as char);
        out.push(VOWELS[digit % VOWELS.len()] as char);
    }
    out
}

/// A rank-indexed vocabulary with Zipf(`s`) sampling: rank 0 is the most
/// frequent word, weight `1 / (rank + 1)^s`.
#[derive(Clone, Debug)]
pub struct Vocabulary {
    /// Words in rank order (rank = index).
    words: Vec<String>,
    /// Cumulative Zipf weights, normalized to end at 1.0.
    cumulative: Vec<f64>,
}

impl Vocabulary {
    /// Builds `len` words starting at index offset `salt` with Zipf
    /// exponent `s`.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0` (an empty vocabulary cannot be sampled).
    pub fn new(salt: usize, len: usize, s: f64) -> Self {
        assert!(len > 0, "empty vocabulary");
        let words = (0..len).map(|i| word_for(salt + i)).collect();
        let mut cumulative = Vec::with_capacity(len);
        let mut total = 0.0f64;
        for rank in 0..len {
            total += 1.0 / ((rank + 1) as f64).powf(s);
            cumulative.push(total);
        }
        for c in &mut cumulative {
            *c /= total;
        }
        Self { words, cumulative }
    }

    /// Number of words.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// `true` when the vocabulary holds no words (never — construction
    /// forbids it; present for API completeness).
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The word at a rank.
    ///
    /// # Panics
    ///
    /// Panics if `rank >= len()`.
    pub fn word(&self, rank: usize) -> &str {
        &self.words[rank]
    }

    /// Samples a rank by the Zipf distribution.
    pub fn sample_rank(&self, rng: &mut Rng) -> usize {
        let x = rng.f64();
        self.cumulative
            .partition_point(|&c| c < x)
            .min(self.words.len() - 1)
    }

    /// Samples a word by the Zipf distribution.
    pub fn sample(&self, rng: &mut Rng) -> &str {
        let rank = self.sample_rank(rng);
        &self.words[rank]
    }
}
