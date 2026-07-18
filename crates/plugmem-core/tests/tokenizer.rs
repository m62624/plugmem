//! Tokenizer v2 tests (specs/04 test plan): a unit table per data-format
//! class (prose, identifiers, numbers, URLs, diacritics, CJK scripts),
//! the concatenation property, and canonical-token invariants.

use plugmem_core::tokenizer::{MAX_TOKEN_BYTES, Tokenizer};
use proptest::prelude::*;

fn tokens(text: &str) -> Vec<String> {
    let mut tk = Tokenizer::new();
    let mut out = Vec::new();
    tk.tokenize(text, &mut |t| out.push(t.to_owned()));
    out
}

#[test]
fn prose_and_case() {
    let cases: &[(&str, &[&str])] = &[
        ("", &[]),
        ("  \t\n ... !!!", &[]),
        ("Hello, World!", &["hello", "world"]),
        ("Предпочитает Tokio", &["предпочитает", "tokio"]),
        // UAX #29: apostrophe joins letters (Lucene/ICU behavior).
        ("don't stop O'Clock", &["don't", "stop", "o'clock"]),
        // Hyphen splits; digits tokenize as-is.
        ("gpt-4o издание 2026", &["gpt", "4o", "издание", "2026"]),
        // Emoji and modifiers separate.
        ("rust🦀lang 👍🏽ok", &["rust", "lang", "ok"]),
        // Mixed-script run stays one token.
        ("сloud42x", &["сloud42x"]),
        // The one two-char lowercase (İ) folds to a clean "i".
        ("İstanbul", &["istanbul"]),
        ("STRAẞE", &["straße"]),
    ];
    for (input, want) in cases {
        assert_eq!(&tokens(input), want, "input: {input:?}");
    }
}

#[test]
fn identifiers_numbers_and_urls() {
    let cases: &[(&str, &[&str])] = &[
        // Underscore joins (UAX #29 ExtendNumLet) — code identifiers
        // survive whole.
        ("snake_case_id CamelCase", &["snake_case_id", "camelcase"]),
        // '.' and ',' join digits: decimals and versions survive.
        ("pi=3.14, v1.2.3 1,000", &["pi", "3.14", "v1.2.3", "1,000"]),
        // Domains survive; '@' and '/' split.
        (
            "see docs.rs/plugmem or a@b.com",
            &["see", "docs.rs", "plugmem", "or", "a", "b.com"],
        ),
    ];
    for (input, want) in cases {
        assert_eq!(&tokens(input), want, "input: {input:?}");
    }
}

#[test]
fn nfkc_and_diacritic_folding() {
    let cases: &[(&str, &[&str])] = &[
        // NFKC: fullwidth forms and ligatures normalize before anything.
        ("ＦＵＬＬ　ｗｉｄｔｈ４２", &["full", "width42"]),
        ("ﬁle ﬂow", &["file", "flow"]),
        // Latin diacritics stripped (FTS5 remove_diacritics class).
        (
            "café naïve Zürich piñata",
            &["cafe", "naive", "zurich", "pinata"],
        ),
        // Decomposed input composes first, then folds the same way.
        ("cafe\u{301}", &["cafe"]),
        // Russian: ё folds to е…
        ("Ёлка мёд", &["елка", "мед"]),
        // …but й is a distinct letter and must NOT lose its breve (the
        // classic unicode61 remove_diacritics=2 mistake).
        ("йод майка", &["йод", "майка"]),
        // Greek keeps its marks (folding is Latin-only by design).
        ("ἀγορά", &["ἀγορά"]),
    ];
    for (input, want) in cases {
        assert_eq!(&tokens(input), want, "input: {input:?}");
    }
}

#[test]
fn cjk_bigrams_and_word_scripts() {
    let cases: &[(&str, &[&str])] = &[
        // Han: overlapping bigrams (Lucene CJKBigramFilter scheme).
        ("東京都", &["東京", "京都"]),
        // A lone ideograph stays a unigram.
        ("水 ok", &["水", "ok"]),
        // Two runs separated by a latin word: adjacency resets.
        ("東京tower大阪", &["東京", "tower", "大阪"]),
        // Hiragana joins the bigram machine…
        ("すしが好き", &["すし", "しが", "が好", "好き"]),
        // …while Katakana and Hangul segment into words by UAX #29.
        ("トーキョー タワー", &["トーキョー", "タワー"]),
        ("안녕하세요 세계", &["안녕하세요", "세계"]),
        // Full sentence: punctuation resets adjacency too.
        (
            "彼は寿司が好きだ。",
            &["彼は", "は寿", "寿司", "司が", "が好", "好き", "きだ"],
        ),
    ];
    for (input, want) in cases {
        assert_eq!(&tokens(input), want, "input: {input:?}");
    }
}

#[test]
fn oversized_token_is_truncated_at_a_char_boundary() {
    // 100 ASCII bytes → exactly MAX_TOKEN_BYTES survive.
    let long = "a".repeat(100);
    assert_eq!(tokens(&long), ["a".repeat(MAX_TOKEN_BYTES)]);

    // Cyrillic is 2 bytes/char: the cut at 64 lands on a boundary.
    let cyr = "ж".repeat(33);
    let got = tokens(&cyr);
    assert_eq!(got, ["ж".repeat(32)]);
    assert_eq!(got[0].len(), MAX_TOKEN_BYTES);

    // Devanagari is 3 bytes/char: byte 64 is mid-char, the cut retreats
    // to the boundary at 63.
    let deva = "अ".repeat(22);
    let got = tokens(&deva);
    assert_eq!(got, ["अ".repeat(21)]);
    assert_eq!(got[0].len(), 63);
}

#[test]
fn scratch_buffers_are_reused_not_leaked() {
    let mut tk = Tokenizer::new();
    let mut first = Vec::new();
    tk.tokenize("alpha beta 東京", &mut |t| first.push(t.to_owned()));
    let mut second = Vec::new();
    tk.tokenize("gamma", &mut |t| second.push(t.to_owned()));
    assert_eq!(first, ["alpha", "beta", "東京"]);
    assert_eq!(second, ["gamma"]);
    // Clone carries config-free state; a cloned tokenizer works alone.
    let mut third = Vec::new();
    tk.clone()
        .tokenize("δ delta", &mut |t| third.push(t.to_owned()));
    assert_eq!(third, ["δ", "delta"]);
}

proptest! {
    // Concatenating two texts with a separating space yields exactly the
    // tokens of both texts, in order: the separator kills any cross-text
    // segment or CJK adjacency, and NFKC cannot compose across a space.
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn concatenation_with_separator_preserves_tokens(a in ".{0,40}", b in ".{0,40}") {
        let joined = format!("{a} {b}");
        let mut want = tokens(&a);
        want.extend(tokens(&b));
        prop_assert_eq!(tokens(&joined), want);
    }

    // Emitted tokens are canonical: non-empty, within the byte cap, and a
    // fixed point — re-tokenizing a token reproduces exactly it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn tokens_are_canonical(text in ".{0,80}") {
        for t in tokens(&text) {
            prop_assert!(!t.is_empty());
            prop_assert!(t.len() <= MAX_TOKEN_BYTES);
            let again = tokens(&t);
            prop_assert_eq!(again.len(), 1, "token {:?} re-split into {:?}", t, again);
            prop_assert_eq!(&again[0], &t, "token is not a fixed point");
        }
    }
}
