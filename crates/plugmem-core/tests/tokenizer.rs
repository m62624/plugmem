//! Tokenizer v2 tests (test plan): a unit table per data-format
//! class (prose, identifiers, numbers, URLs, diacritics, CJK scripts),
//! the concatenation property, and canonical-token invariants.

use plugmem_core::tokenizer::{MAX_TOKEN_BYTES, Tokenizer};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::*;
#[cfg(not(target_family = "wasm"))]
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};

fn tokens(text: &str) -> Vec<String> {
    let mut tk = Tokenizer::new();
    let mut out = Vec::new();
    tk.tokenize(text, &mut |t| out.push(t.to_owned()));
    out
}

#[cfg(not(target_family = "wasm"))]
fn stress_char() -> impl Strategy<Value = char> {
    prop_oneof![
        8 => (b'a' as u32..=b'z' as u32).prop_map(|n| char::from_u32(n).unwrap()),
        4 => (b'A' as u32..=b'Z' as u32).prop_map(|n| char::from_u32(n).unwrap()),
        4 => (b'0' as u32..=b'9' as u32).prop_map(|n| char::from_u32(n).unwrap()),
        3 => (0x0041u32..=0x024F).prop_map(|n| char::from_u32(n).unwrap()),
        3 => (0x0300u32..=0x036F).prop_map(|n| char::from_u32(n).unwrap()),
        3 => (0x0370u32..=0x052F).prop_map(|n| char::from_u32(n).unwrap()),
        2 => (0x0590u32..=0x06FF).prop_map(|n| char::from_u32(n).unwrap()),
        2 => (0x0900u32..=0x0DFF).prop_map(|n| char::from_u32(n).unwrap()),
        3 => (0x3041u32..=0x30FF).prop_map(|n| char::from_u32(n).unwrap()),
        3 => (0x3400u32..=0x9FFF).prop_map(|n| char::from_u32(n).unwrap()),
        3 => (0xAC00u32..=0xD7AF).prop_map(|n| char::from_u32(n).unwrap()),
        2 => (0x1F300u32..=0x1FAFF).prop_map(|n| char::from_u32(n).unwrap()),
        2 => any::<char>(),
        3 => prop::sample::select(vec![
            '\u{00AD}', '\u{200B}', '\u{200D}', '\u{200E}', '\u{200F}', '\u{202E}',
            '\u{2060}', '\u{FEFF}',
        ]),
        4 => prop::sample::select(vec![
            ':', ';', '!', '?', '.', ',', '_', '\'', '\u{2019}', '/', '\\', '-', '+', '=',
            '#', '@', '%', '&',
        ]),
        3 => prop::sample::select(vec![
            'º', 'K', 'ﬁ', 'ﬂ', 'Ａ', '１', 'Å', 'é', 'ï', 'İ', 'ñ', 'й', 'ё', 'ἀ',
        ]),
    ]
}

#[cfg(not(target_family = "wasm"))]
fn stress_text() -> impl Strategy<Value = String> {
    prop::collection::vec(stress_char(), 0..192).prop_map(|chars| chars.into_iter().collect())
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
        // Ignorable format chars vanish from inside words: soft hyphen,
        // zero-width joiner.
        ("co\u{AD}operate ab\u{200D}cd", &["cooperate", "abcd"]),
        // An alphabetic combining mark glued (via UAX #29 Extend) onto a
        // space: the space is dropped, the mark stands alone — exactly
        // what the mark yields without a space in front.
        (" \u{F71}", &["\u{F71}"]),
        ("\u{F71}", &["\u{F71}"]),
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

#[test]
fn regression_canonical_token_from_ordinal_and_modifier_symbols() {
    let emitted = tokens("º:˥");
    assert_eq!(emitted, ["o"]);

    let retokenized = tokens(&emitted[0]);
    assert_eq!(retokenized, emitted, "emitted token must be a fixed point");
}

#[test]
fn word_joiners_are_internal_and_canonical() {
    let cases: &[(&str, &[&str])] = &[
        ("don't o'clock", &["don't", "o'clock"]),
        (
            "3.14 v1.2.3 example.com 1,000",
            &["3.14", "v1.2.3", "example.com", "1,000"],
        ),
        ("snake_case", &["snake_case"]),
        ("word. word, word_", &["word", "word", "word"]),
    ];
    for (input, want) in cases {
        let got = tokens(input);
        assert_eq!(&got, want, "input: {input:?}");
        for token in got {
            let again = tokens(&token);
            assert_eq!(
                again.as_slice(),
                std::slice::from_ref(&token),
                "token is not canonical: {token:?}"
            );
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn unicode_stress_tokens_are_canonical() {
    let config = ProptestConfig {
        cases: 1024,
        max_shrink_iters: 4096,
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);
    runner
        .run(&stress_text(), |text| {
            let emitted = tokens(&text);
            for token in &emitted {
                if token.is_empty() || token.len() > MAX_TOKEN_BYTES {
                    return Err(TestCaseError::fail(format!(
                        "invalid token {token:?} from input {text:?}"
                    )));
                }
                let again = tokens(token);
                if again != [token.clone()] {
                    return Err(TestCaseError::fail(format!(
                        "non-canonical token {token:?} from input {text:?}; retokenized as {again:?}"
                    )));
                }
            }
            Ok(())
        })
        .expect("Unicode stress properties failed");
}

#[cfg(not(target_family = "wasm"))]
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
