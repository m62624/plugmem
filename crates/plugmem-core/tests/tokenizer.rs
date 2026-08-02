//! Tokenizer v2 tests (test plan): a unit table per data-format
//! class (prose, identifiers, numbers, URLs, diacritics, CJK scripts),
//! the concatenation property, and canonical-token invariants.

use plugmem_core::tokenizer::{MAX_TOKEN_BYTES, Tokenizer, TokenizerPolicy};
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

fn tokens_with_policy(text: &str, policy: TokenizerPolicy) -> Vec<String> {
    let mut tk = Tokenizer::with_policy(policy);
    let mut out = Vec::new();
    tk.tokenize(text, &mut |t| out.push(t.to_owned()));
    out
}

#[cfg(not(target_family = "wasm"))]
fn scalar_from_u32(value: u32) -> char {
    char::from_u32(value).expect("the strategy must never generate UTF-16 surrogates")
}

#[cfg(not(target_family = "wasm"))]
fn full_unicode_scalar() -> impl Strategy<Value = char> {
    prop_oneof![
        1 => (0x0000u32..=0xD7FFu32).prop_map(scalar_from_u32),
        1 => (0xE000u32..=0x10FFFFu32).prop_map(scalar_from_u32),
    ]
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
        // This branch spans every Unicode scalar value, including scripts and
        // blocks not listed above. The focused branches keep common boundary
        // classes frequent enough to exercise their interactions as well.
        8 => full_unicode_scalar(),
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
    prop::collection::vec(stress_char(), 0..=256).prop_map(|chars| chars.into_iter().collect())
}

#[cfg(not(target_family = "wasm"))]
fn full_unicode_text() -> impl Strategy<Value = String> {
    prop::collection::vec(full_unicode_scalar(), 0..=256)
        .prop_map(|chars| chars.into_iter().collect())
}

#[cfg(not(target_family = "wasm"))]
fn cjk_boundary_char() -> impl Strategy<Value = char> {
    prop_oneof![
        2 => (0x3041u32..=0x309Fu32).prop_map(scalar_from_u32),
        3 => (0x3400u32..=0x4DBFu32).prop_map(scalar_from_u32),
        3 => (0x4E00u32..=0x9FFFu32).prop_map(scalar_from_u32),
    ]
}

#[cfg(not(target_family = "wasm"))]
fn non_cjk_boundary_char() -> impl Strategy<Value = char> {
    prop::sample::select(vec!['a', '7', 'é', 'က', 'ก', 'न', 'م', '한'])
}

#[cfg(not(target_family = "wasm"))]
fn mixed_policy_text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        (cjk_boundary_char(), non_cjk_boundary_char(), any::<bool>()),
        1..=64,
    )
    .prop_map(|runs| {
        let mut text = String::new();
        for (cjk, non_cjk, cjk_first) in runs {
            if cjk_first {
                text.push(cjk);
                text.push(non_cjk);
            } else {
                text.push(non_cjk);
                text.push(cjk);
            }
        }
        text
    })
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
fn explicit_policies_keep_language_folding_out_of_the_scanner() {
    assert_eq!(
        tokens_with_policy("Ёлка café", TokenizerPolicy::search()),
        ["елка", "cafe"]
    );
    assert_eq!(
        tokens_with_policy("Ёлка café", TokenizerPolicy::unicode()),
        ["ёлка", "café"]
    );
    assert_eq!(Tokenizer::new().policy(), TokenizerPolicy::search());
}

#[test]
fn every_ignorable_format_table_entry_is_removed_inside_words() {
    let chars = [
        '\u{00AD}', '\u{200B}', '\u{200C}', '\u{200D}', '\u{200E}', '\u{200F}', '\u{202A}',
        '\u{202B}', '\u{202C}', '\u{202D}', '\u{202E}', '\u{2060}', '\u{2061}', '\u{2062}',
        '\u{2063}', '\u{2064}', '\u{FEFF}',
    ];
    for c in chars {
        let got = tokens(&format!("a{c}b"));
        assert!(
            got.iter().all(|token| !token.contains(c)),
            "format character survived: U+{:04X} in {got:?}",
            c as u32
        );
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

    for input in [
        "\u{3400}\u{3401}",
        "\u{F900}\u{F901}",
        "\u{20000}\u{20001}",
        "\u{2F800}\u{2F801}",
        "\u{30000}\u{30001}",
        "\u{31350}\u{31351}",
        "\u{323B0}\u{323B1}",
    ] {
        let got = tokens(input);
        assert_eq!(got.len(), 1, "table boundary input: {input:?}");
        assert_eq!(got[0].chars().count(), 2, "table boundary input: {input:?}");
        assert_eq!(
            tokens(&got[0]),
            got,
            "table boundary token is not canonical"
        );
    }

    // The gaps between extension ranges are not lexical characters and must
    // reset adjacency instead of becoming synthetic CJK bigram members.
    assert_eq!(tokens("\u{4E00}\u{2A6E0}\u{4E01}"), ["一", "丁"]);
}

#[test]
fn complex_unicode_scripts_use_icu_word_boundaries() {
    assert_eq!(tokens("ทุกสองสัปดาห์"), ["ทุก", "สอง", "สัปดาห์"]);
    assert_eq!(tokens("नमस्ते दुनिया"), ["नमस्ते", "दुनिया"]);
    assert_eq!(tokens("مرحبا بالعالم"), ["مرحبا", "بالعالم"]);
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
fn mixed_cjk_and_myanmar_runs_are_canonical() {
    for input in ["におက", "ကにお", "東京က大阪", "한東京က"] {
        let emitted = tokens(input);
        for token in emitted {
            let retokenized = tokens(&token);
            assert_eq!(retokenized.as_slice(), std::slice::from_ref(&token));
        }
    }

    let emitted = tokens("におက");
    assert_eq!(emitted, ["にお", "က"]);
    for token in emitted {
        let retokenized = tokens(&token);
        assert_eq!(retokenized.as_slice(), std::slice::from_ref(&token));
    }
}

#[test]
fn regression_canonical_token_with_a_leading_combining_mark() {
    let emitted = tokens("׳ೳ.\u{300}º");
    assert_eq!(emitted, ["o"]);
    assert_eq!(tokens(&emitted[0]), emitted);
}

#[test]
fn leading_joiners_after_marks_are_canonical() {
    let emitted = tokens("׳\u{363}_a");
    assert_eq!(emitted, ["a"]);
    assert_eq!(tokens(&emitted[0]), emitted);
}

#[test]
fn marks_after_trailing_joiners_are_canonical() {
    let emitted = tokens("a'\u{300}\u{200D}🌀");
    assert_eq!(emitted, ["a"]);
    assert_eq!(tokens(&emitted[0]), emitted);
}

#[test]
fn nonspacing_marks_with_zero_combining_class_are_canonical() {
    let emitted = tokens("a_\u{D81}_🌀");
    assert_eq!(emitted, ["a"]);
    assert_eq!(tokens(&emitted[0]), emitted);
}

#[test]
fn mark_only_filler_runs_are_not_contextual_tokens() {
    let emitted = tokens("\u{363}\u{16FE4}");
    for token in emitted {
        assert_eq!(tokens(&token), [token]);
    }
    assert_eq!(tokens("\u{F71}"), ["\u{F71}"]);
}

#[test]
fn apostrophe_joiner_requires_letters_after_folding() {
    let emitted = tokens("a'\u{115F}0");
    assert_eq!(emitted, ["a", "0"]);
    for token in emitted {
        assert_eq!(tokens(&token), [token]);
    }
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn combining_marks_after_joiners_are_canonical() {
    assert_canonical_tokens("_\u{0610}_\u{0300}").unwrap();
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
fn assert_canonical_tokens(text: &str) -> Result<(), TestCaseError> {
    let emitted = tokens(text);
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
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn unicode_stress_tokens_are_canonical() {
    let config = ProptestConfig {
        cases: 2048,
        max_shrink_iters: 4096,
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);
    runner
        .run(&stress_text(), |text| assert_canonical_tokens(&text))
        .expect("Unicode stress properties failed");
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn full_unicode_scalar_texts_are_canonical() {
    let config = ProptestConfig {
        cases: 512,
        max_shrink_iters: 4096,
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);
    runner
        .run(&full_unicode_text(), |text| assert_canonical_tokens(&text))
        .expect("full Unicode property failed");
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn mixed_script_policy_runs_are_canonical() {
    let config = ProptestConfig {
        cases: 512,
        max_shrink_iters: 4096,
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);
    runner
        .run(&mixed_policy_text(), |text| assert_canonical_tokens(&text))
        .expect("mixed script policy property failed");
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
        prop_assert!(assert_canonical_tokens(&text).is_ok());
    }

    // The small ASCII property above remains a quick regression check. This
    // companion uses arbitrary Unicode scalar values and stresses mixed
    // scripts, combining marks, format controls, punctuation, and NFKC forms.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn full_unicode_tokens_are_canonical(text in full_unicode_text()) {
        prop_assert!(assert_canonical_tokens(&text).is_ok());
    }
}
