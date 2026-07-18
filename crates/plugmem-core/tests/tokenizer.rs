//! Tokenizer tests (specs/04 test plan): the unit-case table
//! (rus/eng/digits/emoji/CJK/empty/oversized) plus the concatenation
//! property — joining texts with a separator must not change the token
//! multiset.

use plugmem_core::tokenizer::{MAX_TOKEN_BYTES, tokenize};
use proptest::prelude::*;

fn tokens(text: &str) -> Vec<String> {
    let mut buf = String::new();
    let mut out = Vec::new();
    tokenize(text, &mut buf, |t| out.push(t.to_owned()));
    out
}

#[test]
fn unit_case_table() {
    let cases: &[(&str, &[&str])] = &[
        // Empty and separator-only inputs.
        ("", &[]),
        ("  \t\n ... !!!", &[]),
        // English with punctuation and case.
        ("Hello, World!", &["hello", "world"]),
        // Russian full Unicode lowercase.
        ("Предпочитает Tokio", &["предпочитает", "tokio"]),
        // Digits are tokens; inner '-' and '\'' split (spec: v1 simplicity).
        ("gpt-4o o'clock 2026", &["gpt", "4o", "o", "clock", "2026"]),
        // Emoji separate, skin-tone modifiers and all.
        ("rust🦀lang 👍🏽ok", &["rust", "lang", "ok"]),
        // CJK ideographs: one token per character, latin run unaffected.
        ("東京tower", &["東", "京", "tower"]),
        // Mixed script inside one run stays one token.
        ("сloud42x", &["сloud42x"]),
        // German sharp s lowercases via the simple mapping (ẞ → ß).
        ("STRAẞE", &["straße"]),
        // 'İ' is the one char whose lowercase is two chars ("i" + a
        // combining mark); the mark is dropped to keep tokens canonical.
        ("İstanbul", &["istanbul"]),
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

    // Cyrillic is 2 bytes/char: 33 chars = 66 bytes → the cut at 64 lands
    // exactly on a char boundary and keeps 32 chars.
    let cyr = "ж".repeat(33);
    let got = tokens(&cyr);
    assert_eq!(got, ["ж".repeat(32)]);
    assert_eq!(got[0].len(), MAX_TOKEN_BYTES);

    // Devanagari is 3 bytes/char: 22 chars = 66 bytes → byte 64 is mid-char
    // and the cut must retreat to the boundary at 63.
    let deva = "अ".repeat(22);
    let got = tokens(&deva);
    assert_eq!(got, ["अ".repeat(21)]);
    assert_eq!(got[0].len(), 63);
}

#[test]
fn scratch_buffer_is_reused_not_leaked() {
    let mut buf = String::new();
    let mut first = Vec::new();
    tokenize("alpha beta", &mut buf, |t| first.push(t.to_owned()));
    // A second call on the same buffer must not see leftovers.
    let mut second = Vec::new();
    tokenize("gamma", &mut buf, |t| second.push(t.to_owned()));
    assert_eq!(first, ["alpha", "beta"]);
    assert_eq!(second, ["gamma"]);
}

proptest! {
    // Concatenating two texts with a separator yields exactly the tokens
    // of both texts, in order (the property from specs/04 — strengthened
    // from multiset to sequence, which our tokenizer guarantees).
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn concatenation_with_separator_preserves_tokens(a in ".{0,40}", b in ".{0,40}") {
        let joined = format!("{a} {b}");
        let mut want = tokens(&a);
        want.extend(tokens(&b));
        prop_assert_eq!(tokens(&joined), want);
    }

    // Tokens never exceed the byte cap, are never empty, and are always
    // fully lowercase (re-tokenizing a token is the identity when it is
    // within the cap).
    #[test]
    #[cfg_attr(miri, ignore)]
    fn token_shape_invariants(text in ".{0,80}") {
        for t in tokens(&text) {
            prop_assert!(!t.is_empty());
            prop_assert!(t.len() <= MAX_TOKEN_BYTES);
            let again = tokens(&t);
            prop_assert_eq!(again.concat(), t);
        }
    }
}
