//! Recall tests: per-source scenarios,
//! source agreement/disagreement, temporality filters, the rendered
//! block as a golden contract, determinism, and the allow-set property.

use plugmem_core::memory::source;
use plugmem_core::{Config, FactId, MemStorage, Memory, RecallQuery, RememberInput};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::*;

fn cfg() -> Config {
    let mut cfg = Config::default();
    cfg.shards_facts = 8;
    cfg.shards_entities = 4;
    cfg.shards_edges = 4;
    cfg.shards_temporal = 4;
    cfg.shards_postings = 16;
    cfg
}

/// A fixed fixture: a user with preferences, a project with links, and
/// an unrelated fact. Timestamps are days for readability.
const DAY: u64 = 86_400_000;

fn fixture() -> (Memory<'static>, MemStorage) {
    let mut mem = Memory::new(cfg()).unwrap();
    let mut store = MemStorage::new();
    // f0: user pref, tagged.
    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("user"),
            tags: &["pref"],
            ..RememberInput::text(100 * DAY, "prefers tokio for async work")
        },
    )
    .unwrap();
    // f1: project fact with a link user —works_on→ plugmem.
    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("user"),
            links: &[("works_on", "plugmem")],
            ..RememberInput::text(200 * DAY, "building a memory engine")
        },
    )
    .unwrap();
    // f2: fact about the project itself.
    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("plugmem"),
            tags: &["project:plugmem"],
            ..RememberInput::text(300 * DAY, "plugmem stores facts in flat arenas")
        },
    )
    .unwrap();
    // f3: unrelated noise.
    mem.remember(
        &mut store,
        RememberInput::text(400 * DAY, "the weather was nice"),
    )
    .unwrap();
    (mem, store)
}

fn ids(result: &plugmem_core::RecallResult) -> Vec<u32> {
    result.facts.iter().map(|f| f.id.0).collect()
}

#[test]
fn lexical_recall_ranks_by_relevance() {
    let (mem, _) = fixture();
    let r = mem
        .recall(RecallQuery::text(500 * DAY, "tokio async"))
        .unwrap();
    assert_eq!(ids(&r)[0], 0, "the tokio fact must rank first");
    assert!(r.facts[0].sources & source::BM25 != 0);
    assert!(!r.truncated);
}

#[test]
fn graph_source_pulls_neighbor_facts_and_edges() {
    let (mem, _) = fixture();
    // Anchoring at the *project* pulls the user's facts through the edge.
    let r = mem
        .recall(RecallQuery {
            text: None,
            entities: &["plugmem"],
            ..RecallQuery::text(500 * DAY, "")
        })
        .unwrap();
    let got = ids(&r);
    assert!(got.contains(&2), "the project's own fact");
    assert!(got.contains(&1), "the neighbor's fact through works_on");
    assert!(!got.contains(&3), "unrelated noise stays out");
    assert_eq!(r.edges.len(), 1);
    assert_eq!(r.edges[0].provenance, FactId(1));
    assert!(r.facts.iter().all(|f| f.sources & source::GRAPH != 0));
    // The project's own facts outrank depth-1 neighbors.
    assert_eq!(got[0], 2);
}

#[test]
fn sources_agree_and_disagree() {
    let (mem, _) = fixture();
    // Text matches f2; the entity anchor also reaches f2 → agreement
    // ranks it above facts that only one source saw.
    let r = mem
        .recall(RecallQuery {
            entities: &["plugmem"],
            ..RecallQuery::text(500 * DAY, "flat arenas")
        })
        .unwrap();
    assert_eq!(ids(&r)[0], 2);
    let both = source::BM25 | source::GRAPH;
    assert_eq!(r.facts[0].sources & both, both);
}

#[test]
fn temporal_range_recalls_the_window_recent_first() {
    let (mem, _) = fixture();
    let r = mem
        .recall(RecallQuery {
            text: None,
            range: Some((150 * DAY, 350 * DAY)),
            ..RecallQuery::text(500 * DAY, "")
        })
        .unwrap();
    assert_eq!(ids(&r), [2, 1], "inside the window, recent first");
    assert!(r.facts.iter().all(|f| f.sources == source::TIME));
}

#[test]
fn tag_and_temporal_filters_intersect_exactly() {
    let (mem, _) = fixture();
    let r = mem
        .recall(RecallQuery {
            text: None,
            tags: &["pref"],
            range: Some((50 * DAY, 150 * DAY)),
            ..RecallQuery::text(500 * DAY, "")
        })
        .unwrap();
    assert_eq!(ids(&r), [0]);
    assert!(r.facts.iter().all(|f| f.sources == source::TIME));

    let r = mem
        .recall(RecallQuery {
            text: None,
            tags: &["pref"],
            range: Some((200 * DAY, 350 * DAY)),
            ..RecallQuery::text(500 * DAY, "")
        })
        .unwrap();
    assert!(r.facts.is_empty());
}

#[test]
fn tag_filter_is_an_intersection_with_every_source() {
    let (mem, _) = fixture();
    let r = mem
        .recall(RecallQuery {
            tags: &["pref"],
            ..RecallQuery::text(500 * DAY, "tokio weather arenas")
        })
        .unwrap();
    assert_eq!(ids(&r), [0], "only the tagged fact survives");
    // An unknown tag matches nothing at all.
    let r = mem
        .recall(RecallQuery {
            tags: &["nope"],
            ..RecallQuery::text(500 * DAY, "tokio")
        })
        .unwrap();
    assert!(r.facts.is_empty());
    assert!(
        r.rendered.is_empty(),
        "empty result renders as empty string"
    );
}

#[test]
fn as_of_and_tombstones_gate_candidates() {
    let (mut mem, mut store) = fixture();
    // Before f0 was recorded, it is unknown.
    let r = mem
        .recall(RecallQuery {
            as_of: Some(50 * DAY),
            ..RecallQuery::text(500 * DAY, "tokio")
        })
        .unwrap();
    assert!(r.facts.is_empty());
    // Revise f0: the old version leaves default recall, include_closed
    // shows the whole chain.
    mem.revise(
        &mut store,
        FactId(0),
        RememberInput {
            entity: Some("user"),
            tags: &["pref"],
            ..RememberInput::text(450 * DAY, "prefers async-std now")
        },
    )
    .unwrap();
    let r = mem.recall(RecallQuery::text(500 * DAY, "prefers")).unwrap();
    assert_eq!(ids(&r), [4], "only the successor is live");
    let r = mem
        .recall(RecallQuery {
            include_closed: true,
            ..RecallQuery::text(500 * DAY, "prefers")
        })
        .unwrap();
    assert_eq!(ids(&r).len(), 2, "include_closed shows the chain");
    // Forget hides from recall immediately.
    mem.forget(&mut store, 500 * DAY, FactId(3)).unwrap();
    let r = mem.recall(RecallQuery::text(500 * DAY, "weather")).unwrap();
    assert!(r.facts.is_empty());
}

#[test]
fn rendered_block_is_a_golden_contract() {
    let (mut mem, mut store) = fixture();
    mem.revise(
        &mut store,
        FactId(0),
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(450 * DAY, "prefers async-std now")
        },
    )
    .unwrap();
    let r = mem
        .recall(RecallQuery {
            entities: &["plugmem"],
            include_closed: true,
            k: 3,
            ..RecallQuery::text(500 * DAY, "prefers tokio arenas")
        })
        .unwrap();
    // The exact block, byte for byte: ids, names, month intervals,
    // open/closed markers, tags, link lines. Order: the successor fact is
    // both lexically matched and the most recent (recency boost), the
    // project fact agrees across two sources, the closed ancestor ranks
    // last.
    let want = "\
## memory
- [f4] user: prefers async-std now (1971-03; active)
- [f2] plugmem: plugmem stores facts in flat arenas (1970-10; active) #project:plugmem
- [f0] user: prefers tokio for async work (1970-04 → 1971-03; closed) #pref
- links: user —works_on→ plugmem
";
    assert_eq!(r.rendered, want);
}

#[test]
fn k_and_token_budget_truncate() {
    let (mem, _) = fixture();
    let r = mem
        .recall(RecallQuery {
            k: 1,
            ..RecallQuery::text(500 * DAY, "tokio arenas weather")
        })
        .unwrap();
    assert_eq!(r.facts.len(), 1);
    assert!(r.truncated);
    // A fact costs len(text)/4 + 8 tokens; a 20-token budget fits exactly
    // one of the fixture facts.
    let r = mem
        .recall(RecallQuery {
            token_budget: Some(20),
            ..RecallQuery::text(500 * DAY, "tokio arenas weather")
        })
        .unwrap();
    assert_eq!(r.facts.len(), 1, "one fact fits a 20-token budget");
    assert!(r.truncated);
}

#[test]
fn recall_is_deterministic_and_pure() {
    let (mem, _) = fixture();
    let q = RecallQuery {
        entities: &["user"],
        ..RecallQuery::text(500 * DAY, "tokio engine")
    };
    let a = mem.recall(q).unwrap();
    let b = mem.recall(q).unwrap();
    assert_eq!(a.rendered, b.rendered);
    assert_eq!(ids(&a), ids(&b));
    assert_eq!(a.edges, b.edges);
    // Purity: recall must not have grown the vocabulary or the arenas
    // (querying unknown words does not intern them).
    let before = mem.facts_len();
    mem.recall(RecallQuery::text(500 * DAY, "completely unknown words"))
        .unwrap();
    assert_eq!(mem.facts_len(), before);
}

/// Graph expansion stops once its caps are full instead of decoding a hub's
/// whole edge list, and the prefix it keeps is the deterministic one each
/// traversal order defines:
///
/// - the current graph is keyed by `(src, rel, dst)`, so it yields the
///   lowest-id neighbours first;
/// - `as_of` walks history backwards from the queried instant, so it yields
///   the edges that most recently became true — the ones a caller asking
///   "what was linked then" actually wants at the top.
#[test]
fn hub_expansion_is_capped_at_the_exhaustive_prefix() {
    const LEAVES: u32 = 400; // far past the 128-edge / 64-entity caps
    let mut mem = Memory::new(cfg()).unwrap();
    let mut store = MemStorage::new();
    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("hub"),
            ..RememberInput::text(DAY, "hub anchor")
        },
    )
    .unwrap();
    // Leaves are created in id order, and `link` keys edges by
    // `(src, rel, dst)`, so the expected walk order is leaf id order.
    for i in 0..LEAVES {
        let leaf = format!("leaf-{i:04}");
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some("hub"),
                links: &[("touches", leaf.as_str())],
                ..RememberInput::text((2 + u64::from(i)) * DAY, "hub edge fact")
            },
        )
        .unwrap();
    }

    let now = (LEAVES as u64 + 10) * DAY;
    for as_of in [None, Some(now)] {
        let result = mem
            .recall(RecallQuery {
                entities: &["hub"],
                as_of,
                k: 64,
                token_budget: Some(4096),
                ..RecallQuery::text(now, "")
            })
            .unwrap();
        let label = if as_of.is_some() { "as_of" } else { "current" };
        assert_eq!(result.edges.len(), 128, "{label}: edge cap");
        let dsts: Vec<u32> = result.edges.iter().map(|e| e.dst.0).collect();
        let hub = result.edges[0].src;
        assert!(
            result.edges.iter().all(|e| e.src == hub),
            "{label}: every edge leaves the hub"
        );
        // Leaf entities were created in ascending id order, one per round, so
        // id order and time order are the same sequence read from either end.
        let start = dsts[0];
        let expected: Vec<u32> = match as_of {
            None => (start..start + 128).collect(),
            Some(_) => (start - 127..=start).rev().collect(),
        };
        assert_eq!(dsts, expected, "{label}: prefix of the exhaustive walk");
    }
}

#[test]
fn reused_result_buffers_are_equivalent_to_fresh_ones() {
    let (mem, _) = fixture();
    let q = RecallQuery {
        entities: &["plugmem"],
        ..RecallQuery::text(500 * DAY, "arenas")
    };
    let fresh = mem.recall(q).unwrap();
    let mut reused = plugmem_core::RecallResult::default();
    let mut scratch = plugmem_core::RecallScratch::new();
    mem.recall_into(
        RecallQuery::text(500 * DAY, "tokio"),
        &mut scratch,
        &mut reused,
    )
    .unwrap();
    mem.recall_into(q, &mut scratch, &mut reused).unwrap();
    assert_eq!(fresh.rendered, reused.rendered);
    assert_eq!(ids(&fresh), ids(&reused));
}

#[cfg(not(target_family = "wasm"))]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    // With a tag filter, every recalled fact carries all query tags —
    // the result is a subset of the allow-set (property).
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn results_are_subsets_of_the_allow_set(
        tagged in proptest::collection::vec(proptest::collection::vec(0usize..4, 0..3), 1..20),
        query_tags in proptest::collection::vec(0usize..4, 1..3),
    ) {
        let pool = ["a", "b", "c", "d"];
        let mut mem = Memory::new(cfg()).unwrap();
        let mut store = MemStorage::new();
        for (i, tags) in tagged.iter().enumerate() {
            let tags: Vec<&str> = tags.iter().map(|&t| pool[t]).collect();
            mem.remember(&mut store, RememberInput {
                tags: &tags,
                ..RememberInput::text((i as u64 + 1) * DAY, "common searchable text")
            }).unwrap();
        }
        let query: Vec<&str> = query_tags.iter().map(|&t| pool[t]).collect();
        let r = mem.recall(RecallQuery {
            tags: &query,
            k: 64,
            ..RecallQuery::text(100 * DAY, "common searchable text")
        }).unwrap();
        let mut tag_buf = Vec::new();
        for fact in &r.facts {
            tag_buf.clear();
            mem.tags_of(fact.id, &mut tag_buf);
            let names: Vec<&str> = tag_buf.iter().map(|&t| {
                // resolve returns &str borrowed from mem; collect names
                // eagerly per fact.
                mem.term(t)
            }).collect();
            for q in &query {
                prop_assert!(names.contains(q), "fact {:?} misses tag {q}", fact.id);
            }
        }
    }
}
