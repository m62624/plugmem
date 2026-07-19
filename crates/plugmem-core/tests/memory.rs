//! Engine-verb contract tests (specs/02 §tests, specs/05): temporality
//! scenarios, entity resolution, edges, errors — and the persistence
//! property: replaying the journal reproduces the direct execution.

use plugmem_core::{
    Config, Error, FactId, LinkInput, MemStorage, Memory, RememberInput, Storage, VALID_TO_OPEN,
};
use proptest::prelude::*;

/// A small-sharded config (tests don't need 1024-shard arenas).
fn cfg() -> Config {
    let mut cfg = Config::default();
    cfg.shards_facts = 8;
    cfg.shards_entities = 4;
    cfg.shards_edges = 4;
    cfg.shards_temporal = 4;
    cfg.shards_postings = 16;
    cfg
}

fn engine() -> (Memory, MemStorage) {
    (Memory::new(cfg()).unwrap(), MemStorage::new())
}

#[test]
fn remember_assigns_dense_ids_and_stores_the_record() {
    let (mut mem, mut store) = engine();
    let a = mem
        .remember(&mut store, RememberInput::text(100, "prefers tokio"))
        .unwrap();
    let b = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("User"),
                tags: &["pref"],
                valid_from: Some(50),
                ..RememberInput::text(200, "lives in Moscow")
            },
        )
        .unwrap();
    assert_eq!(a.id, FactId(0));
    assert_eq!(b.id, FactId(1));
    assert_eq!(a.entity, None);
    assert!(b.entity.is_some());

    let view = mem.get(b.id).unwrap();
    assert_eq!(view.text, "lives in Moscow");
    assert_eq!(view.record.recorded_at, 200);
    assert_eq!(view.record.valid_from, 50);
    assert_eq!(view.record.valid_to, VALID_TO_OPEN);
    assert!(view.record.revises.is_none());
    assert!(view.record.is_live_at(300));
    assert!(!view.record.is_live_at(150), "recorded_at gates knowledge");

    let mut tags = Vec::new();
    mem.tags_of(b.id, &mut tags);
    assert_eq!(tags.len(), 1);
    assert_eq!(mem.term(tags[0]), "pref");
    assert_eq!(mem.facts_len(), 2);
}

#[test]
fn revise_closes_the_target_and_chains() {
    let (mut mem, mut store) = engine();
    let old = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(100, "lives in Moscow")
            },
        )
        .unwrap();
    let new = mem
        .revise(
            &mut store,
            old.id,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(500, "lives in Berlin")
            },
        )
        .unwrap();

    let closed = mem.get(old.id).unwrap().record;
    assert!(closed.is_closed());
    assert_eq!(closed.valid_to, 500, "old validity ends where new begins");
    assert!(
        closed.is_live_at(300),
        "history stays true for its interval"
    );
    assert!(!closed.is_live_at(600));
    let head = mem.get(new.id).unwrap().record;
    assert_eq!(head.revises, old.id);
    assert!(head.is_live_at(600));

    // Revising a closed fact is a typed error; so is a missing target.
    assert_eq!(
        mem.revise(&mut store, old.id, RememberInput::text(700, "x"))
            .unwrap_err(),
        Error::AlreadyClosed(old.id)
    );
    assert_eq!(
        mem.revise(&mut store, FactId(99), RememberInput::text(700, "x"))
            .unwrap_err(),
        Error::NotFound(FactId(99))
    );
}

#[test]
fn forget_hides_immediately() {
    let (mut mem, mut store) = engine();
    let f = mem
        .remember(&mut store, RememberInput::text(100, "secret"))
        .unwrap();
    assert!(mem.get(f.id).is_some());
    assert_eq!(mem.forget(&mut store, 200, f.id), Ok(true));
    assert!(mem.get(f.id).is_none(), "tombstone hides from get");
    assert_eq!(mem.forget(&mut store, 300, f.id), Ok(false), "idempotent");
    assert_eq!(
        mem.forget(&mut store, 300, FactId(9)),
        Err(Error::NotFound(FactId(9)))
    );
    // A tombstoned fact cannot be revised.
    assert_eq!(
        mem.revise(&mut store, f.id, RememberInput::text(400, "y"))
            .unwrap_err(),
        Error::NotFound(f.id)
    );
}

#[test]
fn entities_deduplicate_by_normalized_name() {
    let (mut mem, mut store) = engine();
    let a = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("Проект  PlugMem"),
                ..RememberInput::text(1, "a")
            },
        )
        .unwrap();
    let b = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("проект plugmem"),
                ..RememberInput::text(2, "b")
            },
        )
        .unwrap();
    assert_eq!(a.entity, b.entity, "same normalized name, same entity");
    assert_eq!(mem.entities_len(), 1);
    assert_eq!(mem.entity("ПРОЕКТ   PLUGMEM"), a.entity);
    assert_eq!(mem.entity("unknown thing"), None);
    assert_eq!(mem.entity("!!!"), None, "no tokens, no entity");
    assert_eq!(mem.cfg().shards_entities, 4);
    // A name with no indexable characters is invalid.
    assert_eq!(
        mem.remember(
            &mut store,
            RememberInput {
                entity: Some("!!!"),
                ..RememberInput::text(3, "c")
            },
        )
        .unwrap_err(),
        Error::Invalid("entity name has no indexable characters")
    );
}

#[test]
fn links_create_edges_and_input_limits_hold() {
    let (mut mem, mut store) = engine();
    // Links without a subject are rejected.
    assert_eq!(
        mem.remember(
            &mut store,
            RememberInput {
                links: &[("works_on", "plugmem")],
                ..RememberInput::text(1, "x")
            },
        )
        .unwrap_err(),
        Error::Invalid("links require a subject entity")
    );
    let f = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                links: &[("works_on", "plugmem")],
                ..RememberInput::text(2, "user builds plugmem")
            },
        )
        .unwrap();
    assert_eq!(mem.entities_len(), 2, "link target created lazily");
    // Standalone link verb: upsert (same edge, new provenance) plus a new
    // edge, both directions resolvable later through recall's graph
    // source; here we assert through entity resolution only.
    mem.link(
        &mut store,
        LinkInput {
            now: 3,
            src: "user",
            rel: "works_on",
            dst: "plugmem",
            provenance: Some(f.id),
        },
    )
    .unwrap();
    mem.link(
        &mut store,
        LinkInput {
            now: 4,
            src: "plugmem",
            rel: "depends_on",
            dst: "tokio",
            provenance: None,
        },
    )
    .unwrap();
    assert_eq!(mem.entities_len(), 3);

    // Size limits are typed errors.
    let tags: Vec<&str> = (0..33).map(|_| "t").collect();
    let err = mem
        .remember(
            &mut store,
            RememberInput {
                tags: &tags,
                ..RememberInput::text(5, "x")
            },
        )
        .unwrap_err();
    assert!(matches!(err, Error::TooLarge { what: "tags", .. }));
    let big = "a".repeat(5000);
    let err = mem
        .remember(&mut store, RememberInput::text(6, &big))
        .unwrap_err();
    assert!(matches!(err, Error::TooLarge { what: "text", .. }));
    let links: Vec<(&str, &str)> = (0..17).map(|_| ("r", "e")).collect();
    let err = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                links: &links,
                ..RememberInput::text(6, "x")
            },
        )
        .unwrap_err();
    assert!(matches!(err, Error::TooLarge { what: "links", .. }));
    assert_eq!(
        mem.remember(
            &mut store,
            RememberInput {
                tags: &["ok", ""],
                ..RememberInput::text(7, "x")
            },
        )
        .unwrap_err(),
        Error::Invalid("empty tag")
    );
}

#[test]
fn repeated_tokens_accumulate_term_frequency() {
    // "tokio tokio tokio" exercises the tf-accumulation arm; observable
    // through get() only indirectly, so this is a smoke that the path
    // runs (BM25 ranking checks live in the index tests).
    let (mut mem, mut store) = engine();
    mem.remember(&mut store, RememberInput::text(1, "tokio tokio tokio"))
        .unwrap();
    assert_eq!(mem.facts_len(), 1);
}

#[test]
fn duplicate_tags_collapse() {
    let (mut mem, mut store) = engine();
    let f = mem
        .remember(
            &mut store,
            RememberInput {
                tags: &["pref", "pref", "health"],
                ..RememberInput::text(1, "x")
            },
        )
        .unwrap();
    let mut tags = Vec::new();
    mem.tags_of(f.id, &mut tags);
    assert_eq!(tags.len(), 2);
}

#[test]
fn a_vector_config_is_accepted() {
    let mut c = cfg();
    c.dim = 384;
    // The vector layer is built now: a positive dim constructs cleanly
    // (full vector behavior is exercised in tests/vectors.rs).
    assert!(Memory::new(c).is_ok());
}

/// Replay equivalence: run a scripted mix of verbs, reopen from the
/// journal alone, compare every observable.
#[test]
fn journal_replay_reproduces_direct_execution() {
    let (mut mem, mut store) = engine();
    let a = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                tags: &["pref"],
                ..RememberInput::text(100, "prefers tokio")
            },
        )
        .unwrap();
    let b = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                links: &[("works_on", "plugmem")],
                ..RememberInput::text(200, "builds a memory engine")
            },
        )
        .unwrap();
    mem.revise(
        &mut store,
        a.id,
        RememberInput {
            entity: Some("user"),
            tags: &["pref"],
            ..RememberInput::text(300, "prefers async-std now")
        },
    )
    .unwrap();
    mem.forget(&mut store, 400, b.id).unwrap();
    mem.link(
        &mut store,
        LinkInput {
            now: 500,
            src: "plugmem",
            rel: "depends_on",
            dst: "tokio",
            provenance: None,
        },
    )
    .unwrap();

    let (reopened, report) = Memory::open(&mut store, cfg()).unwrap();
    assert_eq!(report.replayed, 5);
    assert_eq!(report.skipped, 0);
    assert!(!report.truncated_tail);
    assert_observably_equal(&mem, &reopened);
}

/// Compares every fact view, tag set and entity count of two engines.
fn assert_observably_equal(a: &Memory, b: &Memory) {
    assert_eq!(a.facts_len(), b.facts_len());
    assert_eq!(a.entities_len(), b.entities_len());
    for id in 0..a.facts_len() as u32 {
        let id = FactId(id);
        match (a.get(id), b.get(id)) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(x.text, y.text, "fact {id:?}");
                assert_eq!(x.record, y.record, "fact {id:?}");
                let (mut ta, mut tb) = (Vec::new(), Vec::new());
                a.tags_of(id, &mut ta);
                b.tags_of(id, &mut tb);
                assert_eq!(ta, tb, "tags of {id:?}");
            }
            (x, y) => panic!("fact {id:?} presence differs: {x:?} vs {y:?}"),
        }
    }
}

#[test]
fn open_on_empty_storage_is_a_fresh_database() {
    let mut store = MemStorage::new();
    let (mem, report) = Memory::open(&mut store, cfg()).unwrap();
    assert_eq!(mem.facts_len(), 0);
    assert_eq!(report, plugmem_core::OpenReport::default());
}

#[test]
fn torn_journal_tail_is_recovered_and_reported() {
    let (mut mem, mut store) = engine();
    mem.remember(&mut store, RememberInput::text(1, "kept"))
        .unwrap();
    mem.remember(&mut store, RememberInput::text(2, "torn"))
        .unwrap();
    let journal = store.read_journal().unwrap();
    let mut torn = MemStorage::new();
    torn.append_journal(&journal[..journal.len() - 3]).unwrap();
    let (reopened, report) = Memory::open(&mut torn, cfg()).unwrap();
    assert_eq!(reopened.facts_len(), 1);
    assert!(report.truncated_tail);
    assert_eq!(report.replayed, 1);
}

/// One step of the property workload.
#[derive(Debug, Clone)]
enum Step {
    Remember { entity: Option<u8>, tags: Vec<u8> },
    Revise { target: u8 },
    Forget { target: u8 },
    Link { src: u8, dst: u8 },
}

fn step_strategy() -> impl Strategy<Value = Step> {
    prop_oneof![
        4 => (proptest::option::of(0u8..5), proptest::collection::vec(0u8..6, 0..3))
            .prop_map(|(entity, tags)| Step::Remember { entity, tags }),
        2 => (any::<u8>(),).prop_map(|(target,)| Step::Revise { target }),
        1 => (any::<u8>(),).prop_map(|(target,)| Step::Forget { target }),
        1 => (0u8..5, 0u8..5).prop_map(|(src, dst)| Step::Link { src, dst }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    // Any interleaving of verbs (including failing ones) leaves a journal
    // whose replay reproduces the direct execution observably.
    #[test]
    #[cfg_attr(miri, ignore)] // proptest persistence calls getcwd — forbidden under miri isolation
    fn replay_equivalence_holds_for_random_workloads(steps in proptest::collection::vec(step_strategy(), 0..40)) {
        let (mut mem, mut store) = engine();
        let names = ["user", "plugmem", "кот Барсик", "tokio", "работа"];
        let tag_pool = ["pref", "health", "project:plugmem", "a", "b", "c"];
        let mut now = 0u64;
        for step in steps {
            now += 10;
            match step {
                Step::Remember { entity, tags } => {
                    let tags: Vec<&str> = tags.iter().map(|&t| tag_pool[t as usize]).collect();
                    let _ = mem.remember(&mut store, RememberInput {
                        entity: entity.map(|e| names[e as usize]),
                        tags: &tags,
                        ..RememberInput::text(now, "some fact text")
                    });
                }
                Step::Revise { target } => {
                    let _ = mem.revise(&mut store, FactId(u32::from(target)),
                        RememberInput::text(now, "revised text"));
                }
                Step::Forget { target } => {
                    let _ = mem.forget(&mut store, now, FactId(u32::from(target)));
                }
                Step::Link { src, dst } => {
                    let _ = mem.link(&mut store, LinkInput {
                        now,
                        src: names[src as usize],
                        rel: "rel",
                        dst: names[dst as usize],
                        provenance: None,
                    });
                }
            }
        }
        let (reopened, _) = Memory::open(&mut store, cfg()).unwrap();
        assert_observably_equal(&mem, &reopened);
    }
}

#[test]
fn similar_detection_surfaces_conflicts_but_never_acts() {
    let (mut mem, mut store) = engine();
    let first = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(100, "lives in Moscow city")
            },
        )
        .unwrap();
    assert!(first.similar.is_empty(), "nothing to be similar to yet");
    // A near-duplicate statement about the same entity.
    let second = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(200, "lives in Moscow now")
            },
        )
        .unwrap();
    assert_eq!(second.similar.len(), 1, "the overlap must surface");
    assert_eq!(second.similar[0].id, first.id);
    assert!(second.similar[0].score > 0.5);
    assert_eq!(
        second.similar[0].reason,
        plugmem_core::SimilarReason::LexicalOverlap
    );
    // The engine did NOT revise anything by itself.
    assert!(!mem.get(first.id).unwrap().record.is_closed());

    // A different entity or unrelated text stays silent.
    let other = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("кот Барсик"),
                ..RememberInput::text(300, "lives in Moscow too")
            },
        )
        .unwrap();
    assert!(other.similar.is_empty(), "different entity, no hint");
    let unrelated = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(400, "prefers strong coffee")
            },
        )
        .unwrap();
    assert!(unrelated.similar.is_empty(), "no lexical overlap, no hint");
    // Closed facts are not conflict candidates: revise the first, then a
    // third "lives in Moscow" statement flags only the live successor.
    let successor = mem
        .revise(
            &mut store,
            first.id,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(500, "lives in Moscow region")
            },
        )
        .unwrap();
    let third = mem
        .remember(
            &mut store,
            RememberInput {
                entity: Some("user"),
                ..RememberInput::text(600, "lives in Moscow region still")
            },
        )
        .unwrap();
    let flagged: Vec<_> = third.similar.iter().map(|s| s.id).collect();
    assert!(flagged.contains(&successor.id));
    assert!(
        !flagged.contains(&first.id),
        "closed facts are history, not conflicts"
    );
}

#[test]
fn remember_batch_imports_and_skips_similar() {
    let (mut mem, mut store) = engine();
    let inputs = [
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(100, "likes green tea a lot")
        },
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(200, "likes green tea very much")
        },
    ];
    let outcomes = mem.remember_batch(&mut store, &inputs, true).unwrap();
    assert_eq!(outcomes.len(), 2);
    assert!(
        outcomes[1].similar.is_empty(),
        "skip_similar suppresses hints"
    );
    // The same near-duplicates without the skip do produce a hint.
    let outcomes = mem
        .remember_batch(
            &mut store,
            &[RememberInput {
                entity: Some("user"),
                ..RememberInput::text(300, "likes green tea very")
            }],
            false,
        )
        .unwrap();
    assert!(!outcomes[0].similar.is_empty());
    // Batch is journaled per record: reopen sees all three facts.
    let (reopened, _) = Memory::open(&mut store, cfg()).unwrap();
    assert_eq!(reopened.facts_len(), 3);
}

#[test]
fn stats_report_engine_counters() {
    let (mut mem, mut store) = engine();
    let empty = mem.stats();
    assert_eq!(
        (empty.facts, empty.entities, empty.terms, empty.edges),
        (0, 0, 0, 0)
    );
    assert_eq!((empty.next_fact, empty.next_entity), (0, 0));
    assert_eq!(empty.vectors, 0);
    assert_eq!(empty.pool_bytes, 0);

    mem.remember(
        &mut store,
        RememberInput {
            entity: Some("user"),
            tags: &["pref"],
            links: &[("works_on", "plugmem")],
            ..RememberInput::text(1, "likes strongly typed engines")
        },
    )
    .unwrap();
    mem.remember(&mut store, RememberInput::text(2, "second fact"))
        .unwrap();

    let s = mem.stats();
    assert_eq!((s.facts, s.entities, s.edges), (2, 2, 1));
    assert_eq!((s.next_fact, s.next_entity), (2, 2));
    assert!(s.terms > 0, "tokens, tags and names were interned");
    assert!(s.pool_bytes > 0, "pools hold the records and texts");
}
