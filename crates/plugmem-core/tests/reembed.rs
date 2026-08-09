//! Vector-space identity and explicit vector-axis replacement.

use plugmem_core::snapshot::SnapshotSink;
use plugmem_core::{
    Config, Error, MemScratch, MemStorage, Memory, RecallQuery, ReembedError, RememberInput,
    Scratch, Storage,
};

fn cfg(dim: usize) -> Config {
    let mut cfg = Config::default();
    cfg.dim = dim;
    cfg.flat_to_hnsw = 2;
    cfg
}

fn vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| seed + i as f32 + 1.0).collect()
}

#[test]
fn vector_space_claim_is_journaled_and_mismatch_is_typed() {
    let mut mem = Memory::new(cfg(2)).unwrap();
    let mut store = MemStorage::new();
    assert!(mem.claim_vector_space(&mut store, "model/old").unwrap());
    assert!(!mem.claim_vector_space(&mut store, "model/old").unwrap());
    assert!(
        mem.claim_vector_space(&mut store, "model/new").unwrap(),
        "an empty pool has no semantic data to protect"
    );
    mem.remember(
        &mut store,
        RememberInput {
            vector: Some(&[1.0, 2.0]),
            ..RememberInput::text(1, "establish the new space")
        },
    )
    .unwrap();
    assert!(matches!(
        mem.claim_vector_space(&mut store, "model/third"),
        Err(plugmem_core::Error::VectorSpaceMismatch { .. })
    ));

    let journal = store.read_journal().unwrap();
    let (replayed, report) = Memory::from_bytes(None, &journal, cfg(2)).unwrap();
    assert_eq!(report.replayed, 3);
    assert_eq!(replayed.vector_space(), Some("model/new"));
}

#[test]
fn untracked_nonempty_legacy_pool_cannot_be_claimed() {
    let mut mem = Memory::new(cfg(2)).unwrap();
    let mut store = MemStorage::new();
    mem.remember(
        &mut store,
        RememberInput {
            vector: Some(&[1.0, 2.0]),
            ..RememberInput::text(1, "legacy vector")
        },
    )
    .unwrap();
    assert_eq!(
        mem.claim_vector_space(&mut store, "guessed/model"),
        Err(plugmem_core::Error::UntrackedVectorSpace)
    );
}

#[test]
fn reembed_changes_only_the_vector_axis_and_survives_auto_maintain() {
    let mut source = Memory::new(cfg(2)).unwrap();
    let mut journal = MemStorage::new();
    source
        .claim_vector_space(&mut journal, "model/old")
        .unwrap();
    let old0 = vector(2, 1.0);
    let first = source
        .remember(
            &mut journal,
            RememberInput {
                tags: &["keep"],
                vector: Some(&old0),
                ..RememberInput::text(10, "first historical text")
            },
        )
        .unwrap()
        .id;
    let old1 = vector(2, 2.0);
    let second = source
        .revise(
            &mut journal,
            first,
            RememberInput {
                tags: &["keep"],
                vector: Some(&old1),
                ..RememberInput::text(20, "second current text")
            },
        )
        .unwrap()
        .id;
    let doomed_vec = vector(2, 3.0);
    let doomed = source
        .remember(
            &mut journal,
            RememberInput {
                vector: Some(&doomed_vec),
                ..RememberInput::text(30, "private forgotten text")
            },
        )
        .unwrap()
        .id;
    source.forget(&mut journal, 40, doomed).unwrap();

    let before_first = source.get(first).unwrap();
    let before_second = source.get(second).unwrap();
    let mut scratch = MemScratch::new();
    let mut snapshot = Vec::new();
    let mut seen = Vec::new();
    let report = source
        .write_reembedded_snapshot(
            50,
            3,
            "model/new",
            1,
            &mut scratch,
            &mut snapshot,
            |texts| {
                seen.extend(texts.iter().map(|s| (*s).to_string()));
                Ok::<_, ()>(
                    texts
                        .iter()
                        .enumerate()
                        .map(|(i, _)| vector(3, 10.0 + i as f32))
                        .collect(),
                )
            },
        )
        .unwrap();
    assert_eq!(report.previous_space.as_deref(), Some("model/old"));
    assert_eq!(report.new_space, "model/new");
    assert_eq!((report.previous_dim, report.new_dim), (2, 3));
    assert_eq!(report.embedded, 2);
    assert_eq!(report.tombstones_skipped, 1);
    assert!(!seen.iter().any(|s| s == "private forgotten text"));

    let (mut loaded, _) = Memory::from_bytes(Some(&snapshot), &[], cfg(3)).unwrap();
    loaded.verify().unwrap();
    assert_eq!(loaded.vector_space(), Some("model/new"));
    assert_eq!(loaded.get(first).unwrap().text, before_first.text);
    assert_eq!(loaded.get(second).unwrap().text, before_second.text);
    assert!(loaded.get(doomed).is_none());
    let mut tags = Vec::new();
    loaded.tags_of(second, &mut tags);
    assert_eq!(tags.len(), 1);
    assert_eq!(loaded.term(tags[0]), "keep");

    let query_vector = vector(3, 10.0);
    let mut query = RecallQuery::text(60, "");
    query.text = None;
    query.vector = Some(&query_vector);
    query.include_closed = true;
    let answer = loaded.recall(query).unwrap();
    assert!(answer.facts.iter().any(|fact| fact.id == first));

    // Auto maintenance may optimize/compact the new vectors, but it must never
    // infer or change an embedding space on its own.
    let mut after = MemStorage::new();
    loaded.maintain(&mut after, 70).unwrap();
    assert_eq!(loaded.vector_space(), Some("model/new"));
    let roundtrip = loaded.snapshot_bytes(80);
    let (again, _) = Memory::from_bytes(Some(&roundtrip), &[], cfg(3)).unwrap();
    assert_eq!(again.vector_space(), Some("model/new"));
}

#[test]
fn provider_failure_leaves_source_untouched() {
    let mut source = Memory::new(cfg(2)).unwrap();
    let mut store = MemStorage::new();
    source.claim_vector_space(&mut store, "model/old").unwrap();
    source
        .remember(
            &mut store,
            RememberInput {
                vector: Some(&[1.0, 2.0]),
                ..RememberInput::text(1, "unchanged")
            },
        )
        .unwrap();
    let before = source.snapshot_bytes(2);
    let mut scratch = MemScratch::new();
    let mut staged = Vec::new();
    let result = source.write_reembedded_snapshot(
        3,
        4,
        "model/broken",
        8,
        &mut scratch,
        &mut staged,
        |_| Err::<Vec<Vec<f32>>, _>("offline"),
    );
    assert!(matches!(
        result,
        Err(plugmem_core::ReembedError::Embedder("offline"))
    ));
    assert_eq!(source.snapshot_bytes(2), before);
    assert_eq!(source.vector_space(), Some("model/old"));
}

#[test]
fn malformed_space_and_batch_are_rejected_before_callback() {
    let source = Memory::new(cfg(2)).unwrap();
    let mut scratch = MemScratch::new();
    let mut staged = Vec::new();
    let mut calls = 0;
    let result =
        source.write_reembedded_snapshot(1, 3, "bad\nspace", 0, &mut scratch, &mut staged, |_| {
            calls += 1;
            Ok::<_, ()>(Vec::new())
        });
    assert!(matches!(
        result,
        Err(plugmem_core::ReembedError::Engine(
            plugmem_core::Error::Invalid(_)
        ))
    ));
    assert_eq!(calls, 0);
}

#[test]
fn empty_source_validates_the_provider_and_shape_errors_are_typed() {
    let source = Memory::new(cfg(2)).unwrap();

    let mut scratch = MemScratch::new();
    let mut snapshot = Vec::new();
    let report = source
        .write_reembedded_snapshot(
            1,
            3,
            "model/empty",
            2,
            &mut scratch,
            &mut snapshot,
            |texts| {
                assert_eq!(texts, &[""]);
                Ok::<_, ()>(vec![vector(3, 1.0)])
            },
        )
        .unwrap();
    assert_eq!(report.embedded, 0);
    assert_eq!(report.vector_bytes, 0);
    assert_eq!(report.new_dim, 3);
    let (loaded, _) = Memory::from_bytes(Some(&snapshot), &[], cfg(3)).unwrap();
    assert_eq!(loaded.vector_space(), Some("model/empty"));

    let mut scratch = MemScratch::new();
    let mut snapshot = Vec::new();
    let wrong_probe = source.write_reembedded_snapshot(
        1,
        3,
        "model/empty",
        2,
        &mut scratch,
        &mut snapshot,
        |_| Ok::<_, ()>(Vec::new()),
    );
    assert!(matches!(
        wrong_probe,
        Err(ReembedError::Engine(Error::Invalid(_)))
    ));

    for (dim, batch) in [(0, 1), (3, 0)] {
        let mut scratch = MemScratch::new();
        let mut snapshot = Vec::new();
        let rejected = source.write_reembedded_snapshot(
            1,
            dim,
            "model/valid",
            batch,
            &mut scratch,
            &mut snapshot,
            |_| Ok::<_, ()>(Vec::new()),
        );
        assert!(matches!(
            rejected,
            Err(ReembedError::Engine(Error::Invalid(_)))
        ));
    }
}

#[test]
fn batches_are_bounded_and_provider_output_is_checked_per_batch() {
    let mut source = Memory::new(cfg(2)).unwrap();
    let mut store = MemStorage::new();
    for (now, text) in [(1, "one"), (2, "two"), (3, "three")] {
        source
            .remember(&mut store, RememberInput::text(now, text))
            .unwrap();
    }

    let mut scratch = MemScratch::new();
    let mut snapshot = Vec::new();
    let mut batches = Vec::new();
    let report = source
        .write_reembedded_snapshot(
            4,
            3,
            "model/batched",
            2,
            &mut scratch,
            &mut snapshot,
            |texts| {
                batches.push(texts.len());
                Ok::<_, ()>(texts.iter().map(|_| vector(3, 4.0)).collect())
            },
        )
        .unwrap();
    assert_eq!(batches, [2, 1]);
    assert_eq!(report.embedded, 3);
    assert!(report.hnsw_indexed >= 2);

    let mut scratch = MemScratch::new();
    let mut snapshot = Vec::new();
    let wrong_count = source.write_reembedded_snapshot(
        4,
        3,
        "model/bad-count",
        2,
        &mut scratch,
        &mut snapshot,
        |_| Ok::<_, ()>(Vec::new()),
    );
    assert!(matches!(
        wrong_count,
        Err(ReembedError::Engine(Error::Invalid(_)))
    ));

    let mut scratch = MemScratch::new();
    let mut snapshot = Vec::new();
    let wrong_dim = source.write_reembedded_snapshot(
        4,
        3,
        "model/bad-dim",
        2,
        &mut scratch,
        &mut snapshot,
        |texts| Ok::<_, ()>(texts.iter().map(|_| vector(2, 4.0)).collect()),
    );
    assert!(matches!(
        wrong_dim,
        Err(ReembedError::Engine(Error::DimMismatch { got: 2, want: 3 }))
    ));
}

#[derive(Debug)]
enum ScratchFailure {
    Write,
    Freeze,
}

struct FailingScratch {
    fail: ScratchFailure,
    bytes: Vec<u8>,
}

impl Scratch for FailingScratch {
    type Error = &'static str;

    fn write(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        if matches!(self.fail, ScratchFailure::Write) {
            return Err("write failed");
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn freeze(&mut self) -> Result<&[u8], Self::Error> {
        if matches!(self.fail, ScratchFailure::Freeze) {
            return Err("freeze failed");
        }
        Ok(&self.bytes)
    }
}

struct FailingSink;

impl SnapshotSink for FailingSink {
    fn write(&mut self, _bytes: &[u8]) -> Result<(), Error> {
        Err(Error::Storage("sink write failed".into()))
    }

    fn patch(&mut self, _at: u64, _bytes: &[u8]) -> Result<(), Error> {
        Err(Error::Storage("sink patch failed".into()))
    }
}

#[test]
fn scratch_and_snapshot_failures_remain_engine_errors() {
    let mut source = Memory::new(cfg(2)).unwrap();
    let mut store = MemStorage::new();
    source
        .remember(&mut store, RememberInput::text(1, "one"))
        .unwrap();
    let mut scratch = FailingScratch {
        fail: ScratchFailure::Write,
        bytes: Vec::new(),
    };
    let mut snapshot = Vec::new();
    let write_error = source.write_reembedded_snapshot(
        2,
        3,
        "model/write-failure",
        1,
        &mut scratch,
        &mut snapshot,
        |texts| Ok::<_, ()>(texts.iter().map(|_| vector(3, 1.0)).collect()),
    );
    assert!(matches!(
        write_error,
        Err(ReembedError::Engine(Error::Storage(_)))
    ));

    let empty = Memory::new(cfg(2)).unwrap();
    let mut scratch = FailingScratch {
        fail: ScratchFailure::Freeze,
        bytes: Vec::new(),
    };
    let mut snapshot = Vec::new();
    let freeze_error = empty.write_reembedded_snapshot(
        2,
        3,
        "model/freeze-failure",
        1,
        &mut scratch,
        &mut snapshot,
        |_| Ok::<_, ()>(vec![vector(3, 1.0)]),
    );
    assert!(matches!(
        freeze_error,
        Err(ReembedError::Engine(Error::Storage(_)))
    ));

    let mut scratch = MemScratch::new();
    let sink_error = empty.write_reembedded_snapshot(
        2,
        3,
        "model/sink-failure",
        1,
        &mut scratch,
        FailingSink,
        |_| Ok::<_, ()>(vec![vector(3, 1.0)]),
    );
    assert!(matches!(
        sink_error,
        Err(ReembedError::Engine(Error::Storage(_)))
    ));
}
