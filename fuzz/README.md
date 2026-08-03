# Fuzz targets

Two of plugmem's files are untrusted input: the **snapshot** and the
**journal**. Both live on disk where anything could have written to them, and
a crash can truncate the journal mid-record.

This matters more than it usually does, because the engine's read accessors
panic on contract violations *by design* — `term()` resolves an id it assumes
exists, a slot lookup assumes the fact is there. What makes that sound is the
load path: it range-checks every persisted id before handing back an engine,
so after a successful open no stored reference can violate a contract. The
argument is only as strong as the checking, and the checking is what these
targets attack.

| target | what it feeds |
|---|---|
| `snapshot_open` | arbitrary bytes into `from_bytes`, `from_bytes_borrowed` and `from_bytes_overlay`, then drives the accessors that trust the load — including `verify`, a recall across every source, and a re-emit |
| `journal_replay` | arbitrary bytes into replay, alone and as a tail behind a fuzzer-shaped snapshot |
| `snapshot_scrub` | the container parser and its budgeted checksum walk, which turn attacker-controlled offsets into slice ranges before anything is validated |

`snapshot_open` covers the borrowed and overlay opens on purpose: there the
pools alias the input rather than an owned copy, so a missed bound reads the
fuzzer's buffer.

## Running

Needs nightly (libFuzzer) and `cargo install cargo-fuzz`.

```bash
# corpus/ first (written to, and it must exist), seeds/ second (read-only)
mkdir -p fuzz/corpus/snapshot_open
cargo +nightly fuzz run snapshot_open fuzz/corpus/snapshot_open fuzz/seeds/snapshot_open \
  -- -max_total_time=60 -rss_limit_mb=4096
cargo +nightly fuzz list                      # the targets
cargo +nightly fuzz run snapshot_open fuzz/artifacts/snapshot_open/crash-…
```

Keep `-rss_limit_mb` on: a malformed header that makes the loader over-allocate
should be reported as a finding, not hang the machine.

CI runs all three for 60 s each on every PR — enough to catch a regression,
not a campaign. Finding something new means a long local run:

```bash
cargo +nightly fuzz run snapshot_open fuzz/corpus/snapshot_open fuzz/seeds/snapshot_open \
  -- -max_total_time=3600 -rss_limit_mb=4096
```

## Seeds and corpus

`seeds/` is committed and hand-made: real images produced by the CLI — a
populated database (entities, tags, links, an unlink, a revision, a tombstone,
non-Latin text), an empty one, and a dirty journal. They exist because random
bytes never get past the magic number; without seeds the fuzzer only ever
exercises the header check and everything behind it stays unreached.

Seeds are kept **across format changes**, not replaced by them. `full.snap` and
`empty.snap` predate the per-document term-set summary, so opening them runs
the migration that widens those records — a path the fuzzer would otherwise
never reach, and one that turns untrusted bytes into an allocation size.
`summarized.snap` is the same shape of database in the current format. When a
layout changes, add an image in the new one and leave the old ones alone.

`corpus/` is the fuzzer's own working set. It is **ignored by git**: a single
run grows it by thousands of files, all derived from the seeds. Pass it first
and `seeds/` second so libFuzzer writes new finds into the corpus and treats
the seeds as read-only.

Regenerate a seed the way it was made — write a database with the CLI,
checkpoint it, and copy `<db>.snap.N` (or `<db>.journal` for the replay
target) into the matching `seeds/` directory.

`artifacts/` (failing inputs), `corpus/` and `target/` are ignored; the
targets and the seeds are not.

## Adding a target

Add the source under `fuzz_targets/`, declare a `[[bin]]` in `Cargo.toml`, and
add the name to the loop in the CI job. Prefer targets that *use* what they
built: a parser that returns `Ok` has proved nothing until the structures it
returned are exercised.
