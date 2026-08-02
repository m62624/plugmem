# Local guide: `plugmem-testgen`

## Role

`plugmem-testgen` is internal deterministic workload generation for core tests and benchmarks. It is not part of the production storage path and is not published. The generator must be reproducible across machines and targets.

The stream is a pure function of `(seed, Profile)`. `rng.rs` implements the PRNG in place so corpus bytes do not depend on an external random crate's stream stability. `words.rs` provides deterministic pronounceable words and a Zipf-like vocabulary.

## Workload model

`Profile` controls dictionary size, tags, vector dimension/clusters, entity creation share, mean text length, operation weights, and time-axis step. The default profile has vectors disabled, realistic text/tags/entities, and no maintenance operations. Set `Profile::dim` to the same value as the consuming engine's `Config::dim`.

Generated operations are owned `GenOp` values: remember, revise, forget, standalone link, and maintain. The generator tracks live/open facts and entities so a revise targets an open fact and a forget targets a live fact. Links, tags, vectors, and timestamps are generated as valid inputs rather than arbitrary malformed noise.

The corpus intentionally includes skew: Zipf vocabulary/tags, hub entities, clustered unit vectors, and increasing time density. Do not describe it as uniformly random. Record the seed and complete `Profile` whenever publishing a benchmark result.

## Applying operations

`apply` is the single mapping from owned `GenOp` values to `plugmem-core` verbs. Keep it aligned with core API semantics. If a new engine verb is added, add a matching operation, generator bookkeeping, apply mapping, and deterministic test.

Do not use wall-clock time or an unseeded source of randomness in this crate. Avoid changing the PRNG algorithm or vocabulary formulas casually: that changes historical corpus identity and benchmark comparability.

## Tests

```bash
cargo test -p plugmem-testgen
cargo test -p plugmem-testgen --test gen
```

Tests should assert repeatability, valid target selection, profile dimension consistency, operation distributions, and stable edge cases. Prefer small deterministic seeds for debugging and keep large workloads in benchmarks rather than unit tests.
