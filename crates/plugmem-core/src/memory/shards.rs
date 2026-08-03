//! How many shards each arena gets.
//!
//! There is no correct constant. Too few shards on a large database lengthens
//! the sorted page directory each insert memmoves within; too many on a small
//! one buys a 4 KiB page for every shard that holds a single record, which is
//! how a thousand facts came to occupy fourteen megabytes. The right number
//! follows the data, so it is derived here rather than configured.
//!
//! Three properties this module owes the rest of the engine:
//!
//! 1. **It is a pure function of engine state.** `maintain` may re-shard, and
//!    the journal records only *that* a maintenance pass ran, not what layout
//!    it chose — so replay recomputes the layout from the same state and must
//!    reach the same answer. A rule that consulted the clock, the caller's
//!    config, or anything else outside the engine would make a journal replay
//!    diverge from the run it replays.
//! 2. **It does not depend on pointer width.** Every product below overflows a
//!    32-bit `usize` at sizes a database can really reach (90M facts is enough),
//!    so the arithmetic is `u64` throughout and the clamp happens before the
//!    single cast. wasm32 and a 64-bit host must agree, for the same replay
//!    reason.
//! 3. **It reads only true payload, never occupancy.** Sizing from
//!    `pool_bytes()` would feed page slack back into the rule: an over-sharded
//!    arena reports more bytes, which would ask for more shards still. Counts
//!    times slot widths are exact and have no such loop.
//!
//! Only the *sharded* structures matter. A [`PostingStore`](crate::index) keeps
//! its bulk in an unsharded chunk pool and shards only the per-term handles, so
//! the postings count follows the number of terms and documents, not the size
//! of the posting lists.

use plugmem_arena::Slot;

use crate::config::{Config, MAX_SHARDS, MIN_SHARDS, SHARD_TARGET_BYTES};
use crate::index::bm25::DocLenSlot;
use crate::index::postings::IdListSlot;
use crate::model::{
    EdgeHistorySlot, EdgeSlot, EntityByName, EntityRecord, FactAux, FactRecord, TemporalSlot,
};

/// Shards for one arena group holding `bytes` of slot payload.
///
/// Clamping before `next_power_of_two` is deliberate: it keeps the result
/// inside `[MIN_SHARDS, MAX_SHARDS]` (both powers of two) without the rounding
/// ever overflowing, and makes the final cast provably in range on a 32-bit
/// target.
fn shards_for(bytes: u64) -> usize {
    let want = bytes
        .div_ceil(SHARD_TARGET_BYTES as u64)
        .clamp(MIN_SHARDS as u64, MAX_SHARDS as u64);
    debug_assert!(want <= MAX_SHARDS as u64);
    want.next_power_of_two() as usize
}

/// Payload of `count` records of one slot type, in `u64` so the product cannot
/// overflow the way `count * T::SIZE` would in a 32-bit `usize`.
fn payload<T: Slot>(count: u64) -> u64 {
    count.saturating_mul(T::SIZE as u64)
}

/// What the engine holds, in records — the only input the layout rule takes.
///
/// Counts, not bytes: see the module note on why occupancy must not feed back
/// into the rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Population {
    /// Fact records that survive a purge (tombstones excluded).
    pub facts: u64,
    /// Entities. Never purged.
    pub entities: u64,
    /// Currently open edges, counted once per mirror arena.
    pub edges: u64,
    /// Edge history versions, open and closed.
    pub edge_versions: u64,
    /// Interned terms carrying postings.
    pub terms: u64,
    /// Distinct tags carrying id lists.
    pub tags: u64,
    /// Documents in the BM25 length arena.
    pub documents: u64,
}

/// Shard counts for every arena group.
///
/// Public because it is observable: it appears in [`Stats`](crate::Stats) and
/// in a [`MaintainReport`](crate::MaintainReport), which is how a caller sees
/// that a database re-sharded itself and what it moved to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShardLayout {
    /// Facts and their cold sidecar.
    pub facts: usize,
    /// Entities, the by-name index and the per-entity fact lists.
    pub entities: usize,
    /// Both current edge mirrors and both history mirrors.
    pub edges: usize,
    /// The `recorded_at` index.
    pub temporal: usize,
    /// BM25 term handles and document lengths, plus the tag lists.
    pub postings: usize,
}

impl Default for ShardLayout {
    /// The floor — what an empty database is laid out with.
    fn default() -> Self {
        Self {
            facts: MIN_SHARDS,
            entities: MIN_SHARDS,
            edges: MIN_SHARDS,
            temporal: MIN_SHARDS,
            postings: MIN_SHARDS,
        }
    }
}

impl ShardLayout {
    /// The layout `population` calls for.
    ///
    /// Arenas that share a shard count are sized by the largest of them: that
    /// one sets the page-directory length, and the smaller ones simply end up
    /// with fewer pages per shard, which costs nothing.
    pub(crate) fn for_population(population: &Population) -> Self {
        let Population {
            facts,
            entities,
            edges,
            edge_versions,
            terms,
            tags,
            documents,
        } = *population;
        Self {
            // `facts` and `fact_aux`.
            facts: shards_for(payload::<FactRecord>(facts).max(payload::<FactAux>(facts))),
            // `entities`, `by_name` and the `entity_facts` id lists.
            entities: shards_for(
                payload::<EntityRecord>(entities)
                    .max(payload::<EntityByName>(entities))
                    .max(payload::<IdListSlot>(entities)),
            ),
            // Four arenas: both current mirrors and both history mirrors.
            edges: shards_for(
                payload::<EdgeSlot>(edges).max(payload::<EdgeHistorySlot>(edge_versions)),
            ),
            temporal: shards_for(payload::<TemporalSlot>(facts)),
            // BM25 term handles, BM25 document lengths, and the tag id lists.
            postings: shards_for(
                payload::<IdListSlot>(terms)
                    .max(payload::<DocLenSlot>(documents))
                    .max(payload::<IdListSlot>(tags)),
            ),
        }
    }

    /// The layout a config records.
    pub(crate) fn of_config(cfg: &Config) -> Self {
        Self {
            facts: cfg.shards_facts,
            entities: cfg.shards_entities,
            edges: cfg.shards_edges,
            temporal: cfg.shards_temporal,
            postings: cfg.shards_postings,
        }
    }

    /// Writes this layout into `cfg`.
    pub(crate) fn apply(&self, cfg: &mut Config) {
        cfg.shards_facts = self.facts;
        cfg.shards_entities = self.entities;
        cfg.shards_edges = self.edges;
        cfg.shards_temporal = self.temporal;
        cfg.shards_postings = self.postings;
    }

    /// Whether one group's move from `have` to `want` earns a rebuild.
    ///
    /// Asymmetric on purpose. Growing protects against a cost that compounds —
    /// a page directory that keeps lengthening — so one doubling is enough to
    /// act on. Shrinking only returns memory, so it waits for a fourfold
    /// overshoot. The gap between the two is what stops a database sitting
    /// near a boundary from rebuilding itself on every maintenance pass, and
    /// what guarantees the count a rebuild lands on is itself stable.
    fn group_earns_rebuild(have: usize, want: usize) -> bool {
        want >= have.saturating_mul(GROW_FACTOR) || have >= want.saturating_mul(SHRINK_FACTOR)
    }

    /// Whether the groups a compacting pass rebuilds are far enough out.
    ///
    /// Deliberately excludes the edges: a compaction does not touch the edge
    /// arenas, so including them here would make every pass report work,
    /// rebuild everything else, leave the edges as they were, and be asked
    /// again immediately.
    pub(crate) fn compacted_groups_earn_rebuild(&self, target: &Self) -> bool {
        [
            (self.facts, target.facts),
            (self.entities, target.entities),
            (self.temporal, target.temporal),
            (self.postings, target.postings),
        ]
        .iter()
        .any(|&(have, want)| Self::group_earns_rebuild(have, want))
    }

    /// Whether the four edge arenas are far enough out to be worth repacking.
    pub(crate) fn edges_earn_rebuild(&self, target: &Self) -> bool {
        Self::group_earns_rebuild(self.edges, target.edges)
    }

    /// This layout with the groups a compaction rebuilt taken from `target`,
    /// and the edges taken from it only when they too were rebuilt.
    ///
    /// What a pass may claim in the config is exactly what it actually laid
    /// out. Claiming more would leave the config describing a shape the file
    /// does not have — the loader would then read those arenas with the wrong
    /// shard count, which is corruption, not inefficiency.
    pub(crate) fn realized(&self, target: &Self, edges_rebuilt: bool) -> Self {
        Self {
            facts: target.facts,
            entities: target.entities,
            temporal: target.temporal,
            postings: target.postings,
            edges: if edges_rebuilt {
                target.edges
            } else {
                self.edges
            },
        }
    }
}

/// How far a group must outgrow its shard count before growing it: one
/// doubling.
const GROW_FACTOR: usize = 2;
/// How far it must fall below before shrinking: two doublings. Wider than
/// [`GROW_FACTOR`] so the two thresholds cannot both be true at once, which is
/// what makes the decision stable.
const SHRINK_FACTOR: usize = 4;

impl super::Memory<'_> {
    /// What this engine currently holds, as the layout rule counts it.
    ///
    /// Facts are counted **live**: a rebuild does not re-insert tombstoned
    /// records, so sizing for them would leave the fresh arenas over-sharded
    /// by exactly the amount just purged.
    pub(crate) fn population(&self) -> Population {
        Population {
            // `saturating_sub` rather than `-`: the tombstone count is carried
            // state, and a wrapped subtraction here would not be caught, it
            // would be *acted on* — a huge count sends the rule straight to
            // `MAX_SHARDS` and the next pass allocates for it.
            facts: self.facts.len().saturating_sub(self.tombstones) as u64,
            entities: self.entities.len() as u64,
            edges: self.edges_out.len() as u64,
            edge_versions: self.edges_hist_out.len() as u64,
            terms: self.bm25.postings().keys() as u64,
            tags: self.tags_idx.keys() as u64,
            documents: self.bm25.doc_len_arena().len() as u64,
        }
    }

    /// The layout this engine's contents call for.
    pub(crate) fn target_layout(&self) -> ShardLayout {
        ShardLayout::for_population(&self.population())
    }

    /// Whether the arenas are laid out for a database this one is no longer.
    ///
    /// **O(1)** — every input is a stored record count, so a host may ask on
    /// every write. That is the point: without a trigger of its own, a growing
    /// database would keep the layout it was created with until somebody ran
    /// `maintain` by hand, and the automatic maintenance that does exist
    /// (`maintain_every_forgets`) is off by default.
    ///
    /// Self-limiting, which is what makes it safe to act on: the thresholds are
    /// a doubling up and a fourfold drop, so this turns true a handful of times
    /// over a database's whole life, not continuously.
    pub fn shard_layout_is_stale(&self) -> bool {
        let stored = ShardLayout::of_config(&self.cfg);
        let target = self.target_layout();
        stored.compacted_groups_earn_rebuild(&target) || stored.edges_earn_rebuild(&target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A population, spelled out so the table below reads as data.
    fn population(
        facts: u64,
        entities: u64,
        edges: u64,
        edge_versions: u64,
        terms: u64,
        tags: u64,
    ) -> Population {
        Population {
            facts,
            entities,
            edges,
            edge_versions,
            terms,
            tags,
            // Every live fact is one BM25 document.
            documents: facts,
        }
    }

    fn layout(p: Population) -> (usize, usize, usize, usize, usize) {
        let l = ShardLayout::for_population(&p);
        (l.facts, l.entities, l.edges, l.temporal, l.postings)
    }

    /// The rule, pinned. Change these numbers only on purpose: they decide how
    /// much memory every database on every host occupies, and a silent shift
    /// re-shards every database in the field on its next maintenance pass.
    #[test]
    fn the_layout_rule_is_a_fixed_table() {
        // Empty: nothing to hold, so every group sits on the floor.
        assert_eq!(layout(population(0, 0, 0, 0, 0, 0)), (4, 4, 4, 4, 4));
        // One fact. Same floor — a shard costs a page, and one page is plenty.
        assert_eq!(layout(population(1, 1, 0, 0, 3, 1)), (4, 4, 4, 4, 4));
        // A personal memory: thousands of facts, still entirely on the floor.
        // This is the case the whole change exists for.
        assert_eq!(
            layout(population(1_000, 50, 20, 25, 1_500, 5)),
            (4, 4, 4, 4, 4)
        );
        assert_eq!(
            layout(population(5_000, 200, 100, 120, 6_000, 12)),
            (4, 4, 4, 4, 4)
        );
        // 100k facts.
        assert_eq!(
            layout(population(100_000, 4_096, 30_000, 60_000, 34_000, 32)),
            (32, 4, 16, 8, 8)
        );
        // The 1M benchmark corpus, the point the target was calibrated on.
        assert_eq!(
            layout(population(910_051, 4_096, 287_972, 549_482, 34_604, 64)),
            (256, 4, 128, 64, 64)
        );
        // Ten million: the rule keeps pages-per-shard flat rather than letting
        // either the directory or the page floor run away.
        assert_eq!(
            layout(population(
                10_000_000, 40_000, 3_000_000, 6_000_000, 200_000, 128
            )),
            (2048, 4, 2048, 512, 1024)
        );
    }

    /// Products in this rule pass `u32::MAX` at populations a database can
    /// really reach, so they are computed in `u64` and the clamp happens before
    /// the one cast. On wasm32 a `usize` product would have wrapped here, and a
    /// wrapped layout means a journal replays differently there than on the
    /// host that wrote it.
    #[test]
    fn the_rule_does_not_depend_on_pointer_width() {
        // 100M facts x 48 B = 4.8 GB, past u32::MAX.
        let big = population(100_000_000, 1_000, 0, 0, 1_000_000, 256);
        assert!(u64::from(u32::MAX) < 100_000_000u64 * 48);
        assert_eq!(layout(big).0, 32768);
        // Saturating all the way up still lands on the ceiling, not on garbage.
        let absurd = population(u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(
            layout(absurd),
            (MAX_SHARDS, MAX_SHARDS, MAX_SHARDS, MAX_SHARDS, MAX_SHARDS)
        );
    }

    /// Every value the rule can produce is a legal shard count.
    #[test]
    fn every_produced_count_is_a_usable_shard_count() {
        let mut facts = 0u64;
        while facts < 40_000_000 {
            let l = ShardLayout::for_population(&population(facts, facts / 8, 0, facts, facts, 16));
            for n in [l.facts, l.entities, l.edges, l.temporal, l.postings] {
                assert!(n.is_power_of_two(), "{n} is not a power of two");
                assert!((MIN_SHARDS..=MAX_SHARDS).contains(&n), "{n} out of range");
            }
            facts = (facts + 1) * 3 / 2;
        }
    }

    fn at(n: usize) -> ShardLayout {
        ShardLayout {
            facts: n,
            entities: n,
            edges: n,
            temporal: n,
            postings: n,
        }
    }

    #[test]
    fn growth_is_eager_and_shrinking_is_lazy() {
        // Unchanged: nothing to do.
        assert!(!at(64).compacted_groups_earn_rebuild(&at(64)));
        // One doubling up is acted on; anything short of it is not.
        assert!(at(64).compacted_groups_earn_rebuild(&at(128)));
        // Shrinking waits for four times over, so the two thresholds leave a
        // band in which neither fires and a database near a boundary rests.
        assert!(!at(64).compacted_groups_earn_rebuild(&at(32)));
        assert!(at(64).compacted_groups_earn_rebuild(&at(16)));
        // And the band means a rebuild cannot bounce back: having acted, the
        // new count is itself stable against the population that caused it.
        assert!(!at(16).compacted_groups_earn_rebuild(&at(16)));
        assert!(!at(128).compacted_groups_earn_rebuild(&at(128)));
        // One group out of step is enough.
        let mut skewed = at(64);
        skewed.postings = 8;
        assert!(at(64).compacted_groups_earn_rebuild(&skewed));
    }

    /// The edges are asked about separately because only a repack rebuilds
    /// them. Were they folded into the compaction question, a database whose
    /// edges alone were out of layout would rebuild everything else on every
    /// pass, never fix the edges, and be asked again forever.
    #[test]
    fn the_edge_arenas_are_judged_on_their_own() {
        let mut edges_only = at(64);
        edges_only.edges = 512;
        assert!(!at(64).compacted_groups_earn_rebuild(&edges_only));
        assert!(at(64).edges_earn_rebuild(&edges_only));
    }

    /// A pass may only claim the layout it actually built.
    #[test]
    fn a_pass_claims_only_what_it_laid_out() {
        let stored = at(64);
        let target = at(512);
        // Compaction alone: four groups move, the untouched edges do not.
        let without = stored.realized(&target, false);
        assert_eq!(
            (
                without.facts,
                without.entities,
                without.temporal,
                without.postings
            ),
            (512, 512, 512, 512)
        );
        assert_eq!(without.edges, stored.edges);
        // With a repack, the edges move too.
        assert_eq!(stored.realized(&target, true), target);
    }

    /// The floor and the ceiling are both reachable, and neither is crossed.
    #[test]
    fn the_clamp_holds_at_both_ends() {
        assert_eq!(shards_for(0), MIN_SHARDS);
        assert_eq!(shards_for(1), MIN_SHARDS);
        assert_eq!(
            shards_for(SHARD_TARGET_BYTES as u64 * MIN_SHARDS as u64),
            MIN_SHARDS
        );
        assert_eq!(shards_for(u64::MAX), MAX_SHARDS);
        // Just past the floor the rule starts following the data.
        assert_eq!(
            shards_for(SHARD_TARGET_BYTES as u64 * MIN_SHARDS as u64 + 1),
            MIN_SHARDS * 2
        );
    }
}
