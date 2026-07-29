//! Deterministic corpus and workload generator for plugmem tests and
//! benches. Internal tooling, never published.
//!
//! Everything is a pure function of `(seed, Profile)`: the same pair
//! yields the same operation stream forever, on every machine — the
//! repo-wide law "no randomness without a seed". The generated shape
//! follows the corpus passport of:
//!
//! - a Zipf(≈1.07) vocabulary of pronounceable syllable words;
//! - text lengths normally distributed around a configured mean;
//! - a lazily growing entity pool (~`entity_share` per remember) with
//!   Zipf popularity — hub entities exist by construction;
//! - 1–4 tags per fact from a Zipf tag pool;
//! - links on some remembers plus standalone link operations;
//! - revisions and forgets against the generator's own book-keeping, so
//!   every emitted operation is *valid* by construction (a revise always
//!   targets an open fact, a forget a live one);
//! - unit vectors drawn from Gaussian clusters on the sphere (checks
//!   both semantic recall and quantization);
//! - a time axis whose steps shrink as the run grows — density rises
//!   toward the newest operations, like real memory does.
//!
//! The stream is consumed either directly (benches map [`GenOp`] onto
//! engine calls) or through the [`apply`] helper, which keeps the
//! mapping to `plugmem-core` verbs in one place.

mod rng;
mod words;

pub use rng::Rng;
pub use words::{Vocabulary, word_for};

use plugmem_core::{
    Error, FactId, LinkInput, MaintainReport, Memory, RememberInput, RememberOutcome, Storage,
};

/// Index-space offset of the tag vocabulary (disjoint from the text
/// dictionary, which starts at 0).
const TAG_SALT: usize = 1 << 24;
/// Index-space offset of the entity-name vocabulary.
const ENTITY_SALT: usize = 1 << 25;
/// Relation names used by generated links.
const RELS: &[&str] = &[
    "works_on",
    "depends_on",
    "knows",
    "owns",
    "likes",
    "uses",
    "part_of",
    "located_in",
];
/// Standard deviation of the per-dimension cluster noise (the cluster
/// centers are unit vectors; this keeps members recognizably close).
const CLUSTER_NOISE: f64 = 0.35;
/// Zipf exponent of the corpus passport.
const ZIPF_S: f64 = 1.07;

/// Shape parameters of a generated workload. Construct with
/// [`Profile::default`] and override fields.
#[derive(Clone, Debug)]
pub struct Profile {
    /// Text dictionary size (: 30k).
    pub dict_words: usize,
    /// Tag pool size (: 500).
    pub tag_pool: usize,
    /// Vector dimension; `0` generates no vectors. Must match the
    /// consuming engine's `Config::dim`.
    pub dim: usize,
    /// Number of Gaussian clusters on the unit sphere (: 64).
    pub vector_clusters: usize,
    /// Probability that a remember mints a *new* entity instead of
    /// reusing one (the pool grows to roughly this share of remembers).
    pub entity_share: f64,
    /// Mean words per fact text (normally distributed around it).
    pub mean_words: usize,
    /// Operation-mix weight of `remember`.
    pub w_remember: u32,
    /// Operation-mix weight of `revise`.
    pub w_revise: u32,
    /// Operation-mix weight of `forget`.
    pub w_forget: u32,
    /// Operation-mix weight of a standalone `link`.
    pub w_link: u32,
    /// Operation-mix weight of `maintain` (default 0 — property tests
    /// and churn benches opt in).
    pub w_maintain: u32,
    /// Initial time-axis step in milliseconds; later steps shrink, so
    /// operation density rises toward the end of the run.
    pub start_step_ms: u64,
}

impl Default for Profile {
    /// The corpus passport with vectors off and no
    /// maintains.
    fn default() -> Self {
        Self {
            dict_words: 30_000,
            tag_pool: 500,
            dim: 0,
            vector_clusters: 64,
            entity_share: 0.05,
            mean_words: 24,
            w_remember: 85,
            w_revise: 6,
            w_forget: 5,
            w_link: 4,
            w_maintain: 0,
            start_step_ms: 6 * 60 * 60 * 1000, // six hours
        }
    }
}

/// One generated operation, mirroring the engine verbs with owned data.
/// Fact ids are the engine's own deterministic sequential ids — the
/// generator allocates them exactly like the engine will.
#[derive(Clone, Debug, PartialEq)]
pub enum GenOp {
    /// A new fact.
    Remember {
        /// Host timestamp.
        now: u64,
        /// Fact text.
        text: String,
        /// Subject entity name, if any.
        entity: Option<String>,
        /// Tags.
        tags: Vec<String>,
        /// `(rel, target_entity)` links (requires `entity`).
        links: Vec<(String, String)>,
        /// Optional unit embedding of `Profile::dim` components.
        vector: Option<Vec<f32>>,
    },
    /// A revision of an open fact; carries the replacement fact's
    /// content (same shape as a remember).
    Revise {
        /// Host timestamp.
        now: u64,
        /// The open fact being revised.
        target: u32,
        /// Replacement text.
        text: String,
        /// Replacement subject entity.
        entity: Option<String>,
        /// Replacement tags.
        tags: Vec<String>,
        /// Optional unit embedding.
        vector: Option<Vec<f32>>,
    },
    /// A tombstone of a live fact.
    Forget {
        /// Host timestamp.
        now: u64,
        /// The fact being forgotten.
        fact: u32,
    },
    /// A standalone typed edge.
    Link {
        /// Host timestamp.
        now: u64,
        /// Source entity name.
        src: String,
        /// Relation.
        rel: String,
        /// Destination entity name.
        dst: String,
        /// Optional provenance fact (live at emission time).
        provenance: Option<u32>,
    },
    /// A maintenance pass.
    Maintain {
        /// Host timestamp.
        now: u64,
    },
}

/// The workload generator. See the crate docs for the guarantees.
#[derive(Clone, Debug)]
pub struct Gen {
    rng: Rng,
    profile: Profile,
    dict: Vocabulary,
    tags: Vocabulary,
    /// Entity-name vocabulary; `entities` counts how many of its ranks
    /// have been minted so far (Zipf reuse draws from that prefix).
    entity_names: Vocabulary,
    entities: usize,
    /// Cluster centers (unit vectors), present when `dim > 0`.
    centers: Vec<Vec<f32>>,
    /// The engine-mirroring fact allocator.
    next_fact: u32,
    /// Facts that are open (revisable): not closed, not forgotten.
    open: Vec<u32>,
    /// Facts that are live (forgettable): not forgotten.
    live: Vec<u32>,
    /// The advancing time cursor (milliseconds).
    now: u64,
    /// Operations emitted so far (drives the shrinking time step).
    emitted: u64,
}

impl Gen {
    /// Creates a generator.
    ///
    /// # Panics
    ///
    /// Panics on a malformed profile (zero vocabulary sizes, a zero
    /// operation mix, `dim > 4096`, or vectors without clusters) — this
    /// is internal tooling, misconfiguration is a bug in the caller.
    pub fn new(seed: u64, profile: Profile) -> Self {
        assert!(profile.dict_words > 0, "dict_words must be positive");
        assert!(profile.tag_pool > 0, "tag_pool must be positive");
        assert!(profile.mean_words > 0, "mean_words must be positive");
        assert!(profile.dim <= 4096, "dim must be <= 4096 (engine limit)");
        assert!(
            profile.dim == 0 || profile.vector_clusters > 0,
            "vectors need at least one cluster"
        );
        assert!(
            profile.w_remember
                + profile.w_revise
                + profile.w_forget
                + profile.w_link
                + profile.w_maintain
                > 0,
            "the operation mix must have a positive total weight"
        );
        let mut rng = Rng::new(seed);
        let centers = (0..if profile.dim == 0 {
            0
        } else {
            profile.vector_clusters
        })
            .map(|_| {
                let mut c: Vec<f32> = (0..profile.dim).map(|_| rng.normal() as f32).collect();
                normalize(&mut c);
                c
            })
            .collect();
        Self {
            dict: Vocabulary::new(0, profile.dict_words, ZIPF_S),
            tags: Vocabulary::new(TAG_SALT, profile.tag_pool, ZIPF_S),
            // The entity vocabulary is sized generously; only the minted
            // prefix is ever drawn from.
            entity_names: Vocabulary::new(ENTITY_SALT, 4096, ZIPF_S),
            entities: 0,
            centers,
            next_fact: 0,
            open: Vec::new(),
            live: Vec::new(),
            now: 0,
            emitted: 0,
            rng,
            profile,
        }
    }

    /// Generates the next `n` operations of the stream.
    pub fn ops(&mut self, n: usize) -> Vec<GenOp> {
        (0..n).map(|_| self.next_op()).collect()
    }

    fn next_op(&mut self) -> GenOp {
        self.advance_clock();
        let p = &self.profile;
        let total = p.w_remember + p.w_revise + p.w_forget + p.w_link + p.w_maintain;
        let mut roll = self.rng.below(total as usize) as u32;
        for (weight, kind) in [
            (p.w_remember, 0u8),
            (p.w_revise, 1),
            (p.w_forget, 2),
            (p.w_link, 3),
            (p.w_maintain, 4),
        ] {
            if roll < weight {
                return match kind {
                    1 if !self.open.is_empty() => self.gen_revise(),
                    2 if !self.live.is_empty() => self.gen_forget(),
                    3 => self.gen_link(),
                    4 => GenOp::Maintain { now: self.now },
                    // Remember proper, and the fallback when a revise or
                    // forget has no valid target yet.
                    _ => self.gen_remember(),
                };
            }
            roll -= weight;
        }
        unreachable!("roll is below the total weight");
    }

    /// Advances the time cursor by a jittered, shrinking step: early
    /// operations are days apart, late ones minutes — density rises
    /// toward "today".
    fn advance_clock(&mut self) {
        self.emitted += 1;
        let base = self.profile.start_step_ms;
        let step = (base / (1 + self.emitted / 64)).max(1);
        self.now += 1 + self.rng.next_u64() % step;
    }

    fn gen_text(&mut self) -> String {
        let mean = self.profile.mean_words as f64;
        let n = (mean + self.rng.normal() * mean * 0.4).round().max(3.0) as usize;
        let mut out = String::new();
        for i in 0..n {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(self.dict.sample(&mut self.rng));
        }
        out
    }

    fn gen_tags(&mut self) -> Vec<String> {
        let n = 1 + self.rng.below(4);
        let mut tags: Vec<String> = Vec::with_capacity(n);
        for _ in 0..n {
            let tag = self.tags.sample(&mut self.rng);
            if !tags.iter().any(|t| t == tag) {
                tags.push(tag.to_string());
            }
        }
        tags
    }

    /// Mints a new entity or reuses one Zipf-style (hubs emerge from the
    /// low ranks).
    fn gen_entity(&mut self) -> String {
        if self.entities == 0 || self.rng.chance(self.profile.entity_share) {
            let rank = self.entities.min(self.entity_names.len() - 1);
            self.entities = (self.entities + 1).min(self.entity_names.len());
            return self.entity_names.word(rank).to_string();
        }
        let rank = self
            .entity_names
            .sample_rank(&mut self.rng)
            .min(self.entities - 1);
        self.entity_names.word(rank).to_string()
    }

    fn gen_vector(&mut self) -> Option<Vec<f32>> {
        if self.profile.dim == 0 {
            return None;
        }
        let center = &self.centers[self.rng.below(self.centers.len())];
        let mut v: Vec<f32> = center
            .iter()
            .map(|&c| c + (self.rng.normal() * CLUSTER_NOISE) as f32)
            .collect();
        normalize(&mut v);
        Some(v)
    }

    fn alloc_fact(&mut self) -> u32 {
        let id = self.next_fact;
        self.next_fact += 1;
        self.open.push(id);
        self.live.push(id);
        id
    }

    fn gen_remember(&mut self) -> GenOp {
        let entity = self.rng.chance(0.8).then(|| self.gen_entity());
        let links = match &entity {
            Some(_) => {
                let n = self.rng.below(4).saturating_sub(1); // 0,0,1,2
                (0..n)
                    .map(|_| {
                        (
                            RELS[self.rng.below(RELS.len())].to_string(),
                            self.gen_entity(),
                        )
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        let op = GenOp::Remember {
            now: self.now,
            text: self.gen_text(),
            entity,
            tags: self.gen_tags(),
            links,
            vector: self.gen_vector(),
        };
        self.alloc_fact();
        op
    }

    fn gen_revise(&mut self) -> GenOp {
        let at = self.rng.below(self.open.len());
        let target = self.open.swap_remove(at); // closed: no longer open
        let op = GenOp::Revise {
            now: self.now,
            target,
            text: self.gen_text(),
            entity: self.rng.chance(0.8).then(|| self.gen_entity()),
            tags: self.gen_tags(),
            vector: self.gen_vector(),
        };
        self.alloc_fact();
        op
    }

    fn gen_forget(&mut self) -> GenOp {
        let at = self.rng.below(self.live.len());
        let fact = self.live.swap_remove(at);
        self.open.retain(|&f| f != fact);
        GenOp::Forget {
            now: self.now,
            fact,
        }
    }

    fn gen_link(&mut self) -> GenOp {
        let provenance = (!self.live.is_empty() && self.rng.chance(0.5))
            .then(|| self.live[self.rng.below(self.live.len())]);
        GenOp::Link {
            now: self.now,
            src: self.gen_entity(),
            rel: RELS[self.rng.below(RELS.len())].to_string(),
            dst: self.gen_entity(),
            provenance,
        }
    }
}

/// L2-normalizes `v` in place (regenerating is unnecessary: a Gaussian
/// draw is zero with probability zero, and cluster members sit near a
/// unit center anyway).
fn normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt() as f32;
    for x in v {
        *x /= norm;
    }
}

/// Applies one generated operation to an engine — the single place that
/// maps [`GenOp`] onto `plugmem-core` verbs. Returns the remember/revise
/// outcome when there is one.
///
/// The generator only emits valid operations (see the crate docs), so on
/// an engine driven exclusively by this stream — created with the same
/// `Config::dim` as the profile — every call is expected to succeed.
pub fn apply<S: Storage>(
    mem: &mut Memory,
    store: &mut S,
    op: &GenOp,
) -> Result<Option<RememberOutcome>, Error> {
    match op {
        GenOp::Remember {
            now,
            text,
            entity,
            tags,
            links,
            vector,
        } => {
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let link_refs: Vec<(&str, &str)> = links
                .iter()
                .map(|(r, d)| (r.as_str(), d.as_str()))
                .collect();
            mem.remember(
                store,
                RememberInput {
                    now: *now,
                    text,
                    entity: entity.as_deref(),
                    tags: &tag_refs,
                    links: &link_refs,
                    vector: vector.as_deref(),
                    valid_from: None,
                    metadata: None,
                },
            )
            .map(Some)
        }
        GenOp::Revise {
            now,
            target,
            text,
            entity,
            tags,
            vector,
        } => {
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            mem.revise(
                store,
                FactId(*target),
                RememberInput {
                    now: *now,
                    text,
                    entity: entity.as_deref(),
                    tags: &tag_refs,
                    links: &[],
                    vector: vector.as_deref(),
                    valid_from: None,
                    metadata: None,
                },
            )
            .map(Some)
        }
        GenOp::Forget { now, fact } => mem.forget(store, *now, FactId(*fact)).map(|_| None),
        GenOp::Link {
            now,
            src,
            rel,
            dst,
            provenance,
        } => mem
            .link(
                store,
                LinkInput {
                    now: *now,
                    src,
                    rel,
                    dst,
                    provenance: provenance.map(FactId),
                },
            )
            .map(|()| None),
        GenOp::Maintain { now } => mem.maintain(store, *now).map(|MaintainReport { .. }| None),
    }
}
