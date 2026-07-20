//! HNSW graph over the quantized vector pool (specs/10, specs/04 §5
//! phase 2).
//!
//! The port follows the specs/10 blueprint: the rust-cv skeleton (flat
//! layers, an external reusable scratch) with the hnswlib algorithmics
//! (the Algorithm-4 neighbor heuristic with `keep_pruned`, the classic
//! early-stopped beam search). Four systematic replacements make it fit
//! this engine:
//!
//! - **levels are a pure function of the fact id** — `level_of` counts
//!   precomputed integer thresholds below `xxh3(fact_id)`, so there is
//!   no PRNG state and journal replay reproduces the graph bit for bit;
//! - **no hash sets** — the visited set is an epoch-stamped `u32` array
//!   in the scratch (reset is one increment);
//! - **distances are quantized cosines** read straight from the
//!   [`VecPool`] slots (higher = closer; every comparison breaks ties by
//!   slot id, so the whole build is deterministic);
//! - **storage is flat**: level 0 is one `Vec<u32>` of `m0`-wide
//!   neighbor blocks indexed by vector-slot id; upper levels live as
//!   [`ChunkPool`] lists behind an [`Arena`] of `[slot | level]` keyed
//!   handles. Both dump/load with the same canonical-image contract as
//!   every other structure.
//!
//! Graph nodes are **vector-slot indices** (`u32`). Only slots below
//! `indexed` are in the graph; slots appended after the last build form
//! the *flat tail* the engine scans separately and merges (specs/04:
//! inserts amortize into `maintain`, `remember` stays microseconds).
//!
//! The beam search does not consult admission filters while walking —
//! connectivity must not depend on the query — the engine filters the
//! returned candidates.

use alloc::vec::Vec;

#[cfg(feature = "counters")]
use core::cell::Cell;

use plugmem_arena::{Arena, ArenaCfg, ChunkPool, ChunkPoolCfg, ListHandle, ShardMode, Slot, key};
use xxhash_rust::xxh3::xxh3_64;

use crate::error::Error;
use crate::id::NONE_U32;
use crate::index::vecpool::{VecPool, dot_i8};

/// Deepest level tracked. With `m = 16` the probability of level 8 is
/// `16^-8 ≈ 2^-32` — unreachable at the capacity passport; 16 is sheer
/// paranoia at 4 bytes a threshold.
const MAX_LEVEL: usize = 16;

/// Shard count of the (small) upper-level handle arena.
const UPPER_SHARDS: usize = 64;

/// Upper-level adjacency handle: key `[slot BE | level BE]`, payload the
/// neighbor list's chunk handle. 20-byte slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UpperSlot {
    /// Graph node (vector-slot id).
    slot: u32,
    /// Level (1-based; level 0 lives in the flat blocks).
    level: u32,
    /// The node's neighbor list at that level.
    handle: ListHandle,
}

impl Slot for UpperSlot {
    const SIZE: usize = 20;
    const KEY_LEN: usize = 8;

    fn write(&self, out: &mut [u8]) {
        key::write_u32(out, self.slot);
        key::write_u32(&mut out[4..], self.level);
        out[8..20].copy_from_slice(&self.handle.to_bytes());
    }

    fn read(bytes: &[u8]) -> Self {
        Self {
            slot: key::read_u32(bytes),
            level: key::read_u32(&bytes[4..]),
            handle: ListHandle::from_bytes(bytes[8..20].try_into().unwrap()),
        }
    }
}

/// Reusable search/build scratch (zero allocations after warm-up).
#[derive(Debug, Default)]
pub struct HnswScratch {
    /// Epoch-stamped visited marks per slot.
    visited: Vec<u32>,
    /// Current epoch; bumping it clears `visited` in O(1).
    epoch: u32,
    /// Candidate beam, ascending by `(sim, slot)` order — the maximum is
    /// popped from the tail.
    cand: Vec<(f32, u32)>,
    /// The `ef` best found so far, ascending — the worst sits at 0.
    found: Vec<(f32, u32)>,
    /// Neighbor staging buffer.
    nbrs: Vec<u32>,
    /// Heuristic selection output.
    sel: Vec<u32>,
    /// Heuristic rejects (recycled by `keep_pruned`).
    pruned: Vec<u32>,
    /// Per-candidate sims staged for a relink prune.
    relink: Vec<(f32, u32)>,
}

/// The deterministic candidate order: higher cosine wins, ties go to the
/// smaller slot id.
#[inline]
fn better(a: (f32, u32), b: (f32, u32)) -> core::cmp::Ordering {
    a.0.total_cmp(&b.0).then(b.1.cmp(&a.1))
}

/// The navigable small-world graph. See the module docs.
///
/// The upper-level pools carry a lifetime so a read-only open can borrow
/// them from an mmap (specs/16); `level0` stays owned (it is rebuilt into
/// a `Vec<u32>` on load — small metadata, not content-scaled). The owned
/// build/load paths are `'static`.
pub struct HnswGraph<'a> {
    /// Upper-level degree cap.
    m: usize,
    /// Level-0 degree cap.
    m0: usize,
    /// `indexed × m0` neighbor slots, `NONE_U32`-padded.
    level0: Vec<u32>,
    /// Upper-level handles.
    upper: Arena<'a, UpperSlot>,
    /// Upper-level neighbor lists (4-byte LE slot ids).
    lists: ChunkPool<'a>,
    /// Entry point (highest-level node), or `NONE_U32` when empty.
    entry: u32,
    /// Slots `[0, indexed)` are in the graph; the rest are the flat tail.
    indexed: u32,
    /// `thresholds[k]` = `2^64 / m^(k+1)`: a fact whose hash falls below
    /// it reaches level `k + 1`.
    thresholds: [u64; MAX_LEVEL],
    /// Exact distance evaluations (feature `counters`) — the
    /// deterministic cost metric of graph search.
    #[cfg(feature = "counters")]
    dist_evals: Cell<u64>,
}

impl<'a> HnswGraph<'a> {
    /// An empty graph for the given degree caps (`m >= 2`, `m0 >= m` —
    /// enforced by `Config::validate`).
    pub fn new(m: usize, m0: usize, max_bytes: usize) -> Result<Self, Error> {
        let mut thresholds = [0u64; MAX_LEVEL];
        let mut t = u64::MAX;
        for slot in &mut thresholds {
            t /= m as u64;
            *slot = t;
        }
        Ok(Self {
            m,
            m0,
            level0: Vec::new(),
            upper: Arena::new(
                ArenaCfg::new(UPPER_SHARDS, ShardMode::Uniform).with_max_bytes(max_bytes),
            )?,
            lists: ChunkPool::new(ChunkPoolCfg::new().with_max_bytes(max_bytes)),
            entry: NONE_U32,
            indexed: 0,
            thresholds,
            #[cfg(feature = "counters")]
            dist_evals: Cell::new(0),
        })
    }

    /// Number of slots covered by the graph (the flat tail starts here).
    pub fn indexed(&self) -> u32 {
        self.indexed
    }

    /// The level of a fact: a pure function of its id, so replay and
    /// rebuilds reproduce the same graph geometry.
    fn level_of(&self, fact: u32) -> usize {
        let h = xxh3_64(&fact.to_le_bytes());
        self.thresholds.iter().take_while(|&&t| h < t).count()
    }

    /// Quantized cosine between an external query and a slot.
    #[inline]
    fn sim_q(&self, pool: &VecPool<'_>, q: (f32, &[u8]), slot: u32) -> f32 {
        #[cfg(feature = "counters")]
        self.dist_evals.set(self.dist_evals.get() + 1);
        let (s, qb) = pool.quant(slot as usize);
        q.0 * s * dot_i8(q.1, qb) as f32
    }

    /// The level-0 neighbor block of a node.
    #[inline]
    fn block(&self, slot: u32) -> &[u32] {
        let at = slot as usize * self.m0;
        &self.level0[at..at + self.m0]
    }

    /// Copies the neighbors of `slot` at `level` into `out`.
    fn neighbors_into(&self, slot: u32, level: usize, out: &mut Vec<u32>) {
        out.clear();
        if level == 0 {
            out.extend(
                self.block(slot)
                    .iter()
                    .copied()
                    .take_while(|&n| n != NONE_U32),
            );
            return;
        }
        let mut kb = [0u8; 8];
        key::write_u32(&mut kb, slot);
        key::write_u32(&mut kb[4..], level as u32);
        let Some(entry) = self.upper.get(&kb) else {
            return;
        };
        for chunk in self.lists.iter(&entry.handle) {
            for raw in chunk.chunks_exact(4) {
                out.push(u32::from_le_bytes(raw.try_into().unwrap()));
            }
        }
    }

    /// Marks `slot` visited; `true` if it already was.
    #[inline]
    fn visit(scratch: &mut HnswScratch, slot: u32) -> bool {
        let at = slot as usize;
        if scratch.visited[at] == scratch.epoch {
            return true;
        }
        scratch.visited[at] = scratch.epoch;
        false
    }

    /// Classic beam search inside one level (specs/10 Alg. 2 with the
    /// early stop). Results land in `scratch.found`, ascending by the
    /// deterministic order — best last.
    fn search_layer(
        &self,
        pool: &VecPool<'_>,
        q: (f32, &[u8]),
        level: usize,
        ep: u32,
        ef: usize,
        scratch: &mut HnswScratch,
    ) {
        scratch.epoch = scratch.epoch.wrapping_add(1);
        if scratch.epoch == 0 {
            // The epoch wrapped: stamp everything stale explicitly.
            scratch.visited.fill(u32::MAX);
            scratch.epoch = 1;
        }
        scratch
            .visited
            .resize(self.indexed as usize, scratch.epoch.wrapping_sub(1));
        scratch.cand.clear();
        scratch.found.clear();

        Self::visit(scratch, ep);
        let s = self.sim_q(pool, q, ep);
        scratch.cand.push((s, ep));
        scratch.found.push((s, ep));

        while let Some(best) = scratch.cand.pop() {
            // Early stop: the best remaining candidate cannot improve the
            // worst of the ef found.
            if scratch.found.len() >= ef && better(best, scratch.found[0]).is_lt() {
                break;
            }
            let nbrs = core::mem::take(&mut scratch.nbrs);
            let mut nbrs = nbrs;
            self.neighbors_into(best.1, level, &mut nbrs);
            for &nb in &nbrs {
                if Self::visit(scratch, nb) {
                    continue;
                }
                let s = self.sim_q(pool, q, nb);
                let entry = (s, nb);
                if scratch.found.len() < ef || better(entry, scratch.found[0]).is_gt() {
                    let at = scratch.found.partition_point(|&e| better(e, entry).is_lt());
                    scratch.found.insert(at, entry);
                    if scratch.found.len() > ef {
                        scratch.found.remove(0);
                    }
                    let at = scratch.cand.partition_point(|&e| better(e, entry).is_lt());
                    scratch.cand.insert(at, entry);
                }
            }
            scratch.nbrs = nbrs;
        }
    }

    /// The Algorithm-4 neighbor heuristic with `keep_pruned` (specs/10):
    /// walks `scratch.found` best-first, keeps a candidate only if it is
    /// closer to the query than to any already-kept neighbor (spread over
    /// clusters), then tops up from the rejects. Result in `scratch.sel`.
    fn select_neighbors(&self, pool: &VecPool<'_>, cap: usize, scratch: &mut HnswScratch) {
        scratch.sel.clear();
        scratch.pruned.clear();
        for i in (0..scratch.found.len()).rev() {
            let (sim, cand) = scratch.found[i];
            if scratch.sel.len() >= cap {
                break;
            }
            let dominated = scratch.sel.iter().any(|&kept| {
                #[cfg(feature = "counters")]
                self.dist_evals.set(self.dist_evals.get() + 1);
                pool.sim(cand, kept) > sim
            });
            if dominated {
                scratch.pruned.push(cand);
            } else {
                scratch.sel.push(cand);
            }
        }
        for &p in scratch.pruned.iter() {
            if scratch.sel.len() >= cap {
                break;
            }
            scratch.sel.push(p);
        }
    }

    /// Overwrites the neighbor list of `slot` at `level` with
    /// `scratch.sel`.
    fn write_list(&mut self, slot: u32, level: usize, sel: &[u32]) -> Result<(), Error> {
        if level == 0 {
            let at = slot as usize * self.m0;
            let block = &mut self.level0[at..at + self.m0];
            block.fill(NONE_U32);
            block[..sel.len()].copy_from_slice(sel);
            return Ok(());
        }
        let mut kb = [0u8; 8];
        key::write_u32(&mut kb, slot);
        key::write_u32(&mut kb[4..], level as u32);
        let mut handle = match self.upper.get(&kb) {
            Some(entry) => {
                let mut h = entry.handle;
                self.lists.free(&mut h);
                h
            }
            None => ListHandle::EMPTY,
        };
        for &n in sel {
            self.lists.push(&mut handle, &n.to_le_bytes())?;
        }
        let updated = UpperSlot {
            slot,
            level: level as u32,
            handle,
        };
        if self.upper.contains(&kb) {
            let payload = self.upper.payload_mut(&kb).expect("checked above");
            let mut full = [0u8; UpperSlot::SIZE];
            updated.write(&mut full);
            payload.copy_from_slice(&full[UpperSlot::KEY_LEN..]);
        } else {
            self.upper.insert(&updated)?;
        }
        Ok(())
    }

    /// Adds `new` to `v`'s list at `level`, pruning with the heuristic
    /// when the degree cap overflows (the hnswlib backlink rule).
    fn add_link(
        &mut self,
        pool: &VecPool<'_>,
        v: u32,
        new: u32,
        level: usize,
        scratch: &mut HnswScratch,
    ) -> Result<(), Error> {
        let cap = if level == 0 { self.m0 } else { self.m };
        let mut nbrs = core::mem::take(&mut scratch.nbrs);
        self.neighbors_into(v, level, &mut nbrs);
        if nbrs.len() < cap {
            nbrs.push(new);
            let sel = core::mem::take(&mut scratch.sel);
            let mut sel = sel;
            sel.clear();
            sel.extend_from_slice(&nbrs);
            let res = self.write_list(v, level, &sel);
            scratch.sel = sel;
            scratch.nbrs = nbrs;
            return res;
        }
        // Overflow: re-select the cap best around `v` from old + new.
        let vq = pool.quant(v as usize);
        scratch.relink.clear();
        for &n in nbrs.iter().chain(core::iter::once(&new)) {
            let s = self.sim_q(pool, (vq.0, vq.1), n);
            scratch.relink.push((s, n));
        }
        scratch.nbrs = nbrs;
        scratch.relink.sort_unstable_by(|a, b| better(*a, *b));
        scratch.found.clear();
        scratch.found.extend_from_slice(&scratch.relink);
        self.select_neighbors(pool, cap, scratch);
        let sel = core::mem::take(&mut scratch.sel);
        let res = self.write_list(v, level, &sel);
        scratch.sel = sel;
        res
    }

    /// Inserts every slot in `[indexed, upto)` into the graph (specs/10
    /// Alg. 1). Called from `maintain` — bulk, deterministic, never from
    /// `remember`.
    pub fn insert_bulk(
        &mut self,
        pool: &VecPool<'_>,
        upto: u32,
        ef_construction: usize,
        scratch: &mut HnswScratch,
    ) -> Result<(), Error> {
        debug_assert!(upto as usize <= pool.len());
        self.level0.resize(upto as usize * self.m0, NONE_U32);
        for slot in self.indexed..upto {
            self.insert_one(pool, slot, ef_construction, scratch)?;
            // Advance per slot: a capacity failure leaves a graph that is
            // still consistent for the slots it admits.
            self.indexed = slot + 1;
        }
        Ok(())
    }

    fn insert_one(
        &mut self,
        pool: &VecPool<'_>,
        slot: u32,
        ef_construction: usize,
        scratch: &mut HnswScratch,
    ) -> Result<(), Error> {
        let level = self.level_of(pool.slot_fact(slot as usize));
        if self.entry == NONE_U32 {
            self.entry = slot;
            return Ok(());
        }
        let q = pool.quant(slot as usize);
        let q = (q.0, q.1);
        let top = self.level_of(pool.slot_fact(self.entry as usize));
        let mut ep = self.entry;
        // Greedy descent through the levels above the new node's.
        let mut lev = top;
        while lev > level {
            self.search_layer(pool, q, lev, ep, 1, scratch);
            ep = scratch.found.last().expect("entry is always found").1;
            lev -= 1;
        }
        // Beam-search and link downward from min(level, top) to 0.
        let mut lev = level.min(top);
        loop {
            self.search_layer(pool, q, lev, ep, ef_construction, scratch);
            ep = scratch.found.last().expect("entry is always found").1;
            let cap = if lev == 0 { self.m0 } else { self.m };
            self.select_neighbors(pool, cap, scratch);
            let sel = core::mem::take(&mut scratch.sel);
            self.write_list(slot, lev, &sel)?;
            for &nb in &sel {
                self.add_link(pool, nb, slot, lev, scratch)?;
            }
            scratch.sel = sel;
            if lev == 0 {
                break;
            }
            lev -= 1;
        }
        if level > top {
            self.entry = slot;
        }
        Ok(())
    }

    /// k-NN query over a raw embedding: quantizes it through the pool
    /// (into `vec_scratch`) and runs the level-descending beam search.
    /// Results land in `out` as `(slot, cosine)`, best first — the
    /// caller filters and merges with the flat tail.
    ///
    /// # Errors
    ///
    /// The pool's quantization errors ([`crate::Error::DimMismatch`],
    /// [`crate::Error::Invalid`] for non-finite or zero vectors).
    pub fn search(
        &self,
        pool: &VecPool<'_>,
        query: &[f32],
        ef: usize,
        vec_scratch: &mut crate::index::vecpool::VecScratch,
        scratch: &mut HnswScratch,
        out: &mut Vec<(u32, f32)>,
    ) -> Result<(), Error> {
        pool.quantize_query(query, vec_scratch)?;
        let q = pool.quantized(vec_scratch);
        self.search_quantized(pool, q, ef, scratch, out);
        Ok(())
    }

    /// k-NN query (specs/10 Alg. 5): greedy descent to level 1, one beam
    /// search on level 0 with `ef`, results appended to `out` as
    /// `(slot, cosine)`, best first.
    pub(crate) fn search_quantized(
        &self,
        pool: &VecPool<'_>,
        q: (f32, &[u8]),
        ef: usize,
        scratch: &mut HnswScratch,
        out: &mut Vec<(u32, f32)>,
    ) {
        out.clear();
        if self.entry == NONE_U32 {
            return;
        }
        let mut ep = self.entry;
        let top = self.level_of(pool.slot_fact(self.entry as usize));
        for lev in (1..=top).rev() {
            self.search_layer(pool, q, lev, ep, 1, scratch);
            ep = scratch.found.last().expect("entry is always found").1;
        }
        self.search_layer(pool, q, 0, ep, ef.max(1), scratch);
        for &(sim, slot) in scratch.found.iter().rev() {
            out.push((slot, sim));
        }
    }

    /// Bytes held by the graph's pools.
    pub(crate) fn pool_bytes(&self) -> usize {
        self.level0.len() * 4 + self.upper.pool_bytes() + self.lists.pool_bytes()
    }

    /// Carries the graph across a `maintain` compaction: `map[old_slot]`
    /// is the new slot id or [`NONE_U32`] for a purged vector. Neighbor
    /// lists are remapped in place-order (dead neighbors drop out), the
    /// entry follows the map or is re-picked deterministically (highest
    /// level, smallest slot). The caller then bulk-inserts the tail.
    pub(crate) fn remapped(
        &self,
        map: &[u32],
        new_pool: &VecPool<'_>,
        max_bytes: usize,
    ) -> Result<HnswGraph<'static>, Error> {
        // The remapped graph is freshly built (owned), so it is `'static`
        // and swaps into a `Memory<'a>` field by covariance (specs/16).
        let mut g: HnswGraph<'static> = HnswGraph::new(self.m, self.m0, max_bytes)?;
        let old_indexed = self.indexed as usize;
        let new_indexed = map[..old_indexed]
            .iter()
            .filter(|&&m| m != NONE_U32)
            .count() as u32;
        g.level0 = alloc::vec![NONE_U32; new_indexed as usize * self.m0];
        g.indexed = new_indexed;
        let mut nbrs = Vec::new();
        let mut sel = Vec::new();
        for old in 0..old_indexed as u32 {
            let new = map[old as usize];
            if new == NONE_U32 {
                continue;
            }
            self.neighbors_into(old, 0, &mut nbrs);
            sel.clear();
            sel.extend(
                nbrs.iter()
                    .map(|&n| map[n as usize])
                    .filter(|&n| n != NONE_U32),
            );
            g.write_list(new, 0, &sel)?;
            let levels = g.level_of(new_pool.slot_fact(new as usize));
            for level in 1..=levels {
                self.neighbors_into(old, level, &mut nbrs);
                if nbrs.is_empty() {
                    continue;
                }
                sel.clear();
                sel.extend(
                    nbrs.iter()
                        .map(|&n| map[n as usize])
                        .filter(|&n| n != NONE_U32),
                );
                g.write_list(new, level, &sel)?;
            }
        }
        g.entry = if self.entry != NONE_U32 && map[self.entry as usize] != NONE_U32 {
            map[self.entry as usize]
        } else {
            // The old entry died: the highest-level survivor takes over
            // (ties to the smallest slot — scan order settles it).
            let mut best = NONE_U32;
            let mut best_level = 0usize;
            for slot in 0..new_indexed {
                let level = g.level_of(new_pool.slot_fact(slot as usize));
                if best == NONE_U32 || level > best_level {
                    best = slot;
                    best_level = level;
                }
            }
            best
        };
        Ok(g)
    }

    /// The graph-header section: `[entry u32 LE][indexed u32 LE]`.
    pub(crate) fn dump_meta(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&self.entry.to_le_bytes());
        out.extend_from_slice(&self.indexed.to_le_bytes());
        out
    }

    /// The level-0 section: the neighbor blocks as `u32 LE` values.
    pub(crate) fn dump_level0(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.level0.len() * 4);
        for &n in &self.level0 {
            out.extend_from_slice(&n.to_le_bytes());
        }
        out
    }

    /// The four upper-level sections (handle arena + list pool).
    pub(crate) fn dump_upper(&self) -> [Vec<u8>; 4] {
        let (mut am, mut ap) = (Vec::new(), Vec::new());
        self.upper.dump_meta(&mut am);
        self.upper.dump_pool(&mut ap);
        let (mut cm, mut cp) = (Vec::new(), Vec::new());
        self.lists.dump_meta(&mut cm);
        self.lists.dump_pool(&mut cp);
        [am, ap, cm, cp]
    }

    /// Rebuilds a graph from its six dumped sections. Structural framing
    /// only — the reference validation that needs the vector pool happens
    /// in [`HnswGraph::validate`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        m: usize,
        m0: usize,
        max_bytes: usize,
        meta: &[u8],
        level0: &[u8],
        upper_meta: &[u8],
        upper_pool: &[u8],
        lists_meta: &[u8],
        lists_pool: &[u8],
    ) -> Result<Self, Error> {
        let mut g = Self::new(m, m0, max_bytes)?;
        if meta.len() != 8 {
            return Err(Error::Corrupt("hnsw meta section has a wrong length"));
        }
        g.entry = u32::from_le_bytes(meta[0..4].try_into().unwrap());
        g.indexed = u32::from_le_bytes(meta[4..8].try_into().unwrap());
        if level0.len() as u64 != u64::from(g.indexed) * m0 as u64 * 4 {
            return Err(Error::Corrupt("hnsw level0 length mismatch"));
        }
        g.level0 = level0
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        g.upper = Arena::load(
            ArenaCfg::new(UPPER_SHARDS, ShardMode::Uniform).with_max_bytes(max_bytes),
            upper_meta,
            upper_pool,
        )?;
        g.lists = ChunkPool::load(
            ChunkPoolCfg::new().with_max_bytes(max_bytes),
            lists_meta,
            lists_pool,
        )?;
        Ok(g)
    }

    /// Zero-copy sibling of [`HnswGraph::from_parts`]: the upper-level
    /// arena pool and the list pool borrow their mmap'd sections instead
    /// of copying (specs/16). `level0` and the small arena/chunk metadata
    /// are still rebuilt owned. Same framing checks as `from_parts`; the
    /// lifetime ties the graph to `upper_pool`/`lists_pool`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts_borrowed(
        m: usize,
        m0: usize,
        max_bytes: usize,
        meta: &[u8],
        level0: &[u8],
        upper_meta: &[u8],
        upper_pool: &'a [u8],
        lists_meta: &[u8],
        lists_pool: &'a [u8],
    ) -> Result<Self, Error> {
        let mut g = Self::new(m, m0, max_bytes)?;
        if meta.len() != 8 {
            return Err(Error::Corrupt("hnsw meta section has a wrong length"));
        }
        g.entry = u32::from_le_bytes(meta[0..4].try_into().unwrap());
        g.indexed = u32::from_le_bytes(meta[4..8].try_into().unwrap());
        if level0.len() as u64 != u64::from(g.indexed) * m0 as u64 * 4 {
            return Err(Error::Corrupt("hnsw level0 length mismatch"));
        }
        g.level0 = level0
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        g.upper = Arena::load_borrowed(
            ArenaCfg::new(UPPER_SHARDS, ShardMode::Uniform).with_max_bytes(max_bytes),
            upper_meta,
            upper_pool,
        )?;
        g.lists = ChunkPool::load_borrowed(
            ChunkPoolCfg::new().with_max_bytes(max_bytes),
            lists_meta,
            lists_pool,
        )?;
        Ok(g)
    }

    /// Full reference validation against the pool (the load-time
    /// panic-free contract, O(edges)): the entry and every neighbor in
    /// range, no self-links, canonical `NONE` padding of level-0 blocks,
    /// upper handles keyed inside the graph with levels their fact's hash
    /// admits, list chains exclusive and cycle-free, no orphan chunks,
    /// degrees within the caps.
    pub(crate) fn validate(&self, pool: &VecPool<'_>) -> Result<(), Error> {
        if self.indexed as usize > pool.len() {
            return Err(Error::Corrupt("hnsw indexes more slots than the pool"));
        }
        if self.level0.len() != self.indexed as usize * self.m0 {
            return Err(Error::Corrupt("hnsw level0 disagrees with indexed"));
        }
        if self.indexed == 0 {
            if self.entry != NONE_U32 || !self.upper.is_empty() || self.lists.chunks() != 0 {
                return Err(Error::Corrupt("hnsw empty graph carries state"));
            }
            return Ok(());
        }
        if self.entry >= self.indexed {
            return Err(Error::Corrupt("hnsw entry out of range"));
        }
        for slot in 0..self.indexed {
            let block = self.block(slot);
            let mut ended = false;
            for &n in block {
                if n == NONE_U32 {
                    ended = true;
                    continue;
                }
                if ended {
                    return Err(Error::Corrupt("hnsw level0 padding is not canonical"));
                }
                if n >= self.indexed || n == slot {
                    return Err(Error::Corrupt("hnsw level0 neighbor out of range"));
                }
            }
        }
        let mut visited = alloc::vec![false; self.lists.chunks()];
        for entry in self.upper.iter() {
            if entry.slot >= self.indexed {
                return Err(Error::Corrupt("hnsw upper handle out of range"));
            }
            let max_level = self.level_of(pool.slot_fact(entry.slot as usize));
            if entry.level == 0 || entry.level as usize > max_level {
                return Err(Error::Corrupt("hnsw upper level disagrees with the hash"));
            }
            self.lists.validate_chain(&entry.handle, &mut visited)?;
            let mut count = 0u32;
            for chunk in self.lists.iter(&entry.handle) {
                if !chunk.len().is_multiple_of(4) {
                    return Err(Error::Corrupt("hnsw upper list is not a slot sequence"));
                }
                for raw in chunk.chunks_exact(4) {
                    let n = u32::from_le_bytes(raw.try_into().unwrap());
                    if n >= self.indexed || n == entry.slot {
                        return Err(Error::Corrupt("hnsw upper neighbor out of range"));
                    }
                    count += 1;
                }
            }
            if count != entry.handle.len() || count as usize > self.m {
                return Err(Error::Corrupt("hnsw upper list disagrees with its handle"));
            }
        }
        if self.lists.orphan_count(&visited) != 0 {
            return Err(Error::Corrupt("hnsw list pool has orphan chunks"));
        }
        Ok(())
    }

    /// Exact distance evaluations so far (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn dist_evals(&self) -> u64 {
        self.dist_evals.get()
    }

    /// Resets the distance counter (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn reset_dist_evals(&self) {
        self.dist_evals.set(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::FactId;
    use alloc::vec;

    /// Deterministic LCG for cluster corpora (the vecpool tests' twin).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// A pool of `n` vectors in `clusters` clusters on the sphere.
    fn cluster_pool(n: usize, dim: usize, clusters: usize, seed: u64) -> VecPool<'static> {
        let mut rng = Lcg(seed);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.next()).collect())
            .collect();
        let mut pool = VecPool::new(dim, usize::MAX);
        for i in 0..n {
            let c = &centers[i % clusters];
            let v: Vec<f32> = c.iter().map(|&x| x + rng.next() * 0.3).collect();
            pool.push(FactId(i as u32), &v).unwrap();
        }
        pool
    }

    /// Builds a graph over the whole pool.
    fn build(pool: &VecPool<'_>, m: usize, m0: usize) -> HnswGraph<'static> {
        let mut g = HnswGraph::new(m, m0, usize::MAX).unwrap();
        let mut scratch = HnswScratch::default();
        g.insert_bulk(pool, pool.len() as u32, 200, &mut scratch)
            .unwrap();
        g
    }

    /// Brute-force top-k by exact quantized cosine.
    fn brute_force(pool: &VecPool<'_>, q: u32, k: usize) -> Vec<u32> {
        let mut all: Vec<(f32, u32)> = (0..pool.len() as u32)
            .map(|i| (pool.sim(q, i), i))
            .collect();
        all.sort_unstable_by(|a, b| better(*b, *a));
        all.into_iter().take(k).map(|(_, s)| s).collect()
    }

    /// Levels follow the geometric distribution: ~1/m of the nodes reach
    /// level 1, ~1/m² level 2, and the counts are exact across runs.
    #[test]
    #[cfg_attr(miri, ignore)] // data-heavy; the small graphs cover the logic
    fn levels_are_geometric_and_pure() {
        let g = HnswGraph::new(16, 32, usize::MAX).unwrap();
        let n = 100_000u32;
        let mut per_level = [0usize; 4];
        for fact in 0..n {
            let l = g.level_of(fact).min(3);
            per_level[l] += 1;
        }
        // Expected ~6250 at ≥1 for m=16; allow a generous band.
        let at_least_1: usize = per_level[1..].iter().sum();
        assert!(
            (4_000..9_000).contains(&at_least_1),
            "level>=1 count {at_least_1} is out of band"
        );
        let at_least_2: usize = per_level[2..].iter().sum();
        assert!(
            (150..800).contains(&at_least_2),
            "level>=2 count {at_least_2} is out of band"
        );
        // Purity: the same fact maps to the same level, always.
        assert_eq!(g.level_of(42), g.level_of(42));
    }

    /// Graph search agrees with brute force: recall@10 >= 0.9 at ef 64
    /// (the specs/10 verification gate).
    #[test]
    #[cfg_attr(miri, ignore)] // data-heavy; the small graphs cover the logic
    fn recall_against_brute_force() {
        let dim = 32;
        let pool = cluster_pool(2_000, dim, 64, 0xA11CE);
        let g = build(&pool, 16, 32);
        let mut scratch = HnswScratch::default();
        let mut out = Vec::new();
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in (0..2_000u32).step_by(97) {
            let truth = brute_force(&pool, q, 10);
            let (scale, qb) = pool.quant(q as usize);
            g.search_quantized(&pool, (scale, qb), 64, &mut scratch, &mut out);
            let got: Vec<u32> = out.iter().take(10).map(|&(s, _)| s).collect();
            hits += truth.iter().filter(|t| got.contains(t)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.9, "recall@10 {recall} below the 0.9 gate");
    }

    /// Two builds over the same pool are byte-identical (determinism is
    /// what makes maintain-replay reproduce the graph).
    #[test]
    #[cfg_attr(miri, ignore)] // data-heavy; the small graphs cover the logic
    fn build_is_deterministic() {
        let pool = cluster_pool(600, 24, 16, 7);
        let a = build(&pool, 8, 16);
        let b = build(&pool, 8, 16);
        assert_eq!(a.level0, b.level0);
        assert_eq!(a.entry, b.entry);
        assert_eq!(a.indexed, b.indexed);
        let (mut am, mut bm) = (Vec::new(), Vec::new());
        a.upper.dump_meta(&mut am);
        b.upper.dump_meta(&mut bm);
        assert_eq!(am, bm);
        let (mut ap, mut bp) = (Vec::new(), Vec::new());
        a.lists.dump_pool(&mut ap);
        b.lists.dump_pool(&mut bp);
        assert_eq!(ap, bp);
    }

    /// Degree caps hold everywhere.
    #[test]
    #[cfg_attr(miri, ignore)] // data-heavy; the small graphs cover the logic
    fn degree_caps_hold() {
        let pool = cluster_pool(800, 16, 8, 3);
        let g = build(&pool, 6, 12);
        let mut nbrs = Vec::new();
        for slot in 0..g.indexed() {
            g.neighbors_into(slot, 0, &mut nbrs);
            assert!(nbrs.len() <= 12);
            // No self-links, no duplicates, all in range.
            assert!(!nbrs.contains(&slot));
            let mut sorted = nbrs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), nbrs.len());
            assert!(nbrs.iter().all(|&n| n < g.indexed()));
            for level in 1..=g.level_of(pool.slot_fact(slot as usize)) {
                g.neighbors_into(slot, level, &mut nbrs);
                assert!(nbrs.len() <= 6, "level {level} degree overflow");
            }
        }
    }

    /// Remap carries the graph across a compaction: dead nodes (the
    /// entry included) drop out, survivors stay searchable, and the
    /// result passes reference validation.
    #[test]
    #[cfg_attr(miri, ignore)] // data-heavy; the small graphs cover the logic
    fn remap_survives_a_dead_entry() {
        let pool = cluster_pool(300, 16, 8, 21);
        let g = build(&pool, 6, 12);
        let entry = g.entry;
        // Compaction map: the entry and every 7th slot die; survivors
        // keep their relative order (as maintain produces).
        let mut map = alloc::vec![NONE_U32; 300];
        let mut new_pool = VecPool::new(16, usize::MAX);
        let mut next = 0u32;
        for old in 0..300u32 {
            if old == entry || old % 7 == 0 {
                continue;
            }
            map[old as usize] = next;
            new_pool.copy_slot(&pool, old);
            next += 1;
        }
        let remapped = g.remapped(&map, &new_pool, usize::MAX).unwrap();
        assert_eq!(remapped.indexed(), next);
        assert_ne!(remapped.entry, NONE_U32, "a survivor takes the entry");
        remapped.validate(&new_pool).unwrap();
        // A surviving vector still finds itself.
        let probe_old = 1u32; // 1 % 7 != 0; if it was the entry pick 2
        let probe_old = if probe_old == entry { 2 } else { probe_old };
        let probe_new = map[probe_old as usize];
        let (scale, qb) = new_pool.quant(probe_new as usize);
        let mut scratch = HnswScratch::default();
        let mut out = Vec::new();
        remapped.search_quantized(&new_pool, (scale, qb), 32, &mut scratch, &mut out);
        assert_eq!(out[0].0, probe_new);
    }

    /// Reference validation rejects structural lies: an out-of-range
    /// entry, a non-canonical level-0 padding, a neighbor past
    /// `indexed`, and state on an allegedly empty graph.
    #[test]
    fn validate_rejects_malformed_graphs() {
        let pool = cluster_pool(50, 8, 4, 5);
        let g = build(&pool, 4, 8);
        let dump = (g.dump_meta(), g.dump_level0(), g.dump_upper());
        let load = |meta: &[u8], level0: &[u8]| {
            HnswGraph::from_parts(
                4,
                8,
                usize::MAX,
                meta,
                level0,
                &dump.2[0],
                &dump.2[1],
                &dump.2[2],
                &dump.2[3],
            )
        };
        // The clean image round-trips and validates.
        load(&dump.0, &dump.1).unwrap().validate(&pool).unwrap();
        // Entry out of range.
        let mut meta = dump.0.clone();
        meta[0..4].copy_from_slice(&999u32.to_le_bytes());
        assert!(load(&meta, &dump.1).unwrap().validate(&pool).is_err());
        // A neighbor past `indexed`.
        let mut level0 = dump.1.clone();
        level0[0..4].copy_from_slice(&500u32.to_le_bytes());
        assert!(load(&dump.0, &level0).unwrap().validate(&pool).is_err());
        // Non-canonical padding: NONE followed by a live neighbor.
        let mut level0 = dump.1.clone();
        level0[0..4].copy_from_slice(&NONE_U32.to_le_bytes());
        level0[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(load(&dump.0, &level0).unwrap().validate(&pool).is_err());
        // A wrong level0 length is rejected at framing already.
        assert!(load(&dump.0, &dump.1[..dump.1.len() - 4]).is_err());
        // An "empty" graph carrying an entry is corrupt.
        let mut meta = dump.0.clone();
        meta[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(load(&meta, &[]).unwrap().validate(&pool).is_err());
    }

    /// An empty graph answers empty; a single node answers itself.
    #[test]
    fn tiny_graphs() {
        let pool = cluster_pool(1, 8, 1, 1);
        let mut g = HnswGraph::new(4, 8, usize::MAX).unwrap();
        let mut scratch = HnswScratch::default();
        let mut out = vec![(0u32, 0.0f32)];
        let (scale, qb) = pool.quant(0);
        g.search_quantized(&pool, (scale, qb), 8, &mut scratch, &mut out);
        assert!(out.is_empty(), "an empty graph must answer empty");
        g.insert_bulk(&pool, 1, 50, &mut scratch).unwrap();
        g.search_quantized(&pool, (scale, qb), 8, &mut scratch, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
    }
}
