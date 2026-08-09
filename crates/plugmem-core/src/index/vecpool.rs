//! The vector index: flat quantized-vector storage with a two-phase
//! flat search.
//!
//! Vectors are the fourth recall source. Storage is deliberately *not* an
//! arena or a blob heap but one contiguous `Vec<u8>` of fixed-stride
//! slots: flat search reads every live slot, so perfect locality matters
//! more than sorted lookup. Slots are append-only; dead ones are dropped
//! by `maintain`, never reused in place.
//!
//! One slot, all little-endian, `stride = 8 + 8·words + dim` bytes where
//! `words = ceil(dim / 64)`:
//!
//! | off | size | field |
//! |---|---|---|
//! | 0 | 4 | `fact` — owning [`FactId`] |
//! | 4 | 4 | `scale` — f32 quantization scale |
//! | 8 | 8·words | `sig` — sign signature, bit `i` set iff `q[i] >= 0` |
//! | 8+8·words | dim | `q` — the i8 components |
//!
//! Quantization is symmetric i8 over the L2-normalized vector
//! (`scale = max|x/‖x‖| / 127`, `q = round((x/‖x‖) / scale)`), so a
//! quantized cosine is `scale_a · scale_b · Σ qa·qb`. It is a pure
//! function of the input — journal replay re-quantizes and reproduces
//! every slot byte for byte.
//!
//! Search is two-phase: the sign signatures give a cheap Hamming
//! prefilter (popcount over u64 words, SIMD-friendly), then only the best
//! `max(4·k, 64)` candidates pay an exact quantized-cosine rescore. The
//! `counters` feature counts those exact dot products — the deterministic
//! cost metric the perf gate holds to `min(candidates, live)`.

use alloc::vec::Vec;

#[cfg(feature = "counters")]
use core::cell::Cell;

use crate::error::Error;
use crate::id::FactId;

/// Serialized width of a slot's owning `fact` id (little-endian `u32`).
const FACT_BYTES: usize = core::mem::size_of::<u32>();

/// Serialized width of a slot's quantization `scale` (little-endian `f32`).
const SCALE_BYTES: usize = core::mem::size_of::<f32>();

/// Fixed slot header: `fact u32` + `scale f32`.
const HEAD: usize = FACT_BYTES + SCALE_BYTES;

/// Bytes of one signature word (a little-endian `u64`): the sign signature is
/// packed into `words(dim)` of these.
const SIG_WORD_BYTES: usize = core::mem::size_of::<u64>();

/// Reusable vector-search scratch, owned by the engine (the zero-alloc
/// recall invariant: after warm-up a search allocates nothing).
#[derive(Debug, Default)]
pub struct VecScratch {
    /// `(hamming, slot)` prefilter buffer.
    cand: Vec<(u32, u32)>,
    /// `(cosine, fact)` rescore buffer.
    top: Vec<(f32, u32)>,
    /// The quantized query slot (`stride` bytes; `fact` unused).
    query: Vec<u8>,
}

impl VecScratch {
    /// Empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Flat store of quantized vectors.
///
/// The slot bytes are an **overlay** of a borrowed base and an owned tail:
/// the owned path (`new`/`push`/`from_parts`) keeps `base = &[]` with every
/// slot in `tail`, byte-for-byte unchanged; the borrowed/overlay path
/// (`from_parts_borrowed`/`from_parts_overlay`) maps an mmap'd section as
/// `base` and appends new slots to `tail`. Because dead slots are
/// dropped by `maintain` and never rewritten in place, a slot is wholly in
/// `base` or wholly in `tail`, so a heap opened over a multi-gigabyte mmap
/// grows without cloning it — reads dispatch on one comparison per slot.
#[derive(Debug)]
pub struct VecPool<'a> {
    /// Borrowed base slots (an mmap'd section) — empty on the owned path.
    base: &'a [u8],
    /// Owned append tail; `push`/`copy_slot` extend this, never `base`.
    tail: Vec<u8>,
    dim: usize,
    max_bytes: usize,
    /// Exact dot products computed by searches (feature `counters`).
    #[cfg(feature = "counters")]
    dots: Cell<u64>,
}

impl<'a> VecPool<'a> {
    /// Signature words for a dimension.
    #[inline]
    fn words(dim: usize) -> usize {
        dim.div_ceil(64)
    }

    /// Creates an empty pool for `dim`-dimensional vectors (`dim == 0`
    /// leaves the layer inert — the pool stays empty).
    pub fn new(dim: usize, max_bytes: usize) -> Self {
        Self {
            base: &[],
            tail: Vec::new(),
            dim,
            max_bytes,
            #[cfg(feature = "counters")]
            dots: Cell::new(0),
        }
    }

    /// Byte stride of one slot.
    #[inline]
    pub fn stride(&self) -> usize {
        HEAD + Self::words(self.dim) * SIG_WORD_BYTES + self.dim
    }

    /// Total bytes of the logical pool (`base` + `tail`).
    #[inline]
    fn pool_len(&self) -> usize {
        self.base.len() + self.tail.len()
    }

    /// The `stride` bytes of slot `i`, dispatched to `base` or `tail`. The
    /// base is a whole number of slots (framed on load), so slot `i` never
    /// straddles the boundary. `pub(crate)` so the disk-first rebuild can stream
    /// a survivor slot straight into a `Scratch`.
    #[inline]
    pub(crate) fn slot_bytes(&self, i: usize) -> &[u8] {
        let stride = self.stride();
        let start = i * stride;
        let base_len = self.base.len();
        if start < base_len {
            &self.base[start..start + stride]
        } else {
            let at = start - base_len;
            &self.tail[at..at + stride]
        }
    }

    /// Number of stored slots.
    #[inline]
    pub fn len(&self) -> usize {
        let pool_len = self.pool_len();
        if pool_len == 0 {
            0
        } else {
            pool_len / self.stride()
        }
    }

    /// `true` when no vector is stored.
    pub fn is_empty(&self) -> bool {
        self.pool_len() == 0
    }

    /// Total bytes held.
    pub fn pool_bytes(&self) -> usize {
        self.pool_len()
    }

    /// The owning fact of slot `i`.
    #[inline]
    pub fn slot_fact(&self, i: usize) -> u32 {
        let slot = self.slot_bytes(i);
        u32::from_le_bytes(slot[..FACT_BYTES].try_into().unwrap())
    }

    /// The scale of slot `i`.
    #[inline]
    fn slot_scale(&self, i: usize) -> f32 {
        let slot = self.slot_bytes(i);
        f32::from_le_bytes(slot[FACT_BYTES..HEAD].try_into().unwrap())
    }

    /// The `(scale, q)` pair of slot `i` — the quantized payload a
    /// distance evaluation needs. Shared with the HNSW graph, which
    /// stores only neighbor ids and reads vectors from here.
    #[inline]
    pub(crate) fn quant(&self, i: usize) -> (f32, &[u8]) {
        let stride = self.stride();
        let q_off = HEAD + Self::words(self.dim) * SIG_WORD_BYTES;
        let slot = self.slot_bytes(i);
        let scale = f32::from_le_bytes(slot[FACT_BYTES..HEAD].try_into().unwrap());
        (scale, &slot[q_off..stride])
    }

    /// Quantized cosine of two slots by index — the graph's edge metric.
    /// Both indices must be in range (graph neighbors are validated on
    /// load).
    #[inline]
    pub(crate) fn sim(&self, a: u32, b: u32) -> f32 {
        self.cosine_at(a as usize, b as usize)
    }

    /// Quantizes `v` into `out` (which is sized to `stride`), writing the
    /// full slot for `fact`. A pure, deterministic function of `v`.
    ///
    /// # Errors
    ///
    /// [`Error::DimMismatch`] if `v.len() != dim`; [`Error::Invalid`] if
    /// `v` is not finite or is the zero vector (no direction to encode).
    fn encode_slot(&self, fact: u32, v: &[f32], out: &mut [u8]) -> Result<(), Error> {
        if v.len() != self.dim {
            return Err(Error::DimMismatch {
                got: v.len(),
                want: self.dim,
            });
        }
        let mut norm_sq = 0.0f32;
        for &x in v {
            if !x.is_finite() {
                return Err(Error::Invalid("vector must be finite"));
            }
            norm_sq += x * x;
        }
        // `norm` is finite and non-negative (inputs are finite), so this
        // rejects exactly the zero vector.
        let norm = libm::sqrtf(norm_sq);
        if norm <= 0.0 {
            return Err(Error::Invalid("vector must be nonzero"));
        }
        let inv_norm = 1.0 / norm;
        let mut max_abs = 0.0f32;
        for &x in v {
            max_abs = max_abs.max(libm::fabsf(x * inv_norm));
        }
        // Nonzero norm guarantees a nonzero max component, so scale > 0.
        let scale = max_abs / 127.0;
        out[..FACT_BYTES].copy_from_slice(&fact.to_le_bytes());
        out[FACT_BYTES..HEAD].copy_from_slice(&scale.to_le_bytes());
        let words = Self::words(self.dim);
        let q_off = HEAD + words * SIG_WORD_BYTES;
        for (i, &x) in v.iter().enumerate() {
            let qf = libm::roundf((x * inv_norm) / scale);
            let qi = qf.clamp(-127.0, 127.0) as i32 as i8;
            out[q_off + i] = qi as u8;
        }
        // Sign signature from the quantized components.
        for w in 0..words {
            let mut word = 0u64;
            for b in 0..64 {
                let i = w * 64 + b;
                if i >= self.dim {
                    break;
                }
                if out[q_off + i] as i8 >= 0 {
                    word |= 1 << b;
                }
            }
            out[HEAD + w * SIG_WORD_BYTES..HEAD + w * SIG_WORD_BYTES + SIG_WORD_BYTES]
                .copy_from_slice(&word.to_le_bytes());
        }
        Ok(())
    }

    /// Quantizes `v` and appends it as `fact`'s slot, returning the slot
    /// index. Ids are not stored sorted — the fact record keeps the index.
    ///
    /// # Errors
    ///
    /// [`Error::DimMismatch`]/[`Error::Invalid`] from quantization, or
    /// [`Error::CapacityExceeded`] at the byte ceiling.
    pub fn push(&mut self, fact: FactId, v: &[f32]) -> Result<u32, Error> {
        let stride = self.stride();
        let pool_len = self.pool_len();
        if pool_len + stride > self.max_bytes {
            return Err(Error::CapacityExceeded { what: "vectors" });
        }
        let index = u32::try_from(pool_len / stride).map_err(|_| Error::CapacityExceeded {
            what: "vector slots",
        })?;
        // Take the tail out so `encode_slot(&self, ...)` can borrow `self`
        // while we write into the (now-detached) buffer; the new slot always
        // lands in the tail, never the borrowed base.
        let mut tail = core::mem::take(&mut self.tail);
        let at = tail.len();
        tail.resize(at + stride, 0);
        let res = match self.encode_slot(fact.0, v, &mut tail[at..]) {
            Ok(()) => Ok(index),
            Err(e) => {
                // Roll the failed append back so the pool stays canonical.
                tail.truncate(at);
                Err(e)
            }
        };
        self.tail = tail;
        res
    }

    /// The `(scale, q)` view of a query previously quantized into
    /// `scratch` by [`VecPool::quantize_query`] — what the graph search
    /// and the tail scan consume.
    pub(crate) fn quantized<'s>(&self, scratch: &'s VecScratch) -> (f32, &'s [u8]) {
        let stride = self.stride();
        let q_off = HEAD + Self::words(self.dim) * SIG_WORD_BYTES;
        debug_assert_eq!(scratch.query.len(), stride);
        (
            f32::from_le_bytes(scratch.query[FACT_BYTES..HEAD].try_into().unwrap()),
            &scratch.query[q_off..stride],
        )
    }

    /// Quantizes a query vector into `scratch.query` (sized to `stride`,
    /// `fact` left as `0`). Reused across searches.
    pub fn quantize_query(&self, v: &[f32], scratch: &mut VecScratch) -> Result<(), Error> {
        let stride = self.stride();
        scratch.query.clear();
        scratch.query.resize(stride, 0);
        let mut buf = core::mem::take(&mut scratch.query);
        let res = self.encode_slot(0, v, &mut buf);
        scratch.query = buf;
        res
    }

    /// Copies slot `i` of `src` verbatim (already quantized) into `self`,
    /// returning its new index. Used by `maintain` compaction — the
    /// quantized bytes are reproduced exactly, so a compacted snapshot is
    /// byte-identical to a replayed one. `src` must share this pool's `dim`.
    pub(crate) fn copy_slot(&mut self, src: &VecPool<'_>, i: u32) -> u32 {
        debug_assert_eq!(self.dim, src.dim, "copy_slot across differing dims");
        let stride = self.stride();
        let index = (self.pool_len() / stride) as u32;
        self.tail.extend_from_slice(src.slot_bytes(i as usize));
        index
    }

    /// Appends an exact copy of one of this pool's slots for a new owning
    /// fact. Retag revisions preserve embeddings without a model round trip or
    /// a lossy dequantize/requantize cycle.
    pub(crate) fn clone_slot_for_fact(&mut self, fact: FactId, source: u32) -> Result<u32, Error> {
        let stride = self.stride();
        let pool_len = self.pool_len();
        if source as usize >= self.len() {
            return Err(Error::Corrupt("retag vector slot is out of range"));
        }
        if pool_len + stride > self.max_bytes {
            return Err(Error::CapacityExceeded { what: "vectors" });
        }
        let index = u32::try_from(pool_len / stride).map_err(|_| Error::CapacityExceeded {
            what: "vector slots",
        })?;
        let source_start = source as usize * stride;
        if source_start < self.base.len() {
            self.tail
                .extend_from_slice(&self.base[source_start..source_start + stride]);
        } else {
            let at = source_start - self.base.len();
            let dst = self.tail.len();
            self.tail.resize(dst + stride, 0);
            self.tail.copy_within(at..at + stride, dst);
        }
        let at = self.tail.len() - stride;
        self.tail[at..at + FACT_BYTES].copy_from_slice(&fact.0.to_le_bytes());
        Ok(index)
    }

    /// The exact quantized cosine of two slots by their scales and i8
    /// components.
    fn cosine_at(&self, a: usize, b: usize) -> f32 {
        let stride = self.stride();
        let q_off = HEAD + Self::words(self.dim) * SIG_WORD_BYTES;
        let (sa, sb) = (self.slot_bytes(a), self.slot_bytes(b));
        let dot = dot_i8(&sa[q_off..stride], &sb[q_off..stride]);
        self.slot_scale(a) * self.slot_scale(b) * dot as f32
    }

    /// Quantized cosine of slots `a` and `b` (similar-detection uses it on
    /// two stored facts). Returns `0.0` if either index is out of range.
    pub fn cosine_slots(&self, a: u32, b: u32) -> f32 {
        let n = self.len();
        if a as usize >= n || b as usize >= n {
            return 0.0;
        }
        self.cosine_at(a as usize, b as usize)
    }

    /// Flat two-phase search: Hamming prefilter on signatures, then exact
    /// quantized-cosine rescore of the best `max(4·k, 64)` candidates.
    /// `admit` filters by the shared recall rule; writes the top `k`
    /// `(fact, cosine)` into `out`, descending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        admit: &mut dyn FnMut(FactId) -> bool,
        scratch: &mut VecScratch,
        out: &mut Vec<(FactId, f32)>,
    ) -> Result<(), Error> {
        out.clear();
        let n = self.len();
        if n == 0 || k == 0 {
            return Ok(());
        }
        self.quantize_query(query, scratch)?;
        let stride = self.stride();
        let words = Self::words(self.dim);
        let q_off = HEAD + words * SIG_WORD_BYTES;

        // Phase 1: Hamming distance of every slot's signature to the query.
        let VecScratch {
            cand, top, query, ..
        } = scratch;
        let q_sig = &query[HEAD..HEAD + words * SIG_WORD_BYTES];
        cand.clear();
        cand.reserve(n);
        for i in 0..n {
            let slot = self.slot_bytes(i);
            let s_sig = &slot[HEAD..HEAD + words * SIG_WORD_BYTES];
            let mut ham = 0u32;
            for w in 0..words {
                let a = u64::from_le_bytes(
                    q_sig[w * SIG_WORD_BYTES..w * SIG_WORD_BYTES + SIG_WORD_BYTES]
                        .try_into()
                        .unwrap(),
                );
                let b = u64::from_le_bytes(
                    s_sig[w * SIG_WORD_BYTES..w * SIG_WORD_BYTES + SIG_WORD_BYTES]
                        .try_into()
                        .unwrap(),
                );
                ham += (a ^ b).count_ones();
            }
            cand.push((ham, i as u32));
        }
        let c = (4 * k).max(64).min(n);
        if cand.len() > c {
            cand.select_nth_unstable(c - 1);
        }

        // Phase 2: exact quantized cosine on the survivors.
        let q_scale = f32::from_le_bytes(query[FACT_BYTES..HEAD].try_into().unwrap());
        let q_q = &query[q_off..q_off + self.dim];
        top.clear();
        #[cfg(feature = "counters")]
        let mut dots = 0u64;
        for &(_, slot) in cand[..c].iter() {
            let sb = self.slot_bytes(slot as usize);
            let fact = FactId(u32::from_le_bytes(sb[..FACT_BYTES].try_into().unwrap()));
            if !admit(fact) {
                continue;
            }
            let s_scale = f32::from_le_bytes(sb[FACT_BYTES..HEAD].try_into().unwrap());
            let dot = dot_i8(q_q, &sb[q_off..stride]);
            top.push((q_scale * s_scale * dot as f32, fact.0));
            #[cfg(feature = "counters")]
            {
                dots += 1;
            }
        }
        #[cfg(feature = "counters")]
        self.dots.set(self.dots.get() + dots);
        top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(score, id) in top.iter().take(k) {
            out.push((FactId(id), score));
        }
        Ok(())
    }

    /// The vector section as one contiguous buffer (`base ++ tail`).
    /// Byte-identical to the owned pool holding the same slots, so an overlay
    /// snapshot is canonical. Test-only: production streams via
    /// [`VecPool::pieces`], which needs no owned copy.
    #[cfg(test)]
    pub(crate) fn dump(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pool_len());
        out.extend_from_slice(self.base);
        out.extend_from_slice(&self.tail);
        out
    }

    /// The vector section as its two contiguous pieces (`base`, `tail`)
    /// without concatenating them — lets the streaming snapshot writer
    /// emit the dominant pool with no owned full-section copy.
    /// Concatenated, the pieces equal the section (`base ++ tail`).
    pub(crate) fn pieces(&self) -> [&[u8]; 2] {
        [self.base, &self.tail]
    }

    /// Rebuilds a pool from its dumped section, checking only the framing
    /// (length is a whole number of slots, fits the ceiling). Slot content
    /// — scales, signatures, the fact bijection — is validated by the
    /// engine's `validate_references` (it needs the fact records too).
    pub(crate) fn from_parts(dim: usize, max_bytes: usize, bytes: &[u8]) -> Result<Self, Error> {
        Self::frame_check(dim, max_bytes, bytes.len())?;
        let mut pool = Self::new(dim, max_bytes);
        pool.tail = bytes.to_vec();
        Ok(pool)
    }

    /// Zero-copy sibling of [`VecPool::from_parts`]: the pool borrows the
    /// dumped section (an mmap'd byte range) as its base instead of copying
    /// it. Same framing checks; the lifetime ties the pool to `bytes`
    /// Under the overlay write path a later [`VecPool::push`]
    /// appends to an owned tail without cloning the base.
    pub(crate) fn from_parts_borrowed(
        dim: usize,
        max_bytes: usize,
        bytes: &'a [u8],
    ) -> Result<Self, Error> {
        Self::frame_check(dim, max_bytes, bytes.len())?;
        let mut pool = Self::new(dim, max_bytes);
        pool.base = bytes;
        Ok(pool)
    }

    /// Frames the dumped section: a length within the ceiling and, for a
    /// live layer, a whole number of slots. Shared by both `from_parts`
    /// constructors so the owned and borrowed paths validate identically.
    fn frame_check(dim: usize, max_bytes: usize, len: usize) -> Result<(), Error> {
        if len > max_bytes {
            return Err(Error::Corrupt("vector pool exceeds the configured ceiling"));
        }
        if dim == 0 {
            if len != 0 {
                return Err(Error::Corrupt("vector pool present with dim 0"));
            }
            return Ok(());
        }
        let stride = HEAD + Self::words(dim) * SIG_WORD_BYTES + dim;
        if !len.is_multiple_of(stride) {
            return Err(Error::Corrupt("vector pool is not a whole number of slots"));
        }
        Ok(())
    }

    /// Structural self-check of every slot (A.4): each scale is
    /// finite and non-negative, and each signature bit agrees with the
    /// sign of its quantized component. Keeps the panic-free contract:
    /// after this, a search over the pool cannot read a malformed slot
    /// into a NaN or disagree with the prefilter.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.dim == 0 {
            return Ok(());
        }
        let words = Self::words(self.dim);
        let q_off = HEAD + words * SIG_WORD_BYTES;
        for i in 0..self.len() {
            let slot = self.slot_bytes(i);
            let scale = f32::from_le_bytes(slot[FACT_BYTES..HEAD].try_into().unwrap());
            if !scale.is_finite() || scale < 0.0 {
                return Err(Error::Corrupt(
                    "vector slot scale is not finite and non-negative",
                ));
            }
            for w in 0..words {
                let stored = u64::from_le_bytes(
                    slot[HEAD + w * SIG_WORD_BYTES..HEAD + w * SIG_WORD_BYTES + SIG_WORD_BYTES]
                        .try_into()
                        .unwrap(),
                );
                let mut expect = 0u64;
                for b in 0..64 {
                    let j = w * 64 + b;
                    if j >= self.dim {
                        break;
                    }
                    if slot[q_off + j] as i8 >= 0 {
                        expect |= 1 << b;
                    }
                }
                if stored != expect {
                    return Err(Error::Corrupt(
                        "vector slot signature disagrees with its components",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Exact dot products computed so far (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn dots(&self) -> u64 {
        self.dots.get()
    }

    /// Resets the dot counter (feature `counters`).
    #[cfg(feature = "counters")]
    pub fn reset_dots(&self) {
        self.dots.set(0);
    }
}

/// Integer dot product of two equal-length i8 slices held as bytes.
/// `dim ≤ 4096` and `|q| ≤ 127`, so the sum fits `i32`
/// (`4096 · 127² < 2³¹`).
#[inline]
pub(crate) fn dot_i8(a: &[u8], b: &[u8]) -> i32 {
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += i32::from(x as i8) * i32::from(y as i8);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A tiny deterministic LCG yielding `f32` in `[-1, 1)` — no rng crate
    /// in the core's test surface, and determinism is the repo law.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vector(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.next()).collect()
        }
    }

    /// True (unquantized) cosine of two vectors.
    fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = libm::sqrtf(a.iter().map(|x| x * x).sum());
        let nb: f32 = libm::sqrtf(b.iter().map(|x| x * x).sum());
        dot / (na * nb)
    }

    /// The quantized cosine tracks the true cosine within the documented
    /// error band across many random pairs.
    #[test]
    fn quantized_cosine_tracks_f32() {
        let dim = 384;
        let mut rng = Lcg(0x1234_5678);
        let mut worst = 0.0f32;
        for i in 0..200u32 {
            let a = rng.vector(dim);
            let b = rng.vector(dim);
            let mut pool = VecPool::new(dim, usize::MAX);
            pool.push(FactId(2 * i), &a).unwrap();
            pool.push(FactId(2 * i + 1), &b).unwrap();
            let q = pool.cosine_slots(0, 1);
            let t = cosine_f32(&a, &b);
            worst = worst.max(libm::fabsf(q - t));
        }
        assert!(
            worst < 0.05,
            "worst quantization error {worst} exceeds 0.05"
        );
    }

    /// Golden: a hand-built dim-4 example. `a` and `b` share a direction,
    /// `c` is orthogonal-ish; quantized cosine ranks them accordingly and
    /// the sign signature matches the component signs.
    #[test]
    fn golden_dim4() {
        let dim = 4;
        let mut pool = VecPool::new(dim, usize::MAX);
        pool.push(FactId(0), &[1.0, 1.0, 0.0, 0.0]).unwrap();
        pool.push(FactId(1), &[2.0, 2.0, 0.0, 0.0]).unwrap(); // same direction
        pool.push(FactId(2), &[0.0, 0.0, 1.0, 1.0]).unwrap(); // orthogonal
        // Parallel vectors → cosine ≈ 1.
        assert!((pool.cosine_slots(0, 1) - 1.0).abs() < 1e-3);
        // Orthogonal vectors → cosine ≈ 0.
        assert!(pool.cosine_slots(0, 2).abs() < 1e-3);
        // Structural self-check passes and the signature is well-formed.
        pool.validate().unwrap();
        // Slot 0 signature: components (positive, positive, +0, +0) → all
        // sign bits set for the first four bits.
        let stride = pool.stride();
        let sig = u64::from_le_bytes(pool.dump()[HEAD..HEAD + SIG_WORD_BYTES].try_into().unwrap());
        assert_eq!(sig & 0b1111, 0b1111);
        assert_eq!(pool.len(), 3);
        // dim 4: one signature word + 4 i8 components.
        assert_eq!(stride, HEAD + SIG_WORD_BYTES + 4);
    }

    /// The two-phase search returns the true nearest neighbor at the top.
    #[test]
    fn search_surfaces_the_nearest() {
        let dim = 64;
        let mut rng = Lcg(0xdead_beef);
        let mut pool = VecPool::new(dim, usize::MAX);
        let target = rng.vector(dim);
        // 200 random vectors, then the target itself as fact 500.
        for i in 0..200u32 {
            pool.push(FactId(i), &rng.vector(dim)).unwrap();
        }
        pool.push(FactId(500), &target).unwrap();
        let mut scratch = VecScratch::new();
        let mut out = Vec::new();
        pool.search(&target, 5, &mut |_| true, &mut scratch, &mut out)
            .unwrap();
        assert_eq!(out[0].0, FactId(500), "exact match must rank first");
        assert!(out[0].1 > 0.99, "self-cosine ≈ 1, got {}", out[0].1);
    }

    /// A zero or non-finite vector has no direction to quantize.
    #[test]
    fn degenerate_vectors_are_invalid() {
        let mut pool = VecPool::new(3, usize::MAX);
        assert_eq!(
            pool.push(FactId(0), &[0.0, 0.0, 0.0]).unwrap_err(),
            Error::Invalid("vector must be nonzero")
        );
        assert_eq!(
            pool.push(FactId(0), &[1.0, f32::NAN, 0.0]).unwrap_err(),
            Error::Invalid("vector must be finite")
        );
        assert!(matches!(
            pool.push(FactId(0), &[1.0, 2.0]).unwrap_err(),
            Error::DimMismatch { got: 2, want: 3 }
        ));
        // A failed push leaves the pool canonical (nothing appended).
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }

    /// Accessors and the edge branches: empty pool, `k == 0`, an
    /// out-of-range cosine, and the byte ceiling.
    #[test]
    fn accessors_and_edges() {
        let dim = 4;
        let mut pool = VecPool::new(dim, usize::MAX);
        assert!(pool.is_empty());
        assert_eq!(pool.pool_bytes(), 0);
        let mut scratch = VecScratch::new();
        let mut out = vec![(FactId(9), 1.0)];
        // Empty pool: search clears the output and returns nothing.
        pool.search(&[1.0; 4], 5, &mut |_| true, &mut scratch, &mut out)
            .unwrap();
        assert!(out.is_empty());

        pool.push(FactId(0), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(!pool.is_empty());
        assert_eq!(pool.pool_bytes(), pool.stride());
        // k == 0 short-circuits.
        pool.search(&[1.0; 4], 0, &mut |_| true, &mut scratch, &mut out)
            .unwrap();
        assert!(out.is_empty());
        // An out-of-range slot index yields a zero cosine, never a panic.
        assert_eq!(pool.cosine_slots(0, 9), 0.0);

        // The byte ceiling: a push past `max_bytes` is a typed error.
        let mut tight = VecPool::new(dim, 4);
        assert_eq!(
            tight.push(FactId(0), &[1.0, 0.0, 0.0, 0.0]).unwrap_err(),
            Error::CapacityExceeded { what: "vectors" }
        );
    }

    /// `from_parts` accepts a whole number of slots and rejects the rest.
    #[test]
    fn from_parts_frames_slots() {
        let dim = 8;
        let mut pool = VecPool::new(dim, usize::MAX);
        pool.push(FactId(0), &vec![0.5; dim]).unwrap();
        pool.push(FactId(1), &vec![-0.5; dim]).unwrap();
        let bytes = pool.dump();
        let rebuilt = VecPool::from_parts(dim, usize::MAX, &bytes).unwrap();
        assert_eq!(rebuilt.len(), 2);
        rebuilt.validate().unwrap();
        // One byte short of a slot boundary is corrupt.
        assert!(VecPool::from_parts(dim, usize::MAX, &bytes[..bytes.len() - 1]).is_err());
        // A non-empty pool with dim 0 is corrupt.
        assert!(VecPool::from_parts(0, usize::MAX, &bytes).is_err());
        // Bytes past the configured ceiling are corrupt.
        assert!(VecPool::from_parts(dim, bytes.len() - 1, &bytes).is_err());
    }

    /// The structural self-check rejects malformed slots — the panic-free
    /// contract for the vector section on hostile input.
    #[test]
    fn validate_rejects_malformed_slots() {
        let dim = 8;
        let mut pool = VecPool::new(dim, usize::MAX);
        pool.push(FactId(0), &vec![0.5; dim]).unwrap();
        let good = pool.dump();

        // A non-finite scale (bytes 4..8) is rejected.
        let mut bad = good.clone();
        bad[FACT_BYTES..HEAD].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(
            VecPool::from_parts(dim, usize::MAX, &bad)
                .unwrap()
                .validate()
                .is_err()
        );

        // A signature bit that disagrees with its component's sign is
        // rejected: flip one i8 component negative without touching sig.
        let mut bad = good.clone();
        let q_off = HEAD + VecPool::words(dim) * SIG_WORD_BYTES;
        bad[q_off] = (-1i8) as u8; // was positive (sig bit 0 set)
        assert!(
            VecPool::from_parts(dim, usize::MAX, &bad)
                .unwrap()
                .validate()
                .is_err()
        );
    }

    /// Overlay open: a pool over a borrowed base grows through an owned tail
    /// without touching the base, and every accessor (fact/scale/cosine/
    /// search) spans the base/tail boundary. The dump is byte-identical to
    /// the fully-owned pool holding the same slots.
    #[test]
    fn overlay_appends_to_tail_and_reads_span_the_boundary() {
        let dim = 16;
        let mut rng = Lcg(0x0ace_1a75);
        let (va, vb, vc) = (rng.vector(dim), rng.vector(dim), rng.vector(dim));

        // Fully-owned reference pool with all three vectors.
        let mut owned = VecPool::new(dim, usize::MAX);
        owned.push(FactId(10), &va).unwrap();
        owned.push(FactId(11), &vb).unwrap();
        owned.push(FactId(12), &vc).unwrap();

        // Base = first two vectors, serialized as if from an mmap; the third
        // is appended through the overlay open.
        let mut seed = VecPool::new(dim, usize::MAX);
        seed.push(FactId(10), &va).unwrap();
        seed.push(FactId(11), &vb).unwrap();
        let base = seed.dump();
        let base_snapshot = base.clone();

        // For this append-only store, the overlay open is `from_parts_borrowed`
        // (a borrowed base that a later `push` extends via the owned tail).
        let mut pool = VecPool::from_parts_borrowed(dim, usize::MAX, &base).unwrap();
        assert_eq!(pool.len(), 2);
        let idx = pool.push(FactId(12), &vc).unwrap();
        assert_eq!(idx, 2);
        assert_eq!(pool.len(), 3);

        // Accessors read base slots (0,1) and the tail slot (2) alike.
        assert_eq!(pool.slot_fact(0), 10); // base
        assert_eq!(pool.slot_fact(2), 12); // tail
        // A cosine between a base slot and the tail slot matches the owned
        // pool's — the overlay changes representation, not values.
        assert!((pool.cosine_slots(0, 2) - owned.cosine_slots(0, 2)).abs() < 1e-6);
        pool.validate().unwrap();

        // Search finds the appended (tail) vector by querying it exactly.
        let mut scratch = VecScratch::new();
        let mut out = Vec::new();
        pool.search(&vc, 1, &mut |_| true, &mut scratch, &mut out)
            .unwrap();
        assert_eq!(out[0].0, FactId(12));

        // The dump is canonical (== owned) and the borrowed base is untouched.
        assert_eq!(pool.dump(), owned.dump());
        assert_eq!(base, base_snapshot);
    }
}
