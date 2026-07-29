//! The generator's own deterministic PRNG: xorshift64*,
//! implemented in place so the corpus depends on nothing but this crate.
//!
//! Not a statistical-quality RNG and not meant to be one — corpus
//! generation needs *reproducibility* (same seed, same bytes, forever)
//! more than it needs randomness quality, and owning the implementation
//! guarantees no dependency upgrade can ever shift the streams.

/// xorshift64* generator state.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Creates a generator from a seed (a zero seed is remapped — the
    /// xorshift state must never be zero).
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform integer in `0..n`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` (an empty range is a caller bug).
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "empty range");
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `true` with probability `p` (clamped to `[0, 1]`).
    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Standard normal sample (Box-Muller; one branch of the pair —
    /// simplicity over throughput, this is a test-data generator).
    pub fn normal(&mut self) -> f64 {
        // f64() is in [0, 1); shift away from 0 so ln is finite.
        let u1 = 1.0 - self.f64();
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}
