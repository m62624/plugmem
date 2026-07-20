//! One-off measurement helper: mean `remember` cost on the testgen
//! corpus (used to source the README number; not a benchmark target).
//!
//! Native-only: `plugmem-testgen` is a dev-dependency gated off wasm
//! targets, so on wasm this example compiles to an empty stub.

#[cfg(not(target_family = "wasm"))]
fn main() {
    use plugmem_core::{Config, MemStorage, Memory};
    use plugmem_testgen::{Gen, GenOp, Profile, apply};

    let profile = Profile {
        dim: 384,
        w_revise: 0,
        w_forget: 0,
        w_link: 0,
        ..Profile::default()
    };
    let ops = Gen::new(1, profile).ops(20_000);
    let mut cfg = Config::default();
    cfg.dim = 384;
    let mut mem = Memory::new(cfg).unwrap();
    let mut store = MemStorage::new();
    // Warm structures with the first half; time the second half.
    let (a, b) = ops.split_at(10_000);
    for op in a {
        apply(&mut mem, &mut store, op).unwrap();
    }
    let t0 = std::time::Instant::now();
    for op in b {
        apply(&mut mem, &mut store, op).unwrap();
    }
    let per = t0.elapsed() / b.len() as u32;
    let with_entity = b
        .iter()
        .filter(|o| {
            matches!(
                o,
                GenOp::Remember {
                    entity: Some(_),
                    ..
                }
            )
        })
        .count();
    println!(
        "remember mean: {per:?} over {} ops ({with_entity} with entity)",
        b.len()
    );
}

#[cfg(target_family = "wasm")]
fn main() {}
