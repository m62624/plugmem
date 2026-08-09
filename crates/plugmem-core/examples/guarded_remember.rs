//! Reject a likely duplicate without creating a fact, then inspect the
//! candidates before deciding whether to revise, forget, or keep both.

use plugmem_core::{Config, GuardedRememberOutcome, MemStorage, Memory, RememberInput, Similar};

fn main() -> Result<(), plugmem_core::Error> {
    let mut storage = MemStorage::new();
    let mut memory = Memory::new(Config::default())?;

    memory.remember(
        &mut storage,
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(1_000, "prefers the tokio runtime")
        },
    )?;

    let decision = memory.remember_guarded(
        &mut storage,
        RememberInput {
            entity: Some("user"),
            ..RememberInput::text(2_000, "prefers tokio runtime")
        },
    )?;

    match decision {
        GuardedRememberOutcome::Stored { outcome } => {
            println!("stored fact {}", outcome.id.0);
        }
        GuardedRememberOutcome::Blocked { similar } => print_candidates(&similar),
    }

    assert_eq!(memory.stats().facts, 1);
    Ok(())
}

fn print_candidates(candidates: &[Similar]) {
    println!("blocked by {} similar fact(s)", candidates.len());
    for candidate in candidates {
        println!(
            "fact {}: score {:.3} ({:?})",
            candidate.id.0, candidate.score, candidate.reason
        );
    }
}
