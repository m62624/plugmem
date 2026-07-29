//! Typed identifiers.
//!
//! All ids are `u32`, allocated monotonically, never reused (journal replay
//! stays deterministic; holes after a purge are normal). `0` is a valid id;
//! "no value" is encoded as [`u32::MAX`] — every newtype exposes it as its
//! `NONE` constant. The 4.29-billion ceiling per kind is far above the 1M
//! capacity passport.

/// The raw "no value" sentinel shared by every id kind.
pub const NONE_U32: u32 = u32::MAX;

/// Declares one id newtype with the shared sentinel plumbing.
macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $name(pub u32);

        impl $name {
            /// The "no value" sentinel ([`u32::MAX`]).
            pub const NONE: Self = Self(NONE_U32);

            /// `true` when this is the [`Self::NONE`] sentinel.
            pub fn is_none(self) -> bool {
                self.0 == NONE_U32
            }

            /// `Some(self)` for a real id, `None` for the sentinel.
            pub fn some(self) -> Option<Self> {
                if self.is_none() { None } else { Some(self) }
            }

            /// Collapses an optional id back into the sentinel encoding.
            pub fn from_opt(opt: Option<Self>) -> Self {
                opt.unwrap_or(Self::NONE)
            }
        }
    };
}

id_type! {
    /// Identifier of a fact — the unit of memory.
    FactId
}

id_type! {
    /// Identifier of an entity — a graph node created lazily on first
    /// mention.
    EntityId
}
