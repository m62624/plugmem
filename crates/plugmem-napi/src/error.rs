//! The thrown-error contract.
//!
//! Every failure this wrapper *decides* — an argument it refuses, a verb called
//! on the wrong kind of handle, a database another process holds — reaches JS as
//! an `Error` carrying a stable `code`, so a caller branches on
//! `err.code === 'PLUGMEM_LOCKED'` instead of matching on prose.
//!
//! A failure the **engine** reports from inside a verb keeps napi's default
//! `GenericFailure` code and the host's own message. That is deliberate, not an
//! oversight: napi-rs' [`napi::Task`] fixes its error type to
//! `napi::Error<Status>`, so an async pass (`maintain`, `checkpoint`,
//! `rememberMany`, `reindex`, `verify`) physically cannot deliver a custom code.
//! Coding the synchronous half alone would make `code` mean one thing on
//! `remember` and another on `checkpoint` — a contract with an invisible seam is
//! worse than a narrower one that always holds. So the rule is:
//!
//! > `code` names why plugmem refused. An engine failure inside a verb is
//! > `GenericFailure` on every verb, sync or async.
//!
//! The open path is fully covered even so: a lock conflict, a missing snapshot
//! and a bad config are all decided while opening, which is always synchronous.

use napi::Error as NapiError;
use plugmem_host::{HostError, SettingsError, WorkspaceError};

/// The error every `#[napi]` method throws. The string status becomes the JS
/// `Error`'s `code`.
pub type Error = NapiError<String>;

/// The result of every `#[napi]` method.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable `code` values on a thrown error. These are API: match on them.
/// The accompanying message is for humans and may be reworded.
pub mod code {
    /// Another process holds the database's writer lock.
    pub const LOCKED: &str = "PLUGMEM_LOCKED";
    /// A read-only open needs a published snapshot and the writer has not
    /// checkpointed yet.
    pub const NEEDS_CHECKPOINT: &str = "PLUGMEM_NEEDS_CHECKPOINT";
    /// The `config.toml` could not be read, or a section of it is invalid.
    pub const CONFIG: &str = "PLUGMEM_CONFIG";
    /// Opening failed for any other reason — IO, or an image that will not load.
    pub const OPEN: &str = "PLUGMEM_OPEN";
    /// An argument was refused before it reached the engine.
    pub const INVALID_ARG: &str = "PLUGMEM_INVALID_ARG";
    /// A memory name is not a usable name.
    pub const INVALID_NAME: &str = "PLUGMEM_INVALID_NAME";
    /// The handle is closed — `close()` was called on it.
    pub const CLOSED: &str = "PLUGMEM_CLOSED";
    /// A write verb was called on a read-only handle.
    pub const READ_ONLY: &str = "PLUGMEM_READ_ONLY";
    /// A read-only-only verb (`generation`, `refresh`) was called on a writer.
    pub const WRITER_ONLY: &str = "PLUGMEM_WRITER_ONLY";
    /// The handle cannot do this right now because another operation on it is
    /// still running.
    pub const BUSY: &str = "PLUGMEM_BUSY";
}

/// napi's own code for an unclassified failure. Taken from napi rather than
/// spelled out, so the engine-error code can never drift from what the async
/// path (which cannot carry a custom one) produces.
fn generic_failure() -> String {
    napi::Status::GenericFailure.as_ref().to_string()
}

/// An error with one of the [`code`] values.
pub fn coded(code: &str, message: impl Into<String>) -> Error {
    Error::new(code.to_string(), message.into())
}

/// An argument the wrapper refused, naming the field and what was wrong.
pub fn invalid_arg(message: impl Into<String>) -> Error {
    coded(code::INVALID_ARG, message)
}

/// An engine failure from inside a verb: the host's message, and the same code
/// the async path produces (see the module note).
pub fn engine(e: HostError) -> Error {
    Error::new(generic_failure(), e.to_string())
}

/// A workspace failure from inside a verb, coded like [`engine`].
pub fn workspace(e: WorkspaceError) -> Error {
    Error::new(generic_failure(), e.to_string())
}

/// A failure while opening a database — always synchronous, so the caller gets
/// the kind as a code. `Locked` and `NeedsCheckpoint` are the two a program
/// actually branches on: retry later, or ask the writer to checkpoint.
pub fn open(e: HostError) -> Error {
    let code = match &e {
        HostError::Locked { .. } => code::LOCKED,
        HostError::NeedsCheckpoint { .. } => code::NEEDS_CHECKPOINT,
        _ => code::OPEN,
    };
    coded(code, e.to_string())
}

/// A config-resolution failure — a missing or invalid `config.toml`, or an
/// `[embedder]` section missing a required field.
pub fn settings(e: SettingsError) -> Error {
    coded(code::CONFIG, e.to_string())
}

/// A result this wrapper could not shape into its typed mirror. Not a caller
/// mistake and not the engine's fault — a bug here — so it carries no code of
/// its own and reads like any other unclassified failure.
pub fn internal(message: impl Into<String>) -> Error {
    Error::new(generic_failure(), message.into())
}

/// Hand an error to a [`napi::Task`] boundary, which cannot carry a custom code.
///
/// Only ever called with errors that are already `GenericFailure` — engine and
/// internal ones — so nothing is lost; a coded refusal is raised synchronously,
/// before any task is scheduled.
pub fn into_napi(e: Error) -> napi::Error {
    napi::Error::from_reason(e.reason)
}

/// A memory name the workspace refused.
pub fn invalid_name(e: WorkspaceError) -> Error {
    coded(code::INVALID_NAME, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_errors_carry_the_kind_a_caller_branches_on() {
        let locked = open(HostError::Locked {
            path: "/tmp/x.plugmem".into(),
        });
        assert_eq!(locked.status, code::LOCKED);
        assert!(!locked.reason.is_empty(), "the host message is preserved");

        let needs = open(HostError::NeedsCheckpoint {
            path: "/tmp/x.plugmem".into(),
        });
        assert_eq!(needs.status, code::NEEDS_CHECKPOINT);

        // Anything else that stops an open is still coded, just not classified.
        let io = open(HostError::Io {
            path: "/tmp/x.plugmem".into(),
            source: std::io::Error::other("disk gone"),
        });
        assert_eq!(io.status, code::OPEN);
    }

    #[test]
    fn engine_failures_use_the_code_the_async_path_cannot_avoid() {
        // The whole point of the rule in the module note: an engine failure has
        // the same code whether it surfaced from a sync verb or from a libuv
        // pass, and that code is whatever napi itself would have produced.
        let failure = || HostError::Embed("no route to host".into());
        let sync = engine(failure());
        // What a libuv task produces for the very same failure (`db::to_napi_err`).
        let asynchronous = napi::Error::from_reason(failure().to_string());
        assert_eq!(sync.status, asynchronous.status.as_ref());
        assert_eq!(sync.reason, asynchronous.reason);
        assert!(sync.reason.contains("no route to host"));
    }

    #[test]
    fn refused_arguments_are_distinguishable_from_engine_failures() {
        let refused = invalid_arg("asOf must be a finite, non-negative number");
        assert_eq!(refused.status, code::INVALID_ARG);
        assert_ne!(refused.status, generic_failure());
    }
}
