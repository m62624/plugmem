//! The thrown-error contract.
//!
//! Every failure this wrapper *decides* — an argument it refuses, a verb called
//! on the wrong kind of handle, a database another process holds — reaches JS as
//! an `Error` carrying a stable `code`, so a caller branches on
//! `err.code === 'PLUGMEM_LOCKED'` instead of matching on prose.
//!
//! The contract holds across the async boundary too, which takes one trick.
//! napi-rs' [`napi::Task`] fixes its error type to `napi::Error<Status>`, whose
//! status is a closed enum — a custom code cannot ride on it. But the promise
//! rejection path calls `JsError::into_value`, and that returns a *stored* JS
//! value verbatim when the value is a real `Error` object. So a task carries its
//! failure in its `Output` (the libuv thread has no `Env` and must not touch
//! JavaScript), and rebuilds it as a genuine JS `Error` with a `code` in
//! `resolve`, which runs back on the JS thread. See [`to_js`].
//!
//! The result is one rule with no seam in it:
//!
//! > every failure plugmem decides carries a `code`, whether the verb was
//! > synchronous or resolved a promise.

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
    /// Local contention: a handle is in use, or every workspace pool slot is
    /// leased by an active operation.
    pub const BUSY: &str = "PLUGMEM_BUSY";
    /// The engine reported a failure — IO, a capacity limit, a rejected input
    /// it alone can judge (a vector that disagrees with `dim`). The message is
    /// the host's own.
    pub const ENGINE: &str = "PLUGMEM_ENGINE";
    /// A bug in this binding: a result it could not shape into its typed
    /// mirror. Not a caller mistake and not the engine's fault.
    pub const INTERNAL: &str = "PLUGMEM_INTERNAL";
}

/// An error with one of the [`code`] values.
pub fn coded(code: &str, message: impl Into<String>) -> Error {
    Error::new(code.to_string(), message.into())
}

/// An argument the wrapper refused, naming the field and what was wrong.
pub fn invalid_arg(message: impl Into<String>) -> Error {
    coded(code::INVALID_ARG, message)
}

/// An engine failure from inside a verb: the host's own message, under the one
/// code that says "this was not your call, it was the engine".
pub fn engine(e: HostError) -> Error {
    coded(code::ENGINE, e.to_string())
}

/// A workspace failure from inside a verb.
///
/// An external writer lock is the same recoverable condition as a direct open;
/// pool capacity and an explicitly released active memory are local contention.
pub fn workspace(e: WorkspaceError) -> Error {
    let status = match &e {
        WorkspaceError::Busy { .. } | WorkspaceError::Host(HostError::Locked { .. }) => {
            code::LOCKED
        }
        WorkspaceError::AtCapacity { .. } | WorkspaceError::InUse { .. } => code::BUSY,
        _ => code::ENGINE,
    };
    coded(status, e.to_string())
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

/// A result this wrapper could not shape into its typed mirror — a bug here.
pub fn internal(message: impl Into<String>) -> Error {
    coded(code::INTERNAL, message)
}

/// Hand an error to a [`napi::Task`] boundary that has no `Env` — the libuv
/// thread. The code is preserved in the task's own `Output` and reattached by
/// [`to_js`]; this is only for the paths where there is nothing to reattach to.
pub fn into_napi(e: Error) -> napi::Error {
    napi::Error::from_reason(e.reason)
}

/// Rebuild a coded error as a real JavaScript `Error` carrying `code`.
///
/// Called from a task's `resolve`, on the JS thread, because it needs an `Env`
/// to make JS values. Returning the result as `Err` rejects the promise with
/// *this* object: napi stores it and `JsError::into_value` hands it back
/// untouched once it confirms the value is an `Error` (napi `error.rs`,
/// `impl_object_methods!`). That is what lets a custom code survive a boundary
/// whose own error type has no room for one.
///
/// If any of the JS calls fail — an environment shutting down — the message
/// still gets through, uncoded. A missing code is worse than a lost failure.
pub fn to_js(env: napi::Env, e: Error) -> napi::Error {
    let Ok(mut object) = env.create_error(napi::Error::from_reason(e.reason.clone())) else {
        return into_napi(e);
    };
    if object
        .set_named_property("code", e.status.as_str())
        .is_err()
    {
        return into_napi(e);
    }
    napi::Error::from(object.into_unknown())
}

/// What a libuv task produces: the value, or a coded failure for [`to_js`] to
/// turn into a JS `Error` once the JS thread is back.
pub type Produced<T> = std::result::Result<T, Error>;

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
    fn an_engine_failure_is_named_and_keeps_the_host_message() {
        let failed = engine(HostError::Embed("no route to host".into()));
        assert_eq!(failed.status, code::ENGINE);
        assert!(failed.reason.contains("no route to host"));
        // Distinguishable from a caller mistake, which is the whole point.
        assert_ne!(failed.status, code::INVALID_ARG);
    }

    #[test]
    fn refused_arguments_are_distinguishable_from_engine_failures() {
        let refused = invalid_arg("asOf must be a finite, non-negative number");
        assert_eq!(refused.status, code::INVALID_ARG);
        assert_ne!(refused.status, code::ENGINE);
    }

    #[test]
    fn workspace_contention_keeps_its_typed_code() {
        let name = plugmem_host::DbName::parse("chat").unwrap();
        assert_eq!(
            workspace(WorkspaceError::Busy { name: name.clone() }).status,
            code::LOCKED
        );
        assert_eq!(
            workspace(WorkspaceError::Host(HostError::Locked {
                path: "/tmp/registry.plugmem".into(),
            }))
            .status,
            code::LOCKED
        );
        assert_eq!(
            workspace(WorkspaceError::AtCapacity { max_open: 1 }).status,
            code::BUSY
        );
        assert_eq!(workspace(WorkspaceError::InUse { name }).status, code::BUSY);
    }
}
