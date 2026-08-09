//! The raised-exception contract.
//!
//! Every failure this wrapper *decides* — an argument it refuses, a verb called
//! on the wrong kind of handle, a database another process holds — reaches
//! Python as an exception class carrying a stable `code`, so a caller branches
//! on the type (or on `err.code`) instead of matching on prose.
//!
//! The codes are `plugmem-napi`'s codes, string for string. A program that
//! learned `PLUGMEM_LOCKED` from the Node binding reads the same value here,
//! and the cross-language documentation stays one table.
//!
//! **Single inheritance, no `ValueError` mixin.** Making `InvalidArgError` also
//! a `ValueError` is idiomatic in isolation, but it would put a second, softer
//! way to catch plugmem failures into a hierarchy whose whole point is that the
//! type names the cause. `except plugmem.InvalidArgError` says what it means;
//! `except ValueError` around a database call would also swallow a caller's own
//! unrelated bug.

use plugmem_host::{HostError, SettingsError, WorkspaceError};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3_stub_gen::create_exception;

create_exception!(
    plugmem._plugmem,
    PlugmemError,
    PyException,
    "Base class for every failure plugmem decides. Carries a stable `code`."
);

create_exception!(
    plugmem._plugmem,
    LockedError,
    PlugmemError,
    "Another process holds the database's writer lock."
);

create_exception!(
    plugmem._plugmem,
    NeedsCheckpointError,
    PlugmemError,
    "A read-only open needs a published snapshot and the writer has not \
     checkpointed yet."
);

create_exception!(
    plugmem._plugmem,
    ConfigError,
    PlugmemError,
    "The `config.toml` could not be read, or a section of it is invalid."
);

create_exception!(
    plugmem._plugmem,
    OpenError,
    PlugmemError,
    "Opening failed for any other reason — IO, or an image that will not load."
);

create_exception!(
    plugmem._plugmem,
    InvalidArgError,
    PlugmemError,
    "An argument was refused before it reached the engine."
);

create_exception!(
    plugmem._plugmem,
    InvalidNameError,
    PlugmemError,
    "A memory name is not a usable name."
);

create_exception!(
    plugmem._plugmem,
    ClosedError,
    PlugmemError,
    "The handle is closed — `close()` was called on it."
);

create_exception!(
    plugmem._plugmem,
    ReadOnlyError,
    PlugmemError,
    "A write verb was called on a read-only handle."
);

create_exception!(
    plugmem._plugmem,
    WriterOnlyError,
    PlugmemError,
    "A read-only-only verb (`generation`, `refresh`) was called on a writer."
);

create_exception!(
    plugmem._plugmem,
    BusyError,
    PlugmemError,
    "Local contention: a handle is in use, or every workspace pool slot is \
     leased by an active operation."
);

create_exception!(
    plugmem._plugmem,
    EngineError,
    PlugmemError,
    "The engine reported a failure — IO, a capacity limit, a rejected input it \
     alone can judge. The message is the host's own."
);

create_exception!(
    plugmem._plugmem,
    StaleCursorError,
    PlugmemError,
    "A tag page cursor no longer names the current catalogue snapshot."
);

// Nothing here raises `InternalError` today: every result is hand-mapped with
// the host type destructured field by field, so there is no "could not shape
// the result" path to fail on — the Node binding has one because it converts
// through serde. The class is still registered, because the code
// `PLUGMEM_INTERNAL` is part of the cross-language contract and a caller that
// branches on codes should not have to ask which binding it is talking to.
create_exception!(
    plugmem._plugmem,
    InternalError,
    PlugmemError,
    "A bug in this binding. Registered for parity with the other bindings' \
     `PLUGMEM_INTERNAL`; this one has no path that raises it."
);

/// Stable `code` values, readable as `err.code` and as a class attribute.
/// These are API: match on them. The accompanying message is for humans and
/// may be reworded.
pub mod code {
    /// Another process holds the database's writer lock.
    pub const LOCKED: &str = "PLUGMEM_LOCKED";
    /// A read-only open needs a published snapshot.
    pub const NEEDS_CHECKPOINT: &str = "PLUGMEM_NEEDS_CHECKPOINT";
    /// The `config.toml` could not be read, or a section of it is invalid.
    pub const CONFIG: &str = "PLUGMEM_CONFIG";
    /// Opening failed for any other reason.
    pub const OPEN: &str = "PLUGMEM_OPEN";
    /// An argument was refused before it reached the engine.
    pub const INVALID_ARG: &str = "PLUGMEM_INVALID_ARG";
    /// A memory name is not a usable name.
    pub const INVALID_NAME: &str = "PLUGMEM_INVALID_NAME";
    /// The handle is closed.
    pub const CLOSED: &str = "PLUGMEM_CLOSED";
    /// A write verb was called on a read-only handle.
    pub const READ_ONLY: &str = "PLUGMEM_READ_ONLY";
    /// A read-only-only verb was called on a writer.
    pub const WRITER_ONLY: &str = "PLUGMEM_WRITER_ONLY";
    /// Local handle or workspace-pool contention.
    pub const BUSY: &str = "PLUGMEM_BUSY";
    /// A tag page cursor is stale after the catalogue changed.
    pub const STALE_CURSOR: &str = "PLUGMEM_STALE_CURSOR";
    /// The engine reported a failure.
    pub const ENGINE: &str = "PLUGMEM_ENGINE";
    /// A bug in this binding.
    pub const INTERNAL: &str = "PLUGMEM_INTERNAL";
}

/// The result of every method on this binding.
pub type Result<T> = std::result::Result<T, PyErr>;

/// An argument the wrapper refused, naming the field and what was wrong.
pub fn invalid_arg(message: impl Into<String>) -> PyErr {
    InvalidArgError::new_err(message.into())
}

/// A memory name the workspace refused.
pub fn invalid_name(e: WorkspaceError) -> PyErr {
    InvalidNameError::new_err(e.to_string())
}

/// An engine failure from inside a verb: the host's own message, under the one
/// class that says "this was not your call, it was the engine".
pub fn engine(e: HostError) -> PyErr {
    match &e {
        HostError::Engine(plugmem_host::Error::StaleCursor) => {
            StaleCursorError::new_err(e.to_string())
        }
        HostError::ReembedBusy => BusyError::new_err(e.to_string()),
        _ => EngineError::new_err(e.to_string()),
    }
}

/// A workspace failure from inside a verb.
pub fn workspace(e: WorkspaceError) -> PyErr {
    match &e {
        WorkspaceError::Busy { .. } | WorkspaceError::Host(HostError::Locked { .. }) => {
            LockedError::new_err(e.to_string())
        }
        WorkspaceError::AtCapacity { .. } | WorkspaceError::InUse { .. } => {
            BusyError::new_err(e.to_string())
        }
        _ => EngineError::new_err(e.to_string()),
    }
}

/// A failure while opening a database. `Locked` and `NeedsCheckpoint` are the
/// two a program actually branches on: retry later, or ask the writer to
/// checkpoint — so they get their own classes and the rest share `OpenError`.
pub fn open(e: HostError) -> PyErr {
    match &e {
        HostError::Locked { .. } => LockedError::new_err(e.to_string()),
        HostError::NeedsCheckpoint { .. } => NeedsCheckpointError::new_err(e.to_string()),
        _ => OpenError::new_err(e.to_string()),
    }
}

/// A config-resolution failure — a missing or invalid `config.toml`, or an
/// `[embedder]` section missing a required field.
pub fn settings(e: SettingsError) -> PyErr {
    ConfigError::new_err(e.to_string())
}

/// The handle was closed by `close()`.
pub fn closed() -> PyErr {
    ClosedError::new_err("memory is closed")
}

/// A write verb on a read-only handle.
pub fn read_only() -> PyErr {
    ReadOnlyError::new_err("memory is open read-only: this verb needs the writer")
}

/// A read-only-only verb on a writer, naming the verb so the message says what
/// to do instead.
pub fn writer_only(verb: &str) -> PyErr {
    WriterOnlyError::new_err(format!(
        "{verb} is a read-only verb: this handle is the writer, which always \
         sees its own latest state"
    ))
}

/// This handle is busy with another operation. Reachable when a second thread
/// calls a verb while one holds the handle's own lock in a way that cannot
/// wait — see the `Scrub` cursor.
pub fn busy(what: &str) -> PyErr {
    BusyError::new_err(format!("{what} is in use by another operation"))
}

/// Add every exception class to the module, each carrying its `code` as a class
/// attribute so `err.code` reads the same string the Node binding puts on a
/// thrown `Error`.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let classes: [(&str, Bound<'_, PyAny>, Option<&str>); 14] = [
        (
            "PlugmemError",
            py.get_type::<PlugmemError>().into_any(),
            None,
        ),
        (
            "LockedError",
            py.get_type::<LockedError>().into_any(),
            Some(code::LOCKED),
        ),
        (
            "NeedsCheckpointError",
            py.get_type::<NeedsCheckpointError>().into_any(),
            Some(code::NEEDS_CHECKPOINT),
        ),
        (
            "ConfigError",
            py.get_type::<ConfigError>().into_any(),
            Some(code::CONFIG),
        ),
        (
            "OpenError",
            py.get_type::<OpenError>().into_any(),
            Some(code::OPEN),
        ),
        (
            "InvalidArgError",
            py.get_type::<InvalidArgError>().into_any(),
            Some(code::INVALID_ARG),
        ),
        (
            "InvalidNameError",
            py.get_type::<InvalidNameError>().into_any(),
            Some(code::INVALID_NAME),
        ),
        (
            "ClosedError",
            py.get_type::<ClosedError>().into_any(),
            Some(code::CLOSED),
        ),
        (
            "ReadOnlyError",
            py.get_type::<ReadOnlyError>().into_any(),
            Some(code::READ_ONLY),
        ),
        (
            "WriterOnlyError",
            py.get_type::<WriterOnlyError>().into_any(),
            Some(code::WRITER_ONLY),
        ),
        (
            "BusyError",
            py.get_type::<BusyError>().into_any(),
            Some(code::BUSY),
        ),
        (
            "EngineError",
            py.get_type::<EngineError>().into_any(),
            Some(code::ENGINE),
        ),
        (
            "StaleCursorError",
            py.get_type::<StaleCursorError>().into_any(),
            Some(code::STALE_CURSOR),
        ),
        (
            "InternalError",
            py.get_type::<InternalError>().into_any(),
            Some(code::INTERNAL),
        ),
    ];

    for (name, class, code) in classes {
        // The base carries `code = None` rather than nothing at all: a caller
        // that reads `err.code` on a subclass it does not know about gets a
        // missing value, not an `AttributeError`.
        match code {
            Some(value) => class.setattr("code", value)?,
            None => class.setattr("code", py.None())?,
        }
        module.add(name, class)?;
    }
    Ok(())
}
