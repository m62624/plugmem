//! A workspace: many databases under one directory, addressed by name.
//!
//! **The default is one database.** Nothing here switches on by itself — a
//! caller that opens a [`crate::Database`] at a path behaves exactly as it did
//! before this module existed. A workspace is what you reach for when one
//! process serves many independent memories (a chat per conversation, a
//! database per tenant) and wants them to stay independent.
//!
//! The layout is deliberately mechanical:
//!
//! ```text
//! <root>/registry.plugmem      the registry — an ordinary plugmem database
//! <root>/db/<name>.plugmem     the databases themselves
//! ```
//!
//! Two properties fall out of it, and both are the point:
//!
//! - **a name is not a path, and cannot become one.** [`DbName`] admits only
//!   `[a-z0-9][a-z0-9_-]*`, so `..`, `/`, a drive letter or an absolute path
//!   are not filtered out — they are unrepresentable. Resolution is then a
//!   join, with nothing to get wrong;
//! - **the directory is the truth.** [`WorkspaceLayout::list`] reads the
//!   filesystem, never the registry. The registry (see the `registry` module)
//!   is a searchable index over descriptions and can be rebuilt from the
//!   databases themselves; losing it costs search, not data.
//!
//! The registry lives in the root while the databases live one level down, so
//! no name can ever collide with it.

mod registry;

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use crate::{Database, HostError};

pub use registry::{
    ARCHIVED_TAG, DbEntry, Description, ENTRY_TAG, ReindexReport, SELF_ENTITY, WorkspaceIssue,
};

/// Longest database name a workspace accepts, in bytes.
///
/// Names are ASCII, so this is also the character count. The limit exists so a
/// name plus the extension stays comfortably inside the shortest filename limit
/// worth caring about (255 bytes on ext4/APFS/NTFS) with room for the sidecar
/// suffixes (`.lock`, `.jrnl`, `.snap.N`) the storage layer appends.
pub const MAX_DB_NAME: usize = 64;

/// Directory, below the workspace root, holding the databases.
const DB_DIR: &str = "db";

/// The registry's file name, directly in the workspace root.
const REGISTRY_FILE: &str = "registry.plugmem";

/// Extension of a database file. The storage layer appends its sidecar
/// suffixes *after* this (`chat-42.plugmem.lock`), so matching on the
/// extension picks out base files and nothing else.
const DB_EXT: &str = "plugmem";

/// Why a string is not a usable database name.
///
/// A typed reason rather than a message, so a caller (and a test) can react to
/// the specific problem instead of matching on prose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NameProblem {
    /// The name was empty.
    Empty,
    /// The name was longer than [`MAX_DB_NAME`].
    TooLong,
    /// The first character was neither a lowercase ASCII letter nor a digit.
    /// Leading `-`, `_` and `.` are refused so a name can never be read as a
    /// flag or as a relative path component.
    LeadingChar,
    /// Some character was outside `[a-z0-9_-]`.
    Character,
}

impl fmt::Display for NameProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("it is empty"),
            Self::TooLong => write!(f, "it is longer than {MAX_DB_NAME} bytes"),
            Self::LeadingChar => f.write_str("it must start with a lowercase letter or a digit"),
            Self::Character => {
                f.write_str("it may hold only lowercase letters, digits, '-' and '_'")
            }
        }
    }
}

/// A validated database name — the only thing a workspace resolves.
///
/// Construct it with [`DbName::parse`]; there is no other way in, which is what
/// makes "a name is not a path" a property of the type rather than a rule
/// somebody has to remember at every call site.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DbName(String);

impl DbName {
    /// Validates `s` as a database name.
    ///
    /// The rule: ASCII, first character `[a-z0-9]`, the rest `[a-z0-9_-]`,
    /// length `1..=`[`MAX_DB_NAME`].
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::BadName`] carrying the specific [`NameProblem`].
    pub fn parse(s: &str) -> Result<Self, WorkspaceError> {
        let bad = |why| {
            Err(WorkspaceError::BadName {
                name: s.to_string(),
                why,
            })
        };
        let Some(&first) = s.as_bytes().first() else {
            return bad(NameProblem::Empty);
        };
        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return bad(NameProblem::LeadingChar);
        }
        if !s.bytes().all(is_name_byte) {
            return bad(NameProblem::Character);
        }
        if s.len() > MAX_DB_NAME {
            return bad(NameProblem::TooLong);
        }
        Ok(DbName(s.to_string()))
    }

    /// The name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DbName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// True for a byte allowed anywhere in a name.
fn is_name_byte(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_'
}

/// Every way a workspace operation can fail.
///
/// Opening one database still fails as a [`HostError`]; this type adds the
/// failures that only exist once databases are addressed by name.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WorkspaceError {
    /// The string is not a usable database name.
    #[error("{name:?} is not a usable database name: {why}")]
    BadName {
        /// The rejected string, as given.
        name: String,
        /// What specifically was wrong with it.
        why: NameProblem,
    },

    /// No database of that name exists, and the caller asked not to create one.
    #[error("no database named {name} in this workspace (looked for {})", path.display())]
    NoSuchDatabase {
        /// The name that did not resolve.
        name: DbName,
        /// Where it would have been.
        path: PathBuf,
    },

    /// The database is open for writing elsewhere.
    ///
    /// Distinct from [`HostError::Locked`] so the message can name the database
    /// rather than a path the caller never typed. One file has one writer, and
    /// in a workspace the other writer is usually a long-running sidecar that
    /// will release it once the handle goes idle.
    #[error(
        "database {name} is in use by another process; it is released once that process closes it (a pooled handle does so after its idle timeout)"
    )]
    Busy {
        /// The database that is held elsewhere.
        name: DbName,
    },

    /// A filesystem operation on the workspace itself failed.
    #[error("i/o on {}: {source}", path.display())]
    Io {
        /// The path the operation touched.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// Opening or using one of the databases failed.
    #[error(transparent)]
    Host(#[from] HostError),
}

impl WorkspaceError {
    /// Shorthand for wrapping an I/O error with its path.
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Where a workspace keeps its files.
///
/// Pure path arithmetic and one directory listing — it opens nothing and locks
/// nothing. A caller that only needs "which file is `work`?" (the CLI resolving
/// `--db work`) wants this and not the handle pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceLayout {
    root: PathBuf,
}

impl WorkspaceLayout {
    /// A layout rooted at `root`. Creates nothing; the directories appear when
    /// a database is first written.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory holding the databases, `<root>/db`.
    pub fn db_dir(&self) -> PathBuf {
        self.root.join(DB_DIR)
    }

    /// The file backing `name`, `<root>/db/<name>.plugmem`.
    ///
    /// A join of validated components, so the result is always inside
    /// [`WorkspaceLayout::db_dir`].
    pub fn path_of(&self, name: &DbName) -> PathBuf {
        self.db_dir().join(format!("{}.{DB_EXT}", name.0))
    }

    /// The registry's file, `<root>/registry.plugmem`. It sits in the root
    /// rather than beside the databases, so no name can collide with it.
    pub fn registry_path(&self) -> PathBuf {
        self.root.join(REGISTRY_FILE)
    }

    /// Whether `name` is a database on disk.
    ///
    /// Asks the storage layer rather than stat-ing the path: a database that
    /// has been written to but not yet checkpointed has no file at its base
    /// path, and treating it as absent would mean creating over it.
    pub fn exists(&self, name: &DbName) -> bool {
        crate::storage::database_exists(&self.path_of(name))
    }

    /// Every database in the workspace, sorted by name.
    ///
    /// Reads the directory — the filesystem is the truth, the registry is only
    /// an index over it. A missing `db/` is an empty workspace, not an error.
    /// Files whose name is not a database's are skipped here and reported by
    /// the registry's `verify`, which is where a person is asking about
    /// consistency rather than about what they can open.
    ///
    /// One database is several files (`chat-42.plugmem`, `.journal`, `.lock`,
    /// `.snap.N`), so the listing folds them back to one name each: a name
    /// cannot contain a dot, so everything up to the first dot is the candidate,
    /// and the storage layer confirms whether a database is really there.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::Io`] if the directory exists but cannot be read.
    pub fn list(&self) -> Result<Vec<DbName>, WorkspaceError> {
        let dir = self.db_dir();
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(WorkspaceError::io(&dir, e)),
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| WorkspaceError::io(&dir, e))?;
            let file_name = entry.file_name();
            let Some(candidate) = file_name.to_str().and_then(|n| n.split('.').next()) else {
                continue;
            };
            if let Ok(name) = DbName::parse(candidate)
                && !names.contains(&name)
                && self.exists(&name)
            {
                names.push(name);
            }
        }
        names.sort_unstable();
        Ok(names)
    }
}

/// How many databases a workspace keeps open, and for how long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceLimits {
    /// Most databases held open at once. The least recently used is closed to
    /// make room. `0` is read as `1` — a pool that holds nothing would reopen
    /// on every call.
    pub max_open: usize,
    /// How long a database may sit unused before [`Workspace::close_idle`]
    /// closes it. `0` disables the sweep, so handles stay until evicted.
    ///
    /// This is not a memory knob, it is a *liveness* knob: an open writer holds
    /// the file's exclusive lock, so nothing else on the machine can touch that
    /// database until the handle goes. A long-running server that never let go
    /// would make its databases permanently unreachable from the CLI.
    pub idle_timeout_ms: u64,
}

/// Handles kept open by default. Small on purpose: reopening a database the
/// size of a chat is milliseconds, so caching is a latency win, not a
/// requirement, and every extra handle is a lock somebody else cannot take.
pub const DEFAULT_MAX_OPEN: usize = 16;

/// Default idle window before a pooled handle is closed.
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 60_000;

impl Default for WorkspaceLimits {
    fn default() -> Self {
        Self {
            max_open: DEFAULT_MAX_OPEN,
            idle_timeout_ms: DEFAULT_IDLE_TIMEOUT_MS,
        }
    }
}

/// Whether resolving a name that has no file yet creates one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IfMissing {
    /// Create the database. What makes a workspace usable without a
    /// registration step: a new chat writes, and its memory exists.
    Create,
    /// Fail with [`WorkspaceError::NoSuchDatabase`]. What reads should do, so a
    /// misspelled name is diagnosed instead of silently answering nothing.
    Fail,
}

/// Opens one database at a path. Supplied by the caller so this module stays
/// out of the business of configuration — [`crate::Settings`] builds one that
/// applies the engine config, the maintenance policy and a shared embedder.
pub type Opener = Box<dyn Fn(&Path) -> Result<Database, HostError> + Send + Sync>;

/// One open database and when it was last handed out.
struct Pooled {
    name: DbName,
    db: Database,
    last_used_ms: u64,
}

/// Many databases in one directory, opened on demand and kept open for a while.
///
/// The pool is a flat `Vec` rather than a map: `max_open` is a handful, and a
/// linear scan of a handful is cheaper than hashing — the same reason the
/// engine's own structures stay flat.
///
/// **A returned [`Database`] outlives its pool entry.** The handle is an `Arc`,
/// so eviction drops the pool's copy and nothing else; the file lock is
/// released when the *last* clone goes. A caller that parks a handle for hours
/// keeps the database locked for hours, whatever the idle timeout says. Hold
/// one for the length of a request, as the MCP server does, and this never
/// comes up.
pub struct Workspace {
    layout: WorkspaceLayout,
    open: Opener,
    limits: WorkspaceLimits,
    pool: Mutex<Vec<Pooled>>,
    registry: Mutex<Option<Database>>,
}

impl Workspace {
    /// A workspace over `layout`, opening databases with `open`.
    pub fn new(layout: WorkspaceLayout, open: Opener, limits: WorkspaceLimits) -> Self {
        Self {
            layout,
            open,
            limits,
            pool: Mutex::new(Vec::new()),
            registry: Mutex::new(None),
        }
    }

    /// The registry database, opened on first use.
    ///
    /// Lazily, and that matters: a process that only ever resolves names it was
    /// given never opens the registry, so it neither creates the file nor holds
    /// a lock on it. The registry is a search index — a caller that is not
    /// searching should not pay for it, and two processes that never search can
    /// share one workspace without contending over it.
    ///
    /// It lives outside the handle pool because it is not one of the databases:
    /// it has no [`DbName`], it is never evicted, and it is never handed out by
    /// [`Workspace::get`].
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::Io`] if the root cannot be created, or whatever the
    /// open reports — including [`HostError::Locked`] if another process holds
    /// the registry.
    pub fn registry(&self) -> Result<Database, WorkspaceError> {
        let mut slot = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(db) = slot.as_ref() {
            return Ok(db.clone());
        }
        let root = self.layout.root();
        std::fs::create_dir_all(root).map_err(|e| WorkspaceError::io(root, e))?;
        let db = (self.open)(&self.layout.registry_path())?;
        *slot = Some(db.clone());
        Ok(db)
    }

    /// Closes the registry handle, if one is open. Returns whether there was
    /// one. The same liveness concern as [`Workspace::close_idle`]: a held
    /// registry is a registry no other process can write.
    pub fn close_registry(&self) -> bool {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .is_some()
    }

    /// Where the files are.
    pub fn layout(&self) -> &WorkspaceLayout {
        &self.layout
    }

    /// The limits in force.
    pub fn limits(&self) -> WorkspaceLimits {
        self.limits
    }

    /// How many databases are open right now. Observability for tests and
    /// `stats`; not a number to make decisions on, since it moves.
    pub fn open_count(&self) -> usize {
        self.pooled().len()
    }

    /// Resolves `name` to an open database, opening it if it is not pooled.
    ///
    /// `now_ms` is the host clock (unix milliseconds), used only for the idle
    /// bookkeeping — it is passed in rather than read here for the same reason
    /// every verb takes `now`: the host owns time.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::NoSuchDatabase`] when the file is absent and `missing`
    /// is [`IfMissing::Fail`]; [`WorkspaceError::Busy`] when another process
    /// holds the writer; [`WorkspaceError::Io`] if the directory cannot be
    /// created; [`WorkspaceError::Host`] for anything the open itself rejects.
    pub fn get(
        &self,
        name: &DbName,
        now_ms: u64,
        missing: IfMissing,
    ) -> Result<Database, WorkspaceError> {
        // The lock is held across the open, deliberately. Releasing it first
        // would let two callers race to open the same file, and the loser would
        // see a spurious `Busy` — from its own process. An open is bounded
        // (the lock is tried, never waited on), so the cost is a short queue.
        let mut pool = self.pooled();

        if let Some(slot) = pool.iter_mut().find(|p| &p.name == name) {
            slot.last_used_ms = now_ms;
            return Ok(slot.db.clone());
        }

        let path = self.layout.path_of(name);
        if !self.layout.exists(name) {
            if missing == IfMissing::Fail {
                return Err(WorkspaceError::NoSuchDatabase {
                    name: name.clone(),
                    path,
                });
            }
            let dir = self.layout.db_dir();
            std::fs::create_dir_all(&dir).map_err(|e| WorkspaceError::io(&dir, e))?;
        }

        // Make room *before* opening, so the ceiling counts this database too.
        let ceiling = self.limits.max_open.max(1);
        while pool.len() >= ceiling {
            let lru = Self::least_recently_used(&pool);
            pool.remove(lru);
        }

        let db = (self.open)(&path).map_err(|e| match e {
            HostError::Locked { .. } => WorkspaceError::Busy { name: name.clone() },
            other => WorkspaceError::Host(other),
        })?;
        pool.push(Pooled {
            name: name.clone(),
            db: db.clone(),
            last_used_ms: now_ms,
        });
        Ok(db)
    }

    /// Closes every database unused for longer than the idle timeout, returning
    /// how many were closed. A no-op when the timeout is `0`.
    ///
    /// Call it on a timer. Nothing else releases a file lock a server is
    /// holding on a database nobody is asking about.
    pub fn close_idle(&self, now_ms: u64) -> usize {
        let timeout = self.limits.idle_timeout_ms;
        if timeout == 0 {
            return 0;
        }
        let mut pool = self.pooled();
        let before = pool.len();
        // `saturating_sub`: a clock that stepped backwards leaves handles open
        // rather than closing all of them at once.
        pool.retain(|p| now_ms.saturating_sub(p.last_used_ms) < timeout);
        before - pool.len()
    }

    /// Closes every open handle. The pool's copies, that is — see the note on
    /// [`Workspace`] about clones the caller still holds.
    pub fn close_all(&self) -> usize {
        let mut pool = self.pooled();
        std::mem::take(&mut *pool).len()
    }

    /// The pool guard. A panic in a verb cannot leave the pool half-updated
    /// (every mutation here is a single push, remove or retain), so a poisoned
    /// lock is recovered — the same rule the engine lock follows.
    fn pooled(&self) -> MutexGuard<'_, Vec<Pooled>> {
        self.pool.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Index of the least recently used entry. The pool is never empty here.
    fn least_recently_used(pool: &[Pooled]) -> usize {
        let mut oldest = 0;
        for (i, p) in pool.iter().enumerate() {
            if p.last_used_ms < pool[oldest].last_used_ms {
                oldest = i;
            }
        }
        oldest
    }
}

impl fmt::Debug for Workspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Workspace")
            .field("root", &self.layout.root())
            .field("limits", &self.limits)
            .field("open", &self.open_count())
            .finish()
    }
}

/// Fixtures shared by this module's tests and the registry's.
#[cfg(test)]
pub(crate) mod testkit {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{DbName, Opener, Workspace, WorkspaceLayout, WorkspaceLimits};
    use crate::Database;

    /// A unique temp directory; removed on drop.
    pub(crate) struct TempDir(pub PathBuf);

    impl TempDir {
        pub(crate) fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-workspace-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A workspace of plain default databases, plus a count of how many times
    /// the opener actually ran — the only way to tell a pool hit from a reopen.
    pub(crate) fn workspace(
        tmp: &TempDir,
        limits: WorkspaceLimits,
    ) -> (Workspace, Arc<AtomicUsize>) {
        let opens = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&opens);
        let open: Opener = Box::new(move |path: &std::path::Path| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(Database::open(path, crate::Config::default())?.0)
        });
        (
            Workspace::new(WorkspaceLayout::new(&tmp.0), open, limits),
            opens,
        )
    }

    /// A name that must parse.
    pub(crate) fn name(s: &str) -> DbName {
        DbName::parse(s).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::testkit::{TempDir, name, workspace};
    use super::*;

    fn problem(s: &str) -> NameProblem {
        match DbName::parse(s) {
            Err(WorkspaceError::BadName { why, .. }) => why,
            other => panic!("expected {s:?} to be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_name_admits_only_the_safe_alphabet() {
        for ok in [
            "a",
            "0",
            "chat-42",
            "common",
            "x_y-9",
            &"a".repeat(MAX_DB_NAME),
        ] {
            assert_eq!(DbName::parse(ok).unwrap().as_str(), ok);
        }

        assert_eq!(problem(""), NameProblem::Empty);
        assert_eq!(problem(&"a".repeat(MAX_DB_NAME + 1)), NameProblem::TooLong);

        // A leading character that could be read as a flag, a path component or
        // a hidden file is refused before anything else looks at the string.
        // A non-ASCII name trips this same check, because its first *byte* is
        // already outside the alphabet.
        for bad in ["-x", "_x", ".x", "..", "/x", "Ab", "чат"] {
            assert_eq!(problem(bad), NameProblem::LeadingChar, "{bad:?}");
        }

        // Path separators, dots, spaces, uppercase and non-ASCII are not
        // filtered late — they simply are not names.
        for bad in ["a/b", "a\\b", "a.b", "a b", "aB", "a:b", "aчат", "a\0b"] {
            assert_eq!(problem(bad), NameProblem::Character, "{bad:?}");
        }
    }

    #[test]
    fn a_name_prints_as_itself() {
        assert_eq!(DbName::parse("chat-42").unwrap().to_string(), "chat-42");
        assert_eq!(NameProblem::Empty.to_string(), "it is empty");
        assert_eq!(
            NameProblem::TooLong.to_string(),
            format!("it is longer than {MAX_DB_NAME} bytes")
        );
        assert!(NameProblem::LeadingChar.to_string().contains("start with"));
        assert!(NameProblem::Character.to_string().contains("lowercase"));
    }

    #[test]
    fn the_layout_puts_the_registry_out_of_reach_of_names() {
        let layout = WorkspaceLayout::new("/ws");
        let name = DbName::parse("chat-42").unwrap();

        assert_eq!(layout.root(), Path::new("/ws"));
        assert_eq!(layout.db_dir(), Path::new("/ws/db"));
        assert_eq!(layout.path_of(&name), Path::new("/ws/db/chat-42.plugmem"));
        assert_eq!(layout.registry_path(), Path::new("/ws/registry.plugmem"));

        // The registry is one level above the databases, so no name — not even
        // one spelled like the registry file — can resolve onto it.
        let lookalike = DbName::parse("registry").unwrap();
        assert_ne!(layout.path_of(&lookalike), layout.registry_path());
    }

    #[test]
    fn listing_reads_the_directory_and_ignores_what_is_not_a_database() {
        let tmp = TempDir::new("list");
        let layout = WorkspaceLayout::new(&tmp.0);

        // A workspace nobody has written to yet lists nothing rather than failing.
        assert!(layout.list().unwrap().is_empty());

        std::fs::create_dir_all(layout.db_dir()).unwrap();
        for file in [
            "chat-42.plugmem",
            "common.plugmem",
            // Sidecars of a database already counted: folded back to its name,
            // not listed again.
            "chat-42.plugmem.lock",
            "chat-42.plugmem.journal",
            "chat-42.plugmem.snap.3",
            // Neither is an unrelated file, nor one whose stem is not a name.
            "notes.txt",
            "Chat-43.plugmem",
        ] {
            std::fs::write(layout.db_dir().join(file), b"").unwrap();
        }

        let names: Vec<String> = layout
            .list()
            .unwrap()
            .iter()
            .map(DbName::to_string)
            .collect();
        assert_eq!(names, ["chat-42", "common"]);

        assert!(layout.exists(&DbName::parse("chat-42").unwrap()));
        assert!(!layout.exists(&DbName::parse("nope").unwrap()));
    }

    #[test]
    fn a_database_exists_before_its_first_checkpoint() {
        let tmp = TempDir::new("list-uncheckpointed");
        let layout = WorkspaceLayout::new(&tmp.0);
        let fresh = DbName::parse("fresh").unwrap();
        std::fs::create_dir_all(layout.db_dir()).unwrap();

        // The base path holds the published snapshot, and that is written by
        // the first checkpoint — so a brand-new database has a journal and no
        // base file. Reporting it as absent would mean creating over live data.
        let (db, _) = Database::open(layout.path_of(&fresh), crate::Config::default()).unwrap();
        db.remember(crate::RememberInput::text(1_000, "not yet checkpointed"))
            .unwrap();
        assert!(!layout.path_of(&fresh).exists());
        assert!(layout.exists(&fresh));
        assert_eq!(layout.list().unwrap(), [fresh]);
    }

    #[test]
    fn an_unreadable_directory_is_an_error_not_an_empty_workspace() {
        let tmp = TempDir::new("list-io");
        let layout = WorkspaceLayout::new(&tmp.0);
        // `db` is a *file*, so reading it as a directory fails with something
        // other than NotFound — the caller must hear about it.
        std::fs::write(layout.db_dir(), b"not a directory").unwrap();
        assert!(matches!(layout.list(), Err(WorkspaceError::Io { .. })));
    }

    #[test]
    fn every_failure_names_what_the_caller_typed() {
        let busy = WorkspaceError::Busy {
            name: DbName::parse("chat-42").unwrap(),
        };
        assert!(busy.to_string().contains("chat-42"));

        let host = WorkspaceError::from(HostError::Embed("no".into()));
        assert!(matches!(host, WorkspaceError::Host(HostError::Embed(_))));

        let io = WorkspaceError::io(
            Path::new("/ws"),
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        assert!(io.to_string().contains("/ws"));

        let missing = WorkspaceError::NoSuchDatabase {
            name: DbName::parse("gone").unwrap(),
            path: PathBuf::from("/ws/db/gone.plugmem"),
        };
        assert!(missing.to_string().contains("gone"));
    }

    #[test]
    fn a_pooled_database_is_reused_and_a_missing_one_is_created_only_on_request() {
        let tmp = TempDir::new("pool-reuse");
        let (ws, opens) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");

        // Reading a name nobody has written names the name, not a path the
        // caller never typed.
        let missed = ws.get(&chat, 1_000, IfMissing::Fail).unwrap_err();
        assert!(
            matches!(&missed, WorkspaceError::NoSuchDatabase { name, .. } if name == &chat),
            "{missed}"
        );
        assert_eq!(opens.load(Ordering::SeqCst), 0);
        assert!(!ws.layout().db_dir().exists());

        // Writing creates it, directory and all.
        let db = ws.get(&chat, 1_000, IfMissing::Create).unwrap();
        db.remember(crate::RememberInput::text(1_000, "prefers tokio"))
            .unwrap();
        assert!(ws.layout().exists(&chat));
        assert_eq!(ws.open_count(), 1);

        // The second call is a pool hit: same state, no reopen.
        let again = ws.get(&chat, 2_000, IfMissing::Fail).unwrap();
        assert_eq!(again.stats().facts, 1);
        assert_eq!(opens.load(Ordering::SeqCst), 1);

        assert!(format!("{ws:?}").contains("open: 1"));
    }

    #[test]
    fn databases_in_one_workspace_do_not_see_each_other() {
        let tmp = TempDir::new("isolation");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());

        for (db, text) in [
            ("chat-42", "the sky is blue"),
            ("chat-43", "the sky is red"),
        ] {
            ws.get(&name(db), 1_000, IfMissing::Create)
                .unwrap()
                .remember(crate::RememberInput::text(1_000, text))
                .unwrap();
        }

        // The same query, the same fact id in each — and each answers only for
        // itself. Ids are per database, which is why they can be plain `[f1]`.
        for (db, expected) in [("chat-42", "blue"), ("chat-43", "red")] {
            let out = ws
                .get(&name(db), 2_000, IfMissing::Fail)
                .unwrap()
                .recall(crate::RecallQuery::text(2_000, "sky"))
                .unwrap();
            assert_eq!(out.facts.len(), 1, "{db}");
            assert!(out.rendered.contains(expected), "{db}: {}", out.rendered);
        }
    }

    #[test]
    fn the_ceiling_evicts_the_least_recently_used() {
        let tmp = TempDir::new("pool-evict");
        let (ws, opens) = workspace(
            &tmp,
            WorkspaceLimits {
                max_open: 2,
                ..WorkspaceLimits::default()
            },
        );

        // Three databases through a pool of two, touching `a` in between so it
        // is `b` that is coldest when `c` arrives.
        ws.get(&name("a"), 1_000, IfMissing::Create).unwrap();
        ws.get(&name("b"), 2_000, IfMissing::Create).unwrap();
        ws.get(&name("a"), 3_000, IfMissing::Fail).unwrap();
        ws.get(&name("c"), 4_000, IfMissing::Create).unwrap();
        assert_eq!(ws.open_count(), 2);
        assert_eq!(opens.load(Ordering::SeqCst), 3);

        // `a` is still pooled; `b` was evicted and has to be reopened.
        ws.get(&name("a"), 5_000, IfMissing::Fail).unwrap();
        assert_eq!(opens.load(Ordering::SeqCst), 3);
        ws.get(&name("b"), 6_000, IfMissing::Fail).unwrap();
        assert_eq!(opens.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn a_ceiling_of_zero_still_serves_one_database() {
        let tmp = TempDir::new("pool-zero");
        let (ws, opens) = workspace(
            &tmp,
            WorkspaceLimits {
                max_open: 0,
                idle_timeout_ms: 0,
            },
        );
        ws.get(&name("a"), 1_000, IfMissing::Create).unwrap();
        ws.get(&name("b"), 2_000, IfMissing::Create).unwrap();
        assert_eq!(ws.open_count(), 1);
        assert_eq!(opens.load(Ordering::SeqCst), 2);

        // A zero timeout disables the sweep rather than closing everything.
        assert_eq!(ws.close_idle(u64::MAX), 0);
        assert_eq!(ws.open_count(), 1);
    }

    #[test]
    fn an_idle_database_is_closed_and_its_lock_released() {
        let tmp = TempDir::new("pool-idle");
        let (ws, _) = workspace(
            &tmp,
            WorkspaceLimits {
                max_open: 8,
                idle_timeout_ms: 1_000,
            },
        );
        let chat = name("chat-42");
        let path = ws.layout().path_of(&chat);
        drop(ws.get(&chat, 1_000, IfMissing::Create).unwrap());

        // Inside the window nothing moves, and the file stays locked.
        assert_eq!(ws.close_idle(1_500), 0);
        assert!(matches!(
            Database::open(&path, crate::Config::default()),
            Err(HostError::Locked { .. })
        ));

        // A clock that stepped backwards must not close everything.
        assert_eq!(ws.close_idle(500), 0);

        // Past it, the handle goes and the database is reachable again — this
        // is the whole point of the timeout, not memory.
        assert_eq!(ws.close_idle(2_000), 1);
        assert_eq!(ws.open_count(), 0);
        assert!(Database::open(&path, crate::Config::default()).is_ok());
    }

    #[test]
    fn a_handle_held_by_a_caller_outlives_its_pool_entry() {
        let tmp = TempDir::new("pool-outlive");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");
        let held = ws.get(&chat, 1_000, IfMissing::Create).unwrap();
        let path = ws.layout().path_of(&chat);

        assert_eq!(ws.close_all(), 1);
        assert_eq!(ws.open_count(), 0);

        // The pool let go; the caller did not, so the lock is still taken and
        // the handle still works. Documented on `Workspace`, checked here.
        held.remember(crate::RememberInput::text(2_000, "still mine"))
            .unwrap();
        assert!(matches!(
            Database::open(&path, crate::Config::default()),
            Err(HostError::Locked { .. })
        ));
        drop(held);
        assert!(Database::open(&path, crate::Config::default()).is_ok());
    }

    #[test]
    fn a_database_held_by_another_process_is_reported_by_name() {
        let tmp = TempDir::new("pool-busy");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");
        let path = ws.layout().path_of(&chat);
        std::fs::create_dir_all(ws.layout().db_dir()).unwrap();

        // Stand in for the other process with a handle the pool knows nothing
        // about: the lock is what matters, not who holds it.
        let outsider = Database::open(&path, crate::Config::default()).unwrap().0;
        let e = ws.get(&chat, 1_000, IfMissing::Create).unwrap_err();
        assert!(matches!(&e, WorkspaceError::Busy { name } if name == &chat));
        assert!(e.to_string().contains("chat-42"), "{e}");

        drop(outsider);
        assert!(ws.get(&chat, 2_000, IfMissing::Fail).is_ok());
    }

    #[test]
    fn an_open_that_fails_for_another_reason_keeps_its_own_error() {
        let tmp = TempDir::new("pool-open-err");
        let open: Opener = Box::new(|_| Err(HostError::Embed("no provider".into())));
        let ws = Workspace::new(
            WorkspaceLayout::new(&tmp.0),
            open,
            WorkspaceLimits::default(),
        );
        let e = ws.get(&name("a"), 1_000, IfMissing::Create).unwrap_err();
        assert!(
            matches!(e, WorkspaceError::Host(HostError::Embed(_))),
            "{e}"
        );
    }

    #[test]
    fn a_directory_that_cannot_be_created_is_an_error_not_a_panic() {
        let tmp = TempDir::new("pool-mkdir");
        // `db` is a file, so `create_dir_all` cannot make it a directory.
        std::fs::write(tmp.0.join(DB_DIR), b"in the way").unwrap();
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        assert!(matches!(
            ws.get(&name("a"), 1_000, IfMissing::Create),
            Err(WorkspaceError::Io { .. })
        ));
    }

    proptest::proptest! {
        /// The property the whole design rests on: whatever a caller sends,
        /// either it is refused, or the file it names sits directly inside
        /// `<root>/db` — one component, no traversal, no absolute path, no
        /// device name. Checked over arbitrary strings rather than a list of
        /// attacks somebody thought of.
        #[test]
        fn a_name_that_parses_can_only_resolve_inside_the_workspace(s in ".*") {
            let Ok(name) = DbName::parse(&s) else { return Ok(()) };
            let layout = WorkspaceLayout::new("/ws");
            let path = layout.path_of(&name);

            let rest: Vec<_> = path
                .strip_prefix(layout.db_dir())
                .expect("resolved outside the workspace")
                .components()
                .collect();
            let expected = format!("{s}.{DB_EXT}");
            proptest::prop_assert_eq!(rest.len(), 1);
            proptest::prop_assert_eq!(
                path.file_name().and_then(|n| n.to_str()),
                Some(expected.as_str())
            );
        }
    }
}
