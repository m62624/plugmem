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

use std::fmt;
use std::path::{Path, PathBuf};

use crate::HostError;

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

    /// Whether `name` has a file on disk.
    pub fn exists(&self, name: &DbName) -> bool {
        self.path_of(name).is_file()
    }

    /// Every database in the workspace, sorted by name.
    ///
    /// Reads the directory — the filesystem is the truth, the registry is only
    /// an index over it. A missing `db/` is an empty workspace, not an error.
    /// Files whose stem is not a valid name are skipped here and reported by
    /// the registry's `verify`, which is where a person is asking about
    /// consistency rather than about what they can open.
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
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(DB_EXT) {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && let Ok(name) = DbName::parse(stem)
            {
                names.push(name);
            }
        }
        names.sort_unstable();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory; removed on drop.
    pub(super) struct TempDir(pub PathBuf);
    impl TempDir {
        pub(super) fn new(tag: &str) -> Self {
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
            // Sidecars the storage layer appends: extension is `lock`/`jrnl`,
            // so they are not mistaken for databases.
            "chat-42.plugmem.lock",
            "chat-42.plugmem.jrnl",
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
