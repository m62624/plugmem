//! The registry: which databases exist, and what each one is for.
//!
//! A person with three databases remembers their names. A bot with three
//! hundred does not, and neither does the model driving it — so there has to be
//! a way to ask "which database is the one about releases?" and get a name back.
//! That is a search problem, and this crate already contains a search engine, so
//! the registry is **an ordinary plugmem database**: one fact per database, the
//! description as its text. Tags, the entity graph and bitemporality come along
//! for free, which is why nothing here invents a file format.
//!
//! # The registry is an index, not the truth
//!
//! Every database also **describes itself**, in a fact of its own anchored on a
//! reserved entity. The registry is then derivable: [`Workspace::reindex`] walks
//! the directory, reads each database's own description, and rebuilds it. That
//! ordering is the whole design:
//!
//! - delete the registry and search stops working. Nothing else does, and
//!   `reindex` brings it back;
//! - copy, move or delete a database file and its description travels with it,
//!   because it is inside it;
//! - the registry can be *wrong* — it is a cache — and [`Workspace::verify`]
//!   says how, without quietly fixing anything.
//!
//! A registry that were the source of truth would instead have four ways to
//! disagree with the disk, and every one of them would lose data rather than
//! lose search.
//!
//! Reading a whole directory of databases is only affordable because a small
//! database is small: after the derived shard layout landed, a chat-sized
//! memory is well under a megabyte and opens in milliseconds.
//!
//! # What goes where
//!
//! Metadata is stored opaquely by the engine and **is not searchable**, so it
//! holds only what has to be read back, never what has to be found:
//!
//! | field | where | why |
//! |---|---|---|
//! | description | the fact's text | this is what search matches |
//! | tags | tags | this is what filters |
//! | owner | an `owned-by` edge, and metadata | the edge answers "all of Ann's chats"; the metadata reads back exactly |
//! | name | metadata, and the record's subject entity | not searched, only returned — and the entity makes lookup by name a graph anchor |

use std::collections::BTreeMap;

use crate::{
    Database, ExportedFact, FactId, HostError, IfMissing, RecallQuery, RememberInput, Workspace,
};

use super::{DbName, WorkspaceError};

/// The entity a database's self-description hangs off.
///
/// Entity names are normalized (tokenized, joined by spaces), so this is
/// written the way it is stored — no surprises about what the tokenizer does to
/// punctuation. It is a *reservation*: a caller that gives one of its own facts
/// this exact subject will be mistaken for a self-description, and
/// [`Workspace::verify`] reports the ambiguity rather than guessing.
pub const SELF_ENTITY: &str = "plugmem workspace self";

/// Tag every registry record carries, so the registry's own facts are
/// distinguishable from anything else written into that file.
pub const ENTRY_TAG: &str = "plugmem-db";

/// Tag marking a database as archived: still present, still openable, no longer
/// somewhere new work should go.
pub const ARCHIVED_TAG: &str = "archived";

/// Metadata key naming the database a record is about.
const NAME_KEY: &str = "name";

/// Metadata key naming its owner.
const OWNER_KEY: &str = "owner";

/// Relation from a database to whoever owns it.
const OWNED_BY_REL: &str = "owned-by";

/// How many facts an anchored lookup asks for. More than one, so a duplicate on
/// the reserved anchor is *seen* rather than silently resolved to whichever
/// scored higher.
const ANCHOR_K: usize = 4;

/// What a caller says about a database.
#[derive(Clone, Copy, Debug, Default)]
pub struct Description<'a> {
    /// Free text: what this database is for, in the words someone would search
    /// with. Written by a person or by the model that just created it.
    pub text: &'a str,
    /// Tags to filter by (`kind:chat`, `archived`, whatever a caller means).
    pub tags: &'a [&'a str],
    /// Who it belongs to, if anyone.
    pub owner: Option<&'a str>,
}

/// One database as the registry knows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DbEntry {
    /// The database's name — its identity, and what a caller passes back.
    pub name: DbName,
    /// What it is for.
    pub description: String,
    /// Its tags.
    pub tags: Vec<String>,
    /// Its owner, if recorded.
    pub owner: Option<String>,
}

impl DbEntry {
    /// Whether this database is archived.
    pub fn is_archived(&self) -> bool {
        self.tags.iter().any(|t| t == ARCHIVED_TAG)
    }
}

/// What a [`Workspace::reindex`] pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReindexReport {
    /// Databases whose own description was copied into the registry.
    pub indexed: Vec<DbName>,
    /// Databases that have never been described. Not a fault — a database is
    /// perfectly usable without a description; it just cannot be found by one.
    pub undescribed: Vec<DbName>,
    /// Databases held open elsewhere, so this pass could not read them. Named
    /// rather than skipped silently: the registry is now knowingly incomplete.
    pub busy: Vec<DbName>,
}

/// Something [`Workspace::verify`] found. Reported, never repaired — a workspace
/// is a directory a person can edit, and guessing at their intent is how a
/// consistency check becomes a data-loss bug.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkspaceIssue {
    /// The registry describes a database that is not on disk.
    Missing {
        /// The database the registry believes in.
        name: DbName,
    },
    /// A database on disk that the registry does not describe. It works; it
    /// just cannot be found by description until it is described.
    Undescribed {
        /// The database with no registry record.
        name: DbName,
    },
    /// The registry's description disagrees with the database's own.
    Stale {
        /// The database whose record is out of date.
        name: DbName,
    },
    /// A database could not be read, so nothing about it could be checked.
    Unreadable {
        /// The database that would not open.
        name: DbName,
        /// Why, in the words of the error.
        why: String,
    },
    /// More than one fact claims the reserved self-description anchor.
    AmbiguousSelf {
        /// The database holding them.
        name: DbName,
        /// How many facts were anchored there.
        facts: usize,
    },
}

impl Workspace {
    /// Records what `name` is for, in the database itself and in the registry.
    ///
    /// Creates the database if it does not exist — describing a database into
    /// being is a reasonable thing to want, and the alternative is a two-step
    /// dance where the first step is forgettable.
    ///
    /// Called again for the same database, this **revises** rather than
    /// duplicating: facts are immutable here, so a change is a new revision and
    /// the history of what this database used to be for is kept for free. The
    /// revision has a new fact id, which is exactly why a database's identity is
    /// its name and never a fact id.
    ///
    /// # Errors
    ///
    /// Whatever opening or writing either database reports.
    pub fn describe(
        &self,
        name: &DbName,
        now_ms: u64,
        desc: Description<'_>,
    ) -> Result<(), WorkspaceError> {
        let db = self.get(name, now_ms, IfMissing::Create)?;
        write_self(&db, now_ms, desc).map_err(|e| self.blame(name, e))?;
        drop(db);

        let registry = self.registry()?;
        write_entry(&registry, now_ms, name, desc)?;
        Ok(())
    }

    /// Marks `name` archived, keeping its description. Returns whether anything
    /// changed (`false` when it was already archived).
    ///
    /// Archiving does not close, move or delete the database — it is a label,
    /// and the caller decides what it means. Deleting is deleting a file, and
    /// this crate does not do that on a caller's behalf.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::NoSuchDatabase`] when there is no record to archive,
    /// plus whatever writing the registry reports.
    pub fn archive(&self, name: &DbName, now_ms: u64) -> Result<bool, WorkspaceError> {
        let Some(entry) = self.entry(name)? else {
            return Err(WorkspaceError::NoSuchDatabase {
                name: name.clone(),
                path: self.layout().path_of(name),
            });
        };
        if entry.is_archived() {
            return Ok(false);
        }
        let mut tags: Vec<&str> = entry.tags.iter().map(String::as_str).collect();
        tags.push(ARCHIVED_TAG);
        self.describe(
            name,
            now_ms,
            Description {
                text: &entry.description,
                tags: &tags,
                owner: entry.owner.as_deref(),
            },
        )?;
        Ok(true)
    }

    /// The registry's record for `name`, or `None` if it has none.
    ///
    /// # Errors
    ///
    /// Whatever opening or reading the registry reports.
    pub fn entry(&self, name: &DbName) -> Result<Option<DbEntry>, WorkspaceError> {
        Ok(self.entries()?.into_iter().find(|e| &e.name == name))
    }

    /// Every record in the registry, sorted by name.
    ///
    /// A full dump rather than a query: the registry holds one fact per
    /// database, so this is cheap at the scale where listing is what a caller
    /// wants. Past that scale they want [`Workspace::find`].
    ///
    /// # Errors
    ///
    /// Whatever opening the registry reports.
    pub fn entries(&self) -> Result<Vec<DbEntry>, WorkspaceError> {
        let registry = self.registry()?;
        let mut out: Vec<DbEntry> = registry.export().iter().filter_map(entry_of).collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// The databases whose descriptions best match `query`, best first.
    ///
    /// This is the answer to "I do not know the name": ask in words, get names
    /// back, then work with the name. Results are ranked by the same fused
    /// recall every other search uses — one database's worth of scoring, so the
    /// ranking means something (scores from *different* databases would not be
    /// comparable, which is why nothing here ever merges across them).
    ///
    /// The query doubles as a graph anchor, so a person's name finds what they
    /// own even though an owner is an edge and edges are not text. "Ann" reaches
    /// the Ann entity, the walk crosses `owned-by` in either direction, and the
    /// records on the other side come back. Nothing special-cases owners: it is
    /// the lexical and graph sources doing what they already do, fused.
    ///
    /// # Errors
    ///
    /// Whatever opening or querying the registry reports.
    pub fn find(&self, query: &str, k: usize, now_ms: u64) -> Result<Vec<DbEntry>, WorkspaceError> {
        let registry = self.registry()?;
        let hits = registry.recall(RecallQuery {
            entities: &[query],
            k,
            ..RecallQuery::text(now_ms, query)
        })?;
        let mut by_name: BTreeMap<String, DbEntry> = self
            .entries()?
            .into_iter()
            .map(|e| (e.name.to_string(), e))
            .collect();

        // Ranked order comes from the recall; the entries themselves come from
        // the dump, which is the only place a fact's tags are readable in one
        // pass. `remove` also dedupes, so a database cannot appear twice.
        let mut out = Vec::with_capacity(hits.facts.len());
        for hit in &hits.facts {
            if let Some(snap) = registry.get(hit.id)
                && let Some(name) = snap.metadata.get(NAME_KEY)
                && let Some(entry) = by_name.remove(name)
            {
                out.push(entry);
            }
        }
        Ok(out)
    }

    /// Rebuilds the registry from the databases themselves.
    ///
    /// The repair path, and the reason the registry is allowed to be a cache.
    /// It reads each database's own description and writes it back into the
    /// registry, so a registry that was deleted, corrupted or edited by hand
    /// comes back from the data.
    ///
    /// A database held open by another process **cannot** be read here — one
    /// file has one writer — so it is named in the report rather than skipped
    /// silently. That is a real limit of rebuilding a live workspace, and the
    /// normal path (`describe` keeping the registry current) does not have it.
    ///
    /// # Errors
    ///
    /// Whatever listing the directory or writing the registry reports. A single
    /// unreadable database is reported, not raised.
    pub fn reindex(&self, now_ms: u64) -> Result<ReindexReport, WorkspaceError> {
        let mut report = ReindexReport::default();
        for name in self.layout().list()? {
            let db = match self.get(&name, now_ms, IfMissing::Fail) {
                Ok(db) => db,
                Err(WorkspaceError::Busy { .. }) => {
                    report.busy.push(name);
                    continue;
                }
                Err(e) => return Err(e),
            };
            let found = self_description(&db, now_ms).map_err(|e| self.blame(&name, e))?;
            drop(db);

            match found {
                Some(desc) => {
                    let registry = self.registry()?;
                    let tags: Vec<&str> = desc.tags.iter().map(String::as_str).collect();
                    write_entry(
                        &registry,
                        now_ms,
                        &name,
                        Description {
                            text: &desc.text,
                            tags: &tags,
                            owner: desc.owner.as_deref(),
                        },
                    )?;
                    report.indexed.push(name);
                }
                None => report.undescribed.push(name),
            }
        }
        Ok(report)
    }

    /// Checks the registry against the directory, reporting every disagreement.
    ///
    /// Fixes nothing: see [`WorkspaceIssue`].
    ///
    /// # Errors
    ///
    /// Whatever listing the directory or opening the registry reports.
    pub fn verify(&self, now_ms: u64) -> Result<Vec<WorkspaceIssue>, WorkspaceError> {
        let on_disk = self.layout().list()?;
        let recorded = self.entries()?;
        let mut issues = Vec::new();

        for entry in &recorded {
            if !self.layout().exists(&entry.name) {
                issues.push(WorkspaceIssue::Missing {
                    name: entry.name.clone(),
                });
            }
        }

        for name in on_disk {
            let db = match self.get(&name, now_ms, IfMissing::Fail) {
                Ok(db) => db,
                Err(e) => {
                    issues.push(WorkspaceIssue::Unreadable {
                        name,
                        why: e.to_string(),
                    });
                    continue;
                }
            };
            let anchored = anchored_facts(&db, now_ms).map_err(|e| self.blame(&name, e))?;
            let own = anchored.first().map(|(id, snap)| SelfDescription {
                text: snap.text.clone(),
                tags: db.tags_of(*id),
                owner: snap.metadata.get(OWNER_KEY).cloned(),
            });
            let anchored = anchored.len();
            drop(db);

            if anchored > 1 {
                issues.push(WorkspaceIssue::AmbiguousSelf {
                    name: name.clone(),
                    facts: anchored,
                });
            }
            match (own, recorded.iter().find(|e| e.name == name)) {
                // Nobody has said what this database is for. Perfectly usable;
                // just not findable by description.
                (None, None) => issues.push(WorkspaceIssue::Undescribed { name }),
                // Anything else where the two disagree is what `reindex` exists
                // to settle — including a record for a database that no longer
                // describes itself.
                (Some(own), Some(record)) if own.agrees_with(record) => {}
                _ => issues.push(WorkspaceIssue::Stale { name }),
            }
        }
        Ok(issues)
    }

    /// Attributes a host failure to a named database, so a lock conflict says
    /// which one.
    fn blame(&self, name: &DbName, e: HostError) -> WorkspaceError {
        match e {
            HostError::Locked { .. } => WorkspaceError::Busy { name: name.clone() },
            other => WorkspaceError::Host(other),
        }
    }
}

/// Writes (or revises) the self-description inside a database.
fn write_self(db: &Database, now_ms: u64, desc: Description<'_>) -> Result<(), HostError> {
    let owner = desc.owner.map(|o| [(OWNER_KEY, o)]);
    let input = RememberInput {
        entity: Some(SELF_ENTITY),
        tags: desc.tags,
        metadata: owner.as_ref().map(|m| m.as_slice()),
        ..RememberInput::text(now_ms, desc.text)
    };
    match anchored_facts(db, now_ms)?.first() {
        Some((id, _)) => db.revise(*id, input)?,
        None => db.remember(input)?,
    };
    Ok(())
}

/// Writes (or revises) a database's record in the registry.
fn write_entry(
    registry: &Database,
    now_ms: u64,
    name: &DbName,
    desc: Description<'_>,
) -> Result<(), WorkspaceError> {
    let mut tags: Vec<&str> = Vec::with_capacity(desc.tags.len() + 1);
    tags.push(ENTRY_TAG);
    tags.extend(desc.tags.iter().copied().filter(|t| *t != ENTRY_TAG));

    let mut metadata: Vec<(&str, &str)> = vec![(NAME_KEY, name.as_str())];
    if let Some(owner) = desc.owner {
        metadata.push((OWNER_KEY, owner));
    }
    let links: Vec<(&str, &str)> = desc
        .owner
        .map(|owner| vec![(OWNED_BY_REL, owner)])
        .unwrap_or_default();

    let input = RememberInput {
        // The record's subject is the database itself, so looking one up by
        // name is a graph anchor rather than a scan, and an owner edge makes
        // "everything Ann owns" reachable from either end (expansion walks
        // edges in both directions).
        entity: Some(name.as_str()),
        tags: &tags,
        links: &links,
        metadata: Some(&metadata),
        ..RememberInput::text(now_ms, desc.text)
    };

    match existing_record(registry, now_ms, name)? {
        Some(id) => registry.revise(id, input)?,
        None => registry.remember(input)?,
    };
    Ok(())
}

/// The id of `name`'s registry record, if it has one.
fn existing_record(
    registry: &Database,
    now_ms: u64,
    name: &DbName,
) -> Result<Option<FactId>, HostError> {
    let hits = registry.recall(RecallQuery {
        entities: &[name.as_str()],
        k: ANCHOR_K,
        ..blank(now_ms)
    })?;
    // The anchor can also surface a *neighbour's* facts, so the record is
    // confirmed by the name in its metadata rather than by having been returned.
    for hit in &hits.facts {
        if let Some(snap) = registry.get(hit.id)
            && snap.metadata.get(NAME_KEY).map(String::as_str) == Some(name.as_str())
        {
            return Ok(Some(hit.id));
        }
    }
    Ok(None)
}

/// Facts anchored on the reserved self-description entity, newest first.
fn anchored_facts(
    db: &Database,
    now_ms: u64,
) -> Result<Vec<(FactId, crate::FactSnapshot)>, HostError> {
    let hits = db.recall(RecallQuery {
        entities: &[SELF_ENTITY],
        k: ANCHOR_K,
        ..blank(now_ms)
    })?;
    Ok(hits
        .facts
        .iter()
        .filter_map(|hit| db.get(hit.id).map(|snap| (hit.id, snap)))
        .collect())
}

/// What a database says about itself. Deliberately not a [`DbEntry`]: the name
/// is not in the file, it is the file's place in the directory, so a type that
/// carried one here would have to invent it.
struct SelfDescription {
    text: String,
    tags: Vec<String>,
    owner: Option<String>,
}

impl SelfDescription {
    /// Whether the registry's record says the same thing this database does —
    /// that is, whether a `reindex` would leave the record alone.
    fn agrees_with(&self, record: &DbEntry) -> bool {
        self.text == record.description && self.owner == record.owner && self.tags == record.tags
    }
}

/// A database's own description, if it has one.
fn self_description(db: &Database, now_ms: u64) -> Result<Option<SelfDescription>, HostError> {
    let Some((id, snap)) = anchored_facts(db, now_ms)?.into_iter().next() else {
        return Ok(None);
    };
    Ok(Some(SelfDescription {
        text: snap.text,
        tags: db.tags_of(id),
        owner: snap.metadata.get(OWNER_KEY).cloned(),
    }))
}

/// A query with no text and no vector — just the anchors set by the caller.
/// Anchoring alone is enough: the graph source seeds from the named entities and
/// returns their facts, so a lookup by entity costs no tokenizing and no search.
fn blank(now_ms: u64) -> RecallQuery<'static> {
    RecallQuery {
        text: None,
        ..RecallQuery::text(now_ms, "")
    }
}

/// A registry record read back out of an export, or `None` for a fact that is
/// not one.
fn entry_of(fact: &ExportedFact) -> Option<DbEntry> {
    if !fact.tags.iter().any(|t| t == ENTRY_TAG) {
        return None;
    }
    entry_from(&fact.text, &fact.tags, &fact.metadata)
}

/// Builds an entry from the three places its parts live.
fn entry_from(text: &str, tags: &[String], metadata: &BTreeMap<String, String>) -> Option<DbEntry> {
    let name = DbName::parse(metadata.get(NAME_KEY)?).ok()?;
    Some(DbEntry {
        name,
        description: text.to_string(),
        tags: tags.iter().filter(|t| *t != ENTRY_TAG).cloned().collect(),
        owner: metadata.get(OWNER_KEY).cloned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::testkit::{TempDir, name, workspace};
    use crate::{RememberInput, WorkspaceLimits};

    /// A description with only text — the common case.
    fn about(text: &str) -> Description<'_> {
        Description {
            text,
            ..Description::default()
        }
    }

    fn names(entries: &[DbEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn the_registry_is_not_opened_until_something_needs_it() {
        let tmp = TempDir::new("registry-lazy");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());

        // Resolving a name a caller already knows never touches the registry:
        // no file, and no lock another process would trip over.
        ws.get(&name("chat-42"), 1_000, IfMissing::Create).unwrap();
        assert!(!ws.layout().registry_path().exists());

        ws.entries().unwrap();
        assert!(crate::storage::database_exists(
            &ws.layout().registry_path()
        ));
        assert!(ws.close_registry());
        assert!(!ws.close_registry());
    }

    #[test]
    fn describing_twice_revises_one_record_rather_than_adding_a_second() {
        let tmp = TempDir::new("registry-revise");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");

        ws.describe(&chat, 1_000, about("work chat about plugmem"))
            .unwrap();
        ws.describe(&chat, 2_000, about("work chat about releases"))
            .unwrap();

        let entries = ws.entries().unwrap();
        assert_eq!(names(&entries), ["chat-42"]);
        assert_eq!(entries[0].description, "work chat about releases");

        // The database's own copy moved with it — that is what makes the
        // registry rebuildable.
        let db = ws.get(&chat, 3_000, IfMissing::Fail).unwrap();
        let own = self_description(&db, 3_000).unwrap().unwrap();
        assert_eq!(own.text, "work chat about releases");
    }

    #[test]
    fn describing_a_name_that_has_no_database_creates_it() {
        let tmp = TempDir::new("registry-create");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");
        assert!(!ws.layout().exists(&chat));
        ws.describe(&chat, 1_000, about("a brand new chat"))
            .unwrap();
        assert!(ws.layout().exists(&chat));
    }

    #[test]
    fn a_database_is_found_by_what_it_is_for_and_by_who_owns_it() {
        let tmp = TempDir::new("registry-find");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());

        ws.describe(
            &name("chat-42"),
            1_000,
            Description {
                text: "release planning and performance work on the engine",
                tags: &["kind:chat"],
                owner: Some("ann"),
            },
        )
        .unwrap();
        ws.describe(
            &name("recipes"),
            1_000,
            Description {
                text: "dinner ideas and shopping lists",
                tags: &["kind:notes"],
                owner: Some("bob"),
            },
        )
        .unwrap();

        // By description.
        let hits = ws.find("release planning", 4, 2_000).unwrap();
        assert_eq!(hits.first().map(|e| e.name.as_str()), Some("chat-42"));
        let hits = ws.find("shopping lists", 4, 2_000).unwrap();
        assert_eq!(hits.first().map(|e| e.name.as_str()), Some("recipes"));

        // By owner — which lives in an edge, not in the text. The graph source
        // reaches it from the person's name.
        let ann = ws.find("ann", 4, 2_000).unwrap();
        assert_eq!(names(&ann), ["chat-42"]);

        // Tags and owner survive the round trip.
        let entry = ws.entry(&name("chat-42")).unwrap().unwrap();
        assert_eq!(entry.tags, ["kind:chat"]);
        assert_eq!(entry.owner.as_deref(), Some("ann"));
        assert!(!entry.is_archived());
        assert!(ws.entry(&name("nope")).unwrap().is_none());
    }

    #[test]
    fn archiving_is_a_label_and_is_idempotent() {
        let tmp = TempDir::new("registry-archive");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");

        // Nothing to archive is an error naming the database, not a silent no-op.
        assert!(matches!(
            ws.archive(&chat, 1_000),
            Err(WorkspaceError::NoSuchDatabase { .. })
        ));

        ws.describe(
            &chat,
            1_000,
            Description {
                text: "an old project",
                tags: &["kind:chat"],
                owner: Some("ann"),
            },
        )
        .unwrap();

        assert!(ws.archive(&chat, 2_000).unwrap());
        let entry = ws.entry(&chat).unwrap().unwrap();
        assert!(entry.is_archived());
        // The rest of the record survives being labelled.
        assert_eq!(entry.description, "an old project");
        assert_eq!(entry.owner.as_deref(), Some("ann"));
        assert!(entry.tags.contains(&"kind:chat".to_string()));

        // Already archived: nothing to do, and it says so.
        assert!(!ws.archive(&chat, 3_000).unwrap());
        // Archiving does not close, move or delete anything.
        assert!(ws.layout().exists(&chat));
    }

    #[test]
    fn a_deleted_registry_is_rebuilt_from_the_databases_themselves() {
        let tmp = TempDir::new("registry-reindex");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());

        for (db, text) in [("chat-42", "release planning"), ("recipes", "dinner ideas")] {
            ws.describe(
                &name(db),
                1_000,
                Description {
                    text,
                    tags: &["kind:chat"],
                    owner: Some("ann"),
                },
            )
            .unwrap();
        }
        // A database nobody described: usable, just not findable.
        ws.get(&name("scratch"), 1_000, IfMissing::Create).unwrap();

        // Lose the registry entirely, the way a botched backup would.
        ws.close_registry();
        for entry in std::fs::read_dir(ws.layout().root()).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                std::fs::remove_file(path).unwrap();
            }
        }
        assert!(ws.entries().unwrap().is_empty());

        let report = ws.reindex(2_000).unwrap();
        assert_eq!(
            report
                .indexed
                .iter()
                .map(DbName::to_string)
                .collect::<Vec<_>>(),
            ["chat-42", "recipes"]
        );
        assert_eq!(
            report
                .undescribed
                .iter()
                .map(DbName::to_string)
                .collect::<Vec<_>>(),
            ["scratch"]
        );
        assert!(report.busy.is_empty());

        // Everything is back, including what was never in the registry's text.
        let entry = ws.entry(&name("chat-42")).unwrap().unwrap();
        assert_eq!(entry.description, "release planning");
        assert_eq!(entry.tags, ["kind:chat"]);
        assert_eq!(entry.owner.as_deref(), Some("ann"));
    }

    #[test]
    fn reindex_names_the_databases_it_could_not_read() {
        let tmp = TempDir::new("registry-reindex-busy");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let held = name("chat-42");
        ws.describe(&held, 1_000, about("release planning"))
            .unwrap();
        ws.describe(&name("recipes"), 1_000, about("dinner ideas"))
            .unwrap();

        // Someone else holds the writer, so this pass genuinely cannot read it.
        // It has to say so: the rebuilt registry is knowingly incomplete.
        ws.close_all();
        let outsider = Database::open(ws.layout().path_of(&held), crate::Config::default())
            .unwrap()
            .0;

        let report = ws.reindex(2_000).unwrap();
        assert_eq!(report.busy, std::slice::from_ref(&held));
        assert_eq!(report.indexed, [name("recipes")]);

        drop(outsider);
        assert_eq!(ws.reindex(3_000).unwrap().busy, []);
    }

    #[test]
    fn verify_reports_every_way_the_registry_can_disagree() {
        let tmp = TempDir::new("registry-verify");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());

        // Agreeing: no issue.
        let agreed = name("chat-42");
        ws.describe(&agreed, 1_000, about("release planning"))
            .unwrap();
        assert_eq!(ws.verify(2_000).unwrap(), []);

        // On disk, never described.
        let plain = name("scratch");
        ws.get(&plain, 1_000, IfMissing::Create).unwrap();

        // Described in the database, absent from the registry — exactly what a
        // rebuild would fix.
        let unlisted = name("orphan");
        let db = ws.get(&unlisted, 1_000, IfMissing::Create).unwrap();
        write_self(&db, 1_000, about("known only to itself")).unwrap();
        drop(db);

        // Registry knows a database the disk does not have.
        let registry = ws.registry().unwrap();
        write_entry(&registry, 1_000, &name("ghost"), about("gone")).unwrap();
        drop(registry);

        // Two facts on the reserved anchor: ambiguous, and reported rather than
        // resolved to whichever ranked higher.
        let two = name("twins");
        let db = ws.get(&two, 1_000, IfMissing::Create).unwrap();
        for text in ["first claim", "second claim"] {
            db.remember(RememberInput {
                entity: Some(SELF_ENTITY),
                ..RememberInput::text(1_000, text)
            })
            .unwrap();
        }
        drop(db);

        let issues = ws.verify(2_000).unwrap();
        assert!(issues.contains(&WorkspaceIssue::Missing {
            name: name("ghost")
        }));
        assert!(issues.contains(&WorkspaceIssue::Undescribed { name: plain }));
        assert!(issues.contains(&WorkspaceIssue::Stale { name: unlisted }));
        assert!(issues.iter().any(|i| matches!(
            i,
            WorkspaceIssue::AmbiguousSelf { name, facts: 2 } if name == &two
        )));
        // The one that agrees is not mentioned.
        assert!(!issues.iter().any(|i| matches!(
            i,
            WorkspaceIssue::Stale { name } | WorkspaceIssue::Undescribed { name } if name == &agreed
        )));
    }

    #[test]
    fn verify_reports_a_database_it_could_not_open() {
        let tmp = TempDir::new("registry-verify-busy");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let held = name("chat-42");
        ws.describe(&held, 1_000, about("release planning"))
            .unwrap();
        ws.close_all();
        let _outsider = Database::open(ws.layout().path_of(&held), crate::Config::default())
            .unwrap()
            .0;

        let issues = ws.verify(2_000).unwrap();
        assert!(issues.iter().any(|i| matches!(
            i,
            WorkspaceIssue::Unreadable { name, why } if name == &held && why.contains("chat-42")
        )));
    }

    #[test]
    fn a_stale_record_is_one_a_reindex_would_change() {
        let tmp = TempDir::new("registry-stale");
        let (ws, _) = workspace(&tmp, WorkspaceLimits::default());
        let chat = name("chat-42");
        ws.describe(&chat, 1_000, about("release planning"))
            .unwrap();

        // Edit only the registry, as a hand-edit or a partial restore would.
        let registry = ws.registry().unwrap();
        write_entry(&registry, 2_000, &chat, about("something else")).unwrap();
        drop(registry);
        assert_eq!(
            ws.verify(3_000).unwrap(),
            [WorkspaceIssue::Stale { name: chat.clone() }]
        );

        // And a rebuild settles it, from the database's own copy.
        ws.reindex(4_000).unwrap();
        assert_eq!(ws.verify(5_000).unwrap(), []);
        assert_eq!(
            ws.entry(&chat).unwrap().unwrap().description,
            "release planning"
        );
    }
}
