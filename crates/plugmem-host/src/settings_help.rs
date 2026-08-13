//! The single source of truth for config.toml help.
//!
//! The parser lives in [`super::settings`], while the CLI, the MCP server and
//! the Node and Python bindings are separate surfaces. Keeping the public
//! setting catalogue here lets those surfaces render their own help without
//! copying descriptions or defaults.

use std::fmt::Write as _;

const PLATFORM_DEFAULT_SOURCE: &str = "platform default config path";

/// Something in `config.toml` that was read and then ignored.
///
/// A warning rather than an error, deliberately: refusing an unknown key would
/// mean an older binary could not read a config written for a newer one, which
/// is a worse failure than a typo. But *silence* is worse than both — a
/// misspelled `w_vec` changes no behaviour and says nothing, and the user is
/// left believing they tuned something.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingWarning {
    /// The TOML section it appeared in, without brackets. Empty for an unknown
    /// *section*, where [`Self::key`] is the section's own name.
    pub section: String,
    /// The key (or section) nobody claimed.
    pub key: String,
    /// The closest known name, when one is close enough to be worth offering.
    pub did_you_mean: Option<&'static str>,
}

impl std::fmt::Display for SettingWarning {
    /// One line, ready for a stderr note or a log.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.section.is_empty() {
            write!(f, "unknown config section [{}]", self.key)?;
        } else {
            write!(f, "unknown setting [{}].{}", self.section, self.key)?;
        }
        match self.did_you_mean {
            Some(near) => write!(f, " — did you mean `{near}`?"),
            None => write!(f, " (ignored)"),
        }
    }
}

/// Which runtime surface owns a setting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettingScope {
    /// Parsed by `plugmem-host` and shared by every wrapper.
    Shared,
    /// Read by `plugmem-cli` in addition to the shared settings.
    Cli,
    /// Read by `plugmem-mcp` in addition to the shared settings.
    Mcp,
}

impl SettingScope {
    /// The stable, user-facing scope label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Cli => "CLI",
            Self::Mcp => "MCP",
        }
    }
}

/// Documentation for one supported config.toml key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SettingDoc {
    /// TOML section, without brackets.
    pub section: &'static str,
    /// TOML key inside [`Self::section`].
    pub key: &'static str,
    /// Human-readable value type.
    pub value_type: &'static str,
    /// Default as displayed to users.
    pub default: &'static str,
    /// What the setting controls.
    pub description: &'static str,
    /// A valid TOML value for this key.
    ///
    /// Not a second opinion about the default — several defaults cannot be
    /// written as a value at all ("automatic", "off", "half of available
    /// cores"), and `config.example.toml` still has to show a line somebody
    /// can uncomment. It is also what makes that file testable: the generator
    /// renders an all-keys-set variant, and a test parses it through the real
    /// loader and requires zero warnings.
    pub example: &'static str,
    /// The wrapper(s) that consume the setting.
    pub scope: SettingScope,
}

/// Runtime access to the complete config.toml help catalogue.
#[derive(Clone, Copy, Debug)]
pub struct SettingsHelp {
    docs: &'static [SettingDoc],
    config_path_precedence: &'static [&'static str],
}

impl SettingsHelp {
    /// Every documented config.toml key.
    pub const fn docs(self) -> &'static [SettingDoc] {
        self.docs
    }

    /// Config-file discovery order, from highest to lowest precedence.
    pub const fn config_path_precedence(self) -> &'static [&'static str] {
        self.config_path_precedence
    }

    /// Every section this catalogue knows, in first-appearance order.
    ///
    /// Borrowed `&'static str`s from the catalogue itself: no allocation, and
    /// there are five of them, so a linear scan beats any index.
    fn sections(self) -> impl Iterator<Item = &'static str> {
        self.docs
            .iter()
            .enumerate()
            .filter(|(i, doc)| *i == 0 || self.docs[i - 1].section != doc.section)
            .map(|(_, doc)| doc.section)
    }

    /// The keys documented under `section`.
    fn keys_in(self, section: &str) -> impl Iterator<Item = &'static str> {
        self.docs
            .iter()
            .filter(move |doc| doc.section == section)
            .map(|doc| doc.key)
    }

    /// Reports every section and key in `table` that no surface claims.
    ///
    /// The catalogue is the authority rather than the parser's own key lists,
    /// and it has to be: `[maintenance].batch_size` belongs to the CLI and
    /// `[server].workers` to the MCP server, so a check that only knew what the
    /// shared loader parses would warn about both on every run.
    ///
    /// Allocation-free in the ordinary case — a clean config returns an empty
    /// `Vec`, which allocates nothing. Only a real mistake costs anything.
    pub fn unknown_in(self, table: &toml::Table) -> Vec<SettingWarning> {
        let mut out = Vec::new();
        for (name, value) in table {
            let Some(section) = self.sections().find(|s| s == name) else {
                out.push(SettingWarning {
                    section: String::new(),
                    key: name.clone(),
                    did_you_mean: nearest(name, self.sections()),
                });
                continue;
            };
            // A section given as something other than a table is the parser's
            // business to reject, not this scan's.
            let Some(entries) = value.as_table() else {
                continue;
            };
            for key in entries.keys() {
                if self.keys_in(section).any(|k| k == key) {
                    continue;
                }
                out.push(SettingWarning {
                    section: section.to_string(),
                    key: key.clone(),
                    did_you_mean: nearest(key, self.keys_in(section)),
                });
            }
        }
        out
    }

    /// Renders `config.example.toml`: every supported key, with its default,
    /// its type and one line about what it does.
    ///
    /// Generated rather than written, because a hand-kept example is another
    /// copy of the catalogue that ages without anyone noticing — which is
    /// exactly what happened to the four README samples this replaces. A test
    /// compares the committed file with this output, so the two cannot drift.
    ///
    /// **Every line is commented out.** An example that sets forty keys
    /// explicitly freezes today's defaults into the config of everyone who
    /// copies it: a later release improves `flat_to_hnsw` and they never see
    /// it. Uncomment the two or three you actually mean to change.
    ///
    /// `set` renders the same file with every key *active*, which is what the
    /// test parses through the real loader — an example nothing can parse is
    /// worth less than no example.
    pub fn render_config_example(self, set: bool) -> String {
        // One line per push rather than one long escaped literal: the header
        // is prose a person reads first, and it should be as easy to fix here
        // as it is to read there.
        let mut output = String::new();
        for line in [
            "# plugmem — every supported config.toml key, with its default.",
            "#",
            "# GENERATED from the settings catalogue. Do not edit: run",
            "#   cargo run -p plugmem-host --bin config_example",
            "# and commit the result. A test fails when this file and the",
            "# catalogue disagree, so an edit here is undone by the next run.",
            "#",
            "# Every key below is commented out and shows the default this build",
            "# uses. Uncomment only what you mean to change — a config that sets",
            "# everything explicitly freezes today's defaults, and never picks up",
            "# a better one.",
            "#",
            "# What each setting is FOR, and when changing it is a good idea,",
            "# lives in crates/plugmem-host/SETTINGS.md. This file is the shape;",
            "# that file is the reasoning.",
            "#",
        ] {
            output.push_str(line);
            output.push('\n');
        }
        output.push_str("# Config file precedence, highest first:\n");
        for (index, source) in self.config_path_precedence.iter().enumerate() {
            let _ = writeln!(output, "#   {}. {source}", index + 1);
        }

        let mut section = None;
        for doc in self.docs {
            if section != Some(doc.section) {
                let _ = write!(output, "\n[{}]\n", doc.section);
                section = Some(doc.section);
            }
            let scope = match doc.scope {
                SettingScope::Shared => String::new(),
                other => format!(", read only by {}", other.as_str()),
            };
            let _ = writeln!(
                output,
                "# {} — {}\n# {}, default: {}{}",
                doc.key, doc.description, doc.value_type, doc.default, scope
            );
            let prefix = if set { "" } else { "# " };
            let _ = writeln!(output, "{prefix}{} = {}", doc.key, doc.example);
        }
        output
    }

    /// Render the catalogue for a terminal or a human-facing tool response.
    pub fn render_human(self) -> String {
        let mut output = String::from("plugmem settings\n\n");
        output.push_str("Config file precedence:\n");
        for (index, source) in self.config_path_precedence.iter().enumerate() {
            if *source == PLATFORM_DEFAULT_SOURCE {
                match crate::default_config_path() {
                    Some(path) => {
                        let _ = writeln!(output, "  {}. {}", index + 1, path.display());
                    }
                    None => {
                        let _ = writeln!(output, "  {}. {source} (unavailable)", index + 1);
                    }
                }
            } else {
                let _ = writeln!(output, "  {}. {source}", index + 1);
            }
        }
        output.push('\n');

        let mut section = None;
        for doc in self.docs {
            if section != Some(doc.section) {
                if section.is_some() {
                    output.push('\n');
                }
                let _ = writeln!(output, "[{}]", doc.section);
                section = Some(doc.section);
            }
            let _ = writeln!(
                output,
                "  {} ({}, default: {}) — {} [{}]",
                doc.key,
                doc.value_type,
                doc.default,
                doc.description,
                doc.scope.as_str()
            );
        }

        output
    }
}

/// The closest candidate to `typo`, if one is close enough to suggest.
///
/// One edit always, plus one per four characters: long names tolerate a bigger
/// slip than short ones, and `dim` never gets confused with `url`. Offering a
/// wrong guess is worse than offering none — it sends the reader to fix the
/// wrong line.
///
/// The scale is set by the mistakes people actually make. `w_vector` for
/// `w_vec` is three edits on an eight-character name: the likeliest typo in the
/// whole catalogue, since the field is a vector weight and nobody abbreviates
/// on the first try. A tighter budget looks principled and misses it.
fn nearest(typo: &str, candidates: impl Iterator<Item = &'static str>) -> Option<&'static str> {
    let budget = 1 + typo.chars().count() / 4;
    candidates
        .map(|c| (edit_distance(typo, c), c))
        .filter(|(d, _)| *d <= budget)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

/// Levenshtein distance over `char`s, two rows at a time.
///
/// Two `Vec<usize>` the width of the shorter name — setting names are a handful
/// of characters, and this runs only when something is already wrong.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut row = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            row[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(row[j] + 1);
        }
        core::mem::swap(&mut prev, &mut row);
    }
    prev[b.len()]
}

const CONFIG_PATH_PRECEDENCE: &[&str] = &[
    "--config PATH",
    "$PLUGMEM_CONFIG",
    "platform default config path",
    "built-in defaults",
];

const DOCS: &[SettingDoc] = &[
    SettingDoc {
        section: "database",
        key: "path",
        value_type: "path string",
        default: "platform data directory/memory.plugmem",
        example: "\"/var/lib/plugmem/memory.plugmem\"",
        description: "Persistent database file; an explicit --db or open path and PLUGMEM_DB override it",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "workspace",
        key: "dir",
        value_type: "path string",
        default: "unset (one database, no workspace)",
        example: "\"/var/lib/plugmem/memories\"",
        description: "Directory of named databases; unset means the single-database default",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "workspace",
        key: "max_open",
        value_type: "positive integer",
        default: "16",
        example: "16",
        description: "Hard limit on open workspace databases; an inactive least-recently-used entry is closed, all-active returns Busy",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "workspace",
        key: "idle_timeout_ms",
        value_type: "non-negative integer",
        default: "60000",
        example: "60000",
        description: "Close a workspace database unused this long, releasing its lock; 0 never closes",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "dim",
        value_type: "non-negative integer",
        default: "0",
        example: "768",
        description: "Embedding dimension; 0 disables vector storage",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "max_bytes",
        value_type: "non-negative integer",
        default: "2147483648",
        example: "2147483648",
        description: "Ceiling applied to each byte pool separately, not to their sum",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "max_text",
        value_type: "non-negative integer",
        default: "4096",
        example: "4096",
        description: "Maximum fact text length in bytes",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "max_blob",
        value_type: "non-negative integer",
        default: "65536",
        example: "65536",
        description: "Maximum single blob length in bytes",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "bm25_k1",
        value_type: "number > 0",
        default: "1.2",
        example: "1.2",
        description: "BM25 term-frequency saturation: higher lets a repeated word keep counting",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "bm25_b",
        value_type: "number in [0, 1]",
        default: "0.75",
        example: "0.75",
        description: "BM25 length normalisation: 0 ignores fact length, 1 penalises long facts fully",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "rrf_k",
        value_type: "integer >= 1",
        default: "60",
        example: "60",
        description: "Reciprocal-rank-fusion constant: larger flattens the gap between rank 1 and rank 10",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "w_bm25",
        value_type: "number >= 0",
        default: "1.0",
        example: "1.0",
        description: "Weight of the lexical source in the fused score; 0 switches it off",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "w_vec",
        value_type: "number >= 0",
        default: "1.0",
        example: "1.0",
        description: "Weight of the vector source; 0 switches it off (and costs nothing when dim = 0)",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "w_graph",
        value_type: "number >= 0",
        default: "1.0",
        example: "1.0",
        description: "Weight of the entity-graph source; 0 switches off relational expansion",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "w_time",
        value_type: "number >= 0",
        default: "1.0",
        example: "1.0",
        description: "Weight of the temporal source (the recorded_at window); 0 switches it off",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "w_recency",
        value_type: "number >= 0",
        default: "0.25",
        example: "0.25",
        description: "How much a fact's age discounts it, on top of the sources above",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "half_life_days",
        value_type: "integer >= 1",
        default: "180",
        example: "180",
        description: "Age at which the recency discount has halved; larger keeps old facts competitive",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "graph_depth",
        value_type: "non-negative integer",
        default: "2",
        example: "2",
        description: "Default hops the graph source may follow from an anchor entity; a recall's own `graph_depth` overrides it. Uncapped — the walk is bounded by its entity and edge caps, not by depth",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "graph_decay",
        value_type: "number in (0, 1]",
        default: "0.5",
        example: "0.5",
        description: "How much each extra hop discounts a fact reached through the graph",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "hnsw_ef_search",
        value_type: "integer >= 1",
        default: "64",
        example: "64",
        description: "Default HNSW beam width; higher is more accurate and slower. A recall's own `ef` overrides it, and it does nothing while the index is still flat",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "similar_cos",
        value_type: "number in [0, 1]",
        default: "0.85",
        example: "0.85",
        description: "Cosine above which remember reports an existing fact as possibly conflicting (it never revises on its own)",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "recall",
        key: "similar_jaccard",
        value_type: "number in [0, 1]",
        default: "0.5",
        example: "0.5",
        description: "Token overlap above which remember reports a possible conflict, for memories with no vectors",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "index",
        key: "hnsw_ef_construction",
        value_type: "integer >= hnsw_m (16 by default)",
        default: "200",
        example: "200",
        description: "Beam width while building the vector graph: higher builds a better index, slower",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "index",
        key: "flat_to_hnsw",
        value_type: "integer >= 1",
        default: "24000",
        example: "24000",
        description: "Vector count at which maintenance stops scanning flat and builds the HNSW graph",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "enabled",
        value_type: "boolean",
        default: "automatic",
        example: "true",
        description: "Enable or disable creation and use of the configured OpenAI-compatible embedder",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "url",
        value_type: "string",
        default: "unset",
        example: "\"http://localhost:11434/v1/embeddings\"",
        description: "OpenAI-compatible /v1/embeddings endpoint",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "model",
        value_type: "string",
        default: "unset",
        example: "\"nomic-embed-text\"",
        description: "Embedding model name",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "space_id",
        value_type: "string",
        default: "model",
        example: "\"nomic-embed-text@v1\"",
        description: "Stable semantic-space identity; change it only for incompatible vectors and reembed explicitly",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "api_key_env",
        value_type: "string",
        default: "unset",
        example: "\"OPENAI_API_KEY\"",
        description: "Environment variable containing the bearer token",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "on_error",
        value_type: "\"fail\" | \"degrade\"",
        default: "fail",
        example: "\"fail\"",
        description: "Unreachable provider: fail the verb, or store/answer without a vector and suspend the embedder",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "timeout_ms",
        value_type: "non-negative integer",
        default: "10000",
        example: "10000",
        description: "Deadline for one embeddings request end to end; 0 waits indefinitely",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "retry_after_ms",
        value_type: "non-negative integer",
        default: "unset (1s doubling to retry_max_ms)",
        example: "0",
        description: "Fixed wait before a suspended embedder is called again; 0 waits for an explicit resume",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "retry_max_ms",
        value_type: "non-negative integer",
        default: "60000",
        example: "60000",
        description: "Ceiling the default doubling retry grows to; ignored when retry_after_ms is set",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "snapshot_every_ops",
        value_type: "non-negative integer",
        default: "1024",
        example: "1024",
        description: "Snapshot after this many mutations",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "snapshot_journal_bytes",
        value_type: "non-negative integer",
        default: "4194304",
        example: "4194304",
        description: "Snapshot when the journal reaches this size",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "maintain_every_forgets",
        value_type: "non-negative integer",
        default: "off",
        example: "100",
        description: "Run policy maintenance after this many forgets",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "fsync",
        value_type: "\"each_op\" | \"on_snapshot\"",
        default: "each_op",
        example: "\"each_op\"",
        description: "When journal appends reach the disk. \"each_op\": every acknowledged write \
survives a power cut. \"on_snapshot\": faster, an OS crash may lose the journal tail since the \
last snapshot",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "batch_size",
        value_type: "positive integer",
        default: "128",
        example: "128",
        description: "CLI import facts per embedding request and journal fsync",
        scope: SettingScope::Cli,
    },
    SettingDoc {
        section: "server",
        key: "workers",
        value_type: "positive integer",
        default: "half of available cores",
        example: "4",
        description: "MCP worker threads",
        scope: SettingScope::Mcp,
    },
];

static SETTINGS_HELP: SettingsHelp = SettingsHelp {
    docs: DOCS,
    config_path_precedence: CONFIG_PATH_PRECEDENCE,
};

/// Returns the shared settings catalogue used by host, CLI, MCP and NAPI.
pub const fn settings_help() -> &'static SettingsHelp {
    &SETTINGS_HELP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_holds_at_the_degenerate_ends() {
        // The empty string against anything is that thing's length, from both
        // sides: with `a` empty the inner loop never runs and the seeded first
        // row is the answer; with `b` empty the table is one column wide and
        // only `row[0]` ever moves. Both are easy to get wrong by one.
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("", "dim"), 3);
        assert_eq!(edit_distance("dim", ""), 3);

        // One character each way, same and different.
        assert_eq!(edit_distance("a", "a"), 0);
        assert_eq!(edit_distance("a", "b"), 1);
        assert_eq!(edit_distance("a", ""), 1);
        assert_eq!(edit_distance(" ", ""), 1);
        assert_eq!(edit_distance(" ", "a"), 1);

        // The three edits, each in isolation.
        assert_eq!(edit_distance("dim", "dm"), 1, "deletion");
        assert_eq!(edit_distance("dim", "diim"), 1, "insertion");
        assert_eq!(edit_distance("dim", "dir"), 1, "substitution");

        // Counted in characters, not bytes: a multi-byte name must not read as
        // several edits away from itself.
        assert_eq!(edit_distance("ключ", "ключ"), 0);
        assert_eq!(edit_distance("ключ", "клуч"), 1);
        assert_eq!(edit_distance("ключ", ""), 4);

        // Symmetric, which a two-row implementation can quietly break.
        for (a, b) in [("dim", "max_text"), ("", "fsync"), ("a", "workers")] {
            assert_eq!(edit_distance(a, b), edit_distance(b, a), "{a} vs {b}");
        }
    }

    #[test]
    fn a_suggestion_is_offered_only_when_it_is_worth_offering() {
        let engine = || settings_help().keys_in("engine");

        // Close enough to be the obvious intent.
        assert_eq!(nearest("dm", engine()), Some("dim"));
        assert_eq!(nearest("max_txt", engine()), Some("max_text"));

        // The one the budget exists for: three edits on an eight-character
        // name, and the likeliest typo in the catalogue.
        let recall = || settings_help().keys_in("recall");
        assert_eq!(nearest("w_vector", recall()), Some("w_vec"));
        assert_eq!(nearest("similar_cosine", recall()), Some("similar_cos"));

        // A truncation is not chased. `half_life` is five edits from
        // `half_life_days`, and widening the budget far enough to reach it
        // would start matching keys that share a prefix and nothing else.
        // The warning still names the key; only the guess is withheld.
        assert_eq!(nearest("half_life", recall()), None);

        // Not close to anything: silence beats sending someone to the wrong
        // line. A single character is the sharpest case — the budget floors at
        // one edit, so it must not reach a three-character key.
        assert_eq!(nearest("a", engine()), None);
        assert_eq!(nearest("", engine()), None);
        assert_eq!(nearest(" ", engine()), None);
        assert_eq!(nearest("completely_unrelated", engine()), None);
    }

    /// A config table from lines, so the fixtures indent with the code instead
    /// of being pinned to the file's left margin.
    fn toml_of(lines: &[&str]) -> toml::Table {
        lines.join("\n").parse().expect("valid TOML fixture")
    }

    #[test]
    fn unknown_sections_and_keys_are_reported_with_their_context() {
        let table = toml_of(&[
            "[engine]",
            "dim = 8",
            "max_txt = 10",
            "",
            "[embedder]",
            "enabled = false",
            "",
            "[engin]",
            "dim = 4",
        ]);

        let found = settings_help().unknown_in(&table);
        // A misspelled key inside a real section, and a misspelled section.
        assert_eq!(
            found,
            vec![
                SettingWarning {
                    section: String::new(),
                    key: "engin".to_string(),
                    did_you_mean: Some("engine"),
                },
                SettingWarning {
                    section: "engine".to_string(),
                    key: "max_txt".to_string(),
                    did_you_mean: Some("max_text"),
                },
            ]
        );
        assert!(
            found[0]
                .to_string()
                .contains("unknown config section [engin]")
        );
        assert!(found[1].to_string().contains("[engine].max_txt"));
    }

    #[test]
    fn keys_a_wrapper_owns_are_not_warned_about() {
        // The reason the catalogue is the authority and the parser's own lists
        // are not: host parses neither of these, and warning about them would
        // fire on every CLI and MCP run.
        let table = toml_of(&[
            "[maintenance]",
            "batch_size = 256",
            "",
            "[server]",
            "workers = 4",
        ]);
        assert_eq!(settings_help().unknown_in(&table), vec![]);
    }

    #[test]
    fn a_clean_config_warns_about_nothing() {
        let mut text = String::new();
        let mut section = "";
        for doc in DOCS {
            if doc.section != section {
                let _ = writeln!(text, "[{}]", doc.section);
                section = doc.section;
            }
            // The value is irrelevant here: this scan checks names, and the
            // parser owns types. Every documented key must pass it.
            let _ = writeln!(text, "{} = 0", doc.key);
        }
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(
            settings_help().unknown_in(&table),
            vec![],
            "the catalogue must accept everything it documents"
        );
    }

    #[test]
    fn every_documented_setting_has_a_complete_description() {
        assert!(!DOCS.is_empty());
        for doc in DOCS {
            assert!(!doc.section.is_empty());
            assert!(!doc.key.is_empty());
            assert!(!doc.value_type.is_empty());
            assert!(!doc.default.is_empty());
            assert!(!doc.description.is_empty());
            assert!(!doc.example.is_empty(), "{}.{}", doc.section, doc.key);
        }
    }

    /// The committed example, read from the workspace root.
    fn committed_example() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("config.example.toml");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
    }

    #[test]
    fn the_committed_config_example_matches_the_catalogue() {
        // The gate the whole generated-file arrangement exists for: add a
        // setting, forget the example, and CI says so here rather than a user
        // finding a key nobody documented. The same idiom the Node binding's
        // `index.d.ts` and the Python binding's `.pyi` are held to.
        assert_eq!(
            committed_example(),
            settings_help().render_config_example(false),
            "config.example.toml is stale — run \
             `cargo run -p plugmem-host --bin config_example` and commit it"
        );
    }

    #[test]
    fn the_example_is_a_config_the_loader_accepts() {
        // An example nobody can parse is worth less than no example. Every key
        // active, through the real loader: an unknown key, a value of the wrong
        // shape, or one the engine's own validation refuses all fail here — and
        // the catalogue is where each of them would have come from.
        let table: toml::Table = settings_help()
            .render_config_example(true)
            .parse()
            .expect("the generated example must be valid TOML");
        let settings = crate::Settings::from_table(Some(&table))
            .expect("the generated example must load through the real settings loader");
        assert_eq!(
            settings.warnings,
            vec![],
            "the example names a key no surface claims"
        );
        // The values are the ones the catalogue advertises, not merely some
        // parseable ones.
        assert_eq!(settings.config.dim, 768);
        assert_eq!(settings.embed_error_policy, crate::EmbedErrorPolicy::Fail);
        assert!(settings.embedder.is_some());
    }

    #[test]
    fn every_key_of_the_committed_example_is_commented_out() {
        // A copied example that sets forty keys freezes today's defaults into
        // somebody's config for good, so the committed file must be inert.
        let table: toml::Table = committed_example()
            .parse()
            .expect("the committed example must be valid TOML");
        for (section, entries) in &table {
            assert!(
                entries.as_table().is_some_and(toml::Table::is_empty),
                "[{section}] has an active key; the example must be all comments"
            );
        }
    }

    #[test]
    fn human_help_contains_every_documented_key() {
        let rendered = settings_help().render_human();
        for doc in DOCS {
            assert!(
                rendered.contains(doc.key),
                "missing {}.{}",
                doc.section,
                doc.key
            );
        }
    }
}
