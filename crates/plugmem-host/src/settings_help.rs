//! The single source of truth for config.toml help.
//!
//! The parser lives in [`super::settings`], while CLI/MCP/NAPI are separate
//! surfaces. Keeping the public setting catalogue here lets those surfaces
//! render their own help without copying descriptions or defaults.

use std::fmt::Write as _;

const PLATFORM_DEFAULT_SOURCE: &str = "platform default config path";

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
        description: "Persistent database file; explicit --db/constructor path and PLUGMEM_DB override it",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "dim",
        value_type: "non-negative integer",
        default: "0",
        description: "Embedding dimension; 0 disables vector storage",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "max_bytes",
        value_type: "non-negative integer",
        default: "2147483648",
        description: "Total byte-pool ceiling",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "max_text",
        value_type: "non-negative integer",
        default: "4096",
        description: "Maximum fact text length in bytes",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "max_blob",
        value_type: "non-negative integer",
        default: "65536",
        description: "Maximum single blob length in bytes",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "shards_facts",
        value_type: "power-of-two integer",
        default: "1024",
        description: "Facts arena shard count",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "shards_entities",
        value_type: "power-of-two integer",
        default: "256",
        description: "Entities arena shard count",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "shards_edges",
        value_type: "power-of-two integer",
        default: "512",
        description: "Edges arena shard count",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "shards_temporal",
        value_type: "power-of-two integer",
        default: "512",
        description: "Temporal arena shard count",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "engine",
        key: "shards_postings",
        value_type: "power-of-two integer",
        default: "2048",
        description: "BM25 postings arena shard count",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "kind",
        value_type: "string",
        default: "none",
        description: "Embedding provider: none, ollama, openai, lmstudio, vllm or llamacpp",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "url",
        value_type: "string",
        default: "unset",
        description: "OpenAI-compatible /v1/embeddings endpoint",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "model",
        value_type: "string",
        default: "unset",
        description: "Embedding model name",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "embedder",
        key: "api_key_env",
        value_type: "string",
        default: "unset",
        description: "Environment variable containing the bearer token",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "snapshot_every_ops",
        value_type: "non-negative integer",
        default: "1024",
        description: "Snapshot after this many mutations",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "snapshot_journal_bytes",
        value_type: "non-negative integer",
        default: "4194304",
        description: "Snapshot when the journal reaches this size",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "maintain_every_forgets",
        value_type: "non-negative integer",
        default: "off",
        description: "Run policy maintenance after this many forgets",
        scope: SettingScope::Shared,
    },
    SettingDoc {
        section: "maintenance",
        key: "batch_size",
        value_type: "positive integer",
        default: "128",
        description: "CLI import facts per embedding request and journal fsync",
        scope: SettingScope::Cli,
    },
    SettingDoc {
        section: "server",
        key: "workers",
        value_type: "positive integer",
        default: "half of available cores",
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
    fn every_documented_setting_has_a_complete_description() {
        assert!(!DOCS.is_empty());
        for doc in DOCS {
            assert!(!doc.section.is_empty());
            assert!(!doc.key.is_empty());
            assert!(!doc.value_type.is_empty());
            assert!(!doc.default.is_empty());
            assert!(!doc.description.is_empty());
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
