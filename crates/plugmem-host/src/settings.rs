//! Shared `config.toml` loader (feature `config`): resolve the engine
//! [`Config`], an optional [`Embedder`], and the maintenance policy from a
//! TOML file plus the environment, with precedence **flag/env > config file >
//! default**.
//!
//! This is the loader the CLI and the MCP server share so they agree on config
//! semantics. It is deliberately small: it reads four shared sections —
//! `[database]` (the optional database path), `[engine]` (size-bearing
//! [`Config`] fields), `[embedder]` (an OpenAI-compatible provider), and
//! `[maintenance]` (snapshot/maintain thresholds). Keys a
//! specific wrapper owns — the CLI's `[maintenance].batch_size`, the server's
//! `[server].workers` — are **not** parsed here; a wrapper reads them from the
//! same table via [`read_config`].
//!
//! Library users who build a [`Config`] in code do not need this module (and,
//! with the feature off, do not pull the `toml` parser).

use std::path::{Path, PathBuf};

use crate::{Config, Database, DatabaseBuilder, Embedder, HostError, OpenAiCompatEmbedder};

/// Environment variable naming the config file (below an explicit path).
const ENV_CONFIG: &str = "PLUGMEM_CONFIG";
/// Environment variable selecting the embedder kind (above the config file).
const ENV_EMBEDDER: &str = "PLUGMEM_EMBEDDER";
// Keep these inventories next to the parser. The settings-help tests compare
// them with the public documentation catalogue, so adding a parser key without
// adding its help entry fails loudly.
pub(crate) const ENGINE_SETTING_KEYS: &[&str] = &["dim", "max_bytes", "max_text", "max_blob"];
pub(crate) const DATABASE_SETTING_KEYS: &[&str] = &["path"];
pub(crate) const EMBEDDER_SETTING_KEYS: &[&str] = &["kind", "url", "model", "api_key_env"];
pub(crate) const MAINTENANCE_SETTING_KEYS: &[&str] = &[
    "snapshot_every_ops",
    "snapshot_journal_bytes",
    "maintain_every_forgets",
];

/// A configuration error: malformed TOML, a bad `[engine]` value, or an
/// `[embedder]` section missing a required field. Distinct from [`HostError`]
/// (which covers opening the database once settings are resolved).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SettingsError {
    /// A usage error in the configuration (message is human-facing).
    #[error("{0}")]
    Config(String),
}

impl SettingsError {
    fn config(msg: impl Into<String>) -> Self {
        SettingsError::Config(msg.into())
    }
}

/// Resolved runtime settings: the engine config, an optional embedder, and the
/// maintenance policy. The wrapper-specific knobs (`import` batch size, server
/// workers) are read separately from the same [`read_config`] table.
pub struct Settings {
    /// `[database].path`, if set. Wrapper-specific explicit paths take
    /// precedence over this value; otherwise the platform default is used.
    pub database_path: Option<PathBuf>,
    /// The engine configuration (size-bearing fields from `[engine]`).
    pub config: Config,
    /// The embedder built from `[embedder]`, or `None` (lexical/graph/time
    /// recall still work without one).
    pub embedder: Option<Box<dyn Embedder>>,
    /// `[maintenance].snapshot_every_ops`, if set.
    pub snapshot_every_ops: Option<u64>,
    /// `[maintenance].snapshot_journal_bytes`, if set.
    pub snapshot_journal_bytes: Option<u64>,
    /// `[maintenance].maintain_every_forgets`, if set.
    pub maintain_every_forgets: Option<u64>,
}

impl Settings {
    /// Loads settings from the config file resolved by [`read_config`] (an
    /// explicit `flag` path, else `$PLUGMEM_CONFIG`, else the platform config
    /// path from [`crate::default_config_path`]). Missing config → defaults.
    pub fn load(flag: Option<&Path>) -> Result<Settings, SettingsError> {
        let table = read_config(flag)?;
        Settings::from_table(table.as_ref())
    }

    /// Builds settings from an already-parsed config table (or `None` for
    /// all defaults). `$PLUGMEM_EMBEDDER` overrides `[embedder].kind`. Use
    /// this when the caller also needs its own keys from the same table
    /// (read once via [`read_config`], then passed here).
    pub fn from_table(table: Option<&toml::Table>) -> Result<Settings, SettingsError> {
        let mut config = Config::default();
        let mut database_path = None;
        let mut embedder = EmbedderCfg::default();
        let mut snapshot_every_ops = None;
        let mut snapshot_journal_bytes = None;
        let mut maintain_every_forgets = None;

        if let Some(table) = table {
            if let Some(t) = table.get("database").and_then(toml::Value::as_table) {
                database_path = t
                    .get(DATABASE_SETTING_KEYS[0])
                    .map(|value| {
                        let path = value.as_str().ok_or_else(|| {
                            SettingsError::config("[database].path must be a string")
                        })?;
                        if path.is_empty() {
                            return Err(SettingsError::config("[database].path must not be empty"));
                        }
                        Ok(PathBuf::from(path))
                    })
                    .transpose()?;
            }
            if let Some(t) = table.get("engine").and_then(toml::Value::as_table) {
                apply_engine(&mut config, t)?;
            }
            if let Some(t) = table.get("embedder").and_then(toml::Value::as_table) {
                embedder.merge(t);
            }
            if let Some(t) = table.get("maintenance").and_then(toml::Value::as_table) {
                snapshot_every_ops = table_u64(t, MAINTENANCE_SETTING_KEYS[0]);
                snapshot_journal_bytes = table_u64(t, MAINTENANCE_SETTING_KEYS[1]);
                maintain_every_forgets = table_u64(t, MAINTENANCE_SETTING_KEYS[2]);
            }
        }

        if let Some(kind) = std::env::var_os(ENV_EMBEDDER) {
            embedder.kind = Some(kind.to_string_lossy().into_owned());
        }

        let embedder = embedder.build(config.dim)?;
        Ok(Settings {
            database_path,
            config,
            embedder,
            snapshot_every_ops,
            snapshot_journal_bytes,
            maintain_every_forgets,
        })
    }

    /// Opens a read-write [`Database`], applying the maintenance policy and
    /// embedder to the builder. Consumes `self` (the embedder moves into the
    /// database). For a read-only handle, take [`Settings::embedder`] out
    /// first, then call [`Database::open_readonly`] with [`Settings::config`].
    pub fn open(self, path: &Path) -> Result<Database, HostError> {
        let mut b: DatabaseBuilder = Database::builder(self.config);
        if let Some(v) = self.snapshot_every_ops {
            b = b.snapshot_every_ops(v);
        }
        if let Some(v) = self.snapshot_journal_bytes {
            b = b.snapshot_journal_bytes(v);
        }
        if let Some(v) = self.maintain_every_forgets {
            b = b.maintain_every_forgets(v);
        }
        if let Some(e) = self.embedder {
            b = b.embedder(e);
        }
        Ok(b.open(path)?.0)
    }
}

/// Reads and parses `config.toml`, or `Ok(None)` if none applies. An explicit
/// `flag` path **must** exist (a read error is a usage error); otherwise
/// `$PLUGMEM_CONFIG`, then the platform path from
/// [`crate::default_config_path`], are read only if present. Wrappers call this once, then pass the table to
/// [`Settings::from_table`] and also read their own keys (batch size, workers)
/// from it.
pub fn read_config(flag: Option<&Path>) -> Result<Option<toml::Table>, SettingsError> {
    let text = match read_config_text(flag)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let table: toml::Table = text
        .parse()
        .map_err(|e| SettingsError::config(format!("config.toml is not valid TOML: {e}")))?;
    Ok(Some(table))
}

/// A non-negative integer key from a table as `u64`, or `None`.
pub(crate) fn table_u64(t: &toml::Table, key: &str) -> Option<u64> {
    t.get(key)
        .and_then(toml::Value::as_integer)
        .filter(|n| *n >= 0)
        .map(|n| n as u64)
}

/// Reads the config file text with flag/env/platform-default precedence.
fn read_config_text(flag: Option<&Path>) -> Result<Option<String>, SettingsError> {
    if let Some(p) = flag {
        return std::fs::read_to_string(p)
            .map(Some)
            .map_err(|e| SettingsError::config(format!("reading config {}: {e}", p.display())));
    }
    let candidate = std::env::var_os(ENV_CONFIG)
        .map(PathBuf::from)
        .or_else(crate::default_config_path);
    match candidate {
        Some(p) if p.exists() => std::fs::read_to_string(&p)
            .map(Some)
            .map_err(|e| SettingsError::config(format!("reading config {}: {e}", p.display()))),
        _ => Ok(None),
    }
}

/// Applies the `[engine]` table onto a [`Config`] (the size-bearing fields;
/// tuning parameters keep their defaults). A non-integer or negative value is
/// a usage error.
fn apply_engine(cfg: &mut Config, t: &toml::Table) -> Result<(), SettingsError> {
    let fields: [(&str, &mut usize); ENGINE_SETTING_KEYS.len()] = [
        (ENGINE_SETTING_KEYS[0], &mut cfg.dim),
        (ENGINE_SETTING_KEYS[1], &mut cfg.max_bytes),
        (ENGINE_SETTING_KEYS[2], &mut cfg.max_text),
        (ENGINE_SETTING_KEYS[3], &mut cfg.max_blob),
    ];
    for (key, slot) in fields {
        if let Some(v) = t.get(key) {
            let n = v.as_integer().filter(|n| *n >= 0).ok_or_else(|| {
                SettingsError::config(format!("[engine].{key} must be a non-negative integer"))
            })?;
            *slot = n as usize;
        }
    }
    Ok(())
}

/// The `[embedder]` section, before it is turned into an [`Embedder`].
#[derive(Default)]
struct EmbedderCfg {
    kind: Option<String>,
    url: Option<String>,
    model: Option<String>,
    api_key_env: Option<String>,
}

impl EmbedderCfg {
    fn merge(&mut self, t: &toml::Table) {
        let s = |t: &toml::Table, k: &str| t.get(k).and_then(toml::Value::as_str).map(String::from);
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[0]) {
            self.kind = Some(v);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[1]) {
            self.url = Some(v);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[2]) {
            self.model = Some(v);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[3]) {
            self.api_key_env = Some(v);
        }
    }

    /// Builds the embedder. `kind = "none"` (or unset) → no embedder; an
    /// OpenAI-compatible `kind` (ollama/openai/lmstudio/vllm/llamacpp) needs a
    /// `url`, a `model` and `[engine].dim > 0`; an optional `api_key_env` names
    /// an environment variable holding the bearer token.
    fn build(&self, dim: usize) -> Result<Option<Box<dyn Embedder>>, SettingsError> {
        let kind = self.kind.as_deref().unwrap_or("none");
        match kind {
            "none" | "" => Ok(None),
            "ollama" | "openai" | "openai-compat" | "lmstudio" | "vllm" | "llamacpp" => {
                let url = self.url.clone().ok_or_else(|| {
                    SettingsError::config(format!("[embedder] kind \"{kind}\" needs a url"))
                })?;
                let model = self.model.clone().ok_or_else(|| {
                    SettingsError::config(format!("[embedder] kind \"{kind}\" needs a model"))
                })?;
                if dim == 0 {
                    return Err(SettingsError::config(
                        "[embedder] requires [engine].dim > 0 (the embedding size)",
                    ));
                }
                let mut e = OpenAiCompatEmbedder::new(&url, &model, dim);
                if let Some(env) = &self.api_key_env
                    && let Some(key) = std::env::var_os(env)
                {
                    e = e.with_api_key(key.to_string_lossy().into_owned());
                }
                Ok(Some(Box::new(e)))
            }
            other => Err(SettingsError::config(format!(
                "unknown [embedder] kind: {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory; removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-settings-{tag}-{}-{}",
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

    #[test]
    fn engine_and_maintenance_parse() {
        let text = "\
[engine]
dim = 384
max_text = 2048
[maintenance]
snapshot_every_ops = 50
snapshot_journal_bytes = 8192
maintain_every_forgets = 3
";
        let table: toml::Table = text.parse().unwrap();
        let s = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(s.config.dim, 384);
        assert_eq!(s.config.max_text, 2048);
        assert_eq!(s.snapshot_every_ops, Some(50));
        assert_eq!(s.snapshot_journal_bytes, Some(8192));
        assert_eq!(s.maintain_every_forgets, Some(3));

        let bad: toml::Table = "[engine]\ndim = \"huge\"".parse().unwrap();
        assert!(matches!(
            Settings::from_table(Some(&bad)),
            Err(SettingsError::Config(_))
        ));
    }

    #[test]
    fn defaults_when_no_table() {
        let s = Settings::from_table(None).unwrap();
        assert!(s.database_path.is_none());
        assert_eq!(s.config.dim, Config::default().dim);
        assert!(s.embedder.is_none());
        assert_eq!(s.snapshot_every_ops, None);
    }

    #[test]
    fn embedder_merge_reads_every_field() {
        let text = "\
[embedder]
kind = \"ollama\"
url = \"http://localhost:11434/v1\"
model = \"nomic-embed-text\"
api_key_env = \"SOME_ENV\"
[engine]
dim = 8
";
        let table: toml::Table = text.parse().unwrap();
        // An OpenAI-compatible kind with a url, model and dim > 0 builds.
        let s = Settings::from_table(Some(&table)).unwrap();
        assert!(s.embedder.is_some());
    }

    #[test]
    fn database_path_reads_and_validates_from_config() {
        let table: toml::Table = "[database]\npath = \"/tmp/memory.plugmem\""
            .parse()
            .unwrap();
        let settings = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(
            settings.database_path.as_deref(),
            Some(std::path::Path::new("/tmp/memory.plugmem"))
        );

        let bad: toml::Table = "[database]\npath = 42".parse().unwrap();
        assert!(matches!(
            Settings::from_table(Some(&bad)),
            Err(SettingsError::Config(message)) if message == "[database].path must be a string"
        ));
    }

    #[test]
    fn settings_open_applies_maintenance_and_embedder() {
        // Every maintenance knob set, plus an embedder, so `Settings::open`
        // exercises each builder branch. The embedder is never invoked by a
        // bare open, so an unreachable url is fine here.
        let tmp = TempDir::new("open");
        let mut config = Config::default();
        config.dim = 8;
        let embedder = EmbedderCfg {
            kind: Some("ollama".into()),
            url: Some("http://127.0.0.1:0/v1".into()),
            model: Some("m".into()),
            api_key_env: None,
        }
        .build(8)
        .unwrap();
        assert!(embedder.is_some());
        let settings = Settings {
            database_path: None,
            config,
            embedder,
            snapshot_every_ops: Some(4),
            snapshot_journal_bytes: Some(4096),
            maintain_every_forgets: Some(2),
        };
        let db = settings.open(&tmp.0.join("m.plugmem")).unwrap();
        assert_eq!(db.stats().facts, 0);
    }

    #[test]
    fn embedder_build_rules() {
        assert!(EmbedderCfg::default().build(0).unwrap().is_none());
        let no_url = EmbedderCfg {
            kind: Some("ollama".into()),
            ..Default::default()
        };
        assert!(matches!(no_url.build(384), Err(SettingsError::Config(_))));
        let no_model = EmbedderCfg {
            kind: Some("ollama".into()),
            url: Some("http://x/v1".into()),
            ..Default::default()
        };
        assert!(matches!(no_model.build(384), Err(SettingsError::Config(_))));
        let zero_dim = EmbedderCfg {
            kind: Some("ollama".into()),
            url: Some("http://x/v1".into()),
            model: Some("m".into()),
            api_key_env: None,
        };
        assert!(matches!(zero_dim.build(0), Err(SettingsError::Config(_))));
        let ok = EmbedderCfg {
            kind: Some("openai".into()),
            url: Some("http://x/v1".into()),
            model: Some("m".into()),
            api_key_env: Some("PLUGMEM_TEST_KEY_UNSET".into()),
        };
        assert!(ok.build(384).unwrap().is_some());
        let weird = EmbedderCfg {
            kind: Some("weird".into()),
            ..Default::default()
        };
        assert!(matches!(weird.build(384), Err(SettingsError::Config(_))));
    }

    #[test]
    fn load_reads_the_config_file() {
        let tmp = TempDir::new("load");
        let cfgfile = tmp.0.join("config.toml");
        std::fs::write(
            &cfgfile,
            "[database]\npath = \"memory.plugmem\"\n[engine]\ndim = 512\n[embedder]\nkind = \"none\"\n[maintenance]\nsnapshot_every_ops = 64\n",
        )
        .unwrap();
        let s = Settings::load(Some(&cfgfile)).unwrap();
        assert_eq!(s.database_path, Some(PathBuf::from("memory.plugmem")));
        assert_eq!(s.config.dim, 512);
        assert!(s.embedder.is_none());
        assert_eq!(s.snapshot_every_ops, Some(64));

        // An explicit path that does not exist is a usage error.
        assert!(matches!(
            Settings::load(Some(&tmp.0.join("nope.toml"))),
            Err(SettingsError::Config(_))
        ));
    }

    #[test]
    fn read_config_none_and_batch_extra() {
        // No file → Ok(None); a wrapper reads its own extra key from the table.
        let tmp = TempDir::new("extra");
        let missing = tmp.0.join("absent.toml");
        // An absent *default* (no flag) yields None only if neither env nor the
        // XDG default exists; exercise the explicit-missing-flag error instead.
        assert!(read_config(Some(&missing)).is_err());

        let cfgfile = tmp.0.join("config.toml");
        std::fs::write(&cfgfile, "[maintenance]\nbatch_size = 256\n").unwrap();
        let table = read_config(Some(&cfgfile)).unwrap().unwrap();
        let batch = table
            .get("maintenance")
            .and_then(toml::Value::as_table)
            .and_then(|m| table_u64(m, "batch_size"));
        assert_eq!(batch, Some(256));
    }

    #[test]
    fn every_host_setting_is_documented() {
        let docs = crate::settings_help::settings_help().docs();
        for (section, keys) in [
            ("database", DATABASE_SETTING_KEYS),
            ("engine", ENGINE_SETTING_KEYS),
            ("embedder", EMBEDDER_SETTING_KEYS),
            ("maintenance", MAINTENANCE_SETTING_KEYS),
        ] {
            let documented: Vec<_> = docs
                .iter()
                .filter(|doc| {
                    doc.section == section
                        && doc.scope == crate::settings_help::SettingScope::Shared
                })
                .map(|doc| doc.key)
                .collect();
            assert_eq!(
                documented.as_slice(),
                keys,
                "undocumented {section} setting"
            );
        }
    }
}
