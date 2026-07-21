//! Runtime configuration: the engine [`Config`], an optional embedder, and
//! the maintenance policy, merged from defaults, `config.toml`, and the
//! environment (precedence: flag/env > config file > default, specs/06).

use std::path::{Path, PathBuf};

use plugmem_host::{Config, Database, DatabaseBuilder, Embedder, HostError, OpenAiCompatEmbedder};

use crate::cli::Cli;
use crate::{CONFIG_DIR, CONFIG_FILE, CliError, ENV_CONFIG, ENV_EMBEDDER};

/// Resolved settings: the engine config, an optional embedder, and the
/// maintenance policy.
pub(crate) struct Settings {
    pub(crate) config: Config,
    pub(crate) embedder: Option<Box<dyn Embedder>>,
    pub(crate) maintenance: Maintenance,
}

impl Settings {
    /// Opens a read-write database, applying the maintenance policy and
    /// embedder to the builder.
    pub(crate) fn open(self, path: &Path) -> Result<Database, HostError> {
        let mut b: DatabaseBuilder = Database::builder(self.config);
        if let Some(v) = self.maintenance.snapshot_every_ops {
            b = b.snapshot_every_ops(v);
        }
        if let Some(v) = self.maintenance.snapshot_journal_bytes {
            b = b.snapshot_journal_bytes(v);
        }
        if let Some(v) = self.maintenance.maintain_every_forgets {
            b = b.maintain_every_forgets(v);
        }
        if let Some(e) = self.embedder {
            b = b.embedder(e);
        }
        Ok(b.open(path)?.0)
    }
}

/// The `[maintenance]` policy knobs (all optional; unset = engine default).
#[derive(Default)]
pub(crate) struct Maintenance {
    pub(crate) snapshot_every_ops: Option<u64>,
    pub(crate) snapshot_journal_bytes: Option<u64>,
    pub(crate) maintain_every_forgets: Option<u64>,
}

impl Maintenance {
    fn merge(&mut self, t: &toml::Table) {
        let u = |t: &toml::Table, k: &str| {
            t.get(k)
                .and_then(toml::Value::as_integer)
                .filter(|n| *n >= 0)
                .map(|n| n as u64)
        };
        if let Some(v) = u(t, "snapshot_every_ops") {
            self.snapshot_every_ops = Some(v);
        }
        if let Some(v) = u(t, "snapshot_journal_bytes") {
            self.snapshot_journal_bytes = Some(v);
        }
        if let Some(v) = u(t, "maintain_every_forgets") {
            self.maintain_every_forgets = Some(v);
        }
    }
}

/// The `[embedder]` section, before it is turned into an [`Embedder`].
#[derive(Default)]
pub(crate) struct EmbedderCfg {
    pub(crate) kind: Option<String>,
    pub(crate) url: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) api_key_env: Option<String>,
}

impl EmbedderCfg {
    fn merge(&mut self, t: &toml::Table) {
        let s = |t: &toml::Table, k: &str| t.get(k).and_then(toml::Value::as_str).map(String::from);
        if let Some(v) = s(t, "kind") {
            self.kind = Some(v);
        }
        if let Some(v) = s(t, "url") {
            self.url = Some(v);
        }
        if let Some(v) = s(t, "model") {
            self.model = Some(v);
        }
        if let Some(v) = s(t, "api_key_env") {
            self.api_key_env = Some(v);
        }
    }

    /// Builds the embedder. `kind = "none"` (or unset) → no embedder; an
    /// OpenAI-compatible `kind` (ollama/openai/lmstudio/vllm/llamacpp) needs
    /// a `url`, a `model` and `[engine].dim > 0`; an optional `api_key_env`
    /// names an environment variable holding the bearer token.
    pub(crate) fn build(&self, dim: usize) -> Result<Option<Box<dyn Embedder>>, CliError> {
        let kind = self.kind.as_deref().unwrap_or("none");
        match kind {
            "none" | "" => Ok(None),
            "ollama" | "openai" | "openai-compat" | "lmstudio" | "vllm" | "llamacpp" => {
                let url = self.url.clone().ok_or_else(|| {
                    CliError::Usage(format!("[embedder] kind \"{kind}\" needs a url"))
                })?;
                let model = self.model.clone().ok_or_else(|| {
                    CliError::Usage(format!("[embedder] kind \"{kind}\" needs a model"))
                })?;
                if dim == 0 {
                    return Err(CliError::Usage(
                        "[embedder] requires [engine].dim > 0 (the embedding size)".into(),
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
            other => Err(CliError::Usage(format!("unknown [embedder] kind: {other}"))),
        }
    }
}

/// Loads settings with precedence flag/env > config.toml > default. The
/// engine config, embedder, and maintenance policy come from `[engine]`,
/// `[embedder]` and `[maintenance]`; `$PLUGMEM_EMBEDDER` overrides the kind.
pub(crate) fn load_settings(cli: &Cli) -> Result<Settings, CliError> {
    let mut config = Config::default();
    let mut embedder = EmbedderCfg::default();
    let mut maintenance = Maintenance::default();

    if let Some(text) = read_config_file(cli.config.as_deref())? {
        let table: toml::Table = text
            .parse()
            .map_err(|e| CliError::Usage(format!("config.toml is not valid TOML: {e}")))?;
        if let Some(t) = table.get("engine").and_then(toml::Value::as_table) {
            apply_engine(&mut config, t)?;
        }
        if let Some(t) = table.get("embedder").and_then(toml::Value::as_table) {
            embedder.merge(t);
        }
        if let Some(t) = table.get("maintenance").and_then(toml::Value::as_table) {
            maintenance.merge(t);
        }
    }

    if let Some(kind) = std::env::var_os(ENV_EMBEDDER) {
        embedder.kind = Some(kind.to_string_lossy().into_owned());
    }

    let embedder = embedder.build(config.dim)?;
    Ok(Settings {
        config,
        embedder,
        maintenance,
    })
}

/// Reads the config file: an explicit `--config` must exist; otherwise
/// `$PLUGMEM_CONFIG`, then `$XDG_CONFIG_HOME/plugmem/config.toml`, are read
/// only if present. Absent default → `Ok(None)` (config is optional).
fn read_config_file(flag: Option<&Path>) -> Result<Option<String>, CliError> {
    if let Some(p) = flag {
        return std::fs::read_to_string(p)
            .map(Some)
            .map_err(|e| CliError::Usage(format!("reading config {}: {e}", p.display())));
    }
    let candidate = std::env::var_os(ENV_CONFIG)
        .map(PathBuf::from)
        .or_else(default_config_path);
    match candidate {
        Some(p) if p.exists() => std::fs::read_to_string(&p)
            .map(Some)
            .map_err(|e| CliError::Usage(format!("reading config {}: {e}", p.display()))),
        _ => Ok(None),
    }
}

/// `$XDG_CONFIG_HOME/plugmem/config.toml`, falling back to
/// `$HOME/.config/plugmem/config.toml`.
fn default_config_path() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .map(|base| base.join(CONFIG_DIR).join(CONFIG_FILE))
}

/// Applies the `[engine]` table onto a [`Config`] (the size-bearing fields;
/// tuning parameters keep their defaults). A non-integer or negative value
/// is a usage error.
pub(crate) fn apply_engine(cfg: &mut Config, t: &toml::Table) -> Result<(), CliError> {
    let fields: [(&str, &mut usize); 9] = [
        ("dim", &mut cfg.dim),
        ("max_bytes", &mut cfg.max_bytes),
        ("max_text", &mut cfg.max_text),
        ("max_blob", &mut cfg.max_blob),
        ("shards_facts", &mut cfg.shards_facts),
        ("shards_entities", &mut cfg.shards_entities),
        ("shards_edges", &mut cfg.shards_edges),
        ("shards_temporal", &mut cfg.shards_temporal),
        ("shards_postings", &mut cfg.shards_postings),
    ];
    for (key, slot) in fields {
        if let Some(v) = t.get(key) {
            let n = v.as_integer().filter(|n| *n >= 0).ok_or_else(|| {
                CliError::Usage(format!("[engine].{key} must be a non-negative integer"))
            })?;
            *slot = n as usize;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory; removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "plugmem-cfg-{tag}-{}-{}",
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
shards_facts = 16
[maintenance]
snapshot_every_ops = 50
snapshot_journal_bytes = 8192
maintain_every_forgets = 3
";
        let table: toml::Table = text.parse().unwrap();
        let mut cfg = Config::default();
        apply_engine(&mut cfg, table["engine"].as_table().unwrap()).unwrap();
        assert_eq!(cfg.dim, 384);
        assert_eq!(cfg.shards_facts, 16);
        let mut m = Maintenance::default();
        m.merge(table["maintenance"].as_table().unwrap());
        assert_eq!(m.snapshot_every_ops, Some(50));
        assert_eq!(m.snapshot_journal_bytes, Some(8192));
        assert_eq!(m.maintain_every_forgets, Some(3));

        let bad: toml::Table = "dim = \"huge\"".parse().unwrap();
        assert!(matches!(
            apply_engine(&mut cfg, &bad),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn embedder_merge_reads_every_field() {
        let mut table = toml::Table::new();
        table.insert("kind".into(), "ollama".into());
        table.insert("url".into(), "http://localhost:11434/v1".into());
        table.insert("model".into(), "nomic-embed-text".into());
        table.insert("api_key_env".into(), "SOME_ENV".into());

        let mut e = EmbedderCfg::default();
        e.merge(&table);
        assert_eq!(e.kind.as_deref(), Some("ollama"));
        assert_eq!(e.url.as_deref(), Some("http://localhost:11434/v1"));
        assert_eq!(e.model.as_deref(), Some("nomic-embed-text"));
        assert_eq!(e.api_key_env.as_deref(), Some("SOME_ENV"));
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
            config,
            embedder,
            maintenance: Maintenance {
                snapshot_every_ops: Some(4),
                snapshot_journal_bytes: Some(4096),
                maintain_every_forgets: Some(2),
            },
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
        assert!(matches!(no_url.build(384), Err(CliError::Usage(_))));
        let no_model = EmbedderCfg {
            kind: Some("ollama".into()),
            url: Some("http://x/v1".into()),
            ..Default::default()
        };
        assert!(matches!(no_model.build(384), Err(CliError::Usage(_))));
        let zero_dim = EmbedderCfg {
            kind: Some("ollama".into()),
            url: Some("http://x/v1".into()),
            model: Some("m".into()),
            api_key_env: None,
        };
        assert!(matches!(zero_dim.build(0), Err(CliError::Usage(_))));
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
        assert!(matches!(weird.build(384), Err(CliError::Usage(_))));
    }
}
