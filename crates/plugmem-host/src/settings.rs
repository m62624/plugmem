//! Shared `config.toml` loader (feature `config`): resolve the engine
//! [`Config`], an optional [`Embedder`], and the maintenance policy from a
//! TOML file plus the environment, with precedence **flag/env > config file >
//! default**.
//!
//! This is the loader every wrapper shares, so they agree on config semantics
//! and — more to the point — so a knob added once is a knob every surface
//! offers. It reads six shared sections: `[database]` (the optional database
//! path), `[engine]` (the size-bearing [`Config`] fields a database is built
//! with), `[recall]` (what comes back for a query, and in what order),
//! `[index]` (how the vector index is built), `[embedder]` (an
//! OpenAI-compatible provider), and `[maintenance]` (snapshot/maintain
//! thresholds and the fsync policy).
//!
//! Keys a specific wrapper owns — the CLI's `[maintenance].batch_size`, the
//! server's `[server].workers` — are **not** parsed here; a wrapper reads them
//! from the same table via [`read_config`]. They are still in the catalogue,
//! because that is what tells [`crate::settings_help`] they are not typos.
//!
//! Anything else is reported through [`Settings::warnings`] rather than
//! ignored: a misspelled key changes no behaviour, and saying nothing about it
//! is how someone ends up believing they tuned something.
//!
//! Library users who build a [`Config`] in code do not need this module (and,
//! with the feature off, do not pull the `toml` parser).

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{
    Config, Database, DatabaseBuilder, EmbedErrorPolicy, EmbedRetry, Embedder, FsyncPolicy,
    HostError, MAX_OPEN_CEILING, OpenAiCompatEmbedder, Opener, SettingWarning, SharedEmbedder,
    Workspace, WorkspaceLayout, WorkspaceLimits, settings_help::settings_help,
};

/// Environment variable naming the config file (below an explicit path).
const ENV_CONFIG: &str = "PLUGMEM_CONFIG";
/// Environment variable that overrides `[embedder].enabled`.
const ENV_EMBEDDER_ENABLED: &str = "PLUGMEM_EMBEDDER_ENABLED";
/// Environment variable that overrides `[embedder].on_error`.
const ENV_EMBEDDER_ON_ERROR: &str = "PLUGMEM_EMBEDDER_ON_ERROR";
/// Environment variable that overrides `[embedder].timeout_ms`.
const ENV_EMBEDDER_TIMEOUT_MS: &str = "PLUGMEM_EMBEDDER_TIMEOUT_MS";
/// Environment variable that overrides `[embedder].retry_after_ms`.
const ENV_EMBEDDER_RETRY_AFTER_MS: &str = "PLUGMEM_EMBEDDER_RETRY_AFTER_MS";
/// Environment variable that overrides `[embedder].retry_max_ms`.
const ENV_EMBEDDER_RETRY_MAX_MS: &str = "PLUGMEM_EMBEDDER_RETRY_MAX_MS";
// Keep these inventories next to the parser. The settings-help tests compare
// them with the public documentation catalogue, so adding a parser key without
// adding its help entry fails loudly.
pub(crate) const ENGINE_SETTING_KEYS: &[&str] = &["dim", "max_bytes", "max_text", "max_blob"];
/// `[recall]` — what comes back for a query, and in what order.
///
/// A separate section from `[engine]` because it answers a different question.
/// `[engine]` is about how big things may get; these decide *answers*, and
/// folding twenty of them into one section would bury the four that govern
/// size. Every one of them may differ from what the file was written with:
/// reopening with new weights is how a caller changes the ranking.
pub(crate) const RECALL_SETTING_KEYS: &[&str] = &[
    "bm25_k1",
    "bm25_b",
    "rrf_k",
    "w_bm25",
    "w_vec",
    "w_graph",
    "w_time",
    "w_recency",
    "half_life_days",
    "graph_depth",
    "graph_decay",
    "hnsw_ef_search",
    "similar_cos",
    "similar_jaccard",
];
/// `[index]` — how the vector index is built, and when it stops being flat.
pub(crate) const INDEX_SETTING_KEYS: &[&str] = &["hnsw_ef_construction", "flat_to_hnsw"];
pub(crate) const DATABASE_SETTING_KEYS: &[&str] = &["path"];
pub(crate) const WORKSPACE_SETTING_KEYS: &[&str] = &["dir", "max_open", "idle_timeout_ms"];
pub(crate) const EMBEDDER_SETTING_KEYS: &[&str] = &[
    "enabled",
    "url",
    "model",
    "space_id",
    "api_key_env",
    "on_error",
    "timeout_ms",
    "retry_after_ms",
    "retry_max_ms",
];
pub(crate) const MAINTENANCE_SETTING_KEYS: &[&str] = &[
    "snapshot_every_ops",
    "snapshot_journal_bytes",
    "maintain_every_forgets",
    "fsync",
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
    /// `[embedder].on_error` — what a verb does when the provider cannot be
    /// reached. Defaults to [`EmbedErrorPolicy::Fail`], which is what every
    /// release before this one did.
    pub embed_error_policy: EmbedErrorPolicy,
    /// `[embedder].retry_after_ms` / `retry_max_ms` — when a database that
    /// suspended its own embedder calls it again. Inert unless the policy is
    /// [`EmbedErrorPolicy::Degrade`], since nothing else suspends by itself.
    pub embed_retry: EmbedRetry,
    /// `[maintenance].snapshot_every_ops`, if set.
    pub snapshot_every_ops: Option<u64>,
    /// `[maintenance].snapshot_journal_bytes`, if set.
    pub snapshot_journal_bytes: Option<u64>,
    /// `[maintenance].maintain_every_forgets`, if set.
    pub maintain_every_forgets: Option<u64>,
    /// `[maintenance].fsync`, if set. `None` leaves the engine default
    /// ([`FsyncPolicy::EachOp`]) — every acknowledged write survives a power
    /// cut. This is the largest single lever on write throughput, which is why
    /// changing it is a deliberate config edit and not a per-call flag.
    pub fsync: Option<FsyncPolicy>,
    /// The `[workspace]` section. Its `dir` is `None` unless the file names
    /// one — **the default is a single database**, and nothing turns a
    /// workspace on by itself.
    pub workspace: WorkspaceSettings,
    /// Sections and keys in the file that nothing claimed, in file order.
    ///
    /// Empty for a clean config, which is why this is a field rather than an
    /// error: a typo must not stop a program that was configured correctly
    /// enough to run. **Show them.** A surface that drops these is back to the
    /// silence this exists to end — see [`SettingWarning`].
    pub warnings: Vec<SettingWarning>,
}

/// The `[workspace]` section: where a directory of named databases lives, and
/// how many of them to keep open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSettings {
    /// `[workspace].dir`, if set. Unset is the default and means there is no
    /// workspace: one database, addressed by path, exactly as before.
    pub dir: Option<PathBuf>,
    /// Pool limits, defaulted when the section omits them.
    pub limits: WorkspaceLimits,
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
    /// all defaults). `$PLUGMEM_EMBEDDER_ENABLED` overrides
    /// `[embedder].enabled`. Use this when the caller also needs its own keys
    /// from the same table (read once via [`read_config`], then passed here).
    pub fn from_table(table: Option<&toml::Table>) -> Result<Settings, SettingsError> {
        let mut config = Config::default();
        let mut database_path = None;
        let mut embedder = EmbedderCfg::default();
        let mut snapshot_every_ops = None;
        let mut snapshot_journal_bytes = None;
        let mut maintain_every_forgets = None;
        let mut fsync = None;
        let mut workspace = WorkspaceSettings {
            dir: None,
            limits: WorkspaceLimits::default(),
        };
        let warnings = table
            .map(|t| settings_help().unknown_in(t))
            .unwrap_or_default();

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
            if let Some(t) = table.get("recall").and_then(toml::Value::as_table) {
                apply_recall(&mut config, t)?;
            }
            if let Some(t) = table.get("index").and_then(toml::Value::as_table) {
                apply_index(&mut config, t)?;
            }
            // Ranges are the engine's to judge, and it already knows them: a
            // weight must be finite and non-negative, `similar_cos` must be a
            // cosine. Validating here rather than per-key keeps one definition
            // of "valid" instead of a second copy that can drift from it.
            config
                .validate()
                .map_err(|e| SettingsError::config(format!("config.toml: {e}")))?;
            if let Some(t) = table.get("embedder").and_then(toml::Value::as_table) {
                embedder.merge(t)?;
            }
            if let Some(t) = table.get("maintenance").and_then(toml::Value::as_table) {
                snapshot_every_ops = table_u64(t, MAINTENANCE_SETTING_KEYS[0]);
                snapshot_journal_bytes = table_u64(t, MAINTENANCE_SETTING_KEYS[1]);
                maintain_every_forgets = table_u64(t, MAINTENANCE_SETTING_KEYS[2]);
                fsync = parse_fsync(t)?;
            }
            if let Some(t) = table.get("workspace").and_then(toml::Value::as_table) {
                workspace = parse_workspace(t)?;
            }
        }

        // Environment over file, for every key of this section rather than
        // for one of them: the operational moment these exist for - "the
        // provider is down, run without it for now" - is exactly when editing
        // a config file is the wrong thing to ask of somebody.
        if let Some(enabled) = std::env::var_os(ENV_EMBEDDER_ENABLED) {
            embedder.enabled = Some(parse_embedder_enabled(&enabled.to_string_lossy())?);
        }
        if let Some(policy) = std::env::var_os(ENV_EMBEDDER_ON_ERROR) {
            embedder.on_error = Some(parse_on_error(&policy.to_string_lossy())?);
        }
        if let Some(ms) = std::env::var_os(ENV_EMBEDDER_TIMEOUT_MS) {
            embedder.timeout = Some(parse_timeout_ms(&env_number(
                &ms.to_string_lossy(),
                ENV_EMBEDDER_TIMEOUT_MS,
            )?));
        }
        if let Some(ms) = std::env::var_os(ENV_EMBEDDER_RETRY_AFTER_MS) {
            embedder.retry_after_ms = Some(env_number(
                &ms.to_string_lossy(),
                ENV_EMBEDDER_RETRY_AFTER_MS,
            )?);
        }
        if let Some(ms) = std::env::var_os(ENV_EMBEDDER_RETRY_MAX_MS) {
            embedder.retry_max_ms = Some(env_number(
                &ms.to_string_lossy(),
                ENV_EMBEDDER_RETRY_MAX_MS,
            )?);
        }

        let embed_error_policy = embedder.on_error.unwrap_or_default();
        let embed_retry = embedder.retry();
        let embedder = embedder.build(config.dim)?;
        Ok(Settings {
            database_path,
            config,
            embedder,
            embed_error_policy,
            embed_retry,
            snapshot_every_ops,
            snapshot_journal_bytes,
            maintain_every_forgets,
            fsync,
            workspace,
            warnings,
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
        if let Some(v) = self.fsync {
            b = b.fsync(v);
        }
        if let Some(e) = self.embedder {
            b = b.embedder(e);
        }
        b = b
            .on_embed_error(self.embed_error_policy)
            .embed_retry(self.embed_retry);
        Ok(b.open(path)?.0)
    }

    /// Opens a [`Workspace`] rooted at `root`: many named databases, each built
    /// with these same settings.
    ///
    /// The embedder is shared rather than duplicated — a hundred chats pointed
    /// at one endpoint want one client, not a hundred (see [`SharedEmbedder`]).
    ///
    /// `root` is passed rather than read from [`WorkspaceSettings::dir`] so a
    /// wrapper keeps its own precedence (flag, then environment, then config),
    /// the same way it already does for the database path.
    ///
    /// # Errors
    ///
    /// Nothing yet — the databases open lazily, so a bad root is reported by
    /// the first [`Workspace::get`] rather than here. The signature is
    /// fallible because that is where the failure will move if the root ever
    /// needs validating up front.
    pub fn open_workspace(self, root: &Path) -> Result<Workspace, crate::WorkspaceError> {
        let Settings {
            config,
            embedder,
            embed_error_policy,
            embed_retry,
            snapshot_every_ops,
            snapshot_journal_bytes,
            maintain_every_forgets,
            workspace,
            ..
        } = self;
        let shared = embedder.map(SharedEmbedder::new);

        let open: Opener = Box::new(move |path: &Path| {
            let mut b = Database::builder(config.clone());
            if let Some(v) = snapshot_every_ops {
                b = b.snapshot_every_ops(v);
            }
            if let Some(v) = snapshot_journal_bytes {
                b = b.snapshot_journal_bytes(v);
            }
            if let Some(v) = maintain_every_forgets {
                b = b.maintain_every_forgets(v);
            }
            if let Some(e) = &shared {
                b = b.embedder(Box::new(e.clone()));
            }
            // Every database in a workspace shares one provider, so they must
            // also share what happens when it stops answering; a per-database
            // default here would degrade one memory and fail another against
            // the same dead endpoint.
            b = b
                .on_embed_error(embed_error_policy)
                .embed_retry(embed_retry);
            Ok(b.open(path)?.0)
        });
        Ok(Workspace::new(
            WorkspaceLayout::new(root),
            open,
            workspace.limits,
        ))
    }
}

/// Parses the `[workspace]` section. An out-of-range pool limit is a usage
/// error rather than a silent clamp: a person who wrote a number meant it, and
/// finding out later that it was ignored is worse than being told now.
fn parse_workspace(t: &toml::Table) -> Result<WorkspaceSettings, SettingsError> {
    let mut out = WorkspaceSettings {
        dir: None,
        limits: WorkspaceLimits::default(),
    };
    if let Some(value) = t.get(WORKSPACE_SETTING_KEYS[0]) {
        let dir = value
            .as_str()
            .ok_or_else(|| SettingsError::config("[workspace].dir must be a string"))?;
        if dir.is_empty() {
            return Err(SettingsError::config("[workspace].dir must not be empty"));
        }
        out.dir = Some(PathBuf::from(dir));
    }
    if let Some(n) = table_u64(t, WORKSPACE_SETTING_KEYS[1]) {
        if n == 0 || n > MAX_OPEN_CEILING as u64 {
            return Err(SettingsError::config(format!(
                "[workspace].max_open must be between 1 and {MAX_OPEN_CEILING} \
                 (one open database costs several file descriptors)"
            )));
        }
        // In range by the check above, so the narrowing cannot truncate — the
        // comparison happens in `u64` precisely so it holds where `usize` is 32
        // bits too.
        out.limits.max_open = n as usize;
    }
    if let Some(n) = table_u64(t, WORKSPACE_SETTING_KEYS[2]) {
        out.limits.idle_timeout_ms = n;
    }
    Ok(out)
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
/// Reads `[maintenance].fsync` as a named policy.
///
/// A string rather than a boolean, because the two values are not opposites of
/// one thing: `"each_op"` says *when* a record is durable, `"on_snapshot"` says
/// which window may be lost. A misspelling is refused rather than silently
/// treated as the default — quietly running with weaker durability than the
/// file asks for is the one outcome worth erroring over.
fn parse_fsync(t: &toml::Table) -> Result<Option<FsyncPolicy>, SettingsError> {
    let Some(value) = t.get(MAINTENANCE_SETTING_KEYS[3]) else {
        return Ok(None);
    };
    let name = value.as_str().ok_or_else(|| {
        SettingsError::config("[maintenance].fsync must be \"each_op\" or \"on_snapshot\"")
    })?;
    match name {
        "each_op" => Ok(Some(FsyncPolicy::EachOp)),
        "on_snapshot" => Ok(Some(FsyncPolicy::OnSnapshot)),
        other => Err(SettingsError::config(format!(
            "[maintenance].fsync must be \"each_op\" or \"on_snapshot\", got \"{other}\""
        ))),
    }
}

/// `"fail"` / `"degrade"`, from a file or from the environment.
fn parse_on_error(value: &str) -> Result<EmbedErrorPolicy, SettingsError> {
    match value {
        "fail" => Ok(EmbedErrorPolicy::Fail),
        "degrade" => Ok(EmbedErrorPolicy::Degrade),
        other => Err(SettingsError::config(format!(
            "[embedder].on_error must be \"fail\" or \"degrade\", got \"{other}\""
        ))),
    }
}

/// A non-negative integer from a TOML value, refused rather than ignored.
///
/// [`table_u64`] drops what it cannot read, which is right for a knob whose
/// absence means "engine default". These four decide whether a memory keeps
/// working when its provider dies, and a silently dropped `timeout_ms = "5s"`
/// would leave somebody sure they had bounded a wait they had not.
fn table_number(value: &toml::Value, key: &str) -> Result<u64, SettingsError> {
    value
        .as_integer()
        .filter(|n| *n >= 0)
        .map(|n| n as u64)
        .ok_or_else(|| {
            SettingsError::config(format!(
                "[embedder].{key} must be a non-negative integer number of milliseconds"
            ))
        })
}

/// The same, from an environment variable.
fn env_number(value: &str, var: &str) -> Result<u64, SettingsError> {
    value.trim().parse::<u64>().map_err(|_| {
        SettingsError::config(format!(
            "{var} must be a non-negative integer number of milliseconds, got \"{value}\""
        ))
    })
}

/// `0` means "no timeout" — the one spelling of "wait indefinitely" a TOML
/// integer has, and the behaviour every release before this one had.
fn parse_timeout_ms(ms: &u64) -> Option<Duration> {
    (*ms > 0).then(|| Duration::from_millis(*ms))
}

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

/// A non-negative integer from `[section].key`, or `None` when absent.
fn setting_uint(t: &toml::Table, section: &str, key: &str) -> Result<Option<i64>, SettingsError> {
    let Some(v) = t.get(key) else {
        return Ok(None);
    };
    v.as_integer().filter(|n| *n >= 0).map(Some).ok_or_else(|| {
        SettingsError::config(format!("[{section}].{key} must be a non-negative integer"))
    })
}

/// A number from `[section].key` as `f32`, or `None` when absent.
///
/// An integer is accepted for a float key: `w_vec = 1` is what anyone writes,
/// and refusing it over the missing decimal point would be pedantry.
fn setting_f32(t: &toml::Table, section: &str, key: &str) -> Result<Option<f32>, SettingsError> {
    let Some(v) = t.get(key) else {
        return Ok(None);
    };
    v.as_float()
        .or_else(|| v.as_integer().map(|n| n as f64))
        .map(|n| Some(n as f32))
        .ok_or_else(|| SettingsError::config(format!("[{section}].{key} must be a number")))
}

/// Applies the `[engine]` table onto a [`Config`]: the size-bearing fields,
/// the ones a database is *built* with. See [`ENGINE_SETTING_KEYS`].
fn apply_engine(cfg: &mut Config, t: &toml::Table) -> Result<(), SettingsError> {
    let fields: [(&str, &mut usize); ENGINE_SETTING_KEYS.len()] = [
        (ENGINE_SETTING_KEYS[0], &mut cfg.dim),
        (ENGINE_SETTING_KEYS[1], &mut cfg.max_bytes),
        (ENGINE_SETTING_KEYS[2], &mut cfg.max_text),
        (ENGINE_SETTING_KEYS[3], &mut cfg.max_blob),
    ];
    for (key, slot) in fields {
        if let Some(n) = setting_uint(t, "engine", key)? {
            *slot = n as usize;
        }
    }
    Ok(())
}

/// Applies the `[recall]` table onto a [`Config`]. See
/// [`RECALL_SETTING_KEYS`] for why these are their own section.
fn apply_recall(cfg: &mut Config, t: &toml::Table) -> Result<(), SettingsError> {
    let floats: [(&str, &mut f32); 10] = [
        (RECALL_SETTING_KEYS[0], &mut cfg.bm25_k1),
        (RECALL_SETTING_KEYS[1], &mut cfg.bm25_b),
        (RECALL_SETTING_KEYS[3], &mut cfg.w_bm25),
        (RECALL_SETTING_KEYS[4], &mut cfg.w_vec),
        (RECALL_SETTING_KEYS[5], &mut cfg.w_graph),
        (RECALL_SETTING_KEYS[6], &mut cfg.w_time),
        (RECALL_SETTING_KEYS[7], &mut cfg.w_recency),
        (RECALL_SETTING_KEYS[10], &mut cfg.graph_decay),
        (RECALL_SETTING_KEYS[12], &mut cfg.similar_cos),
        (RECALL_SETTING_KEYS[13], &mut cfg.similar_jaccard),
    ];
    for (key, slot) in floats {
        if let Some(v) = setting_f32(t, "recall", key)? {
            *slot = v;
        }
    }
    let uints: [(&str, &mut u32); 3] = [
        (RECALL_SETTING_KEYS[2], &mut cfg.rrf_k),
        (RECALL_SETTING_KEYS[8], &mut cfg.half_life_days),
        (RECALL_SETTING_KEYS[9], &mut cfg.graph_depth),
    ];
    for (key, slot) in uints {
        if let Some(n) = setting_uint(t, "recall", key)? {
            *slot = n as u32;
        }
    }
    if let Some(n) = setting_uint(t, "recall", RECALL_SETTING_KEYS[11])? {
        cfg.hnsw_ef_search = n as usize;
    }
    Ok(())
}

/// Applies the `[index]` table onto a [`Config`].
fn apply_index(cfg: &mut Config, t: &toml::Table) -> Result<(), SettingsError> {
    let fields: [(&str, &mut usize); INDEX_SETTING_KEYS.len()] = [
        (INDEX_SETTING_KEYS[0], &mut cfg.hnsw_ef_construction),
        (INDEX_SETTING_KEYS[1], &mut cfg.flat_to_hnsw),
    ];
    for (key, slot) in fields {
        if let Some(n) = setting_uint(t, "index", key)? {
            *slot = n as usize;
        }
    }
    Ok(())
}

/// The `[embedder]` section, before it is turned into an [`Embedder`].
#[derive(Default)]
struct EmbedderCfg {
    enabled: Option<bool>,
    url: Option<String>,
    model: Option<String>,
    space_id: Option<String>,
    api_key_env: Option<String>,
    on_error: Option<EmbedErrorPolicy>,
    /// `None` = the provider's own default; `Some(None)` = wait forever.
    timeout: Option<Option<Duration>>,
    /// Milliseconds as written: `None` = backoff, `Some(0)` = manual,
    /// `Some(n)` = a fixed interval. Turned into an [`EmbedRetry`] by
    /// [`EmbedderCfg::retry`], which is the only place that mapping lives.
    retry_after_ms: Option<u64>,
    retry_max_ms: Option<u64>,
}

impl EmbedderCfg {
    fn merge(&mut self, t: &toml::Table) -> Result<(), SettingsError> {
        let s = |t: &toml::Table, k: &str| t.get(k).and_then(toml::Value::as_str).map(String::from);
        if let Some(value) = t.get(EMBEDDER_SETTING_KEYS[0]) {
            self.enabled =
                Some(value.as_bool().ok_or_else(|| {
                    SettingsError::config("[embedder].enabled must be a boolean")
                })?);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[1]) {
            self.url = Some(v);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[2]) {
            self.model = Some(v);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[3]) {
            self.space_id = Some(v);
        }
        if let Some(v) = s(t, EMBEDDER_SETTING_KEYS[4]) {
            self.api_key_env = Some(v);
        }
        if let Some(value) = t.get(EMBEDDER_SETTING_KEYS[5]) {
            let text = value
                .as_str()
                .ok_or_else(|| SettingsError::config("[embedder].on_error must be a string"))?;
            self.on_error = Some(parse_on_error(text)?);
        }
        if let Some(value) = t.get(EMBEDDER_SETTING_KEYS[6]) {
            self.timeout = Some(parse_timeout_ms(&table_number(
                value,
                EMBEDDER_SETTING_KEYS[6],
            )?));
        }
        if let Some(value) = t.get(EMBEDDER_SETTING_KEYS[7]) {
            self.retry_after_ms = Some(table_number(value, EMBEDDER_SETTING_KEYS[7])?);
        }
        if let Some(value) = t.get(EMBEDDER_SETTING_KEYS[8]) {
            self.retry_max_ms = Some(table_number(value, EMBEDDER_SETTING_KEYS[8])?);
        }
        Ok(())
    }

    /// How a suspended embedder comes back, from the two milliseconds keys.
    ///
    /// One function so the file, the environment and the defaults cannot
    /// disagree about what "0" means, and so the mapping is documented in
    /// exactly one place:
    ///
    /// - nothing written -> the default backoff (1s doubling to `retry_max_ms`);
    /// - `retry_after_ms = 0` -> never; the host resumes it explicitly;
    /// - `retry_after_ms = n` -> that interval, every time.
    fn retry(&self) -> EmbedRetry {
        match self.retry_after_ms {
            None => EmbedRetry::Backoff {
                first: crate::DEFAULT_EMBED_RETRY_FIRST,
                max: self
                    .retry_max_ms
                    .map_or(crate::DEFAULT_EMBED_RETRY_MAX, Duration::from_millis),
            },
            Some(0) => EmbedRetry::Manual,
            Some(ms) => EmbedRetry::Fixed(Duration::from_millis(ms)),
        }
    }

    /// Builds the one supported embedder. An explicitly disabled embedder, or
    /// an absent/incomplete section with no activation request, produces no
    /// embedder. An active embedder needs a `url`, a `model` and
    /// `[engine].dim > 0`; an optional `api_key_env` names an environment
    /// variable holding the bearer token.
    fn build(&self, dim: usize) -> Result<Option<Box<dyn Embedder>>, SettingsError> {
        let enabled = self
            .enabled
            .unwrap_or(self.url.is_some() || self.model.is_some());
        if !enabled {
            return Ok(None);
        }
        let url = self
            .url
            .clone()
            .ok_or_else(|| SettingsError::config("[embedder] enabled embedder needs a URL"))?;
        let model = self
            .model
            .clone()
            .ok_or_else(|| SettingsError::config("[embedder] enabled embedder needs a model"))?;
        if dim == 0 {
            return Err(SettingsError::config(
                "[embedder] requires [engine].dim > 0 (the embedding size)",
            ));
        }
        let mut e = OpenAiCompatEmbedder::new(&url, &model, dim);
        if let Some(timeout) = self.timeout {
            e = e.with_timeout(timeout);
        }
        if let Some(space_id) = &self.space_id {
            e = e.with_space_id(space_id);
        }
        if let Some(env) = &self.api_key_env
            && let Some(key) = std::env::var_os(env)
        {
            e = e.with_api_key(key.to_string_lossy().into_owned());
        }
        Ok(Some(Box::new(e)))
    }
}

fn parse_embedder_enabled(value: &str) -> Result<bool, SettingsError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(SettingsError::config(format!(
            "{ENV_EMBEDDER_ENABLED} must be true or false, got \"{other}\""
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config table from lines, so the fixtures indent with the code instead
    /// of being pinned to the file's left margin.
    fn toml_of(lines: &[&str]) -> toml::Table {
        lines.join("\n").parse().expect("valid TOML fixture")
    }

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
        let table = toml_of(&[
            "[engine]",
            "dim = 384",
            "max_text = 2048",
            "[maintenance]",
            "snapshot_every_ops = 50",
            "snapshot_journal_bytes = 8192",
            "maintain_every_forgets = 3",
        ]);
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
        let table = toml_of(&[
            "[embedder]",
            "enabled = true",
            r#"url = "http://localhost:11434/v1/embeddings""#,
            r#"model = "nomic-embed-text""#,
            r#"space_id = "nomic-embed-text@v1""#,
            r#"api_key_env = "SOME_ENV""#,
            "[engine]",
            "dim = 8",
        ]);
        // The shared OpenAI-compatible client builds with a url, model and
        // dim > 0; the server may be OpenAI, Ollama or another compatible one.
        let s = Settings::from_table(Some(&table)).unwrap();
        let embedder = s.embedder.unwrap();
        assert_eq!(embedder.space_id(), "nomic-embed-text@v1");
        assert_eq!(embedder.dim(), 8);
    }

    #[test]
    fn embedder_failure_policy_reads_every_key() {
        let table = toml_of(&[
            "[embedder]",
            "enabled = true",
            r#"url = "http://localhost:11434/v1/embeddings""#,
            r#"model = "nomic-embed-text""#,
            r#"on_error = "degrade""#,
            "timeout_ms = 2500",
            "retry_after_ms = 750",
            "[engine]",
            "dim = 8",
        ]);
        let s = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(s.embed_error_policy, EmbedErrorPolicy::Degrade);
        assert_eq!(s.embed_retry, EmbedRetry::Fixed(Duration::from_millis(750)));
    }

    #[test]
    fn an_unconfigured_embedder_section_keeps_the_old_behaviour() {
        // The default matters more than the feature: somebody who upgrades and
        // changes nothing must still get the error they get today.
        let table = toml_of(&["[engine]", "dim = 8"]);
        let s = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(s.embed_error_policy, EmbedErrorPolicy::Fail);
        assert_eq!(
            s.embed_retry,
            EmbedRetry::Backoff {
                first: crate::DEFAULT_EMBED_RETRY_FIRST,
                max: crate::DEFAULT_EMBED_RETRY_MAX,
            }
        );
    }

    #[test]
    fn retry_keys_map_onto_the_three_shapes() {
        // One table, because the mapping is the thing being tested and it is
        // easy to get one arm of it wrong in isolation.
        let cfg = |lines: &[&str]| {
            let mut lines = lines.to_vec();
            lines.insert(0, "[embedder]");
            Settings::from_table(Some(&toml_of(&lines)))
                .unwrap()
                .embed_retry
        };
        assert_eq!(cfg(&["retry_after_ms = 0"]), EmbedRetry::Manual);
        assert_eq!(
            cfg(&["retry_after_ms = 250"]),
            EmbedRetry::Fixed(Duration::from_millis(250))
        );
        assert_eq!(
            cfg(&["retry_max_ms = 5000"]),
            EmbedRetry::Backoff {
                first: crate::DEFAULT_EMBED_RETRY_FIRST,
                max: Duration::from_millis(5000),
            }
        );
        // A cap without a doubling to cap is not an error, it is simply unused
        // — saying so in a test keeps somebody from "fixing" it later.
        assert_eq!(
            cfg(&["retry_after_ms = 100", "retry_max_ms = 5000"]),
            EmbedRetry::Fixed(Duration::from_millis(100))
        );
    }

    #[test]
    fn a_malformed_failure_key_is_refused_rather_than_ignored() {
        // These four decide whether a memory keeps working when its provider
        // dies. A dropped value would leave somebody sure they configured
        // something they did not.
        for lines in [
            vec!["[embedder]", r#"on_error = "sometimes""#],
            vec!["[embedder]", "on_error = true"],
            vec!["[embedder]", r#"timeout_ms = "5s""#],
            vec!["[embedder]", "timeout_ms = -1"],
            vec!["[embedder]", r#"retry_after_ms = "soon""#],
            vec!["[embedder]", "retry_max_ms = -5"],
        ] {
            let table = toml_of(&lines);
            assert!(
                matches!(
                    Settings::from_table(Some(&table)),
                    Err(SettingsError::Config(_))
                ),
                "accepted {lines:?}"
            );
        }
    }

    #[test]
    fn a_zero_timeout_means_wait_indefinitely() {
        // TOML has no "unset" to write in place of a number, so zero carries
        // it — the same spelling every release before this one had by default.
        assert_eq!(parse_timeout_ms(&0), None);
        assert_eq!(parse_timeout_ms(&1500), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn the_environment_parsers_answer_the_same_way_the_file_does() {
        // The env overrides cannot be exercised through `std::env::set_var`
        // (it is unsafe and racy across test threads), so the parsers they
        // share with the file are tested directly instead.
        assert_eq!(
            parse_on_error("degrade").unwrap(),
            EmbedErrorPolicy::Degrade
        );
        assert_eq!(parse_on_error("fail").unwrap(), EmbedErrorPolicy::Fail);
        assert!(parse_on_error("Degrade").is_err());
        assert_eq!(
            env_number("2500", "PLUGMEM_EMBEDDER_TIMEOUT_MS").unwrap(),
            2500
        );
        assert!(env_number("-1", "PLUGMEM_EMBEDDER_TIMEOUT_MS").is_err());
        assert!(env_number("2.5", "PLUGMEM_EMBEDDER_TIMEOUT_MS").is_err());
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
            enabled: Some(true),
            url: Some("http://127.0.0.1:0/v1/embeddings".into()),
            model: Some("m".into()),
            ..Default::default()
        }
        .build(8)
        .unwrap();
        assert!(embedder.is_some());
        let settings = Settings {
            database_path: None,
            config,
            embedder,
            embed_error_policy: EmbedErrorPolicy::default(),
            embed_retry: EmbedRetry::default(),
            snapshot_every_ops: Some(4),
            snapshot_journal_bytes: Some(4096),
            maintain_every_forgets: Some(2),
            fsync: Some(FsyncPolicy::OnSnapshot),
            workspace: WorkspaceSettings {
                dir: None,
                limits: WorkspaceLimits::default(),
            },
            warnings: Vec::new(),
        };
        let db = settings.open(&tmp.0.join("m.plugmem")).unwrap();
        assert_eq!(db.stats().facts, 0);
    }

    #[test]
    fn the_workspace_section_is_absent_by_default_and_parsed_when_present() {
        // The default is one database: no section, no workspace, nothing to
        // configure. This is the case that must never drift.
        let bare = Settings::from_table(None).unwrap();
        assert_eq!(bare.workspace.dir, None);
        assert_eq!(bare.workspace.limits, WorkspaceLimits::default());

        let table: toml::Table =
            "[workspace]\ndir = \"/srv/bot\"\nmax_open = 4\nidle_timeout_ms = 5000\n"
                .parse()
                .unwrap();
        let s = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(s.workspace.dir, Some(PathBuf::from("/srv/bot")));
        assert_eq!(s.workspace.limits.max_open, 4);
        assert_eq!(s.workspace.limits.idle_timeout_ms, 5_000);

        // A section that only sets the directory keeps the defaults.
        let only_dir: toml::Table = "[workspace]\ndir = \"/srv/bot\"\n".parse().unwrap();
        let s = Settings::from_table(Some(&only_dir)).unwrap();
        assert_eq!(s.workspace.limits, WorkspaceLimits::default());
    }

    #[test]
    fn a_workspace_pool_limit_out_of_range_is_a_usage_error() {
        // Not clamped: a number somebody wrote is a number they meant, and
        // discovering later that it was ignored is worse than being told now.
        for bad in [
            "[workspace]\nmax_open = 0\n".to_string(),
            format!("[workspace]\nmax_open = {}\n", MAX_OPEN_CEILING + 1),
            // Well past what a 32-bit `usize` could hold, so the range check
            // has to happen before the narrowing.
            "[workspace]\nmax_open = 9999999999\n".to_string(),
        ] {
            let table: toml::Table = bad.parse().unwrap();
            assert!(
                matches!(Settings::from_table(Some(&table)), Err(SettingsError::Config(m)) if m.contains("max_open")),
                "{bad}"
            );
        }

        for bad in ["[workspace]\ndir = 42\n", "[workspace]\ndir = \"\"\n"] {
            let table: toml::Table = bad.parse().unwrap();
            assert!(
                matches!(Settings::from_table(Some(&table)), Err(SettingsError::Config(m)) if m.contains("dir")),
                "{bad}"
            );
        }

        // The largest accepted value is accepted.
        let table: toml::Table = format!("[workspace]\nmax_open = {MAX_OPEN_CEILING}\n")
            .parse()
            .unwrap();
        let s = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(s.workspace.limits.max_open, MAX_OPEN_CEILING);
    }

    #[test]
    fn open_workspace_builds_databases_from_the_same_settings() {
        let tmp = TempDir::new("open-workspace");
        let table: toml::Table = "[engine]\ndim = 8\n[maintenance]\nsnapshot_every_ops = 4\n\
             snapshot_journal_bytes = 4096\nmaintain_every_forgets = 2\n"
            .parse()
            .unwrap();
        let settings = Settings::from_table(Some(&table)).unwrap();
        let ws = settings.open_workspace(&tmp.0).unwrap();

        let name = crate::DbName::parse("chat-42").unwrap();
        let db = ws.get(&name, 1_000, crate::IfMissing::Create).unwrap();
        db.remember(crate::RememberInput::text(1_000, "prefers tokio"))
            .unwrap();
        assert_eq!(db.stats().facts, 1);
        assert!(ws.layout().exists(&name));
    }

    #[test]
    fn fsync_policy_is_named_and_a_misspelling_is_refused() {
        let parse = |body: &str| {
            let table: toml::Table = body.parse().unwrap();
            let t = table.get("maintenance").unwrap().as_table().unwrap();
            parse_fsync(t)
        };

        assert_eq!(
            parse("[maintenance]\n").unwrap(),
            None,
            "absent stays default"
        );
        assert_eq!(
            parse("[maintenance]\nfsync = \"each_op\"\n").unwrap(),
            Some(FsyncPolicy::EachOp)
        );
        assert_eq!(
            parse("[maintenance]\nfsync = \"on_snapshot\"\n").unwrap(),
            Some(FsyncPolicy::OnSnapshot)
        );

        // The one thing worth erroring over: a typo must not quietly leave the
        // database running with different durability than the file asks for.
        for bad in [
            "[maintenance]\nfsync = \"on-snapshot\"\n",
            "[maintenance]\nfsync = \"none\"\n",
            "[maintenance]\nfsync = true\n",
            "[maintenance]\nfsync = 1\n",
        ] {
            let Err(err) = parse(bad) else {
                panic!("{bad:?} must be refused");
            };
            assert!(
                err.to_string().contains("each_op"),
                "the message names the legal values: {err}"
            );
        }
    }

    #[test]
    fn fsync_reaches_settings_from_the_config_file() {
        // The gap this closes: `FsyncPolicy` was public in the host and
        // reachable from nowhere else — not a CLI flag, not an MCP argument,
        // not a napi option, not the config file. Only hand-written Rust.
        let table: toml::Table = "[maintenance]\nfsync = \"on_snapshot\"\n".parse().unwrap();
        let settings = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(settings.fsync, Some(FsyncPolicy::OnSnapshot));

        let plain = Settings::from_table(None).unwrap();
        assert_eq!(plain.fsync, None, "no config means the engine default");
    }

    #[test]
    fn embedder_build_rules() {
        assert!(EmbedderCfg::default().build(0).unwrap().is_none());
        let no_url = EmbedderCfg {
            enabled: Some(true),
            ..Default::default()
        };
        assert!(matches!(no_url.build(384), Err(SettingsError::Config(_))));
        let no_model = EmbedderCfg {
            enabled: Some(true),
            url: Some("http://x/v1/embeddings".into()),
            ..Default::default()
        };
        assert!(matches!(no_model.build(384), Err(SettingsError::Config(_))));
        let zero_dim = EmbedderCfg {
            enabled: Some(true),
            url: Some("http://x/v1/embeddings".into()),
            model: Some("m".into()),
            ..Default::default()
        };
        assert!(matches!(zero_dim.build(0), Err(SettingsError::Config(_))));
        let ok = EmbedderCfg {
            enabled: None,
            url: Some("http://x/v1/embeddings".into()),
            model: Some("m".into()),
            api_key_env: Some("PLUGMEM_TEST_KEY_UNSET".into()),
            ..Default::default()
        };
        assert!(ok.build(384).unwrap().is_some());
        let disabled = EmbedderCfg {
            enabled: Some(false),
            url: Some("http://x/v1/embeddings".into()),
            model: Some("m".into()),
            ..Default::default()
        };
        assert!(disabled.build(0).unwrap().is_none());
        assert!(parse_embedder_enabled("true").unwrap());
        assert!(!parse_embedder_enabled("false").unwrap());
        assert!(parse_embedder_enabled("ollama").is_err());
    }

    #[test]
    fn load_reads_the_config_file() {
        let tmp = TempDir::new("load");
        let cfgfile = tmp.0.join("config.toml");
        std::fs::write(
            &cfgfile,
            "[database]\npath = \"memory.plugmem\"\n[engine]\ndim = 512\n[embedder]\nenabled = false\n[maintenance]\nsnapshot_every_ops = 64\n",
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
    fn every_tuning_key_actually_reaches_the_config() {
        // The test the missing one would have caught. Documenting a key and
        // parsing it are two different acts, and for the whole of 0.5.0 the
        // catalogue could have promised a knob that went nowhere: nothing
        // compared a *value* written in the file against the `Config` that came
        // out. Each key here is set to something no default equals, then read
        // back off the resolved config.
        let cfg = Config::default();
        let table = toml_of(&[
            "[recall]",
            "bm25_k1 = 2.5",
            "bm25_b = 0.25",
            "rrf_k = 17",
            "w_bm25 = 3.0",
            "w_vec = 4.0",
            "w_graph = 5.0",
            "w_time = 6.0",
            "w_recency = 0.75",
            "half_life_days = 7",
            "graph_depth = 4",
            "graph_decay = 0.125",
            "hnsw_ef_search = 111",
            "similar_cos = 0.31",
            "similar_jaccard = 0.32",
            "[index]",
            "hnsw_ef_construction = 222",
            "flat_to_hnsw = 333",
        ]);
        let s = Settings::from_table(Some(&table)).unwrap();

        assert_eq!(s.config.bm25_k1, 2.5);
        assert_eq!(s.config.bm25_b, 0.25);
        assert_eq!(s.config.rrf_k, 17);
        assert_eq!(s.config.w_bm25, 3.0);
        assert_eq!(s.config.w_vec, 4.0);
        assert_eq!(s.config.w_graph, 5.0);
        assert_eq!(s.config.w_time, 6.0);
        assert_eq!(s.config.w_recency, 0.75);
        assert_eq!(s.config.half_life_days, 7);
        assert_eq!(s.config.graph_depth, 4);
        assert_eq!(s.config.graph_decay, 0.125);
        assert_eq!(s.config.hnsw_ef_search, 111);
        assert_eq!(s.config.similar_cos, 0.31);
        assert_eq!(s.config.similar_jaccard, 0.32);
        assert_eq!(s.config.hnsw_ef_construction, 222);
        assert_eq!(s.config.flat_to_hnsw, 333);

        // Every value above differs from its default, so the assertions cannot
        // pass on a parser that read nothing at all.
        assert_ne!(s.config.bm25_k1, cfg.bm25_k1);
        assert_ne!(s.config.flat_to_hnsw, cfg.flat_to_hnsw);
        assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    }

    #[test]
    fn an_integer_is_accepted_where_a_float_is_meant() {
        // `w_vec = 1` is what a person writes. Refusing it over the missing
        // decimal point would be pedantry, and the failure would be a warning
        // about a key that is spelled perfectly.
        let table = toml_of(&["[recall]", "w_vec = 2", "graph_decay = 1"]);
        let s = Settings::from_table(Some(&table)).unwrap();
        assert_eq!(s.config.w_vec, 2.0);
        assert_eq!(s.config.graph_decay, 1.0);
    }

    #[test]
    fn a_tuning_value_out_of_range_is_refused_by_name() {
        // The range belongs to the engine, and it names the field it rejected;
        // this only has to carry that through instead of inventing a second,
        // drifting copy of what "valid" means.
        for line in ["graph_decay = 2.0", "similar_cos = -1.0", "w_vec = -0.5"] {
            let table = toml_of(&["[recall]", line]);
            let Err(SettingsError::Config(message)) = Settings::from_table(Some(&table)) else {
                panic!("{line} must be refused");
            };
            let field = line.split(' ').next().unwrap();
            assert!(
                message.contains(field),
                "the message must name the offending field: {message}"
            );
        }

        // A wrong *type* is caught before the engine sees it, and names the
        // section too, since the same key name can live in more than one.
        let table = toml_of(&["[recall]", r#"w_vec = "lots""#]);
        let Err(SettingsError::Config(message)) = Settings::from_table(Some(&table)) else {
            panic!("a string weight must be refused");
        };
        assert!(message.contains("[recall].w_vec"), "{message}");
    }

    #[test]
    fn every_host_setting_is_documented() {
        let docs = crate::settings_help::settings_help().docs();
        for (section, keys) in [
            ("database", DATABASE_SETTING_KEYS),
            ("workspace", WORKSPACE_SETTING_KEYS),
            ("engine", ENGINE_SETTING_KEYS),
            ("recall", RECALL_SETTING_KEYS),
            ("index", INDEX_SETTING_KEYS),
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
