//! CLI-side configuration glue. The engine [`Settings`] (config, embedder,
//! maintenance policy) come from the shared [`plugmem_host`] loader; only the
//! CLI-specific `import` batch size — `[maintenance].batch_size`, not an engine
//! knob — is read here, from the same config table.

use plugmem_host::SettingsError;

use crate::CliError;

/// CLI-owned config keys. The settings-help completeness test compares this
/// inventory with the host catalogue whenever the parser gains a new key.
pub(crate) const CLI_SETTING_KEYS: &[(&str, &str)] = &[("maintenance", "batch_size")];

impl From<SettingsError> for CliError {
    fn from(e: SettingsError) -> Self {
        CliError::Usage(e.to_string())
    }
}

/// The `[maintenance].batch_size` — facts per `import` batch. CLI-only (not an
/// engine knob), read from the shared [`plugmem_host::read_config`] table.
pub(crate) fn read_batch_size(table: Option<&toml::Table>) -> Option<u64> {
    table
        .and_then(|t| t.get(CLI_SETTING_KEYS[0].0))
        .and_then(toml::Value::as_table)
        .and_then(|m| m.get(CLI_SETTING_KEYS[0].1))
        .and_then(toml::Value::as_integer)
        .filter(|n| *n >= 0)
        .map(|n| n as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_reads_from_the_maintenance_table() {
        let table: toml::Table = "[maintenance]\nbatch_size = 256\n".parse().unwrap();
        assert_eq!(read_batch_size(Some(&table)), Some(256));
        // Absent section / absent key / no table → None.
        let empty: toml::Table = "[engine]\ndim = 8\n".parse().unwrap();
        assert_eq!(read_batch_size(Some(&empty)), None);
        assert_eq!(read_batch_size(None), None);
        // A negative value is rejected (treated as unset).
        let neg: toml::Table = "[maintenance]\nbatch_size = -1\n".parse().unwrap();
        assert_eq!(read_batch_size(Some(&neg)), None);
    }

    #[test]
    fn settings_error_maps_to_a_usage_error() {
        let e: CliError = SettingsError::Config("boom".into()).into();
        assert!(matches!(e, CliError::Usage(m) if m == "boom"));
    }

    #[test]
    fn every_cli_setting_is_documented() {
        let docs = plugmem_host::settings_help().docs();
        let documented: Vec<_> = docs
            .iter()
            .filter(|doc| doc.scope == plugmem_host::SettingScope::Cli)
            .map(|doc| (doc.section, doc.key))
            .collect();
        assert_eq!(documented.as_slice(), CLI_SETTING_KEYS);
    }
}
