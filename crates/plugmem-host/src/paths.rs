//! Platform-aware per-user paths shared by the host and its wrappers.

use directories::ProjectDirs;
use std::path::PathBuf;

const CONFIG_FILE: &str = "config.toml";
const DATABASE_FILE: &str = "memory.plugmem";

fn project_dirs() -> Option<ProjectDirs> {
    // Empty qualifier and organization keep the user-facing project name
    // simply `plugmem` on every supported platform.
    ProjectDirs::from("", "", "plugmem")
}

/// The platform's conventional per-user config directory.
pub fn default_config_dir() -> Option<PathBuf> {
    project_dirs().map(|dirs| dirs.config_dir().to_path_buf())
}

/// The platform's conventional per-user data directory.
pub fn default_data_dir() -> Option<PathBuf> {
    // The database is local state, so Windows uses LocalAppData rather than
    // the roaming data directory. On Linux and macOS this is the conventional
    // project data directory.
    project_dirs().map(|dirs| dirs.data_local_dir().to_path_buf())
}

/// The default config file path, when a platform user directory is available.
pub fn default_config_path() -> Option<PathBuf> {
    default_config_dir().map(|dir| dir.join(CONFIG_FILE))
}

/// The default persistent per-user database path.
pub fn default_database_path() -> Option<PathBuf> {
    default_data_dir().map(|dir| dir.join(DATABASE_FILE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_paths_have_stable_app_names() {
        if let Some(path) = default_config_path() {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some(CONFIG_FILE)
            );
            assert!(path.to_string_lossy().contains("plugmem"));
        }
        if let Some(path) = default_database_path() {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some(DATABASE_FILE)
            );
            assert!(path.to_string_lossy().contains("plugmem"));
        }
    }
}
