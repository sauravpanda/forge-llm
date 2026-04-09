//! HuggingFace Hub integration — resolve model IDs to local paths.
//!
//! Supports downloading GGUF model files and tokenizer.json from HF Hub.
//! Uses `huggingface-cli` if available, otherwise provides manual download instructions.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Check if a string looks like a HuggingFace model ID (org/model format).
pub fn is_hf_model_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    parts.len() == 2
        && !s.ends_with(".gguf")
        && !s.ends_with(".json")
        && !s.ends_with(".bin")
        && !Path::new(s).exists()
        && parts[0].len() > 1
        && parts[1].len() > 1
}

/// Resolve a HF model ID to local GGUF and tokenizer paths.
///
/// Returns (gguf_path, tokenizer_path).
/// Downloads files if not already cached.
pub fn resolve_model(model_id: &str) -> Result<(PathBuf, PathBuf), HubError> {
    // Try to find a GGUF variant first
    // Common pattern: if model_id is "HuggingFaceTB/SmolLM2-135M-Instruct",
    // check for "bartowski/SmolLM2-135M-Instruct-GGUF" or the model itself
    let gguf_repo = find_gguf_repo(model_id);

    eprintln!("Downloading from HuggingFace Hub...");

    // Download GGUF file
    let gguf_path = download_gguf(&gguf_repo)?;

    // Download tokenizer from original repo
    let tokenizer_path = download_file(model_id, "tokenizer.json")?;

    Ok((gguf_path, tokenizer_path))
}

/// Find the GGUF repo for a given model ID.
/// Many models have GGUF variants published by bartowski or the original author.
fn find_gguf_repo(model_id: &str) -> String {
    // Check if the model ID already contains GGUF
    if model_id.to_lowercase().contains("gguf") {
        return model_id.to_string();
    }

    // Try common GGUF repo patterns
    let parts: Vec<&str> = model_id.split('/').collect();
    if parts.len() == 2 {
        // Try bartowski variant (most common GGUF publisher)
        let bartowski = format!("bartowski/{}-GGUF", parts[1]);
        if repo_exists(&bartowski) {
            return bartowski;
        }
        // Try original repo with -GGUF suffix
        let original_gguf = format!("{}-GGUF", model_id);
        if repo_exists(&original_gguf) {
            return original_gguf;
        }
    }

    // Fall back to original repo
    model_id.to_string()
}

/// Check if a HF repo exists using huggingface-cli.
fn repo_exists(repo_id: &str) -> bool {
    Command::new("huggingface-cli")
        .args(["repo", "info", repo_id])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Download a GGUF file from a repo, preferring Q8_0 quantization.
fn download_gguf(repo_id: &str) -> Result<PathBuf, HubError> {
    // Try common GGUF filename patterns
    let model_base = repo_id
        .split('/')
        .nth(1)
        .unwrap_or("model")
        .trim_end_matches("-GGUF");

    let patterns = [
        format!("{model_base}-Q8_0.gguf"),
        format!("{model_base}.Q8_0.gguf"),
        format!("{}-Q8_0.gguf", model_base.to_lowercase()),
    ];

    for pattern in &patterns {
        match download_file(repo_id, pattern) {
            Ok(path) => return Ok(path),
            Err(_) => continue,
        }
    }

    // Fall back to downloading any .gguf file with glob pattern
    download_file_glob(repo_id, "*.gguf")
}

/// Download a specific file from a HF repo.
fn download_file(repo_id: &str, filename: &str) -> Result<PathBuf, HubError> {
    let output = Command::new("huggingface-cli")
        .args(["download", repo_id, filename, "--quiet"])
        .output()
        .map_err(|e| HubError::CommandFailed(format!("huggingface-cli not found: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HubError::DownloadFailed(format!(
            "failed to download {filename} from {repo_id}: {stderr}"
        )));
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        return Err(HubError::DownloadFailed(
            "huggingface-cli returned empty path".into(),
        ));
    }

    Ok(PathBuf::from(path))
}

/// Download files matching a glob pattern from a HF repo.
fn download_file_glob(repo_id: &str, pattern: &str) -> Result<PathBuf, HubError> {
    let output = Command::new("huggingface-cli")
        .args(["download", repo_id, "--include", pattern, "--quiet"])
        .output()
        .map_err(|e| HubError::CommandFailed(format!("huggingface-cli not found: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HubError::DownloadFailed(format!(
            "failed to download {pattern} from {repo_id}: {stderr}"
        )));
    }

    // The output is the cache directory; find the GGUF file in it
    let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Look for .gguf files in the output path
    if let Ok(entries) = std::fs::read_dir(&path_str) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|ext| ext == "gguf") {
                return Ok(p);
            }
        }
    }

    // Maybe the output IS the file path
    let path = PathBuf::from(&path_str);
    if path.exists() {
        return Ok(path);
    }

    Err(HubError::DownloadFailed(format!(
        "no GGUF files found in {repo_id}"
    )))
}

/// Errors during HF Hub operations.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("command failed: {0}")]
    CommandFailed(String),

    #[error("download failed: {0}")]
    DownloadFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hf_model_ids() {
        assert!(is_hf_model_id("HuggingFaceTB/SmolLM2-135M-Instruct"));
        assert!(is_hf_model_id("bartowski/SmolLM2-135M-Instruct-GGUF"));
        assert!(is_hf_model_id("Qwen/Qwen2.5-0.5B-Instruct"));
        // Not model IDs:
        assert!(!is_hf_model_id("model.gguf"));
        assert!(!is_hf_model_id("/path/to/model"));
        assert!(!is_hf_model_id("single-name"));
    }

    #[test]
    fn find_gguf_repo_already_gguf() {
        assert_eq!(
            find_gguf_repo("bartowski/SmolLM2-135M-Instruct-GGUF"),
            "bartowski/SmolLM2-135M-Instruct-GGUF"
        );
    }
}
