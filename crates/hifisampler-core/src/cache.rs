//! Feature cache manager.
//!
//! Caches mel spectrograms and HN-SEP results to disk,
//! using read-write locks for thread safety.

use anyhow::Result;
use ndarray::Array2;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, warn};

/// Thread-safe cache manager for mel features and HN-SEP results.
pub struct CacheManager {
    /// Locks per cache file path to prevent concurrent writes.
    locks: RwLock<HashMap<PathBuf, Arc<RwLock<()>>>>,
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a lock for the given path.
    fn get_lock(&self, path: &Path) -> Arc<RwLock<()>> {
        {
            let locks = self.locks.read();
            if let Some(lock) = locks.get(path) {
                return Arc::clone(lock);
            }
        }

        let mut locks = self.locks.write();
        locks
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone()
    }

    /// Build cache file path from input WAV path and flags.
    pub fn cache_path(wav_path: &Path, suffix: &str) -> PathBuf {
        let stem = wav_path.file_stem().unwrap_or_default().to_string_lossy();
        let dir = wav_path.parent().unwrap_or(Path::new("."));
        dir.join(format!("{}{}.hifi.bin", stem, suffix))
    }

    /// Build a flag suffix string for cache differentiation.
    pub fn flag_suffix(g: i32, hb: i32, hv: i32, ht: i32) -> String {
        let mut parts = Vec::new();
        if g != 0 {
            parts.push(format!("_g{}", g));
        }
        if hb != 100 {
            parts.push(format!("_hb{}", hb));
        }
        if hv != 100 {
            parts.push(format!("_hv{}", hv));
        }
        if ht != 0 {
            parts.push(format!("_ht{}", ht));
        }
        parts.join("")
    }

    /// Try to load cached mel features.
    /// Returns (mel, scale) if cache exists and is valid.
    pub fn load_mel_cache(
        &self,
        cache_path: &Path,
    ) -> Option<(Array2<f32>, f32)> {
        let lock = self.get_lock(cache_path);
        let _guard = lock.read();

        if !cache_path.exists() {
            return None;
        }

        match self.read_mel_file(cache_path) {
            Ok(data) => {
                debug!("Cache hit: {}", cache_path.display());
                Some(data)
            }
            Err(e) => {
                warn!("Cache read error for {}: {}", cache_path.display(), e);
                None
            }
        }
    }

    /// Save mel features to cache (atomic write).
    pub fn save_mel_cache(
        &self,
        cache_path: &Path,
        mel: &Array2<f32>,
        scale: f32,
    ) -> Result<()> {
        let lock = self.get_lock(cache_path);
        let _guard = lock.write();

        // Double-check: another thread may have written it
        if cache_path.exists() {
            return Ok(());
        }

        let tmp_path = cache_path.with_extension("tmp");
        self.write_mel_file(&tmp_path, mel, scale)?;
        fs::rename(&tmp_path, cache_path)?;

        debug!("Cache saved: {}", cache_path.display());
        Ok(())
    }

    /// Invalidate (delete) a cache file.
    pub fn invalidate(&self, cache_path: &Path) {
        let lock = self.get_lock(cache_path);
        let _guard = lock.write();

        if cache_path.exists() {
            let _ = fs::remove_file(cache_path);
        }
    }

    // Simple binary format: [rows:u32][cols:u32][scale:f32][data:f32...]
    fn write_mel_file(&self, path: &Path, mel: &Array2<f32>, scale: f32) -> Result<()> {
        let rows = mel.nrows() as u32;
        let cols = mel.ncols() as u32;

        let mut buf = Vec::new();
        buf.extend_from_slice(&rows.to_le_bytes());
        buf.extend_from_slice(&cols.to_le_bytes());
        buf.extend_from_slice(&scale.to_le_bytes());

        for &val in mel.iter() {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &buf)?;
        Ok(())
    }

    fn read_mel_file(&self, path: &Path) -> Result<(Array2<f32>, f32)> {
        let data = fs::read(path)?;
        if data.len() < 12 {
            anyhow::bail!("Cache file too small");
        }

        let rows = u32::from_le_bytes(data[0..4].try_into()?) as usize;
        let cols = u32::from_le_bytes(data[4..8].try_into()?) as usize;
        let scale = f32::from_le_bytes(data[8..12].try_into()?);

        let expected_len = 12 + rows * cols * 4;
        if data.len() < expected_len {
            anyhow::bail!("Cache file truncated");
        }

        let mut mel = Array2::zeros((rows, cols));
        let mut offset = 12;
        for i in 0..rows {
            for j in 0..cols {
                mel[[i, j]] = f32::from_le_bytes(data[offset..offset + 4].try_into()?);
                offset += 4;
            }
        }

        Ok((mel, scale))
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}
