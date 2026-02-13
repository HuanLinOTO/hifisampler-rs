//! HiFiSampler inference server.
//!
//! HTTP server compatible with the original Python implementation.
//! - GET /  → readiness check
//! - POST / → inference request (UTAU params as text body)
//! - GET /stats → performance statistics (for WebUI)
//! - WebUI served from /ui/

mod stats;
mod webui;

use anyhow::Result;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Redirect, Response},
    routing::{get, post, put},
};
use clap::Parser;
use hifisampler_core::{
    cache::CacheManager, config::Config, models::Models, parse_utau::UtauParams, resampler,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::task;
use tracing::{error, info};

use crate::stats::StatsCollector;

const BRIDGE_SERVER_PATH_FILE: &str = "hifisampler-server.path";

#[derive(Parser, Debug)]
#[command(name = "hifisampler-server", about = "HiFiSampler inference server")]
struct Args {
    /// Path to config.yaml
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    /// Override host
    #[arg(long)]
    host: Option<String>,

    /// Override port
    #[arg(long)]
    port: Option<u16>,

    /// Run in bridge-managed mode (auto shutdown when idle)
    #[arg(long, default_value_t = false)]
    managed: bool,

    /// Auto shutdown timeout in seconds for managed mode
    #[arg(long, default_value_t = 600)]
    idle_timeout_secs: u64,
}

/// Decomposed application state — each component is independently wrapped
/// so that inference does NOT block health checks, stats, or config queries.
#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    config_path: Arc<String>,
    models: Arc<Models>,
    cache: Arc<CacheManager>,
    stats: Arc<Mutex<StatsCollector>>,
    ready: Arc<AtomicBool>,
    active_requests: Arc<AtomicU64>,
    managed: bool,
    idle_timeout_secs: u64,
    last_activity_unix: Arc<AtomicU64>,
    shutdown_requested: Arc<AtomicBool>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    info!("HiFiSampler Server v{}", env!("CARGO_PKG_VERSION"));

    // Load config
    let mut config = Config::load_or_default(&args.config);
    if let Some(host) = args.host {
        config.server.host = host;
    }
    if let Some(port) = args.port {
        config.server.port = port;
    }

    let addr = format!("{}:{}", config.server.host, config.server.port);

    // Load models
    let models = Models::load(&config)?;
    let cache = CacheManager::new();
    let stats = StatsCollector::new();
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let state = AppState {
        config: Arc::new(config),
        config_path: Arc::new(args.config.clone()),
        models: Arc::new(models),
        cache: Arc::new(cache),
        stats: Arc::new(Mutex::new(stats)),
        ready: Arc::new(AtomicBool::new(true)),
        active_requests: Arc::new(AtomicU64::new(0)),
        managed: args.managed,
        idle_timeout_secs: args.idle_timeout_secs,
        last_activity_unix: Arc::new(AtomicU64::new(now_unix_secs())),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        shutdown_tx,
    };

    if state.managed {
        info!(
            "Managed mode enabled (idle timeout: {}s)",
            state.idle_timeout_secs
        );
        let watchdog_state = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                if watchdog_state.shutdown_requested.load(Ordering::Relaxed) {
                    break;
                }
                let active = watchdog_state.active_requests.load(Ordering::Relaxed);
                if active > 0 {
                    continue;
                }
                let idle = now_unix_secs()
                    .saturating_sub(watchdog_state.last_activity_unix.load(Ordering::Relaxed));
                if idle >= watchdog_state.idle_timeout_secs {
                    request_shutdown(
                        &watchdog_state,
                        &format!("managed idle timeout reached ({}s)", idle),
                    );
                    break;
                }
            }
        });
    }

    // Build router
    let app = Router::new()
        .route("/", get(health_check))
        .route("/", post(inference_handler))
        .route("/refresh", get(refresh_handler))
        .route("/stats", get(stats_handler))
        .route("/stats/reset", post(stats_reset_handler))
        .route("/config", get(config_handler))
        .route("/config", put(config_save_handler))
        .route("/shutdown", post(shutdown_handler))
        .route("/install-bridge", post(install_bridge_handler))
        .nest_service("/ui", tower_http::services::ServeDir::new("webui"))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    info!("Server listening on http://{}", addr);

    // Try to open a browser window:
    // - if a known Chromium-like browser is present in PATH, launch it with --app="{addr}"
    // - otherwise fall back to the platform default opener (start/open/xdg-open)
    if !args.managed {
        use std::env;
        use std::path::PathBuf;
        use std::process::Command;

        // candidates that accept --app (names normalized below)
        const CANDIDATES: &[&str] = &[
            "chrome.exe",
            "msedge.exe",
            "brave.exe",
            "chromium.exe",
            "google-chrome",
            "browser",
            "edge",
        ];

        // On Windows prefer querying HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths
        #[cfg(windows)]
        fn find_in_app_paths(exe: &str) -> Option<PathBuf> {
            use winreg::RegKey;
            use winreg::enums::HKEY_LOCAL_MACHINE;
            let hk = RegKey::predef(HKEY_LOCAL_MACHINE);
            let base = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths";
            let candidates = [
                format!("{}\\{}", base, exe),
                format!("{}\\{}", base, exe.to_lowercase()),
            ];
            for k in &candidates {
                if let Ok(sub) = hk.open_subkey(k) {
                    // Default value or 'Path' value may contain the full exe path
                    if let Ok(path_str) = sub.get_value::<String, _>("") {
                        let p = PathBuf::from(path_str);
                        if p.exists() {
                            return Some(p);
                        }
                    }
                    if let Ok(path_str) = sub.get_value::<String, _>("Path") {
                        let p = PathBuf::from(path_str);
                        if p.exists() {
                            return Some(p);
                        }
                    }
                }
            }
            None
        }

        fn path_has_exe(name: &str) -> Option<PathBuf> {
            // Windows: check registry App Paths first
            #[cfg(windows)]
            {
                if let Some(p) = find_in_app_paths(name) {
                    return Some(p);
                }
                // Also try with and without .exe variations
                if !name.ends_with(".exe") {
                    let try_name = format!("{}.exe", name);
                    if let Some(p) = find_in_app_paths(&try_name) {
                        return Some(p);
                    }
                }
            }

            if let Ok(pathvar) = env::var("PATH") {
                let paths = env::split_paths(&pathvar);
                #[cfg(windows)]
                let exts: Vec<String> = env::var("PATHEXT")
                    .unwrap_or_default()
                    .split(';')
                    .map(|s| s.to_string())
                    .collect();
                #[cfg(not(windows))]
                let exts: Vec<String> = vec!["".to_string()];

                for p in paths {
                    #[cfg(windows)]
                    {
                        for ext in &exts {
                            let mut candidate = p.join(name);
                            candidate.set_extension(ext.trim_start_matches('.'));
                            if candidate.exists() {
                                return Some(candidate);
                            }
                        }
                    }
                    #[cfg(not(windows))]
                    {
                        let candidate = p.join(name);
                        if candidate.exists() && candidate.is_file() {
                            return Some(candidate);
                        }
                    }
                }
            }
            None
        }

        let url = format!("http://{}", addr);
        // prefer explicit Chromium-like app with --app
        let mut launched = false;
        for &bin in CANDIDATES {
            if let Some(path) = path_has_exe(bin) {
                info!("Launching browser '{}' with --app=...", bin);
                let _ = Command::new(path)
                    .arg(format!("--app={}", url))
                    .arg("--window-size=1280,720")
                    .spawn();
                launched = true;
                break;
            }
        }

        if !launched {
            info!("No Chromium-like browser found in PATH / App Paths — opening default browser");
            // Cross-platform default opener
            #[cfg(target_os = "windows")]
            let _ = Command::new("cmd").args(["/C", "start", &url]).spawn();
            #[cfg(target_os = "macos")]
            let _ = Command::new("open").arg(&url).spawn();
            #[cfg(all(unix, not(target_os = "macos")))]
            let _ = Command::new("xdg-open").arg(&url).spawn();
        }
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let mut shutdown_rx_wait = shutdown_rx.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx_wait.changed().await;
        })
        .await?;

    Ok(())
}

/// GET / - Health check endpoint.
/// Uses AtomicBool — never blocks on inference.
async fn health_check(State(state): State<AppState>) -> Response {
    if state.ready.load(Ordering::Relaxed) {
        // Server is healthy — redirect the browser to the WebUI
        Redirect::temporary("/ui").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Server not ready").into_response()
    }
}

/// POST / - Inference endpoint.
/// Body: UTAU resample parameters as plain text.
///
/// Clones Arc refs into `spawn_blocking` — no global Mutex lock.
/// Models internally use per-model Mutex (only locked for actual ONNX calls).
async fn inference_handler(State(state): State<AppState>, body: String) -> impl IntoResponse {
    state
        .last_activity_unix
        .store(now_unix_secs(), Ordering::Relaxed);
    let body = body.trim().to_string();

    // Parse params (cheap, no lock needed)
    let params = match UtauParams::parse(&body) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to parse params: {}", e);
            return (StatusCode::BAD_REQUEST, format!("Parse error: {}", e));
        }
    };

    // Clone Arc refs for the blocking task — NO Mutex lock held across spawn
    let config = Arc::clone(&state.config);
    let models = Arc::clone(&state.models);
    let cache = Arc::clone(&state.cache);

    state.active_requests.fetch_add(1, Ordering::Relaxed);
    let result =
        task::spawn_blocking(move || resampler::resample(&params, &config, &models, &cache)).await;
    state.active_requests.fetch_sub(1, Ordering::Relaxed);

    match result {
        Ok(Ok(resample_stats)) => {
            // Brief lock to record stats — microseconds, not seconds
            state.stats.lock().record(resample_stats);
            (StatusCode::OK, "OK".to_string())
        }
        Ok(Err(e)) => {
            error!("Inference error: {}", e);
            state.stats.lock().record_error();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {}", e),
            )
        }
        Err(e) => {
            error!("Task error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task error: {}", e),
            )
        }
    }
}

async fn shutdown_handler(State(state): State<AppState>) -> impl IntoResponse {
    request_shutdown(&state, "requested from WebUI");
    (StatusCode::OK, "Shutting down")
}

async fn refresh_handler() -> impl IntoResponse {
    tokio::time::sleep(Duration::from_secs(10)).await;
    (StatusCode::OK, "refresh")
}

/// GET /stats - Performance statistics.
/// Brief lock — only reading stats, no interference with inference.
async fn stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    let active = state.active_requests.load(Ordering::Relaxed);
    let mut summary = state.stats.lock().summary();
    summary.active_requests = active;
    (StatusCode::OK, axum::Json(summary))
}

/// POST /stats/reset - Reset statistics.
async fn stats_reset_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.stats.lock().reset();
    (StatusCode::OK, "Stats reset")
}

fn request_shutdown(state: &AppState, reason: &str) {
    if state
        .shutdown_requested
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        state.ready.store(false, Ordering::Relaxed);
        info!("Shutdown requested: {}", reason);
        let _ = state.shutdown_tx.send(true);
    }
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// GET /config - Current configuration.
/// No lock at all — Config is immutable behind Arc.
async fn config_handler(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, axum::Json((*state.config).clone()))
}

/// PUT /config - Save configuration to file.
/// Writes the submitted config to the config YAML file.
/// Note: runtime config is NOT hot-reloaded; a server restart is needed.
async fn config_save_handler(
    State(state): State<AppState>,
    axum::Json(new_config): axum::Json<Config>,
) -> impl IntoResponse {
    match new_config.save(state.config_path.as_str()) {
        Ok(()) => {
            info!("Config saved to {}", state.config_path);
            (
                StatusCode::OK,
                axum::Json(serde_json::json!({"message": "配置已保存，部分设置重启后生效"})),
            )
        }
        Err(e) => {
            error!("Failed to save config: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": format!("保存失败: {}", e)})),
            )
        }
    }
}

/// POST /install-bridge - Copy bridge executable to target directory.
#[derive(serde::Deserialize)]
struct InstallBridgeRequest {
    path: String,
}

async fn install_bridge_handler(
    axum::Json(req): axum::Json<InstallBridgeRequest>,
) -> impl IntoResponse {
    let target_dir = std::path::Path::new(&req.path);

    // Validate target directory exists
    if !target_dir.is_dir() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": format!("目录不存在: {}", req.path)})),
        );
    }

    // Find bridge executable next to the server executable
    let bridge_name = if cfg!(windows) {
        "hifisampler.exe"
    } else {
        "hifisampler"
    };
    let server_exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": format!("无法获取服务器路径: {}", e)})),
            );
        }
    };
    let bridge_src = server_exe
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join(bridge_name);

    if !bridge_src.exists() {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(
                serde_json::json!({"error": format!("未找到桥接程序: {}", bridge_src.display())}),
            ),
        );
    }

    let dest = target_dir.join(bridge_name);
    match std::fs::copy(&bridge_src, &dest) {
        Ok(_) => {
            let path_file = target_dir.join(BRIDGE_SERVER_PATH_FILE);
            if let Err(e) = std::fs::write(&path_file, format!("{}\n", server_exe.display())) {
                error!("Bridge server path file write failed: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(
                        serde_json::json!({"error": format!("写入 {} 失败: {}", path_file.display(), e)}),
                    ),
                );
            }

            info!(
                "Bridge installed: {} -> {} (server path file: {})",
                bridge_src.display(),
                dest.display(),
                path_file.display()
            );
            (
                StatusCode::OK,
                axum::Json(serde_json::json!({
                    "message": format!("已安装到 {}", dest.display()),
                    "server_path_file": path_file.display().to_string(),
                })),
            )
        }
        Err(e) => {
            error!("Bridge install failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": format!("复制失败: {}", e)})),
            )
        }
    }
}
