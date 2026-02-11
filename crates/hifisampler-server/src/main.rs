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
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use hifisampler_core::{
    cache::CacheManager,
    config::Config,
    models::Models,
    parse_utau::UtauParams,
    resampler,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::task;
use tracing::{error, info};

use crate::stats::StatsCollector;

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
}

/// Decomposed application state — each component is independently wrapped
/// so that inference does NOT block health checks, stats, or config queries.
#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    models: Arc<Models>,
    cache: Arc<CacheManager>,
    stats: Arc<Mutex<StatsCollector>>,
    ready: Arc<AtomicBool>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
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

    let state = AppState {
        config: Arc::new(config),
        models: Arc::new(models),
        cache: Arc::new(cache),
        stats: Arc::new(Mutex::new(stats)),
        ready: Arc::new(AtomicBool::new(true)),
    };

    // Build router
    let app = Router::new()
        .route("/", get(health_check))
        .route("/", post(inference_handler))
        .route("/stats", get(stats_handler))
        .route("/stats/reset", post(stats_reset_handler))
        .route("/config", get(config_handler))
        .nest_service(
            "/ui",
            tower_http::services::ServeDir::new("webui"),
        )
        .layer(
            tower_http::cors::CorsLayer::permissive(),
        )
        .with_state(state);

    info!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// GET / - Health check endpoint.
/// Uses AtomicBool — never blocks on inference.
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    if state.ready.load(Ordering::Relaxed) {
        (StatusCode::OK, "HiFiSampler server is ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Server not ready")
    }
}

/// POST / - Inference endpoint.
/// Body: UTAU resample parameters as plain text.
///
/// Clones Arc refs into `spawn_blocking` — no global Mutex lock.
/// Models internally use per-model Mutex (only locked for actual ONNX calls).
async fn inference_handler(
    State(state): State<AppState>,
    body: String,
) -> impl IntoResponse {
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

    let result = task::spawn_blocking(move || {
        resampler::resample(&params, &config, &models, &cache)
    })
    .await;

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

/// GET /stats - Performance statistics.
/// Brief lock — only reading stats, no interference with inference.
async fn stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    let summary = state.stats.lock().summary();
    (StatusCode::OK, axum::Json(summary))
}

/// POST /stats/reset - Reset statistics.
async fn stats_reset_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.stats.lock().reset();
    (StatusCode::OK, "Stats reset")
}

/// GET /config - Current configuration.
/// No lock at all — Config is immutable behind Arc.
async fn config_handler(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, axum::Json((*state.config).clone()))
}
