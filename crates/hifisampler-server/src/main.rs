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

/// Shared application state.
struct AppState {
    config: Config,
    models: Models,
    cache: CacheManager,
    stats: StatsCollector,
    ready: bool,
}

type SharedState = Arc<Mutex<AppState>>;

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

    // Load models
    let models = Models::load(&config)?;
    let cache = CacheManager::new();
    let stats = StatsCollector::new();

    let state = Arc::new(Mutex::new(AppState {
        config: config.clone(),
        models,
        cache,
        stats,
        ready: true,
    }));

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

    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// GET / - Health check endpoint.
async fn health_check(State(state): State<SharedState>) -> impl IntoResponse {
    let state = state.lock();
    if state.ready {
        (StatusCode::OK, "HiFiSampler server is ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Server not ready")
    }
}

/// POST / - Inference endpoint.
/// Body: UTAU resample parameters as plain text.
async fn inference_handler(
    State(state): State<SharedState>,
    body: String,
) -> impl IntoResponse {
    let body = body.trim().to_string();

    // Parse params
    let params = match UtauParams::parse(&body) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to parse params: {}", e);
            return (StatusCode::BAD_REQUEST, format!("Parse error: {}", e));
        }
    };

    // Run inference in blocking thread (ONNX Runtime is sync)
    let state_clone = Arc::clone(&state);
    let result = task::spawn_blocking(move || {
        let state = state_clone.lock();
        resampler::resample(&params, &state.config, &state.models, &state.cache)
    })
    .await;

    match result {
        Ok(Ok(stats)) => {
            // Record stats
            {
                let mut state = state.lock();
                state.stats.record(stats);
            }
            (StatusCode::OK, "OK".to_string())
        }
        Ok(Err(e)) => {
            error!("Inference error: {}", e);
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
async fn stats_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let state = state.lock();
    let summary = state.stats.summary();
    (StatusCode::OK, axum::Json(summary))
}

/// POST /stats/reset - Reset statistics.
async fn stats_reset_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let mut state = state.lock();
    state.stats.reset();
    (StatusCode::OK, "Stats reset")
}

/// GET /config - Current configuration.
async fn config_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let state = state.lock();
    (StatusCode::OK, axum::Json(state.config.clone()))
}
