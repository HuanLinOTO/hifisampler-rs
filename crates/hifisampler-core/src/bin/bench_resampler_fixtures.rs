//! Run deterministic benchmark fixtures through the full resampler pipeline.

use anyhow::{Context, Result};
use hifisampler_core::{
    cache::CacheManager, config::Config, models::Models, parse_utau::UtauParams, resampler,
};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::info;

const FIXTURE_ROOT: &str = "tests/benchmark_fixtures";
const DEFAULT_CONFIG: &str = "config.benchmark.yaml";
const REQUESTS: &[&str] = &[
    "baseline_short",
    "cache_repeat_short",
    "postprocess_ap_hg",
    "hnsep_tension",
    "loop_mode_long",
    "stereo_48k_keyshift",
];

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    label: String,
    config: String,
    timestamp_unix_ms: u128,
    records: Vec<BenchmarkRecord>,
}

#[derive(Debug, Serialize)]
struct BenchmarkRecord {
    name: String,
    wall_ms: f64,
    total_ms: f64,
    feature_ms: f64,
    synthesis_ms: f64,
    postprocess_ms: f64,
    input_samples: usize,
    output_samples: usize,
    cache_hit: bool,
    output_path: String,
}

fn main() -> Result<()> {
    // Run on a large-stack thread: Burn Module derive (CascadedNet with 5 BaseNets)
    // recurses deeply and overflows the default 1-8MB main stack.
    // CUDA context is thread-local — it gets created on this thread and stays here
    // for both load and forward (predict_from_audio runs on current thread too).
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(real_main)
        .unwrap()
        .join()
        .map_err(|e| anyhow::anyhow!("benchmark thread panicked: {e:?}"))?
}

fn real_main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let label = args.get(1).cloned().unwrap_or_else(|| "run".to_string());
    let config_path = args.get(2).map(String::as_str).unwrap_or(DEFAULT_CONFIG);

    ensure_fixtures_exist()?;
    clear_fixture_caches()?;

    let config = Config::load(config_path)
        .with_context(|| format!("failed to load benchmark config {config_path}"))?;
    let models = Models::load(&config).context("failed to load models")?;
    let cache = CacheManager::new();
    info!("[bench] models loaded, starting fixtures");

    let mut records = Vec::with_capacity(REQUESTS.len());
    for name in REQUESTS {
        info!("[bench] === fixture: {name} ===");
        let request_path = Path::new(FIXTURE_ROOT)
            .join("requests")
            .join(format!("{name}.txt"));
        let raw = fs::read_to_string(&request_path)
            .with_context(|| format!("failed to read {}", request_path.display()))?;
        let params = UtauParams::parse(raw.trim())
            .with_context(|| format!("failed to parse request {name}"))?;

        remove_output(&params.output_path)?;

        let started = Instant::now();
        let stats = resampler::resample(&params, &config, &models, &cache)
            .with_context(|| format!("inference failed for request {name}"))?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

        records.push(BenchmarkRecord {
            name: (*name).to_string(),
            wall_ms,
            total_ms: stats.total_ms,
            feature_ms: stats.feature_ms,
            synthesis_ms: stats.synthesis_ms,
            postprocess_ms: stats.postprocess_ms,
            input_samples: stats.input_samples,
            output_samples: stats.output_samples,
            cache_hit: stats.cache_hit,
            output_path: params.output_path,
        });
    }

    let report = BenchmarkReport {
        label: label.clone(),
        config: config_path.to_string(),
        timestamp_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        records,
    };

    write_report(&label, &report)?;
    print_report(&report);
    Ok(())
}

fn ensure_fixtures_exist() -> Result<()> {
    for name in REQUESTS {
        let request_path = Path::new(FIXTURE_ROOT)
            .join("requests")
            .join(format!("{name}.txt"));
        if !request_path.exists() {
            anyhow::bail!(
                "missing benchmark request {}; run `cargo run -p hifisampler-core --bin gen_benchmark_fixtures` first",
                request_path.display()
            );
        }
    }
    Ok(())
}

fn clear_fixture_caches() -> Result<()> {
    let input_dir = Path::new(FIXTURE_ROOT).join("inputs");
    if !input_dir.exists() {
        anyhow::bail!("missing benchmark input dir {}", input_dir.display());
    }

    for entry in fs::read_dir(&input_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.ends_with(".hifi.bin") {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove cache {}", path.display()))?;
        }
    }

    Ok(())
}

fn remove_output(path: &str) -> Result<()> {
    let output_path = Path::new(path);
    if output_path.exists() {
        fs::remove_file(output_path)
            .with_context(|| format!("failed to remove output {}", output_path.display()))?;
    }
    Ok(())
}

fn write_report(label: &str, report: &BenchmarkReport) -> Result<()> {
    let result_dir = Path::new(FIXTURE_ROOT).join("results");
    fs::create_dir_all(&result_dir)?;

    let json_path = result_dir.join(format!("{label}.json"));
    fs::write(&json_path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    let csv_path = result_dir.join(format!("{label}.csv"));
    fs::write(&csv_path, csv_report(report))
        .with_context(|| format!("failed to write {}", csv_path.display()))?;

    Ok(())
}

fn csv_report(report: &BenchmarkReport) -> String {
    let mut out = String::from(
        "name,wall_ms,total_ms,feature_ms,synthesis_ms,postprocess_ms,input_samples,output_samples,cache_hit,output_path\n",
    );
    for record in &report.records {
        out.push_str(&format!(
            "{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{}\n",
            record.name,
            record.wall_ms,
            record.total_ms,
            record.feature_ms,
            record.synthesis_ms,
            record.postprocess_ms,
            record.input_samples,
            record.output_samples,
            record.cache_hit,
            record.output_path,
        ));
    }
    out
}

fn print_report(report: &BenchmarkReport) {
    println!("benchmark_label,{}", report.label);
    println!("config,{}", report.config);
    println!(
        "name,wall_ms,total_ms,feature_ms,synthesis_ms,postprocess_ms,output_samples,cache_hit"
    );
    for record in &report.records {
        println!(
            "{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{}",
            record.name,
            record.wall_ms,
            record.total_ms,
            record.feature_ms,
            record.synthesis_ms,
            record.postprocess_ms,
            record.output_samples,
            record.cache_hit,
        );
    }
}
