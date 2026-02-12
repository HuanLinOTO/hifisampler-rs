//! Performance statistics collector.

use hifisampler_core::resampler::ResampleStats;
use serde::Serialize;
use std::collections::VecDeque;

const MAX_HISTORY: usize = 1000;

/// Collects and summarizes inference performance statistics.
pub struct StatsCollector {
    history: VecDeque<ResampleStats>,
    total_requests: u64,
    total_errors: u64,
}

#[derive(Debug, Serialize)]
pub struct StatsSummary {
    pub total_requests: u64,
    pub total_errors: u64,
    pub active_requests: u64,
    pub recent_count: usize,
    pub avg_total_ms: f64,
    pub avg_feature_ms: f64,
    pub avg_synthesis_ms: f64,
    pub avg_postprocess_ms: f64,
    pub p50_total_ms: f64,
    pub p95_total_ms: f64,
    pub p99_total_ms: f64,
    pub min_total_ms: f64,
    pub max_total_ms: f64,
    pub cache_hit_rate: f64,
    pub avg_output_samples: f64,
    pub throughput_rps: f64,
    pub recent: Vec<ResampleStats>,
}

impl StatsCollector {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(MAX_HISTORY),
            total_requests: 0,
            total_errors: 0,
        }
    }

    pub fn record(&mut self, stats: ResampleStats) {
        self.total_requests += 1;
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(stats);
    }

    pub fn record_error(&mut self) {
        self.total_errors += 1;
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.total_requests = 0;
        self.total_errors = 0;
    }

    pub fn summary(&self) -> StatsSummary {
        let count = self.history.len();
        if count == 0 {
            return StatsSummary {
                total_requests: self.total_requests,
                total_errors: self.total_errors,
                active_requests: 0,
                recent_count: 0,
                avg_total_ms: 0.0,
                avg_feature_ms: 0.0,
                avg_synthesis_ms: 0.0,
                avg_postprocess_ms: 0.0,
                p50_total_ms: 0.0,
                p95_total_ms: 0.0,
                p99_total_ms: 0.0,
                min_total_ms: 0.0,
                max_total_ms: 0.0,
                cache_hit_rate: 0.0,
                avg_output_samples: 0.0,
                throughput_rps: 0.0,
                recent: Vec::new(),
            };
        }

        let avg_total = self.history.iter().map(|s| s.total_ms).sum::<f64>() / count as f64;
        let avg_feat = self.history.iter().map(|s| s.feature_ms).sum::<f64>() / count as f64;
        let avg_synth = self.history.iter().map(|s| s.synthesis_ms).sum::<f64>() / count as f64;
        let avg_post = self.history.iter().map(|s| s.postprocess_ms).sum::<f64>() / count as f64;

        let mut sorted_total: Vec<f64> = self.history.iter().map(|s| s.total_ms).collect();
        sorted_total.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted_total, 50.0);
        let p95 = percentile(&sorted_total, 95.0);
        let p99 = percentile(&sorted_total, 99.0);
        let min_val = sorted_total.first().copied().unwrap_or(0.0);
        let max_val = sorted_total.last().copied().unwrap_or(0.0);

        let cache_hits = self.history.iter().filter(|s| s.cache_hit).count();
        let cache_hit_rate = cache_hits as f64 / count as f64;

        let avg_output = self
            .history
            .iter()
            .map(|s| s.output_samples as f64)
            .sum::<f64>()
            / count as f64;

        // Simple throughput estimate
        let total_time_s = self.history.iter().map(|s| s.total_ms).sum::<f64>() / 1000.0;
        let throughput = if total_time_s > 0.0 {
            count as f64 / total_time_s
        } else {
            0.0
        };

        // Last 10 for detail view
        let recent: Vec<ResampleStats> = self.history.iter().rev().take(10).cloned().collect();

        StatsSummary {
            total_requests: self.total_requests,
            total_errors: self.total_errors,
            active_requests: 0,
            recent_count: count,
            avg_total_ms: avg_total,
            avg_feature_ms: avg_feat,
            avg_synthesis_ms: avg_synth,
            avg_postprocess_ms: avg_post,
            p50_total_ms: p50,
            p95_total_ms: p95,
            p99_total_ms: p99,
            min_total_ms: min_val,
            max_total_ms: max_val,
            cache_hit_rate,
            avg_output_samples: avg_output,
            throughput_rps: throughput,
            recent,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
