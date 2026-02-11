//! Quick smoke test: load models and run a simple vocoder inference.

use hifisampler_core::config::Config;
use hifisampler_core::vocoder::Vocoder;
use hifisampler_core::hnsep::HnsepModel;
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== HiFiSampler Rust Smoke Test ===\n");

    // Load config
    let config = Config::load("config.yaml")?;
    println!("Config loaded: sr={}, hop={}, mels={}", config.sample_rate, config.hop_size, config.num_mels);
    println!("Vocoder model: {:?}", config.vocoder.model);
    println!("HN-SEP model: {:?}", config.hnsep.model);

    // Test 1: Load vocoder
    println!("\n[1] Loading vocoder...");
    let mut vocoder = Vocoder::load(&config.vocoder.model, &config.performance.device, config.performance.num_threads)?;
    println!("    ✓ Vocoder loaded");

    // Test 2: Run vocoder with dummy data
    println!("\n[2] Running vocoder inference with dummy mel+f0...");
    let n_frames = 100;
    let num_mels = config.num_mels; // 128
    let mel = Array2::<f32>::zeros((num_mels, n_frames));
    let f0 = vec![440.0f32; n_frames]; // A4

    let wav = vocoder.synthesize(&mel, &f0)?;
    println!("    ✓ Output: {} samples ({:.2}s at {}Hz)",
        wav.len(), wav.len() as f64 / config.sample_rate as f64, config.sample_rate);

    let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("    Peak amplitude: {:.6}", peak);

    // Test 3: Load HN-SEP
    if config.hnsep.model.exists() {
        println!("\n[3] Loading HN-SEP model...");
        let mut hnsep = HnsepModel::load(&config.hnsep.model, config.n_fft, config.hop_size, &config.performance.device, config.performance.num_threads)?;
        println!("    ✓ HN-SEP loaded");

        // Test with dummy audio (1 second of sine wave)
        println!("\n[4] Running HN-SEP inference with sine wave...");
        let duration_samples = config.sample_rate as usize;
        let freq = 440.0f32;
        let audio: Vec<f32> = (0..duration_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / config.sample_rate as f32).sin() * 0.5)
            .collect();

        let harmonic = hnsep.predict_from_audio(&audio, config.sample_rate)?;
        println!("    ✓ Output: {} samples", harmonic.len());
        let h_peak = harmonic.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!("    Harmonic peak amplitude: {:.6}", h_peak);
    } else {
        println!("\n[3] HN-SEP model not found, skipping");
    }

    println!("\n=== All tests passed! ===");
    Ok(())
}
