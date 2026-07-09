//! Generate deterministic benchmark fixtures for the HiFiSampler pipeline.
//!
//! The generated request files are raw UTAU argument strings accepted by the
//! server POST `/` endpoint and `UtauParams::parse`.

use anyhow::{Context, Result};
use hifisampler_core::{audio, parse_utau::UtauParams};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::BTreeSet;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

const FIXTURE_ROOT: &str = "tests/benchmark_fixtures";
const SAMPLE_RATE: u32 = 44_100;

struct RequestSpec {
    name: &'static str,
    input: &'static str,
    output: &'static str,
    pitch: &'static str,
    velocity: f32,
    flags: &'static str,
    offset_ms: f32,
    length_ms: f32,
    consonant_ms: f32,
    cutoff_ms: f32,
    volume: f32,
    modulation: f32,
    tempo: f32,
    pitchbend: &'static str,
}

impl RequestSpec {
    fn raw(&self) -> String {
        format!(
            "{} {} {} {:.0} {} {:.0} {:.0} {:.0} {:.0} {:.0} {:.0} !{:.0} {}",
            self.input,
            self.output,
            self.pitch,
            self.velocity,
            self.flags,
            self.offset_ms,
            self.length_ms,
            self.consonant_ms,
            self.cutoff_ms,
            self.volume,
            self.modulation,
            self.tempo,
            self.pitchbend,
        )
    }
}

fn main() -> Result<()> {
    let root = Path::new(FIXTURE_ROOT);
    let input_dir = root.join("inputs");
    let request_dir = root.join("requests");
    let output_dir = root.join("outputs");

    fs::create_dir_all(&input_dir).context("failed to create benchmark input dir")?;
    fs::create_dir_all(&request_dir).context("failed to create benchmark request dir")?;
    fs::create_dir_all(&output_dir).context("failed to create benchmark output dir")?;

    write_mono_wav(
        &input_dir.join("steady_a4_1s_44100_mono.wav"),
        SAMPLE_RATE,
        1.0,
        |t| {
            let env = smooth_envelope(t, 1.0, 0.03, 0.08);
            env * harmonic_voice(t, 440.0, 0.38, 0.06)
        },
    )?;

    write_mono_wav(
        &input_dir.join("expressive_c4_2s_44100_mono.wav"),
        SAMPLE_RATE,
        2.0,
        |t| {
            let vibrato = 2.0f32.powf((18.0 * (2.0 * PI * 5.2 * t).sin()) / 1200.0);
            let f0 = 261.63 * vibrato;
            let env =
                smooth_envelope(t, 2.0, 0.04, 0.18) * (0.82 + 0.18 * (2.0 * PI * 2.0 * t).sin());
            let breath = pseudo_noise(t, SAMPLE_RATE) * 0.018;
            env * harmonic_voice(t, f0, 0.34, 0.09) + breath
        },
    )?;

    write_stereo_wav(
        &input_dir.join("stereo_e4_1s_48000.wav"),
        48_000,
        1.0,
        |t| {
            let env = smooth_envelope(t, 1.0, 0.02, 0.07);
            env * harmonic_voice(t, 329.63, 0.32, 0.04)
        },
        |t| {
            let env = smooth_envelope(t, 1.0, 0.02, 0.07);
            env * harmonic_voice(t, 332.0, 0.30, 0.05)
        },
    )?;

    write_mono_wav(
        &input_dir.join("long_phrase_4s_44100_mono.wav"),
        SAMPLE_RATE,
        4.0,
        |t| {
            let phrase = if t < 1.2 {
                220.0
            } else if t < 2.6 {
                293.66
            } else {
                349.23
            };
            let slide = 2.0f32.powf((24.0 * (2.0 * PI * 0.7 * t).sin()) / 1200.0);
            let env =
                smooth_envelope(t, 4.0, 0.05, 0.25) * (0.88 + 0.12 * (2.0 * PI * 1.1 * t).sin());
            env * harmonic_voice(t, phrase * slide, 0.33, 0.07)
        },
    )?;

    let requests = [
        RequestSpec {
            name: "baseline_short",
            input: "tests/benchmark_fixtures/inputs/steady_a4_1s_44100_mono.wav",
            output: "tests/benchmark_fixtures/outputs/baseline_short.wav",
            pitch: "A4",
            velocity: 100.0,
            flags: "P0",
            offset_ms: 0.0,
            length_ms: 1_000.0,
            consonant_ms: 100.0,
            cutoff_ms: 0.0,
            volume: 100.0,
            modulation: 0.0,
            tempo: 120.0,
            pitchbend: "AA",
        },
        RequestSpec {
            name: "cache_repeat_short",
            input: "tests/benchmark_fixtures/inputs/steady_a4_1s_44100_mono.wav",
            output: "tests/benchmark_fixtures/outputs/cache_repeat_short.wav",
            pitch: "A4",
            velocity: 100.0,
            flags: "P0",
            offset_ms: 0.0,
            length_ms: 1_000.0,
            consonant_ms: 100.0,
            cutoff_ms: 0.0,
            volume: 100.0,
            modulation: 0.0,
            tempo: 120.0,
            pitchbend: "AA",
        },
        RequestSpec {
            name: "postprocess_ap_hg",
            input: "tests/benchmark_fixtures/inputs/expressive_c4_2s_44100_mono.wav",
            output: "tests/benchmark_fixtures/outputs/postprocess_ap_hg.wav",
            pitch: "C4",
            velocity: 95.0,
            flags: "A80HG50P100",
            offset_ms: 0.0,
            length_ms: 1_800.0,
            consonant_ms: 180.0,
            cutoff_ms: 0.0,
            volume: 100.0,
            modulation: 0.0,
            tempo: 128.0,
            pitchbend: "AAEAAAgAAP//AAQA",
        },
        RequestSpec {
            name: "hnsep_tension",
            input: "tests/benchmark_fixtures/inputs/expressive_c4_2s_44100_mono.wav",
            output: "tests/benchmark_fixtures/outputs/hnsep_tension.wav",
            pitch: "C4",
            velocity: 100.0,
            flags: "Hb80Hv120Ht20P0",
            offset_ms: 0.0,
            length_ms: 1_800.0,
            consonant_ms: 180.0,
            cutoff_ms: 0.0,
            volume: 100.0,
            modulation: 0.0,
            tempo: 120.0,
            pitchbend: "AA",
        },
        RequestSpec {
            name: "loop_mode_long",
            input: "tests/benchmark_fixtures/inputs/long_phrase_4s_44100_mono.wav",
            output: "tests/benchmark_fixtures/outputs/loop_mode_long.wav",
            pitch: "D4",
            velocity: 85.0,
            flags: "HeP0",
            offset_ms: 0.0,
            length_ms: 5_000.0,
            consonant_ms: 400.0,
            cutoff_ms: 0.0,
            volume: 100.0,
            modulation: 0.0,
            tempo: 110.0,
            pitchbend: "AAEAAAgAAP//AAQA",
        },
        RequestSpec {
            name: "stereo_48k_keyshift",
            input: "tests/benchmark_fixtures/inputs/stereo_e4_1s_48000.wav",
            output: "tests/benchmark_fixtures/outputs/stereo_48k_keyshift.wav",
            pitch: "E4",
            velocity: 115.0,
            flags: "g-20P0",
            offset_ms: 0.0,
            length_ms: 1_000.0,
            consonant_ms: 120.0,
            cutoff_ms: 0.0,
            volume: 100.0,
            modulation: 0.0,
            tempo: 120.0,
            pitchbend: "AA",
        },
    ];

    let mut input_paths = BTreeSet::new();
    for request in requests {
        let raw = request.raw();
        let params = UtauParams::parse(&raw)
            .with_context(|| format!("generated request {} is not parseable", request.name))?;
        if !Path::new(&params.input_path).exists() {
            anyhow::bail!("generated input does not exist: {}", params.input_path);
        }
        input_paths.insert(params.input_path);

        fs::write(
            request_dir.join(format!("{}.txt", request.name)),
            format!("{}\n", raw),
        )
        .with_context(|| format!("failed to write request {}", request.name))?;
    }

    for input_path in input_paths {
        let samples = audio::read_wav(&input_path, SAMPLE_RATE)
            .with_context(|| format!("generated input is not readable: {input_path}"))?;
        if samples.is_empty() {
            anyhow::bail!("generated input is empty after read_wav: {input_path}");
        }
    }

    fs::write(
        root.join("manifest.txt"),
        "HiFiSampler benchmark fixtures\n\nInputs:\n  steady_a4_1s_44100_mono.wav\n  expressive_c4_2s_44100_mono.wav\n  stereo_e4_1s_48000.wav\n  long_phrase_4s_44100_mono.wav\n\nRequests:\n  baseline_short.txt\n  cache_repeat_short.txt\n  postprocess_ap_hg.txt\n  hnsep_tension.txt\n  loop_mode_long.txt\n  stereo_48k_keyshift.txt\n",
    )?;

    println!("Generated benchmark fixtures under {FIXTURE_ROOT}");
    Ok(())
}

fn write_mono_wav(
    path: &Path,
    sample_rate: u32,
    duration_secs: f32,
    mut sample_fn: impl FnMut(f32) -> f32,
) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let samples = (sample_rate as f32 * duration_secs).round() as usize;
    for i in 0..samples {
        let t = i as f32 / sample_rate as f32;
        writer.write_sample(to_i16(sample_fn(t)))?;
    }
    writer.finalize()?;
    Ok(())
}

fn write_stereo_wav(
    path: &Path,
    sample_rate: u32,
    duration_secs: f32,
    mut left_fn: impl FnMut(f32) -> f32,
    mut right_fn: impl FnMut(f32) -> f32,
) -> Result<()> {
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let samples = (sample_rate as f32 * duration_secs).round() as usize;
    for i in 0..samples {
        let t = i as f32 / sample_rate as f32;
        writer.write_sample(to_i16(left_fn(t)))?;
        writer.write_sample(to_i16(right_fn(t)))?;
    }
    writer.finalize()?;
    Ok(())
}

fn harmonic_voice(t: f32, f0: f32, amplitude: f32, noise_mix: f32) -> f32 {
    let mut sample = 0.0;
    let harmonics = [1.0, 0.42, 0.24, 0.13, 0.08, 0.05];
    for (idx, gain) in harmonics.iter().enumerate() {
        let partial = (idx + 1) as f32;
        sample += gain * (2.0 * PI * f0 * partial * t).sin();
    }
    amplitude * sample / 1.92 + noise_mix * pseudo_noise(t, SAMPLE_RATE)
}

fn smooth_envelope(t: f32, duration: f32, attack: f32, release: f32) -> f32 {
    let attack_gain = (t / attack).clamp(0.0, 1.0);
    let release_gain = ((duration - t) / release).clamp(0.0, 1.0);
    attack_gain.min(release_gain)
}

fn pseudo_noise(t: f32, sample_rate: u32) -> f32 {
    let mut state = (t * sample_rate as f32) as u32;
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let normalized = ((state >> 9) as f32) / ((1u32 << 23) as f32);
    normalized * 2.0 - 1.0
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-0.95, 0.95) * i16::MAX as f32).round() as i16
}
