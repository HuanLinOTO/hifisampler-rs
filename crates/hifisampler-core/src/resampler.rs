//! Core resampler pipeline.
//!
//! Implements the full UTAU resampling pipeline:
//! WAV → Feature extraction → Time stretch → Pitch bend → Vocoder synthesis → Post-processing

use crate::audio::{
    self, interp1d, akima_interp, loudness_normalize, peak, pre_emphasis_tension,
};
use crate::cache::CacheManager;
use crate::config::Config;
use crate::growl::apply_growl;
use crate::mel::dynamic_range_compression;
use crate::models::Models;
use crate::parse_utau::{
    decode_pitchbend, midi_to_hz, note_to_midi, parse_flags, UtauFlags, UtauParams,
};
use anyhow::Result;
use ndarray::Array2;
use std::path::Path;
use std::time::Instant;
use tracing::{debug, warn};

/// Statistics for a single resample operation.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ResampleStats {
    pub total_ms: f64,
    pub feature_ms: f64,
    pub synthesis_ms: f64,
    pub postprocess_ms: f64,
    pub input_samples: usize,
    pub output_samples: usize,
    pub cache_hit: bool,
}

/// Perform the full resample operation.
pub fn resample(
    params: &UtauParams,
    config: &Config,
    models: &Models,
    cache: &CacheManager,
) -> Result<ResampleStats> {
    let total_start = Instant::now();
    let mut stats = ResampleStats::default();

    let flags = parse_flags(&params.flags);
    let sr = config.sample_rate as f32;

    // ── Step 1: Feature extraction ──
    let feat_start = Instant::now();
    let (mel_origin, scale) = get_features(
        &params.input_path,
        config,
        models,
        cache,
        &flags,
    )?;
    stats.feature_ms = feat_start.elapsed().as_secs_f64() * 1000.0;
    stats.cache_hit = false; // TODO: detect from cache

    // ── Step 2: Resample (time stretch + pitch bend + synthesis) ──
    let synth_start = Instant::now();

    let hop_size = config.hop_size;
    let hop_interp = config.hop_size_interp;
    let pad_frames = config.pad_frames;

    // Time axis calculations
    let offset_samples = (params.offset / 1000.0 * sr).round() as i64;
    let consonant_samples = (params.consonant / 1000.0 * sr).round() as i64;
    let length_req_samples = (params.length / 1000.0 * sr).round() as i64;
    
    // Cutoff handling (UTAU convention: negative = from end)
    let cutoff_samples = if params.cutoff < 0.0 {
        (-params.cutoff / 1000.0 * sr).round() as i64
    } else {
        let total_input_samples = mel_origin.ncols() as i64 * hop_interp as i64;
        total_input_samples - (params.cutoff / 1000.0 * sr).round() as i64
    };

    // Convert to mel frame indices (at hop_size_interp resolution)
    let start_frame = (offset_samples as f32 / hop_interp as f32).round() as i64;
    let con_frame = (consonant_samples as f32 / hop_interp as f32).round() as i64;
    let end_frame = (cutoff_samples as f32 / hop_interp as f32).round() as i64;

    // Ensure valid range
    let n_mel_frames = mel_origin.ncols() as i64;
    let start_frame = start_frame.max(0).min(n_mel_frames - 1) as usize;
    let end_frame = end_frame.max(start_frame as i64 + 1).min(n_mel_frames) as usize;
    let con_frame = con_frame.max(0).min((end_frame - start_frame) as i64) as usize;

    // Extract the relevant mel segment
    let mel_segment = mel_origin.slice(ndarray::s![.., start_frame..end_frame]).to_owned();
    let _seg_frames = mel_segment.ncols();

    // Velocity → consonant speed factor
    let velocity = params.velocity.clamp(0.0, 200.0);
    let vel_factor = 2.0f32.powf((100.0 - velocity) / 100.0);

    // Loop mode handling
    let mel_work = if flags.he || config.processing.loop_mode {
        loop_pad_mel(&mel_segment, length_req_samples as usize, hop_interp)
    } else {
        mel_segment.clone()
    };

    // Time stretching: interpolate mel from hop_interp to hop_size grid
    let ratio = hop_interp as f32 / hop_size as f32;
    let work_frames = mel_work.ncols();

    // Target frame count
    let target_frames = if length_req_samples > 0 {
        (length_req_samples as f32 / hop_size as f32).ceil() as usize
    } else {
        (work_frames as f32 / ratio).ceil() as usize
    };

    // Build time mapping (stretch function)
    let stretched_mel = stretch_mel(
        &mel_work,
        con_frame,
        vel_factor,
        target_frames,
        ratio,
    );

    // Add padding frames
    let padded_mel = pad_mel(&stretched_mel, pad_frames);
    let total_frames = padded_mel.ncols();

    // ── Pitch bend ──
    let midi_note = note_to_midi(&params.pitch).unwrap_or(60) as f32;
    let pitchbend_cents = decode_pitchbend(&params.pitchbend);

    // Build F0 curve
    let f0 = build_f0_curve(
        midi_note,
        &pitchbend_cents,
        flags.t as f32,
        total_frames,
        pad_frames,
        hop_size,
        sr,
        params.tempo,
    );

    // ── Vocoder synthesis ──
    let wav = models.vocoder.lock().synthesize(&padded_mel, &f0)?;

    stats.synthesis_ms = synth_start.elapsed().as_secs_f64() * 1000.0;

    // ── Step 3: Post-processing ──
    let post_start = Instant::now();

    // Trim padding
    let pad_samples = pad_frames * hop_size;
    let new_start = pad_samples;
    let new_end = if wav.len() > 2 * pad_samples {
        wav.len() - pad_samples
    } else {
        wav.len()
    };

    let mut output = if new_start < new_end {
        wav[new_start..new_end].to_vec()
    } else {
        wav
    };

    // Trim to requested length
    if length_req_samples > 0 && output.len() > length_req_samples as usize {
        output.truncate(length_req_samples as usize);
    }

    // Amplitude modulation (A flag)
    if flags.a != 0 {
        apply_amplitude_modulation(&mut output, &f0, flags.a as f32, hop_size, sr);
    }

    // Volume recovery (undo scale)
    if scale > 0.0 && scale != 1.0 {
        let inv_scale = 1.0 / scale;
        output.iter_mut().for_each(|x| *x *= inv_scale);
    }

    // Growl effect (HG flag)
    if flags.hg > 0 {
        output = apply_growl(&output, flags.hg as f32, config.sample_rate);
    }

    // Loudness normalization (P flag)
    if config.processing.wave_norm && flags.p > 0 {
        output = loudness_normalize(&output, -16.0, flags.p as f32);
    }

    // Peak limiting
    let peak_val = peak(&output);
    if peak_val > config.processing.peak_limit {
        let gain = config.processing.peak_limit / peak_val;
        output.iter_mut().for_each(|x| *x *= gain);
    }

    // Volume scaling
    let volume = params.volume / 100.0;
    if (volume - 1.0).abs() > 0.001 {
        output.iter_mut().for_each(|x| *x *= volume);
    }

    stats.postprocess_ms = post_start.elapsed().as_secs_f64() * 1000.0;
    stats.output_samples = output.len();

    // ── Save output ──
    audio::save_wav(&params.output_path, &output, config.sample_rate)?;

    stats.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    debug!(
        "Resample complete: total={:.1}ms feat={:.1}ms synth={:.1}ms post={:.1}ms",
        stats.total_ms, stats.feature_ms, stats.synthesis_ms, stats.postprocess_ms
    );

    Ok(stats)
}

/// Extract features: read WAV, apply HN-SEP if needed, compute mel spectrogram.
fn get_features(
    wav_path: &str,
    config: &Config,
    models: &Models,
    cache: &CacheManager,
    flags: &UtauFlags,
) -> Result<(Array2<f32>, f32)> {
    let path = Path::new(wav_path);
    let suffix = CacheManager::flag_suffix(flags.g, flags.hb, flags.hv, flags.ht);
    let cache_path = CacheManager::cache_path(path, &suffix);

    // Check cache (unless G flag forces regeneration)
    if !flags.gen_cache {
        if let Some(cached) = cache.load_mel_cache(&cache_path) {
            return Ok(cached);
        }
    }

    // Read WAV
    let mut wave = audio::read_wav(wav_path, config.sample_rate)?;

    // HN-SEP processing
    if flags.needs_hnsep() {
        if let Some(ref hnsep) = models.hnsep {
            let harmonic = hnsep.lock().predict_from_audio(&wave, config.sample_rate)?;

            // Mix based on Hb/Hv flags
            let noise: Vec<f32> = wave
                .iter()
                .zip(harmonic.iter())
                .map(|(&w, &h)| w - h)
                .collect();

            let hb = flags.hb as f32 / 100.0;
            let hv = flags.hv as f32 / 100.0;

            let mut mixed: Vec<f32> = noise
                .iter()
                .zip(harmonic.iter())
                .map(|(&n, &h)| hb * n + hv * h)
                .collect();

            // Apply tension filter to harmonic component
            if flags.ht != 0 {
                let tension_harmonic = pre_emphasis_tension(
                    &harmonic,
                    flags.ht as f32,
                    config.sample_rate,
                    config.n_fft,
                    config.hop_size,
                );
                mixed = noise
                    .iter()
                    .zip(tension_harmonic.iter())
                    .map(|(&n, &h)| hb * n + hv * h)
                    .collect();
            }

            wave = mixed;
        } else {
            warn!("HN-SEP requested but model not loaded");
        }
    }

    // Volume scaling (prevent clipping during mel computation)
    let peak_val = peak(&wave);
    let scale = if peak_val > 0.5 {
        0.5 / peak_val
    } else {
        1.0
    };
    if scale != 1.0 {
        wave.iter_mut().for_each(|x| *x *= scale);
    }

    // Gender/formant shift
    let key_shift = flags.g as f32 / 100.0;
    let speed = config.hop_size_interp as f32 / config.hop_size as f32;

    // Compute mel spectrogram
    let mel = models.mel_analyzer.mel_spectrogram(&wave, key_shift, speed);

    // Dynamic range compression (log mel)
    let mel = dynamic_range_compression(&mel, 1.0);

    // Save to cache
    let _ = cache.save_mel_cache(&cache_path, &mel, scale);

    Ok((mel, scale))
}

/// Stretch mel spectrogram: consonant at velocity speed, vowel stretched to fill.
fn stretch_mel(
    mel: &Array2<f32>,
    con_frames: usize,
    vel_factor: f32,
    target_frames: usize,
    ratio: f32,
) -> Array2<f32> {
    let src_frames = mel.ncols();
    let num_mels = mel.nrows();

    if src_frames == 0 || target_frames == 0 {
        return Array2::zeros((num_mels, target_frames.max(1)));
    }

    // Source time axis (in hop_interp units)
    let x_src: Vec<f32> = (0..src_frames).map(|i| i as f32).collect();

    // Build target time mapping
    let mut x_target: Vec<f32> = Vec::with_capacity(target_frames);

    // Consonant region: stretched by velocity factor
    let con_target = (con_frames as f32 * vel_factor / ratio).round() as usize;
    let con_target = con_target.min(target_frames);

    for i in 0..con_target {
        let src_pos = i as f32 * ratio / vel_factor;
        x_target.push(src_pos);
    }

    // Vowel region: stretch to fill remaining frames
    let vowel_src_frames = src_frames.saturating_sub(con_frames);
    let vowel_target_frames = target_frames.saturating_sub(con_target);

    if vowel_target_frames > 0 && vowel_src_frames > 0 {
        let vowel_ratio = vowel_src_frames as f32 / vowel_target_frames as f32;
        for i in 0..vowel_target_frames {
            let src_pos = con_frames as f32 + i as f32 * vowel_ratio;
            x_target.push(src_pos);
        }
    }

    // Fill remaining if needed
    while x_target.len() < target_frames {
        x_target.push((src_frames - 1) as f32);
    }
    x_target.truncate(target_frames);

    // Interpolate each mel band
    let mut stretched = Array2::zeros((num_mels, target_frames));
    for mel_idx in 0..num_mels {
        let y_src: Vec<f32> = (0..src_frames).map(|i| mel[[mel_idx, i]]).collect();
        let y_target = interp1d(&x_src, &y_src, &x_target);
        for (i, &v) in y_target.iter().enumerate() {
            if i < target_frames {
                stretched[[mel_idx, i]] = v;
            }
        }
    }

    stretched
}

/// Pad mel spectrogram with pad_frames of repeated edge values.
fn pad_mel(mel: &Array2<f32>, pad_frames: usize) -> Array2<f32> {
    let num_mels = mel.nrows();
    let n_frames = mel.ncols();
    let total = n_frames + 2 * pad_frames;

    let mut padded = Array2::zeros((num_mels, total));

    // Left pad (replicate first frame)
    for i in 0..pad_frames {
        for m in 0..num_mels {
            padded[[m, i]] = mel[[m, 0]];
        }
    }

    // Copy original
    for i in 0..n_frames {
        for m in 0..num_mels {
            padded[[m, pad_frames + i]] = mel[[m, i]];
        }
    }

    // Right pad (replicate last frame)
    for i in 0..pad_frames {
        for m in 0..num_mels {
            padded[[m, pad_frames + n_frames + i]] = mel[[m, n_frames.saturating_sub(1)]];
        }
    }

    padded
}

/// Loop-pad mel spectrogram for loop mode (He flag).
fn loop_pad_mel(mel: &Array2<f32>, target_samples: usize, hop_size: usize) -> Array2<f32> {
    let target_frames = (target_samples as f32 / hop_size as f32).ceil() as usize + 1;
    let src_frames = mel.ncols();

    if src_frames == 0 || target_frames <= src_frames {
        return mel.clone();
    }

    let num_mels = mel.nrows();
    let mut looped = Array2::zeros((num_mels, target_frames));

    for i in 0..target_frames {
        // Reflect-pad: 0,1,2,...,n-1,n-2,...,1,0,1,...
        let period = 2 * src_frames.max(1) - 2;
        let pos = if period > 0 {
            let p = i % period;
            if p < src_frames {
                p
            } else {
                2 * (src_frames - 1) - p
            }
        } else {
            0
        };

        for m in 0..num_mels {
            looped[[m, i]] = mel[[m, pos]];
        }
    }

    looped
}

/// Build F0 curve from MIDI note + pitchbend.
fn build_f0_curve(
    midi_note: f32,
    pitchbend_cents: &[i32],
    t_offset: f32,
    total_frames: usize,
    pad_frames: usize,
    hop_size: usize,
    sr: f32,
    tempo: f32,
) -> Vec<f32> {
    // Pitchbend resolution: 5ms per point (UTAU standard)
    let pb_interval_ms = if tempo > 0.0 {
        // tempo is in BPM, UTAU pitch tick = 5ms
        5.0
    } else {
        5.0
    };
    let pb_interval_samples = pb_interval_ms / 1000.0 * sr;

    // Frame time positions (in samples)
    let frame_times: Vec<f32> = (0..total_frames)
        .map(|i| {
            let frame_in_content = i as i64 - pad_frames as i64;
            frame_in_content as f32 * hop_size as f32
        })
        .collect();

    if pitchbend_cents.is_empty() {
        // Constant pitch
        let midi_final = midi_note + t_offset / 100.0;
        let hz = midi_to_hz(midi_final);
        return vec![hz; total_frames];
    }

    // Pitchbend time positions
    let pb_times: Vec<f32> = (0..pitchbend_cents.len())
        .map(|i| i as f32 * pb_interval_samples)
        .collect();

    let pb_midi: Vec<f32> = pitchbend_cents
        .iter()
        .map(|&c| midi_note + c as f32 / 100.0 + t_offset / 100.0)
        .collect();

    // Interpolate to frame times using Akima
    let midi_interp = akima_interp(&pb_times, &pb_midi, &frame_times);

    // Convert to Hz
    midi_interp.iter().map(|&m| midi_to_hz(m)).collect()
}

/// Apply amplitude modulation based on pitch change rate (A flag).
fn apply_amplitude_modulation(
    audio: &mut [f32],
    f0: &[f32],
    strength: f32,
    hop_size: usize,
    _sr: f32,
) {
    if f0.len() < 2 || strength.abs() < 0.01 {
        return;
    }

    // Compute pitch change rate (derivative of log f0)
    let mut rates: Vec<f32> = Vec::with_capacity(f0.len());
    rates.push(0.0);
    for i in 1..f0.len() {
        if f0[i] > 0.0 && f0[i - 1] > 0.0 {
            rates.push(((f0[i] / f0[i - 1]).ln()).abs());
        } else {
            rates.push(0.0);
        }
    }

    // Apply gain modulation
    let factor = strength / 100.0;
    for (i, sample) in audio.iter_mut().enumerate() {
        let frame_idx = (i / hop_size).min(rates.len() - 1);
        let gain = 1.0 + factor * rates[frame_idx];
        *sample *= gain;
    }
}
