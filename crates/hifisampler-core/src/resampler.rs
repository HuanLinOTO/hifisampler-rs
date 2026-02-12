//! Core resampler pipeline — **exact match** with Python `backend/resampler.py`.
//!
//! All time-axis arithmetic, interpolation calls, and mel/f0 tensor shapes
//! are ported line-by-line from the Python implementation.

use crate::audio::{
    self, akima_interp_f64, interp1d, loudness_normalize, peak, pre_emphasis_tension,
};
use crate::cache::CacheManager;
use crate::config::Config;
use crate::growl::apply_growl;
use crate::mel::dynamic_range_compression;
use crate::models::Models;
use crate::parse_utau::{
    UtauFlags, UtauParams, decode_pitchbend, midi_to_hz_f64, note_to_midi, parse_flags,
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

/// Perform the full resample operation — matches Python `Resampler.render()`.
pub fn resample(
    params: &UtauParams,
    config: &Config,
    models: &Models,
    cache: &CacheManager,
) -> Result<ResampleStats> {
    let total_start = Instant::now();
    let mut stats = ResampleStats::default();

    let flags = parse_flags(&params.flags);
    let sr = config.sample_rate as f64;

    // ── Step 1: Feature extraction (matches Python generate_features) ──
    let feat_start = Instant::now();
    let (mel_origin, scale) = get_features(&params.input_path, config, models, cache, &flags)?;
    stats.feature_ms = feat_start.elapsed().as_secs_f64() * 1000.0;

    // ── Step 2: Resample (matches Python Resampler.resample) ──
    let synth_start = Instant::now();

    if params.output_path == "nul" || params.output_path == "/dev/null" {
        debug!("Null output file, skipping synthesis");
        stats.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        return Ok(stats);
    }

    let _mod_factor = params.modulation / 100.0;

    // Python time calculations (all in seconds):
    //   thop_origin = CONFIG.origin_hop_size / CONFIG.sample_rate
    //   thop        = CONFIG.hop_size / CONFIG.sample_rate
    let thop_origin = config.hop_size_interp as f64 / sr; // origin_hop_size=128 → 0.002902s
    let thop = config.hop_size as f64 / sr; // hop_size=512 → 0.01161s

    // t_area_origin = arange(mel_origin.shape[1]) * thop_origin + thop_origin / 2
    let n_mel_frames = mel_origin.ncols();
    let t_area_origin: Vec<f64> = (0..n_mel_frames)
        .map(|i| i as f64 * thop_origin + thop_origin / 2.0)
        .collect();
    let total_time = t_area_origin.last().copied().unwrap_or(0.0) + thop_origin / 2.0;

    // Velocity, offset, cutoff (in seconds)
    let vel = 2.0f64.powf(1.0 - params.velocity as f64 / 100.0);
    let offset = params.offset as f64 / 1000.0;
    let cutoff = params.cutoff as f64 / 1000.0;
    let start = offset;

    // End time calculation
    let end = if params.cutoff < 0.0 {
        start - cutoff // cutoff is negative → start - (-|cutoff|/1000) = start + |cutoff|/1000
    } else {
        total_time - cutoff
    };
    let con = start + params.consonant as f64 / 1000.0;

    let length_req = params.length as f64 / 1000.0;
    let mut stretch_length = end - con;

    // ── Loop mode (He flag) ──
    let (mel_work, t_area_work, total_time_work) = if config.processing.loop_mode || flags.he {
        let con_frame = ((con + thop_origin / 2.0) / thop_origin) as usize;
        let end_frame = ((end + thop_origin / 2.0) / thop_origin) as usize;
        let con_frame = con_frame.min(n_mel_frames);
        let end_frame = end_frame.min(n_mel_frames);

        let mel_loop = mel_origin
            .slice(ndarray::s![.., con_frame..end_frame])
            .to_owned();
        let pad_loop_size = (length_req / thop_origin) as usize + 1;

        // Reflect-pad mel_loop
        let loop_frames = mel_loop.ncols();
        let mut padded_mel = Array2::zeros((mel_origin.nrows(), loop_frames + pad_loop_size));
        for t in 0..(loop_frames + pad_loop_size) {
            let src_t = reflect_index_mel(t, loop_frames);
            for m in 0..mel_origin.nrows() {
                padded_mel[[m, t]] = mel_loop[[m, src_t]];
            }
        }

        // Concatenate: mel_origin[:, :con_frame] + padded_mel
        let mut mel_new = Array2::zeros((mel_origin.nrows(), con_frame + padded_mel.ncols()));
        for t in 0..con_frame {
            for m in 0..mel_origin.nrows() {
                mel_new[[m, t]] = mel_origin[[m, t]];
            }
        }
        for t in 0..padded_mel.ncols() {
            for m in 0..mel_origin.nrows() {
                mel_new[[m, con_frame + t]] = padded_mel[[m, t]];
            }
        }

        stretch_length = pad_loop_size as f64 * thop_origin;

        let new_n = mel_new.ncols();
        let new_t_area: Vec<f64> = (0..new_n)
            .map(|i| i as f64 * thop_origin + thop_origin / 2.0)
            .collect();
        let new_total = new_t_area.last().copied().unwrap_or(0.0) + thop_origin / 2.0;

        (mel_new, new_t_area, new_total)
    } else {
        (mel_origin.clone(), t_area_origin.clone(), total_time)
    };

    // ── Interpolator for mel ──
    // mel_interp = interp.interp1d(t_area_origin, mel_origin, axis=1)
    // (We'll interpolate each mel band separately below)

    // Scaling ratio
    let scaling_ratio = if stretch_length < length_req {
        length_req / stretch_length
    } else {
        1.0
    };

    // stretch function: maps render time → source time
    // def stretch(t, con, scaling_ratio):
    //     return np.where(t < vel*con, t/vel, con + (t - vel*con) / scaling_ratio)
    let stretch_fn = |t: f64| -> f64 {
        if t < vel * con {
            t / vel
        } else {
            con + (t - vel * con) / scaling_ratio
        }
    };

    // stretched_n_frames = (con*vel + (total_time - con) * scaling_ratio) // thop + 1
    let stretched_n_frames =
        ((con * vel + (total_time_work - con) * scaling_ratio) / thop + 1.0) as usize;

    // stretched_t_mel = arange(stretched_n_frames) * thop + thop / 2
    let stretched_t_mel: Vec<f64> = (0..stretched_n_frames)
        .map(|i| i as f64 * thop + thop / 2.0)
        .collect();

    // Cut frames calculation (fill = pad_frames)
    let fill = config.pad_frames as f64;

    // start_left_mel_frames = (start*vel + thop/2)//thop
    let start_left_mel_frames = ((start * vel + thop / 2.0) / thop).floor();
    let cut_left_mel_frames = if start_left_mel_frames > fill {
        start_left_mel_frames - fill
    } else {
        0.0
    };

    // end_right_mel_frames = stretched_n_frames - (length_req+con*vel + thop/2)//thop
    let end_right_mel_frames =
        stretched_n_frames as f64 - ((length_req + con * vel + thop / 2.0) / thop).floor();
    let cut_right_mel_frames = if end_right_mel_frames > fill {
        end_right_mel_frames - fill
    } else {
        0.0
    };

    // Trim stretched_t_mel
    let cut_left = cut_left_mel_frames as usize;
    let cut_right = cut_right_mel_frames as usize;
    let trimmed_end = stretched_n_frames.saturating_sub(cut_right);
    let cut_left = cut_left.min(trimmed_end);
    let stretched_t_mel_trimmed: Vec<f64> = stretched_t_mel[cut_left..trimmed_end].to_vec();

    if stretched_t_mel_trimmed.is_empty() {
        anyhow::bail!("No frames to render after trimming");
    }

    // Apply stretch function and clamp to valid range
    let t_last = t_area_work.last().copied().unwrap_or(0.0);
    let stretch_t_mel: Vec<f64> = stretched_t_mel_trimmed
        .iter()
        .map(|&t| stretch_fn(t).clamp(0.0, t_last))
        .collect();

    // new_start / new_end (relative to the trimmed mel)
    let new_start = start * vel - cut_left_mel_frames * thop;
    let new_end = (length_req + con * vel) - cut_left_mel_frames * thop;

    // Interpolate mel: mel_render = mel_interp(stretch_t_mel)
    let render_frames = stretch_t_mel.len();
    let num_mels = mel_work.nrows();
    let mut mel_render = Array2::zeros((num_mels, render_frames));

    let t_area_f32: Vec<f32> = t_area_work.iter().map(|&t| t as f32).collect();
    let stretch_t_f32: Vec<f32> = stretch_t_mel.iter().map(|&t| t as f32).collect();

    for m in 0..num_mels {
        let y_src: Vec<f32> = (0..mel_work.ncols()).map(|i| mel_work[[m, i]]).collect();
        let y_interp = interp1d(&t_area_f32, &y_src, &stretch_t_f32);
        for (i, &v) in y_interp.iter().enumerate() {
            mel_render[[m, i]] = v;
        }
    }

    // ── Pitch / F0 calculation ──
    // t = arange(mel_render.shape[1]) * thop
    let t_render: Vec<f64> = (0..render_frames).map(|i| i as f64 * thop).collect();

    let midi_note = note_to_midi(&params.pitch).unwrap_or(60) as f64;
    let pitchbend_cents = decode_pitchbend(&params.pitchbend);

    // pitch = pitchbend / 100 + self.pitch
    let mut pitch_data: Vec<f64> = pitchbend_cents
        .iter()
        .map(|&c| c as f64 / 100.0 + midi_note)
        .collect();

    // Apply t flag offset
    if flags.t != 0 {
        let t_offset = flags.t as f64 / 100.0;
        pitch_data.iter_mut().for_each(|p| *p += t_offset);
    }

    // If no pitchbend data, use constant pitch
    if pitch_data.is_empty() {
        pitch_data = vec![midi_note + flags.t as f64 / 100.0];
    }

    // t_pitch = 60 * arange(len(pitch)) / (tempo * 96) + new_start
    let tempo = if params.tempo > 0.0 {
        params.tempo as f64
    } else {
        120.0
    };
    let t_pitch: Vec<f64> = (0..pitch_data.len())
        .map(|i| 60.0 * i as f64 / (tempo * 96.0) + new_start)
        .collect();

    // Akima interpolation of pitch — use f64 throughout to match Python/SciPy precision
    // pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
    // pitch_render = pitch_interp(np.clip(t, new_start, t_pitch[-1]))
    let t_pitch_last = t_pitch.last().copied().unwrap_or(0.0);
    let t_render_clamped: Vec<f64> = t_render
        .iter()
        .map(|&t| t.clamp(new_start, t_pitch_last))
        .collect();

    let pitch_render_f64 = akima_interp_f64(&t_pitch, &pitch_data, &t_render_clamped);

    // f0_render = midi_to_hz(pitch_render) — compute in f64, then convert to f32 for ONNX
    let f0_render: Vec<f32> = pitch_render_f64
        .iter()
        .map(|&p| midi_to_hz_f64(p) as f32)
        .collect();
    let pitch_render: Vec<f32> = pitch_render_f64.iter().map(|&p| p as f32).collect();

    // ── Vocoder synthesis ──
    // Python ONNX path:
    //   mel = mel_render [num_mels, render_frames]
    //   mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)  → [1, render_frames, num_mels]
    //   f0 = np.expand_dims(f0_render, axis=0)                → [1, render_frames]
    //   output = ort_session.run(['waveform'], {'mel': mel, 'f0': f0})[0]
    //   wav_con = output[0]  → [total_samples]
    //
    // Our Vocoder::synthesize expects mel [num_mels, n_frames] and transposes internally.
    let wav_con = models.vocoder.lock().synthesize(&mel_render, &f0_render)?;

    stats.synthesis_ms = synth_start.elapsed().as_secs_f64() * 1000.0;

    // ── Step 3: Post-processing ──
    let post_start = Instant::now();

    // Cut output:
    //   render = wav_con[int(new_start * sr):int(new_end * sr)]
    let cut_start = (new_start * sr) as usize;
    let cut_end = (new_end * sr) as usize;
    let cut_start = cut_start.min(wav_con.len());
    let cut_end = cut_end.min(wav_con.len());

    let mut render = if cut_start < cut_end {
        wav_con[cut_start..cut_end].to_vec()
    } else {
        wav_con.clone()
    };

    // Amplitude modulation (A flag)
    if flags.a != 0 {
        apply_amplitude_modulation(
            &mut render,
            &pitch_render,
            &t_render,
            new_start,
            new_end,
            flags.a as f64,
        );
    }

    // Volume recovery: render = render / scale
    if scale > 0.0 && scale != 1.0 {
        let inv_scale = 1.0 / scale;
        render.iter_mut().for_each(|x| *x *= inv_scale as f32);
    }

    let new_max = peak(&render);

    // Growl effect (HG flag)
    if flags.hg > 0 {
        render = apply_growl(&render, flags.hg as f32 / 100.0, config.sample_rate);
    }

    // Loudness normalization (P flag)
    if config.processing.wave_norm {
        if flags.p > 0 {
            render = loudness_normalize(&render, -16.0, flags.p as f32);
        }
    }

    // Peak limiting
    if new_max > config.processing.peak_limit {
        let gain = config.processing.peak_limit / new_max;
        render.iter_mut().for_each(|x| *x *= gain);
    }

    // Volume scaling
    let volume = params.volume as f64 / 100.0;
    if (volume - 1.0).abs() > 0.001 {
        render.iter_mut().for_each(|x| *x *= volume as f32);
    }

    stats.postprocess_ms = post_start.elapsed().as_secs_f64() * 1000.0;
    stats.output_samples = render.len();

    // Save output
    audio::save_wav(&params.output_path, &render, config.sample_rate)?;

    stats.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    debug!(
        "Resample complete: total={:.1}ms feat={:.1}ms synth={:.1}ms post={:.1}ms",
        stats.total_ms, stats.feature_ms, stats.synthesis_ms, stats.postprocess_ms
    );

    Ok(stats)
}

/// Extract features — matches Python `Resampler.generate_features`.
fn get_features(
    wav_path: &str,
    config: &Config,
    models: &Models,
    cache: &CacheManager,
    flags: &UtauFlags,
) -> Result<(Array2<f32>, f64)> {
    let path = Path::new(wav_path);
    let suffix = CacheManager::flag_suffix(flags.g, flags.hb, flags.hv, flags.ht);
    let cache_path = CacheManager::cache_path(path, &suffix);

    // Check cache
    if !flags.gen_cache {
        if let Some((mel, scale)) = cache.load_mel_cache(&cache_path) {
            return Ok((mel, scale as f64));
        }
    }

    // Read WAV
    let wave = audio::read_wav(wav_path, config.sample_rate)?;

    // HN-SEP processing
    let mut wave = if flags.needs_hnsep() {
        if let Some(ref hnsep) = models.hnsep {
            let harmonic = hnsep.lock().predict_from_audio(&wave, config.sample_rate)?;

            let noise: Vec<f32> = wave
                .iter()
                .zip(harmonic.iter())
                .map(|(&w, &h)| w - h)
                .collect();

            let hb = flags.hb as f32 / 100.0;
            let hv = flags.hv as f32 / 100.0;

            if flags.ht != 0 {
                // Python: wave = (breath/100)*(wave - seg_output) +
                //                pre_emphasis_base_tension((voicing/100)*seg_output, -tension/50)
                let scaled_harmonic: Vec<f32> = harmonic.iter().map(|&h| hv * h).collect();
                let tension_harmonic = pre_emphasis_tension(
                    &scaled_harmonic,
                    -(flags.ht as f32) / 50.0,
                    config.sample_rate,
                    config.n_fft,
                    config.hop_size,
                );
                noise
                    .iter()
                    .zip(tension_harmonic.iter())
                    .map(|(&n, &h)| hb * n + h)
                    .collect()
            } else {
                // Python: wave = (breath/100)*(wave - seg_output) + (voicing/100)*seg_output
                noise
                    .iter()
                    .zip(harmonic.iter())
                    .map(|(&n, &h)| hb * n + hv * h)
                    .collect()
            }
        } else {
            warn!("HN-SEP requested but model not loaded");
            wave
        }
    } else {
        wave
    };

    // Python: wave = wave.squeeze(0).squeeze(0).cpu().numpy()
    //         wave = torch.from_numpy(wave).unsqueeze(0)  # shape [1, T]
    //         wave_max = torch.max(torch.abs(wave))
    //         if wave_max >= 0.5:
    //             scale = 0.5 / wave_max
    //             wave = wave * scale
    //         else:
    //             scale = 1.0
    let wave_max = peak(&wave);
    let scale: f64 = if wave_max >= 0.5 {
        let s = 0.5 / wave_max as f64;
        wave.iter_mut().for_each(|x| *x *= s as f32);
        s
    } else {
        1.0
    };

    // Gender / key_shift
    // Python: mel_origin = models.mel_analyzer(wave, gender/100, 1).squeeze()
    // mel_analyzer.__call__(y, key_shift=gender/100, speed=1.0)
    // mel_analyzer was initialized with hop_length=origin_hop_size (128)
    let key_shift = flags.g as f32 / 100.0;

    // Create mel analyzer with origin_hop_size for feature extraction
    let mel_analyzer = crate::mel::MelAnalyzer::new(
        config.sample_rate,
        config.n_fft,
        config.hop_size_interp, // origin_hop_size = 128
        config.win_size,
        config.num_mels,
        config.fmin,
        config.fmax,
    );

    let mel = mel_analyzer.mel_spectrogram(&wave, key_shift, 1.0);

    // DRC: mel_origin = dynamic_range_compression_torch(mel_origin)
    let mel = dynamic_range_compression(&mel, 1.0);

    // Save to cache
    let _ = cache.save_mel_cache(&cache_path, &mel, scale as f32);

    Ok((mel, scale))
}

/// Reflect index for mel loop padding.
fn reflect_index_mel(idx: usize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let idx = idx % period;
    if idx < len { idx } else { period - idx }
}

/// Apply amplitude modulation (A flag) — matches Python exactly.
///
/// Python:
///   pitch_derivative = np.gradient(pitch_render, t)
///   gain = 5**((10**-4) * A * pitch_derivative)
///   interpolated_gain = np.interp(audio_time, t, gain)
///   render = render * interpolated_gain
fn apply_amplitude_modulation(
    audio: &mut [f32],
    pitch_render: &[f32],
    t_mel: &[f64],
    new_start: f64,
    new_end: f64,
    a_flag: f64,
) {
    if pitch_render.len() < 2 || t_mel.len() < 2 || a_flag.abs() < 0.01 {
        return;
    }

    // np.gradient(pitch_render, t)
    let n = pitch_render.len();
    let mut pitch_derivative = vec![0.0f64; n];

    // Forward difference for first point
    if n >= 2 {
        let dt = t_mel[1] - t_mel[0];
        if dt.abs() > 1e-12 {
            pitch_derivative[0] = (pitch_render[1] as f64 - pitch_render[0] as f64) / dt;
        }
    }
    // Central differences for interior
    for i in 1..n - 1 {
        let dt = t_mel[i + 1] - t_mel[i - 1];
        if dt.abs() > 1e-12 {
            pitch_derivative[i] = (pitch_render[i + 1] as f64 - pitch_render[i - 1] as f64) / dt;
        }
    }
    // Backward difference for last point
    if n >= 2 {
        let dt = t_mel[n - 1] - t_mel[n - 2];
        if dt.abs() > 1e-12 {
            pitch_derivative[n - 1] =
                (pitch_render[n - 1] as f64 - pitch_render[n - 2] as f64) / dt;
        }
    }

    // gain = 5 ** (1e-4 * A * pitch_derivative)
    let gain_at_mel: Vec<f64> = pitch_derivative
        .iter()
        .map(|&d| 5.0f64.powf(1e-4 * a_flag * d))
        .collect();

    // Interpolate gain to audio sample positions
    let num_samples = audio.len();
    let t_mel_f32: Vec<f32> = t_mel.iter().map(|&t| t as f32).collect();
    let gain_f32: Vec<f32> = gain_at_mel.iter().map(|&g| g as f32).collect();

    for (i, sample) in audio.iter_mut().enumerate() {
        let t =
            new_start as f32 + (new_end as f32 - new_start as f32) * i as f32 / num_samples as f32;
        // Linear interp of gain
        let gain = interp1d_single(&t_mel_f32, &gain_f32, t);
        *sample *= gain;
    }
}

/// Single-point linear interpolation with edge clamping.
fn interp1d_single(x: &[f32], y: &[f32], t: f32) -> f32 {
    if x.is_empty() || y.is_empty() {
        return 1.0;
    }
    if t <= x[0] {
        return y[0];
    }
    if t >= *x.last().unwrap() {
        return *y.last().unwrap();
    }

    let idx = match x.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
        Ok(i) => return y[i],
        Err(i) => i.saturating_sub(1),
    };

    let idx = idx.min(x.len() - 2);
    let frac = (t - x[idx]) / (x[idx + 1] - x[idx]);
    y[idx] + frac * (y[idx + 1] - y[idx])
}
