//! UTAU protocol parsing.
//!
//! Handles note name → MIDI, MIDI → Hz, pitchbend decoding, and flag parsing.

use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Note name to MIDI number mapping.
/// C4 = 60, A4 = 69, etc.
pub fn note_to_midi(note: &str) -> Option<i32> {
    static NOTE_MAP: LazyLock<HashMap<&str, i32>> = LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert("C", 0);
        m.insert("C#", 1);
        m.insert("D", 2);
        m.insert("D#", 3);
        m.insert("E", 4);
        m.insert("F", 5);
        m.insert("F#", 6);
        m.insert("G", 7);
        m.insert("G#", 8);
        m.insert("A", 9);
        m.insert("A#", 10);
        m.insert("B", 11);
        m
    });

    // Parse note like "C4", "A#5", etc.
    let note = note.trim();
    if note.len() < 2 {
        return None;
    }

    let (name, octave_str) = if note.len() >= 3 && &note[1..2] == "#" {
        (&note[..2], &note[2..])
    } else {
        (&note[..1], &note[1..])
    };

    let octave: i32 = octave_str.parse().ok()?;
    let semitone = NOTE_MAP.get(name)?;
    Some((octave + 1) * 12 + semitone)
}

/// MIDI number to frequency in Hz (A4 = 440 Hz, equal temperament).
pub fn midi_to_hz(midi: f32) -> f32 {
    440.0 * 2.0f32.powf((midi - 69.0) / 12.0)
}

/// Hz to MIDI number.
pub fn hz_to_midi(hz: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (hz / 440.0).log2()
}

/// Decode UTAU pitchbend string.
///
/// Format: Base64-encoded 12-bit signed integers, with '#' as RLE separator.
///
/// Python reference (`pitch_string_to_cents`):
///   pitch = x.split('#')
///   for i in range(0, len(pitch), 2):
///       p = pitch[i:i+2]
///       if len(p) == 2:
///           pitch_str, rle = p
///           res.extend(to_int12_stream(pitch_str))
///           res.extend([res[-1]] * int(rle))
///       else:
///           res.extend(to_int12_stream(p[0]))
///   return np.concatenate([res, np.zeros(1)])
pub fn decode_pitchbend(pitch_str: &str) -> Vec<i32> {
    const BASE64_CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if pitch_str.is_empty() {
        return vec![0]; // trailing zero only
    }

    // to_int12_stream: decode base64 pairs into 12-bit signed ints
    let decode_b64_stream = |seg: &str| -> Vec<i32> {
        let chars: Vec<char> = seg.chars().collect();
        let mut out = Vec::new();
        let mut j = 0;
        while j + 1 < chars.len() {
            let c1 = BASE64_CHARS.find(chars[j]).unwrap_or(0) as i32;
            let c2 = BASE64_CHARS.find(chars[j + 1]).unwrap_or(0) as i32;
            let uint12 = (c1 << 6) | c2;
            // Two's complement for 12-bit
            let val = if (uint12 >> 11) & 1 == 1 {
                uint12 - 4096
            } else {
                uint12
            };
            out.push(val);
            j += 2;
        }
        out
    };

    let segments: Vec<&str> = pitch_str.split('#').collect();
    let mut result: Vec<i32> = Vec::new();

    // Process pairs: (data_string, rle_count), last may be unpaired
    let mut i = 0;
    while i < segments.len() {
        if i + 1 < segments.len() {
            // Pair: (pitch_str, rle)
            let pitch_data = segments[i];
            let rle_str = segments[i + 1];

            result.extend(decode_b64_stream(pitch_data));

            if let Ok(rle_count) = rle_str.parse::<usize>() {
                let last_val = result.last().copied().unwrap_or(0);
                for _ in 0..rle_count {
                    result.push(last_val);
                }
            }
            i += 2;
        } else {
            // Last unpaired segment
            result.extend(decode_b64_stream(segments[i]));
            i += 1;
        }
    }

    // Append trailing zero (matching Python: np.concatenate([res, np.zeros(1)]))
    result.push(0);

    result
}

/// Parsed UTAU flags.
#[derive(Debug, Clone, Default)]
pub struct UtauFlags {
    /// Gender / formant shift (key_shift in semitones * 100)
    pub g: i32,
    /// Breathiness (HN-SEP noise channel scaling, 0-500, default 100)
    pub hb: i32,
    /// Voicing (HN-SEP harmonic channel scaling, 0-150, default 100)
    pub hv: i32,
    /// Tension (-100 to 100, frequency tilt)
    pub ht: i32,
    /// Growl effect (0-100)
    pub hg: i32,
    /// Pitch offset in cents
    pub t: i32,
    /// Amplitude modulation based on pitch change rate
    pub a: i32,
    /// Loudness normalization strength (0-100, default 100)
    pub p: i32,
    /// Force regenerate cache
    pub gen_cache: bool,
    /// Enable mel loop mode
    pub he: bool,
}

impl UtauFlags {
    pub fn needs_hnsep(&self) -> bool {
        self.hb != 100 || self.hv != 100 || self.ht != 0
    }
}

/// Parse UTAU flags string.
///
/// Format: various flag codes like "g-10Hb80Hv120Ht20HG50t10A5P80GHe"
pub fn parse_flags(flags_str: &str) -> UtauFlags {
    static FLAG_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)(g|Hb|Hv|Ht|HG|He|t|A|P|G)(-?\d+)?").unwrap()
    });

    let mut flags = UtauFlags {
        hb: 100,
        hv: 100,
        p: 100,
        ..Default::default()
    };

    for cap in FLAG_RE.captures_iter(flags_str) {
        let name = &cap[1];
        let value: Option<i32> = cap.get(2).and_then(|m| m.as_str().parse().ok());

        match name {
            "g" => flags.g = value.unwrap_or(0),
            "Hb" | "hb" => flags.hb = value.unwrap_or(100).clamp(0, 500),
            "Hv" | "hv" => flags.hv = value.unwrap_or(100).clamp(0, 150),
            "Ht" | "ht" => flags.ht = value.unwrap_or(0).clamp(-100, 100),
            "HG" | "hg" => flags.hg = value.unwrap_or(0).clamp(0, 100),
            "He" | "he" => flags.he = true,
            "t" if name == "t" => flags.t = value.unwrap_or(0).clamp(-1200, 1200),
            "A" => flags.a = value.unwrap_or(0).clamp(-100, 100),
            "P" => flags.p = value.unwrap_or(100).clamp(0, 100),
            "G" if name == "G" => flags.gen_cache = true,
            _ => {}
        }
    }

    flags
}

/// Full UTAU resample parameters.
#[derive(Debug, Clone)]
pub struct UtauParams {
    pub input_path: String,
    pub output_path: String,
    pub pitch: String,
    pub velocity: f32,
    pub flags: String,
    pub offset: f32,
    pub length: f32,
    pub consonant: f32,
    pub cutoff: f32,
    pub volume: f32,
    pub modulation: f32,
    pub tempo: f32,
    pub pitchbend: String,
}

impl UtauParams {
    /// Parse UTAU resample parameters from the raw argument string.
    /// Format: <input> <output> <pitch> <velocity> <flags> <offset> <length>
    ///         <consonant> <cutoff> <volume> <modulation> <tempo> [pitchbend...]
    pub fn parse(raw: &str) -> Result<Self> {
        // Split by ".wav " to get input and output paths
        let parts: Vec<&str> = raw.splitn(3, ".wav ").collect();

        let (input_path, remainder) = if parts.len() >= 3 {
            (
                format!("{}.wav", parts[0]),
                format!("{}.wav {}", parts[1], parts[2]),
            )
        } else if parts.len() == 2 {
            (format!("{}.wav", parts[0]), parts[1].to_string())
        } else {
            anyhow::bail!("Invalid UTAU params: cannot find .wav separator");
        };

        let args: Vec<&str> = remainder.split_whitespace().collect();
        
        // Extract output path and UTAU params
        // The last 11 args are: pitch velocity flags offset length consonant cutoff volume modulation tempo pitchbend
        if args.len() < 12 {
            anyhow::bail!("Not enough UTAU parameters: got {}, need at least 12", args.len());
        }

        let param_start = args.len() - 11;
        let output_path = args[..param_start].join(" ");
        let output_path = if !output_path.ends_with(".wav") {
            format!("{}.wav", output_path)
        } else {
            output_path
        };

        let parse_f32 = |s: &str| -> f32 { s.parse().unwrap_or(0.0) };

        Ok(Self {
            input_path,
            output_path,
            pitch: args[param_start].to_string(),
            velocity: parse_f32(args[param_start + 1]),
            flags: args[param_start + 2].to_string(),
            offset: parse_f32(args[param_start + 3]),
            length: parse_f32(args[param_start + 4]),
            consonant: parse_f32(args[param_start + 5]),
            cutoff: parse_f32(args[param_start + 6]),
            volume: parse_f32(args[param_start + 7]),
            modulation: parse_f32(args[param_start + 8]),
            tempo: parse_f32(args[param_start + 9]),
            pitchbend: args[param_start + 10].to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_to_midi() {
        assert_eq!(note_to_midi("C4"), Some(60));
        assert_eq!(note_to_midi("A4"), Some(69));
        assert_eq!(note_to_midi("C#4"), Some(61));
        assert_eq!(note_to_midi("B3"), Some(59));
    }

    #[test]
    fn test_midi_to_hz() {
        assert!((midi_to_hz(69.0) - 440.0).abs() < 0.01);
        assert!((midi_to_hz(60.0) - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_decode_pitchbend() {
        // "AA" = decoded [0] + trailing zero = [0, 0]
        let result = decode_pitchbend("AA");
        assert_eq!(result, vec![0, 0]);

        // Basic decoding: "AAAA" = two zeros + trailing zero = [0, 0, 0]
        let result = decode_pitchbend("AAAA");
        assert_eq!(result, vec![0, 0, 0]);

        // Python reference: 'AAAAAA#3#BABA' → [0,0,0, 0,0,0, 64,64, 0] (len=9, includes trailing zero)
        let result = decode_pitchbend("AAAAAA#3#BABA");
        assert_eq!(result, vec![0, 0, 0, 0, 0, 0, 64, 64, 0]);

        // 'BABA' → [64, 64, 0] (decoded + trailing zero)
        let result = decode_pitchbend("BABA");
        assert_eq!(result, vec![64, 64, 0]);
    }

    #[test]
    fn test_parse_flags() {
        let flags = parse_flags("g-10Hb80Hv120Ht20HG50");
        assert_eq!(flags.g, -10);
        assert_eq!(flags.hb, 80);
        assert_eq!(flags.hv, 120);
        assert_eq!(flags.ht, 20);
        assert_eq!(flags.hg, 50);
    }
}
