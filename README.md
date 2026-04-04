# HiFiSampler-RS

A Rust rewrite of [HiFiSampler](https://github.com/openhachimi/hifisampler) - a neural vocoder-based UTAU/OpenUTAU resampler.

## Architecture

```
┌─────────────────────────────────────┐
│  OpenUTAU / UTAU                    │
│  (resampler = hifisampler.exe)      │
└──────────┬──────────────────────────┘
           │ CLI args
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│  hifisampler         │────▶│  hifisampler-server   │
│  (bridge exe)        │HTTP │  (inference server)   │
│  crates/bridge       │     │  crates/server        │
└──────────────────────┘     └──────┬───────────────┘
                                    │
                             ┌──────┴───────────────┐
                             │  hifisampler-core     │
                             │  - ONNX Runtime       │
                             │  - Audio processing   │
                             │  - Mel spectrogram    │
                             │  - UTAU parsing       │
                             └──────────────────────┘
```

## Components

- **hifisampler-core**: Shared library with inference pipeline, audio processing, UTAU protocol parsing
- **hifisampler-server**: HTTP inference server with WebUI for monitoring
- **hifisampler-bridge** (`hifisampler.exe`): OpenUTAU bridge client that forwards requests to the server

## Building

```bash
cargo build --release
```

Binaries will be in `target/release/`:

- `hifisampler-server` (or `.exe` on Windows)
- `hifisampler` (bridge, or `.exe` on Windows)

## Setup

1. Place ONNX models in the `models/` directory:
   - `models/vocoder/model.onnx` - HiFi-GAN vocoder
   - `models/vocoder/model_fp16.onnx` - HiFi-GAN vocoder (FP16/quantized, optional)
   - `models/hnsep/model_fp32_slim.onnx` - HN-SEP model (required when using HN-SEP)
   - `models/hnsep/model_fp16.onnx` - HN-SEP FP16 model (optional, non-CPU devices)

2. Copy `config.default.yaml` to `config.yaml` and adjust settings.

3. Start the server:

   ```bash
   ./hifisampler-server --config config.yaml
   ```

4. Set `hifisampler` as the resampler in OpenUTAU.

## WebUI

Access the monitoring dashboard at `http://127.0.0.1:8572/ui/` when the server is running.

## Common UTAU Flags

The WebUI monitor page now includes a quick-reference table for common flags. Useful ones in HiFiSampler-RS:

- `He`: toggles extension strategy against `processing.loop_mode`
- `HG0..100`: growl strength
- `Ht-100..100`: tension (spectral tilt)
- `Hb0..500`: breathiness (noise amount)
- `Hv0..150`: voicing (harmonic amount)
- `g...`: gender/formant shift
- `A-100..100`: amplitude modulation by pitch-rate
- `P0..100`: loudness normalization strength

`He` behavior matrix:

- `loop_mode=false`, no `He` => stretch
- `loop_mode=false`, `He` => loop
- `loop_mode=true`, no `He` => loop
- `loop_mode=true`, `He` => stretch

## License

MIT
