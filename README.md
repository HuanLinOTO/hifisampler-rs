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
   - `models/hnsep/model.onnx` - HN-SEP model (optional)

2. Copy `config.default.yaml` to `config.yaml` and adjust settings.

3. Start the server:
   ```bash
   ./hifisampler-server --config config.yaml
   ```

4. Set `hifisampler` as the resampler in OpenUTAU.

## WebUI

Access the monitoring dashboard at `http://127.0.0.1:8572/ui/` when the server is running.

## License

MIT
