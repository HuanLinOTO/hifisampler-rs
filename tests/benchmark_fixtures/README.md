# Benchmark Fixtures

Run this from the repository root to regenerate deterministic benchmark samples:

```powershell
cargo run -p hifisampler-core --bin gen_benchmark_fixtures
```

Run full inference over the fixtures with the benchmark configuration:

```powershell
cargo run --release -p hifisampler-core --bin bench_resampler_fixtures -- baseline config.benchmark.yaml
```

Generated layout:

- `inputs/`: synthetic WAV files used as source audio.
- `requests/`: raw UTAU argument strings accepted by `UtauParams::parse()` and the server POST `/` endpoint.
- `outputs/`: target directory for benchmark render outputs.
- `results/`: JSON and CSV timing reports written by `bench_resampler_fixtures`.

The fixture set is intentionally small but covers baseline rendering, cache reuse, post-processing flags, HN-SEP/tension flags, loop mode, and 48 kHz stereo input resampling.
