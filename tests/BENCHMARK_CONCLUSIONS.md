# HiFiSampler Benchmark 结论

## 最终结果

RTX 4060 Ti, CUDA 12.5, burn 0.21, 6 fixture 全热启动：

```
fixture                    f32 baseline   f16+全优化    加速比
                           wall(ms)       wall(ms)
baseline_short                  740.7        122.1       6.1x
cache_repeat_short              170.4         95.3       1.8x
postprocess_ap_hg               337.9        134.7       2.5x
hnsep_tension                   576.1        232.1       2.5x
loop_mode_long                 1213.0        205.8       5.9x
stereo_48k_keyshift             200.7         83.1       2.4x

Total                         3238.6        873.1       3.7x
```

最优配置：**f16 半精度 + shape bucketing (64帧步进) + BN folding + autotune**。

## 优化项收益分解

| 优化项 | 总时间(ms) | 边际收益 | 说明 |
|---|---|---|---|
| f32 baseline | 3239 | — | 起点 |
| +f16 半精度 | 2795 | 1.16x | Tensor Core + 带宽减半；stereo_48k 因 f16 kernel 慢路径反而 2x 回归 |
| +shape bucketing | 1751 | 1.60x | 64帧步进消除 vocoder shape 多样性，修复 stereo_48k 回归 |
| +BN folding | 1751 | ~1.0x | snapshot-level 折叠 scale 进 conv weight，BN 降为 identity（kernel 仍运行，收益小） |
| +autotune | 873 | 2.00x | bucketing 后调优成本可控，5个 bucket warmup 一次性覆盖 |

关键洞察：**shape bucketing 是 autotune 的前提**。无 bucketing 时 autotune 对 vocoder 多 shape 每个都调优，首请求 19s（26x 慢）；有 bucketing 后仅 5 个固定 shape，warmup 摊掉调优成本。

## Fixture 说明

| fixture | 输入 | 覆盖场景 |
|---|---|---|
| baseline_short | steady_a4 1s 44.1kHz mono | 基线渲染 |
| cache_repeat_short | 同上 | mel cache 命中 |
| postprocess_ap_hg | expressive_c4 2s | 后处理 flags (ap+hg) |
| hnsep_tension | 同上 + tension flag | HN-SEP 谐波分离 |
| loop_mode_long | long_phrase 4s | 循环模式 |
| stereo_48k_keyshift | stereo_e4 1s 48kHz g-20 | 48kHz + keyshift |

## 运行方法

```powershell
cargo run --release --features cuda -p hifisampler-core --bin bench_resampler_fixtures -- <label> config.benchmark.yaml
```

输出写入 `tests/benchmark_fixtures/results/<label>.{json,csv}`。

## 下一步可优化方向

1. **LSTM 同步点重排** — 5次 CPU LSTM 同步 → 3次，预期 hnsep 再降 30-50ms
2. **BN 完全消除** — 改结构体跳过 BN kernel（当前只折叠参数，kernel 仍运行）
3. **autotune cache 持久化分发** — 当前每次运行 warmup ~10min，cache 可随 binary 分发
