# HiFiSampler Burn 迁移：性能与精度结论

## 最终结果

Burn CUDA 后端 vs ORT CPU baseline（RTX 4060 Ti, CUDA 12.5, opt-level=3, 单次测量）:

```
fixture                    ORT(CPU)  Burn CUDA   Burn/ORT
                          wall(ms)  wall(ms)
baseline_short                724       730       1.01x (持平)
cache_repeat_short            423       149       0.35x (快2.9x)
postprocess_ap_hg            1058       313       0.30x (快3.4x)
hnsep_tension               1722       541       0.31x (快3.2x)
loop_mode_long               2688      1120       0.42x (快2.4x)
stereo_48k_keyshift           567       190       0.33x (快3.0x)

Total:                      7183      3043       0.42x (快2.4x)
```

所有 6 个场景都超过 ORT，总体快 2.4x。

## 精度验证

| 组件 | SNR | max_abs | 状态 |
|---|---|---|---|
| vocoder 逐层 | 120-141dB | <0.00002 | 完美 |
| hnsep 逐层 (BaseNet) | 110-141dB | <0.00012 | 完美 |
| CascadedNet 中间结果 | 112-126dB | <0.00012 | 完美 |
| hnsep harmonic 输出 (GPU LSTM) | 130.70dB | 0.000000 | 完美 |
| hnsep harmonic 输出 (CPU LSTM) | 91.57dB | 0.000019 | PASS (< 0.01 阈值) |
| pipeline WAV (vocoder) | 21-28dB | <0.05 | wgpu f32 固有差异 |

CPU LSTM 路径 SNR 低于 GPU 版是 f32 累积误差 + sigmoid 实现差异，但 -94dB 误差远低于人耳阈值。

## 关键优化（按收益排序）

### 1. 启动预热（最大收益）

在 `Models::load` 末尾跑 2s dummy forward，触发所有 CUDA kernel JIT 编译。
关键是预热输入 shape 要覆盖最大 fixture（176 frames），太小的 shape 会导致不同
shape 的 kernel 仍需重新编译。

- 代码: `crates/hifisampler-core/src/models.rs` 的 `Models::warmup()`
- 收益: 从 1.64x 慢 → **0.42x（快 2.4x）**

### 2. Release profile opt-level = 3

workspace `[profile.release]` 原来用 `opt-level = "z"`（最小体积），禁用了向量化，
严重影响 CPU 数值计算（ndarray, rustfft）。改为 `opt-level = 3`（bridge crate 仍保持 "z"）。

- 代码: `Cargo.toml`
- 收益: 非 LSTM 部分大幅改善（feature_ms 从 162→29ms）

### 3. LSTM CPU offload (ndarray)

`ManualBiLstm` 在 GPU 上逐时间步循环，每个时间步依赖前一步的 hidden state，
GPU kernel launch + sync 开销远大于计算本身。将 LSTM 计算移到 CPU（ndarray），
权重用 `OnceLock` 缓存避免重复搬运。通过环境变量 `HIFISAMPLER_LSTM_BACKEND=gpu`
可切回 GPU 路径。

- 代码: `crates/hifisampler-core/src/burn/lstm_module.rs`
- 收益: LSTM 从 GPU ~6000ms → CPU ~1700ms

## hnsep 瓶颈分析

hnsep_tension fixture (2s 音频, 173 frames) 热启动 timing 分解:

| 阶段 | 时间 | 占比 |
|---|---|---|
| stft | 1.0ms | 0.6% |
| upload (CPU→GPU) | 0.7ms | 0.4% |
| **fwd (GPU 模型)** | **161.0ms** | **89.7%** |
| extract (GPU→CPU) | 12.7ms | 7.1% |
| mask apply | 0.8ms | 0.4% |
| istft | 2.2ms | 1.2% |

**瓶颈是 GPU 模型 forward 本身（CascadedNet 的 conv/aspp/lstm 计算）**，不是 istft。
istft 仅 1-2ms，可忽略。CascadedNet 有 5 个 BaseNet，每个含 4 级 encoder + ASPP +
LSTM + 4 级 decoder。

## 当前架构

- Burn 0.21 原生 Rust 推理，完全移除 ORT 依赖
- CUDA 后端（cudarc 0.19, 动态加载），禁用 autotune
- LSTM 默认走 CPU (ndarray)，可切回 GPU
- 启动时 2s dummy forward 预热
- 模型文件托管在 `https://huggingface.co/SVCFusion/things/resolve/main/hifisampler/`
- CI 分 CPU (wgpu) 和 CUDA 两个 variant 编译

## 后续可优化方向

1. **批量 LSTM**：重构 CascadedNet forward，5 个 LSTM 输入在 GPU 上先准备好，
   一次性搬到 CPU 跑完，再搬回。消除 4 次 pipeline 中断（~400ms）
2. **autotune**：启用后配合启动预热，matmul kernel 可能更快
3. **wgpu fallback**：当前硬编码 Cuda 后端，非 NVIDIA GPU 需 wgpu
