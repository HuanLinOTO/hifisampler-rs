# REFACTOR: 迁移 ONNX Runtime → Burn 原生推理

## 目标

将 hifisampler-rs 的 ONNX 推理后端从 `ort 2.0.0-rc.11`（onnxruntime 动态库）
迁移到 `burn 0.21.0`（tracel-ai/burn 上游稳定版）原生 Rust 推理。

**动机**：减小二进制体积、移除 C++ 依赖、纯 Rust 单二进制部署。

**策略**：将两个 ONNX 模型用原生 Burn 模块重写，从 PyTorch checkpoint 加载权重。

---

## 已验证的关键事实

### 模型架构（已从源码确认）

**Vocoder (HiFi-GAN NSF, `mini_nsf=true`)**:
- 源码: `D:\Projects\ToooooHifiSampler\hifisampler\util\nsf_hifigan.py` (322行)
- checkpoint: `D:\Projects\ToooooHifiSampler\新建文件夹\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt`
- config: 同目录 `config.json` (已确认 `mini_nsf=true`, `upsample_rates=[8,8,2,2,2]`, `resblock="1"`)
- 结构: conv_pre(128→512,k7) + 5×ConvTranspose1d + 15×ResBlock1(3并行×5阶段) + source_conv(1→128,k1) + conv_post(16→1,k7) + tanh
- **无 BatchNorm、无 LSTM**，纯 Conv1d/ConvTranspose1d + LeakyReLU(0.1)
- weight_norm 在 checkpoint 中是参数化的，需 Python 预处理融合

**HN-SEP (CascadedNet)**:
- 源码: `D:\Projects\ToooooHifiSampler\hifisampler\hnsep\nets.py` (202行) + `layers.py` (166行)
- checkpoint: `D:\Projects\ToooooHifiSampler\新建文件夹\vr\model.pt`
- config: 同目录 `config.yaml` (n_out=32, n_out_lstm=128, is_complex=true, is_mono=true)
- 结构: 3-stage 级联 U-Net，5个 BaseNet，每个含 4级下采样 + ASPP + LSTM + 4级上采样
- **有 BatchNorm2d、双向 LSTM、bilinear upsample** — 比 vocoder 复杂得多

### Burn 0.21 API 验证结果（全部通过）

| 操作 | Burn 0.21 方法 | 状态 |
|---|---|---|
| `torch.arange(1, n+1)` | `Tensor::<B,1,Int>::arange(1..n+1, &device).float()` | ✅ 返回 Int 需 .float() |
| `torch.fmod(x, 1.0)` | `x.fmod_scalar(1.0)` | ✅ |
| `cumsum(dim=1)` | `x.cumsum(1)` | ✅ |
| `torch.sin(x)` | `x.sin()` | ✅ |
| `F.pad(x, (0,0,0,1))` | `x.pad((0,0,0,1), 0.0)` 或 `x.pad([(0,0),(0,1),(0,0)], 0.0)` | ✅ |
| `x.unsqueeze(-1)` | `x.unsqueeze_dims(&[-1])` | ✅ |
| `x.reshape([B,1,-1])` | `x.reshape([batch, 1, -1])` | ✅ |
| `x[:, 1:, :]` | `x.slice(s![.., 1.., ..])` | ✅ |
| 广播 `[B,T,1]*[64]→[B,T,64]` | 自动广播 | ✅ |
| `F.leaky_relu(x, 0.1)` | `burn::tensor::activation::leaky_relu(x, 0.1)` (函数) 或 `LeakyRelu` 模块 | ✅ 无 .leaky_relu() 方法 |
| `x + y` (残差) | `x + y` (Add trait) | ✅ |
| `xs / 3` (MRF均值) | `xs / 3.0` (Div trait) | ✅ |
| `x.tanh()` | `x.tanh()` | ✅ |
| `scalar * tensor` | `2.0 * std::f64::consts::PI * rad` | ✅ PR #3127 |

### ConvTranspose1d 验证（输出长度与 PyTorch 完全一致）

Burn 公式: `(L_in-1)*stride - 2*padding + dilation*(kernel_size-1) + 1 + padding_out`
等价于 PyTorch: `(L_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1`

| Layer | k | s | p | Burn L_out | PyTorch L_out |
|---|---|---|---|---|---|
| ups[0] | 16 | 8 | 4 | 8L | 8L ✅ |
| ups[1] | 16 | 8 | 4 | 8L | 8L ✅ |
| ups[2] | 4 | 2 | 1 | 2L | 2L ✅ |
| ups[3] | 4 | 2 | 1 | 2L | 2L ✅ |
| ups[4] | 4 | 2 | 1 | 2L | 2L ✅ |

总上采样: 8×8×2×2×2 = 512× → `n_samples = n_frames * 512` ✅

权重布局: Burn `[in, out/groups, k]` = PyTorch `[in, out/groups, k]` — **无需转置**

### 权重加载验证

- 使用 `burn-store = "0.21"` crate（**不是**已废弃的 burn-import）
- `PytorchStore::from_file(path).with_top_level_key("generator")` 处理 `cp_dict['generator']` 嵌套
- `model.load_from(&mut store)` 返回 `ApplyResult` 含 `applied/missing/unused/errors`
- `Vec<Module>` 的 key 自动映射为 `ups.0.weight`, `resblocks.0.convs1.0.weight` 等
- Conv1d/ConvTranspose1d 权重**不转置**，只对 Linear 转置

### 当前项目结构

```
hifisampler-rs/
├── Cargo.toml                          # workspace, ort=2.0.0-rc.11, edition=2024
├── crates/
│   ├── hifisampler-core/               # 唯一依赖 ort 的 crate
│   │   ├── Cargo.toml                  # ort.workspace, ndarray.workspace
│   │   └── src/
│   │       ├── ep.rs                   # EP dispatch (147行, 需重写)
│   │       ├── vocoder.rs              # Vocoder struct (81行, Session→Burn)
│   │       ├── hnsep.rs                # HnsepModel (需重写)
│   │       ├── config.rs               # PerformanceConfig (device/device_id)
│   │       └── ...
│   ├── hifisampler-server/             # axum HTTP + WebUI
│   │   ├── Cargo.toml                  # windows=0.58 Win32_Graphics_Dxgi
│   │   └── src/main.rs:530-568         # enumerate_dml_adapters() (需替换为 wgpu)
│   └── hifisampler-bridge/             # 最小 CLI, 无 ort 依赖
├── models/                             # .onnx 文件 (gitignored)
└── tests/                              # 6 benchmark fixtures
```

**无 rust-toolchain.toml** — 需创建以锁定 ≥1.92

---

## 迁移阶段

### Phase 0: 前置准备

**0.1 升级 Rust 工具链**
```powershell
rustup update stable
rustc --version  # 验证 ≥1.92
```

**0.2 创建 `rust-toolchain.toml`** 锁定工具链:
```toml
[toolchain]
channel = "stable"
```

**0.3 创建 spike crate** `crates/burn-spike/`:
- `Cargo.toml`: `burn = { version = "0.21", features = ["wgpu", "vulkan", "fusion", "autotune"] }`, `burn-store = "0.21"`, `anyhow = "1"`
- `src/lib.rs`: `#![recursion_limit = "256"]`
- 验证: `cargo build -p burn-spike --release`

### Phase 1: 权重预处理（Python 脚本）

**创建** `scripts/prepare_vocoder_weights.py`:

```python
import torch, json
from util.nsf_hifigan import Generator
from util.utils import AttrDict

config = json.load(open("新建文件夹/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json"))
h = AttrDict(config)
generator = Generator(h)
cp_dict = torch.load("新建文件夹/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt", map_location='cpu')
generator.load_state_dict(cp_dict['generator'])
generator.eval()
generator.remove_weight_norm()  # 融合 weight_norm → 标准 conv 权重
torch.save({"generator": generator.state_dict()}, "crates/burn-spike/models/vocoder_fused.pt")
print("Saved fused weights")
```

**验证**: 加载后检查 state_dict key 名:
```python
for k, v in generator.state_dict().items():
    print(f"{k}: {v.shape}")
# 期望: conv_pre.weight [512,128,7], ups.0.weight [512,256,16], resblocks.0.convs1.0.weight, ...
```

### Phase 2: Vocoder 原生 Burn 实现（核心 Spike）

**目录结构**:
```
crates/burn-spike/
├── Cargo.toml
├── src/
│   ├── lib.rs              # #![recursion_limit = "256"]
│   ├── vocoder.rs          # HifiGanNsf<B> + HifiGanNsfConfig
│   ├── resblock.rs         # ResBlock1<B>
│   ├── fastsinegen.rs      # fastsinegen() 纯 tensor 函数
│   └── main.rs             # spike 测试入口
└── models/
    └── vocoder_fused.pt
```

**2.1 `resblock.rs` — ResBlock1**:
```rust
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, PaddingConfig1d};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::activation::leaky_relu;

#[derive(Module, Debug)]
pub struct ResBlock1<B: Backend> {
    convs1: Vec<Conv1d<B>>,  // 3 个, dilation=[1,3,5]
    convs2: Vec<Conv1d<B>>,  // 3 个, dilation=1
}

impl<B: Backend> ResBlock1<B> {
    pub fn new(channels: usize, kernel_size: usize, dilations: [usize; 3], device: &B::Device) -> Self {
        let convs1 = dilations.iter().map(|&d| {
            Conv1dConfig::new([channels, channels], kernel_size)
                .with_dilation(d)
                .with_padding(PaddingConfig1d::Explicit(
                    (kernel_size * d - d) / 2  // get_padding(k, d)
                ))
                .init(device)
        }).collect();
        let convs2 = (0..3).map(|_| {
            Conv1dConfig::new([channels, channels], kernel_size)
                .with_dilation(1)
                .with_padding(PaddingConfig1d::Explicit((kernel_size - 1) / 2))
                .init(device)
        }).collect();
        Self { convs1, convs2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = leaky_relu(x.clone(), 0.1);
            let xt = c1.forward(xt);
            let xt = leaky_relu(xt, 0.1);
            let xt = c2.forward(xt);
            x = xt + x;  // residual
        }
        x
    }
}
```

**2.2 `fastsinegen.rs`** — 最高风险部分:
```rust
use burn::tensor::{backend::Backend, Tensor};

pub fn fastsinegen<B: Backend>(
    f0: Tensor<B, 2>,      // [B, T]
    upp: usize,            // 64
    source_sr: f64,        // 5512.5
    device: &B::Device,
) -> Tensor<B, 3> {        // [B, 1, T*64]
    let batch = f0.shape()[0];
    let t = f0.shape()[1];

    // n = arange(1, upp+1) as float
    let n = Tensor::<B, 1, burn_tensor::Int>::arange(1..(upp + 1), device)
        .float();
    // n: [64]

    // s0 = f0.unsqueeze(-1) / source_sr  → [B, T, 1]
    let s0 = f0.unsqueeze_dims(&[-1]).div_scalar(source_sr);

    // ds0 = pad(s0[:,1:,:] - s0[:,:-1,:], (0,0,0,1))  → [B, T, 1]
    let diff = s0.clone().slice([0..batch, 1..t, 0..1])
        - s0.clone().slice([0..batch, 0..(t-1), 0..1]);
    let ds0 = diff.pad((0, 0, 0, 1), 0.0);  // (left,right,top,bottom) pad dim 1 right by 1

    // rad = s0 * n + 0.5 * ds0 * n * (n-1) / upp  → [B, T, 64]
    // 广播: [B,T,1] * [64] → [B,T,64]
    let n_minus_1 = n.clone() - 1.0;
    let rad = s0.clone() * n.clone()
        + 0.5 * ds0 * n.clone() * n_minus_1 / (upp as f64);

    // rad2 = fmod(rad[...,-1:] + 0.5, 1.0) - 0.5  → [B, T, 1]
    let rad_last = rad.clone().slice([0..batch, 0..t, (upp-1)..upp]);
    let rad2 = rad_last.add_scalar(0.5).fmod_scalar(1.0).sub_scalar(0.5);

    // rad_acc = fmod(cumsum(rad2, dim=1), 1.0)  → [B, T, 1]
    let rad_acc = rad2.cumsum(1).fmod_scalar(1.0);

    // rad += pad(rad_acc[:,:-1,:], (0,0,1,0))  → [B, T, 64]
    let rad_acc_shifted = rad_acc.clone()
        .slice([0..batch, 0..(t-1), 0..1])
        .pad((0, 0, 1, 0), 0.0);  // pad dim 1 left by 1
    let rad = rad + rad_acc_shifted;

    // rad = rad.reshape(B, 1, -1)  → [B, 1, T*64]
    let rad = rad.reshape([batch, 1, -1]);

    // sines = sin(2 * pi * rad)
    rad.mul_scalar(2.0 * std::f64::consts::PI).sin()
}
```

**2.3 `vocoder.rs` — HifiGanNsf**:
```rust
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, PaddingConfig1d};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::activation::leaky_relu;
use crate::resblock::ResBlock1;
use crate::fastsinegen::fastsinegen;

#[derive(Module, Debug)]
pub struct HifiGanNsf<B: Backend> {
    conv_pre: Conv1d<B>,
    ups: Vec<ConvTranspose1d<B>>,
    resblocks: Vec<ResBlock1<B>>,
    source_conv: Conv1d<B>,
    conv_post: Conv1d<B>,
    upp: usize,            // 64
    source_sr: f64,        // 5512.5
    num_upsamples: usize,  // 5
    num_kernels: usize,    // 3
}

#[derive(Config, Debug)]
pub struct HifiGanNsfConfig {
    num_mels: usize,                  // 128
    upsample_initial_channel: usize,  // 512
    upsample_rates: Vec<usize>,       // [8,8,2,2,2]
    upsample_kernel_sizes: Vec<usize>,// [16,16,4,4,4]
    resblock_kernel_sizes: Vec<usize>,    // [3,7,11]
    resblock_dilation_sizes: Vec<Vec<usize>>, // [[1,3,5],[1,3,5],[1,3,5]]
    sampling_rate: u32,               // 44100
}

impl HifiGanNsfConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HifiGanNsf<B> {
        let mut ch = self.upsample_initial_channel;
        let mut ups = Vec::new();
        let mut resblocks = Vec::new();
        let mut source_conv = None;

        for (i, (u, k)) in self.upsample_rates.iter()
            .zip(self.upsample_kernel_sizes.iter()).enumerate() {
            ch //= 2;
            ups.push(
                ConvTranspose1dConfig::new([ch * 2, ch], *k)
                    .with_stride(*u)
                    .with_padding((k - u) / 2)
                    .init(device)
            );
            for (k_r, d_r) in self.resblock_kernel_sizes.iter()
                .zip(self.resblock_dilation_sizes.iter()) {
                let dilations = [d_r[0], d_r[1], d_r[2]];
                resblocks.push(ResBlock1::new(ch, *k_r, dilations, device));
            }
            if i == 1 {  // mini_nsf: source_conv at i==1
                source_conv = Some(
                    Conv1dConfig::new([1, ch], 1).init(device)
                );
            }
        }

        let conv_pre = Conv1dConfig::new([self.num_mels, self.upsample_initial_channel], 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);
        let conv_post = Conv1dConfig::new([ch, 1], 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);

        let upp = self.upsample_rates[0] * self.upsample_rates[1];  // 64
        let source_sr = self.sampling_rate as f64
            / (self.upsample_rates[2..].iter().product::<usize>() as f64);  // 5512.5

        HifiGanNsf {
            conv_pre,
            ups,
            resblocks,
            source_conv: source_conv.expect("source_conv must be created at i==1"),
            conv_post,
            upp,
            source_sr,
            num_upsamples: self.upsample_rates.len(),
            num_kernels: self.resblock_kernel_sizes.len(),
        }
    }
}

impl<B: Backend> HifiGanNsf<B> {
    /// mel: [B, num_mels, T], f0: [B, T] → waveform [B, 1, T*512]
    pub fn forward(&self, mel: Tensor<B, 3>, f0: Tensor<B, 2>) -> Tensor<B, 3> {
        let device = mel.device();
        let har_source = fastsinegen(f0, self.upp, self.source_sr, &device);
        // har_source: [B, 1, T*64]

        let mut x = self.conv_pre.forward(mel);  // [B, 512, T]

        for i in 0..self.num_upsamples {
            x = leaky_relu(x, 0.1);
            x = self.ups[i].forward(x);

            if i == 1 {
                x = x + self.source_conv.forward(har_source.clone());
            }

            let mut xs: Option<Tensor<B, 3>> = None;
            for j in 0..self.num_kernels {
                let idx = i * self.num_kernels + j;
                let out = self.resblocks[idx].forward(x.clone());
                xs = match xs {
                    None => Some(out),
                    Some(s) => Some(s + out),
                };
            }
            x = xs.unwrap() / 3.0;
        }

        x = leaky_relu(x, 0.1);
        x = self.conv_post.forward(x);  // [B, 1, T*512]
        x.tanh()
    }

    /// 从 PyTorch .pt 文件加载权重
    pub fn load_from_pt(path: &str, device: &B::Device) -> anyhow::Result<Self> {
        let model = HifiGanNsfConfig::new(
            128, 512,
            vec![8,8,2,2,2], vec![16,16,4,4,4],
            vec![3,7,11], vec![vec![1,3,5],vec![1,3,5],vec![1,3,5]],
            44100
        ).init(device);

        let mut store = burn_store::PytorchStore::from_file(path)
            .with_top_level_key("generator")
            .allow_partial(false);

        let result = model.load_from(&mut store)?;
        if !result.missing.is_empty() {
            anyhow::bail!("missing tensors: {:?}", result.missing);
        }
        if !result.errors.is_empty() {
            anyhow::bail!("load errors: {:?}", result.errors);
        }
        eprintln!("loaded {} tensors", result.applied.len());
        Ok(model)
    }
}
```

**2.4 `main.rs` — spike 测试**:
```rust
use burn::backend::Wgpu;
use burn::tensor::Tensor;

type B = Wgpu;

fn main() -> anyhow::Result<()> {
    let device = Default::default();
    let model = vocoder::HifiGanNsf::load_from_pt("models/vocoder_fused.pt", &device)?;

    // 测试输入: mel [1, 128, 100], f0 [1, 100]
    let mel = Tensor::<B, 3>::zeros([1, 128, 100], &device);
    let f0 = Tensor::<B, 2>::full([1, 100], 440.0, &device);

    let output = model.forward(mel, f0);
    println!("output shape: {:?}", output.shape());
    // 期望: [1, 1, 51200]  (100 * 512)

    Ok(())
}
```

### Phase 3: 数值验证

**3.1 生成 ORT golden output** (用现有 ORT 路径):
```powershell
# 设置 ORT_DYLIB_PATH
$env:ORT_DYLIB_PATH = "<path>/onnxruntime.dll"
cargo run --release -p hifisampler-core --bin bench_resampler_fixtures -- ort_baseline config.benchmark.yaml
```
输出: `tests/benchmark_fixtures/outputs/ort_baseline/*.wav`

**3.2 Burn 推理 + 比较** (spike main.rs 中):
- 从 fixture WAV 提取 mel + f0 (复用 hifisampler-core 的 mel.rs)
- Burn 模型推理
- 与 ORT golden WAV 比较: `max_abs_diff`, `rms_diff`, `SNR_dB`

**容差标准** (沿用 `compare_onnx_pytorch.py`):
- `max_abs_diff < 0.01` ✅ 通过
- `max_abs_diff < 0.1` ⚠️ 可接受
- `max_abs_diff >= 0.1` ❌ 失败，需调试

**3.3 性能 + 体积基准**:
- 推理延迟: `model.forward()` 计时 vs ORT `synthesis_ms`
- 二进制体积: burn-spike.exe vs onnxruntime.dll (50-200MB)

### Phase 4: 集成到主项目（spike 成功后）

**4.1 workspace Cargo.toml 修改**:
```toml
[workspace.dependencies]
# 移除:
# ort = {version = "2.0.0-rc.11", features = ["ndarray", "load-dynamic"]}
# ndarray = "0.17"  # 如果其他地方不再需要
# 添加:
burn = { version = "0.21", features = ["wgpu", "vulkan", "fusion", "autotune"] }
burn-store = "0.21"
```

**4.2 hifisampler-core/Cargo.toml**:
```toml
[dependencies]
# 移除: ort.workspace, ndarray.workspace
# 添加:
burn.workspace = true
burn-store.workspace = true
```

**4.3 重写 `ep.rs`** — 从 EP 列表 → 后端选择:
```rust
// 新的 ep.rs: 返回设备选择信息, 不再构建 EP 列表
pub enum BackendDevice {
    Cpu,
    Vulkan(usize),  // WgpuDevice::DiscreteGpu(index)
    Cuda(usize),
}
pub fn select_backend(device: &str, device_id: i32) -> BackendDevice { ... }
```

**4.4 重写 `vocoder.rs`**:
- `session: Session` → `model: HifiGanNsf<B>`
- `synthesize()` 中 `session.run()` → `model.forward()`
- 输入布局: 当前 ONNX 是 `[1, T, mel]`, Burn 模型期望 `[1, mel, T]` — 需调整转置

**4.5 重写 `hnsep.rs`** (Phase 5, 更复杂):
- CascadedNet 实现: BaseNet + ASPP + LSTM + Decoder
- BatchNorm 加载 running stats
- bilinear interpolate (Burn 的 `F.interpolate` 等价物)

**4.6 替换 DXGI 枚举** (`main.rs:530-568`):
```rust
// 替换 enumerate_dml_adapters()
fn enumerate_gpu_adapters() -> Vec<GpuAdapterInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    instance.enumerate_adapters(wgpu::Backends::VULKAN)
        .into_iter().enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            GpuAdapterInfo {
                id: i as i32,
                name: info.name,
                vendor_id: info.vendor,
                device_id: info.device,
                device_type: info.device_type,
                // dedicated_video_memory: wgpu 不提供
            }
        }).collect()
}
```

**4.7 更新 CI 矩阵** (`.github/workflows/ci.yml`):
- 从 6 变体 (CPU/DML/CUDA/TRT/CoreML) → 3 变体 (CPU/Vulkan/CUDA)
- 移除 ORT_VERSION, onnxruntime.dll 下载
- 移除 `ORT_DYLIB_PATH` 环境变量

**4.8 更新 config schema** (`config.rs`):
- `device` 选项: `auto/cpu/cuda/dml/coreml` → `auto/cpu/vulkan/cuda`
- 移除 `rocm` 文档残留
- `device_id` 语义: DXGI 全局索引 → wgpu DiscreteGpu 索引

**4.9 更新 installer** (`installer/setup.iss`):
- 移除 `onnxruntime.dll` 打包
- 添加 Vulkan 驱动需求说明

### Phase 5: HN-SEP (CascadedNet) 实现

**Phase 4 成功后**, 实现 CascadedNet:
- 结构: 3-stage 级联, 5个 BaseNet, 每个 4级 U-Net + ASPP + LSTM
- 复杂度远高于 vocoder: BatchNorm2d, 双向 LSTM, bilinear upsample, ASPP 5分支
- 权重: `新建文件夹/vr/model.pt` (已有)

### Phase 6: 清理

- 移除 `ndarray` 依赖 (如果 mel.rs 等不再需要)
- 更新 README, 文档
- 移除 `scripts/convert_hnsep_to_onnx.py` 等不再需要的脚本
- 更新 `config.default.yaml`

---

## 风险矩阵

| 风险 | 严重度 | 缓解 |
|---|---|---|
| fastsinegen 的 slice/pad 语义错误 | 中 | Phase 2 先单独测试 fastsinegen, 验证输出形状和数值 |
| weight_norm 融合后 key 名不匹配 | 低 | Python 预处理后打印 state_dict key 验证 |
| Burn wgpu vulkan 在 AMD GPU 不稳定 | 中 | 准备禁用 `fusion` feature 的 fallback 配置 |
| ConvTranspose1d 数值与 PyTorch 有差异 | 低 | 已验证公式一致; Phase 3 数值验证会捕获 |
| CascadedNet 的 bilinear interpolate 在 Burn 中不同 | 中 | Phase 5 重点验证; 可能需手动实现 interpolate |
| burn 0.21 API 在编译期有意外问题 | 低 | Phase 0 先编译 burn-spike crate 验证 |

---

## 验证清单

- [ ] Phase 0: `cargo build -p burn-spike --release` 成功
- [ ] Phase 1: `vocoder_fused.pt` 生成, state_dict key 检查通过
- [ ] Phase 2: `cargo run -p burn-spike` 输出 `[1, 1, 51200]`
- [ ] Phase 3: `max_abs_diff < 0.01` vs ORT golden
- [ ] Phase 3: 推理延迟 ≤ ORT 的 2×
- [ ] Phase 3: burn-spike.exe 体积 < onnxruntime.dll
- [ ] Phase 4: `cargo test --release` 通过
- [ ] Phase 4: 6 benchmark fixtures 输出 WAV 与 ORT golden `max_abs_diff < 0.01`
- [ ] Phase 4: WebUI `/capabilities` 正确枚举 GPU
- [ ] Phase 6: `cargo build --release` 整个 workspace 无 ort 依赖
