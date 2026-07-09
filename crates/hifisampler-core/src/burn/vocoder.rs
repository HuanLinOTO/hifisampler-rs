use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::leaky_relu;
use burn::tensor::{Tensor, backend::Backend};
use burn_store::{BurnpackStore, ModuleSnapshot, ModuleStore, PytorchStore, TensorSnapshot};

use super::fastsinegen::fastsinegen;
use super::resblock::ResBlock1;

/// HiFi-GAN NSF Generator（mini_nsf=true 变体）。
#[derive(Module, Debug)]
pub struct HifiGanNsf<B: Backend> {
    conv_pre: Conv1d<B>,
    ups: Vec<ConvTranspose1d<B>>,
    resblocks: Vec<ResBlock1<B>>,
    source_conv: Conv1d<B>,
    conv_post: Conv1d<B>,
    upp: usize,
    source_sr: f64,
    num_upsamples: usize,
    num_kernels: usize,
}

#[derive(Config, Debug)]
pub struct HifiGanNsfConfig {
    pub num_mels: usize,
    pub upsample_initial_channel: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub sampling_rate: u32,
}

impl HifiGanNsfConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HifiGanNsf<B> {
        let mut ch = self.upsample_initial_channel;
        let mut ups = Vec::new();
        let mut resblocks = Vec::new();
        let mut source_conv = None;

        for (i, (u, k)) in self
            .upsample_rates
            .iter()
            .zip(self.upsample_kernel_sizes.iter())
            .enumerate()
        {
            ch /= 2;
            ups.push(
                ConvTranspose1dConfig::new([ch * 2, ch], *k)
                    .with_stride(*u)
                    .with_padding((k - u) / 2)
                    .init(device),
            );
            for (k_r, d_r) in self
                .resblock_kernel_sizes
                .iter()
                .zip(self.resblock_dilation_sizes.iter())
            {
                let dilations = [d_r[0], d_r[1], d_r[2]];
                resblocks.push(ResBlock1::new(ch, *k_r, dilations, device));
            }
            if i == 1 {
                // mini_nsf: source_conv at i==1（对应 PyTorch 源码中 `elif i == 1`）
                source_conv = Some(Conv1dConfig::new(1, ch, 1).init(device));
            }
        }

        let conv_pre = Conv1dConfig::new(self.num_mels, self.upsample_initial_channel, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .init(device);
        let conv_post = Conv1dConfig::new(ch, 1, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .init(device);

        let upp = self.upsample_rates[0] * self.upsample_rates[1];
        let source_sr =
            self.sampling_rate as f64 / (self.upsample_rates[2..].iter().product::<usize>() as f64);

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
    /// `mel`: `[B, num_mels, T]`, `f0`: `[B, T]` -> 波形 `[B, 1, T*512]`
    pub fn forward(&self, mel: Tensor<B, 3>, f0: Tensor<B, 2>) -> Tensor<B, 3> {
        let device = mel.device();
        let har_source = fastsinegen(f0, self.upp, self.source_sr, &device);

        let mut x = self.conv_pre.forward(mel);

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
        x = self.conv_post.forward(x);
        x.tanh()
    }

    /// 从 PyTorch .pt 文件加载权重。
    pub fn load_from_pt(
        path: impl Into<std::path::PathBuf>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let path = path.into();
        let mut model = HifiGanNsfConfig::new(
            128,
            512,
            vec![8, 8, 2, 2, 2],
            vec![16, 16, 4, 4, 4],
            vec![3, 7, 11],
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            44100,
        )
        .init(device);

        let mut store = PytorchStore::from_file(path)
            .with_top_level_key("generator")
            .allow_partial(false);
        let result = model.load_from(&mut store)?;
        if !result.missing.is_empty() {
            anyhow::bail!("missing tensors: {:?}", result.missing);
        }
        if !result.errors.is_empty() {
            anyhow::bail!("load errors: {:?}", result.errors);
        }
        tracing::info!("loaded {} vocoder tensors", result.applied.len());
        Ok(model)
    }

    /// 从 Burnpack (.bpk) 文件加载权重。
    ///
    /// 与 `load_from_pt` 相比：无需 PyTorchToBurnAdapter，内存映射 + lazy load。
    /// 当后端 float elem 与 .bpk 存储 dtype 不一致时（如 f16 backend 加载 f32 权重），
    /// 手动转换 dtype（burn-store 的 load_from 不自动转换）。
    pub fn load_from_bpk(
        path: impl Into<std::path::PathBuf>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        use burn::tensor::Element;
        let path = path.into();
        let mut model = HifiGanNsfConfig::new(
            128,
            512,
            vec![8, 8, 2, 2, 2],
            vec![16, 16, 4, 4, 4],
            vec![3, 7, 11],
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            44100,
        )
        .init(device);

        let mut store = BurnpackStore::from_file(path);

        let target_dtype = <B::FloatElem as Element>::dtype();
        let snapshots: Vec<TensorSnapshot> = match store.get_all_snapshots() {
            Ok(s) => s.values().cloned().collect(),
            Err(e) => anyhow::bail!("vocoder bpk get_all_snapshots: {:?}", e),
        };
        let materialized: Vec<(Vec<String>, burn::tensor::TensorData)> = snapshots
            .iter()
            .map(|s| {
                let p = s.path_stack.clone().unwrap_or_default();
                let data = s.to_data().map_err(|e| {
                    anyhow::anyhow!("failed to materialize tensor {}: {:?}", p.join("."), e)
                })?;
                let data = if data.dtype != target_dtype {
                    data.convert_dtype(target_dtype)
                } else {
                    data
                };
                Ok((p, data))
            })
            .collect::<anyhow::Result<_>>()?;
        let views: Vec<TensorSnapshot> = materialized
            .into_iter()
            .map(|(p, data)| {
                TensorSnapshot::from_data(data, p, Vec::new(), burn::module::ParamId::new())
            })
            .collect();
        let result = model.apply(views, None, None, true);
        if !result.missing.is_empty() {
            anyhow::bail!("missing tensors: {:?}", result.missing);
        }
        if !result.errors.is_empty() {
            anyhow::bail!("load errors: {:?}", result.errors);
        }
        tracing::info!("loaded {} vocoder tensors (bpk)", result.applied.len());
        Ok(model)
    }

    /// 保存为 Burnpack (.bpk) 格式。
    pub fn save_to_bpk(&self, path: impl Into<std::path::PathBuf>) -> anyhow::Result<()> {
        let path = path.into();
        let mut store = BurnpackStore::from_file(&path).overwrite(true);
        self.save_into(&mut store)?;
        tracing::info!(
            "saved vocoder tensors to {} (bpk)",
            path.display()
        );
        Ok(())
    }
}
