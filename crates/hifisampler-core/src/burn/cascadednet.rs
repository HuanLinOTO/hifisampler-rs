use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::{Tensor, backend::Backend};
use burn_store::{BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, TensorSnapshot};

use super::basetnet::{BaseNet, BaseNetConfig};

/// stg1_low_band_net = Sequential(BaseNet, Conv2dBnActiv)
/// 用单独结构而非 Vec，因为 burn-store 需要固定字段名。
#[derive(Module, Debug)]
pub struct StgLowBandNet<B: Backend> {
    /// 对应 PyTorch Sequential.0 (BaseNet)
    pub inner0: BaseNet<B>,
    /// 对应 PyTorch Sequential.1 (Conv2dBnActiv)
    pub inner1: super::conv2dbnactiv::Conv2dBnActiv<B>,
}

impl<B: Backend> StgLowBandNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.inner0.forward(x);
        self.inner1.forward(h)
    }
}

/// 复刻 `nets.py::CascadedNet`（is_complex=false, is_mono=false, nin=2 路径）.
#[derive(Module, Debug)]
pub struct CascadedNet<B: Backend> {
    pub stg1_low_band_net: StgLowBandNet<B>,
    pub stg1_high_band_net: BaseNet<B>,
    pub stg2_low_band_net: StgLowBandNet<B>,
    pub stg2_high_band_net: BaseNet<B>,
    pub stg3_full_band_net: BaseNet<B>,
    /// out = Conv2d(nout, nin, 1, bias=False)
    pub out: Conv2d<B>,
    /// aux_out = Conv2d(3*nout//4, nin, 1, bias=False)
    pub aux_out: Conv2d<B>,
    pub max_bin: usize,
    pub output_bin: usize,
}

#[derive(Config, Debug)]
pub struct CascadedNetConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub nout: usize,
    pub nout_lstm: usize,
    // nin = 2 (is_complex=false, is_mono=false)
    pub nin: usize,
}

impl CascadedNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CascadedNet<B> {
        let n = self.nout;
        let nin = self.nin;
        let max_bin = self.n_fft / 2;
        let output_bin = self.n_fft / 2 + 1;
        let nin_lstm_low = max_bin / 2 / 2; // self.nin_lstm // 2, where nin_lstm = max_bin//2

        // stg1_low: BaseNet(nin, nout//2, nin_lstm//2, nout_lstm) + Conv2dBnActiv(nout//2, nout//4, 1)
        let stg1_low_band_net = StgLowBandNet {
            inner0: BaseNetConfig::new(nin, n / 2, nin_lstm_low, self.nout_lstm).init(device),
            inner1: super::conv2dbnactiv::Conv2dBnActiv::new(n / 2, n / 4, 1, 1, [0, 0], [1, 1], true, device),
        };
        // stg1_high: BaseNet(nin, nout//4, nin_lstm//2, nout_lstm//2)
        let stg1_high_band_net = BaseNetConfig::new(nin, n / 4, nin_lstm_low, self.nout_lstm / 2).init(device);

        // stg2_low: BaseNet(nout//4 + nin, nout, nin_lstm//2, nout_lstm) + Conv2dBnActiv(nout, nout//2, 1)
        let stg2_low_band_net = StgLowBandNet {
            inner0: BaseNetConfig::new(n / 4 + nin, n, nin_lstm_low, self.nout_lstm).init(device),
            inner1: super::conv2dbnactiv::Conv2dBnActiv::new(n, n / 2, 1, 1, [0, 0], [1, 1], true, device),
        };
        // stg2_high: BaseNet(nout//4 + nin, nout//2, nin_lstm//2, nout_lstm//2)
        let stg2_high_band_net = BaseNetConfig::new(n / 4 + nin, n / 2, nin_lstm_low, self.nout_lstm / 2).init(device);

        // stg3_full: BaseNet(3*nout//4 + nin, nout, nin_lstm, nout_lstm)
        let stg3_full_band_net = BaseNetConfig::new(3 * n / 4 + nin, n, max_bin / 2, self.nout_lstm).init(device);

        let out = Conv2dConfig::new([n, nin], [1, 1]).with_bias(false).init(device);
        let aux_out = Conv2dConfig::new([3 * n / 4, nin], [1, 1]).with_bias(false).init(device);

        CascadedNet {
            stg1_low_band_net,
            stg1_high_band_net,
            stg2_low_band_net,
            stg2_high_band_net,
            stg3_full_band_net,
            out,
            aux_out,
            max_bin,
            output_bin,
        }
    }
}

impl<B: Backend> CascadedNet<B> {
    /// 输入 x: [B, nin, max_bin, T] (实数路径，is_complex=false)
    /// 输出 mask: [B, nin, output_bin, T]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, _nin, _max_bin, t] = x.dims();
        let max_bin = self.max_bin;

        // x = x[:, :, :self.max_bin]  (已由调用方保证)
        let bandw = max_bin / 2;
        let l1_in = x.clone().slice([0..batch, 0..2, 0..bandw, 0..t]);
        let h1_in = x.clone().slice([0..batch, 0..2, bandw..max_bin, 0..t]);

        let l1 = self.stg1_low_band_net.forward(l1_in.clone());
        let h1 = self.stg1_high_band_net.forward(h1_in.clone());
        // aux1 = torch.cat([l1, h1], dim=2)  — 沿频维拼接
        let aux1 = Tensor::cat(vec![l1.clone(), h1.clone()], 2);
        self.dump("aux1", &aux1);

        // l2_in = torch.cat([l1_in, l1], dim=1)  — 沿通道维
        let l2_in = Tensor::cat(vec![l1_in, l1.clone()], 1);
        let h2_in = Tensor::cat(vec![h1_in, h1.clone()], 1);
        let l2 = self.stg2_low_band_net.forward(l2_in);
        let h2 = self.stg2_high_band_net.forward(h2_in);
        let aux2 = Tensor::cat(vec![l2.clone(), h2.clone()], 2);
        self.dump("aux2", &aux2);

        // f3_in = torch.cat([x, aux1, aux2], dim=1)
        let f3_in = Tensor::cat(vec![x.clone(), aux1, aux2], 1);
        self.dump("f3_in", &f3_in);
        let f3 = self.stg3_full_band_net.forward(f3_in);
        self.dump("f3", &f3);

        // ONNX-compatible path: bounded_mask (not sigmoid!)
        // mask = self.out(f3)  → [B, 2, H, W]
        // real = mask[:, :1], imag = mask[:, 1:]
        // mag = sqrt(real^2 + imag^2 + eps)
        // tanh_mag = tanh(mag)
        // real_norm = tanh_mag * real / (mag + eps)
        // imag_norm = tanh_mag * imag / (mag + eps)
        // mask = cat([real_norm, imag_norm], dim=1)
        let mask_raw = self.out.forward(f3);
        let [b, _c, mh, mw] = mask_raw.dims();
        let real_part = mask_raw.clone().slice([0..b, 0..1, 0..mh, 0..mw]);
        let imag_part = mask_raw.clone().slice([0..b, 1..2, 0..mh, 0..mw]);
        // bounded_mask: mag = sqrt(real^2 + imag^2 + eps), mask = tanh(mag) * mask / (mag + eps)
        // eps must be representable in f16 (1e-8 underflows to 0 → division by zero).
        // 1e-4 is the smallest practical eps for f16; numerical impact is negligible
        // (only affects bins where real≈imag≈0, where output is ~0 regardless).
        let eps = 1e-4;
        let real_sq = real_part.clone().powi_scalar(2);
        let imag_sq = imag_part.clone().powi_scalar(2);
        let mag = (real_sq + imag_sq).add_scalar(eps).sqrt();
        let tanh_mag = mag.clone().tanh();
        let mag_eps = mag.clone().add_scalar(eps);
        let real_norm = tanh_mag.clone() * real_part.clone() / mag_eps.clone();
        let imag_norm = tanh_mag * imag_part.clone() / mag_eps;
        let mask = Tensor::cat(vec![real_norm, imag_norm], 1);
        self.dump("mask", &mask);

        // F.pad(mask, (0,0, 0, output_bin - mask.size(2)), mode='replicate')
        let mask_h = mask.dims()[2];
        let pad_h = self.output_bin.saturating_sub(mask_h);
        if pad_h > 0 {
            mask.pad([(0, 0), (0, 0), (0, pad_h), (0, 0)], burn::tensor::ops::PadMode::Edge)
        } else {
            mask
        }
    }

    fn dump(&self, name: &str, t: &Tensor<B, 4>) {
        static DUMP_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let dir = DUMP_DIR.get_or_init(|| std::env::var("BURN_DUMP_LAYERS").ok());
        if let Some(dir) = dir {
            use burn::tensor::ElementConversion;
            let path = std::path::Path::new(dir).join(format!("{name}.raw"));
            let data = t.to_data();
            let slice: Vec<f32> = data
                .as_slice::<<B as burn::tensor::backend::BackendTypes>::FloatElem>()
                .unwrap_or(&[])
                .iter()
                .map(|&v| v.elem::<f32>())
                .collect();
            let dims: Vec<usize> = t.dims().to_vec();
            use std::io::Write;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
            for &d in &dims {
                f.write_all(&(d as u32).to_le_bytes()).unwrap();
            }
            let mut buf = Vec::with_capacity(slice.len() * 4);
            for &v in &slice {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            f.write_all(&buf).unwrap();
        }
    }

    /// 从 PyTorch .pt 文件加载权重（key="model"，自动 remap）。
    ///
    /// CascadedNet 模块深嵌套（5 BaseNet × 11 子模块，全部固定结构体字段）。
    /// burn-store 的 `apply()` 会递归遍历每个字段，在 Windows 默认 1MB 栈上
    /// 会溢出。此外，捕获 175KB 模型的 `move` 闭包会使栈用量翻倍。
    ///
    /// 解决方案：Box 模型（只捕获 8 字节），在主线程物化 TensorData（Send），
    /// 然后在 16MB 栈的工作线程上运行 `apply()`。
    pub fn load_from_pt(
        path: impl Into<std::path::PathBuf>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        use burn_store::ModuleStore;
        let path = path.into();
        let mut model = Box::new(
            CascadedNetConfig::new(2048, 512, 32, 128, 2).init(device),
        );

        let mut store = PytorchStore::from_file(&path)
            .with_top_level_key("model")
            .allow_partial(false);

        // PytorchStore/TensorSnapshot 是 !Send (内部 Rc<dyn Fn>)，
        // 在主线程物化数据，跨线程只发送 Send 的 (Vec<String>, TensorData)。
        let snapshots: Vec<TensorSnapshot> =
            store.get_all_snapshots()?.values().cloned().collect();
        let materialized: Vec<(Vec<String>, burn::tensor::TensorData)> = snapshots
            .iter()
            .map(|s| {
                let p = s.path_stack.clone().unwrap_or_default();
                let data = s.to_data().map_err(|e| {
                    anyhow::anyhow!("failed to materialize tensor {}: {:?}", p.join("."), e)
                })?;
                Ok((p, data))
            })
            .collect::<anyhow::Result<_>>()?;

        let result = std::thread::Builder::new()
            .name("cascadednet-pt-loader".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || -> (Box<CascadedNet<B>>, burn_store::ApplyResult) {
                let views: Vec<TensorSnapshot> = materialized
                    .into_iter()
                    .map(|(p, data)| {
                        TensorSnapshot::from_data(
                            data,
                            p,
                            Vec::new(),
                            burn::module::ParamId::new(),
                        )
                    })
                    .collect();
                let res = model.apply(views, None, Some(Box::new(PyTorchToBurnAdapter)), true);
                (model, res)
            })
            .expect("failed to spawn cascadednet pt load thread")
            .join()
            .expect("cascadednet pt load thread panicked");
        let (model, result) = result;

        if !result.missing.is_empty() {
            anyhow::bail!(
                "missing {} tensors (first: {:?})",
                result.missing.len(),
                result.missing.first()
            );
        }
        if !result.errors.is_empty() {
            anyhow::bail!("load errors: {:?}", result.errors);
        }
        tracing::info!("loaded {} cascadednet tensors (pt)", result.applied.len());
        Ok(*model)
    }

    /// 从 Burnpack (.bpk) 文件加载权重。
    ///
    /// BurnpackStore 是 Send，但 CascadedNet 模块结构深嵌套，`apply()` 仍会
    /// 递归遍历字段，可能在主线程默认 1MB 栈上溢出。所以在 16MB 栈工作线程
    /// 上运行 `apply()`。
    ///
    /// 当后端 float elem 与 .bpk 存储 dtype 不一致时（如 f16 backend 加载 f32 权重），
    /// `load_from()` 会因 DTypeMismatch 失败。这里手动获取 snapshots → 转换 dtype → apply。
    pub fn load_from_bpk(
        path: impl Into<std::path::PathBuf>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        use burn_store::ModuleStore;
        let path = path.into();
        let mut model = Box::new(
            CascadedNetConfig::new(2048, 512, 32, 128, 2).init(device),
        );

        let result = std::thread::Builder::new()
            .name("cascadednet-bpk-loader".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || -> (Box<CascadedNet<B>>, anyhow::Result<burn_store::ApplyResult>) {
                let mut store = BurnpackStore::from_file(&path);

                use burn::tensor::Element;
                let target_dtype = <B::FloatElem as Element>::dtype();

                let snapshots: Vec<TensorSnapshot> = match store.get_all_snapshots() {
                    Ok(s) => s.values().cloned().collect(),
                    Err(e) => return (model, Err(anyhow::anyhow!("bpk get_all_snapshots: {:?}", e))),
                };

                let materialized: anyhow::Result<Vec<(Vec<String>, burn::tensor::TensorData)>> =
                    snapshots
                        .iter()
                        .map(|s| {
                            let p = s.path_stack.clone().unwrap_or_default();
                            let data = s.to_data().map_err(|e| {
                                anyhow::anyhow!(
                                    "failed to materialize tensor {}: {:?}",
                                    p.join("."),
                                    e
                                )
                            })?;
                            let data = if data.dtype != target_dtype {
                                data.convert_dtype(target_dtype)
                            } else {
                                data
                            };
                            Ok((p, data))
                        })
                        .collect();

                let materialized = match materialized {
                    Ok(m) => m,
                    Err(e) => return (model, Err(e)),
                };

                // BN folding: for each "*.conv.conv0.weight", find matching
                // "*.conv.conv1.{gamma,beta,running_mean,running_var}" and fold:
                //   w' = w * (gamma / sqrt(var + eps))
                //   beta' = beta - gamma * mean / sqrt(var + eps)
                //   gamma' = 1, mean' = 0, var' = 1
                // After folding, BN forward becomes approximately identity + bias,
                // and conv weight already includes the scale. The BN kernel still
                // runs but with identity params (minimal compute).
                let materialized = fold_bn_snapshots(materialized);

                let views: Vec<TensorSnapshot> = materialized
                    .into_iter()
                    .map(|(p, data)| {
                        TensorSnapshot::from_data(
                            data,
                            p,
                            Vec::new(),
                            burn::module::ParamId::new(),
                        )
                    })
                    .collect();

                let res = model.apply(views, None, None, true);
                (model, Ok(res))
            })
            .expect("failed to spawn cascadednet bpk load thread")
            .join()
            .expect("cascadednet bpk load thread panicked");
        let (model, res) = result;
        let result = res?;
        if !result.missing.is_empty() {
            anyhow::bail!(
                "missing {} tensors (first: {:?})",
                result.missing.len(),
                result.missing.first()
            );
        }
        if !result.errors.is_empty() {
            anyhow::bail!("load errors: {:?}", result.errors);
        }
        tracing::info!("loaded {} cascadednet tensors (bpk)", result.applied.len());
        Ok(*model)
    }

    /// 保存为 Burnpack (.bpk) 格式。
    pub fn save_to_bpk(&self, path: impl Into<std::path::PathBuf>) -> anyhow::Result<()> {
        let path = path.into();
        let mut store = BurnpackStore::from_file(&path).overwrite(true);
        self.save_into(&mut store)?;
        tracing::info!(
            "saved cascadednet tensors to {} (bpk)",
            path.display()
        );
        Ok(())
    }
}

/// Fold BatchNorm params into Conv2d weight at the snapshot level.
///
/// For each key ending in ".conv.conv0.weight", finds the corresponding
/// ".conv.conv1.{gamma,beta,running_mean,running_var}" and folds:
///   w' = w * scale, where scale = gamma / sqrt(var + eps)
///   beta' = beta - mean * scale
/// Then sets gamma=1, running_mean=0, running_var=1 so BN forward ≈ identity + bias.
///
/// This doesn't eliminate the BN kernel (forward still runs it), but folds
/// the scale into conv weight so BN does minimal work with identity params.
/// Full elimination requires struct changes that break burn-store compat.
fn fold_bn_snapshots(
    mut snaps: Vec<(Vec<String>, burn::tensor::TensorData)>,
) -> Vec<(Vec<String>, burn::tensor::TensorData)> {
    use burn::tensor::ElementConversion;
    use burn::tensor::DType;
    use std::collections::HashMap;

    // Build a lookup: key string → index in snaps
    let mut key_map: HashMap<String, usize> = HashMap::new();
    for (i, (path, _)) in snaps.iter().enumerate() {
        key_map.insert(path.join("."), i);
    }

    let eps: f32 = 1e-5;
    let mut folded_count = 0;

    // Find all conv weight keys
    let conv_weight_indices: Vec<usize> = snaps
        .iter()
        .enumerate()
        .filter(|(_, (path, _))| path.join(".").ends_with(".conv.conv0.weight"))
        .map(|(i, _)| i)
        .collect();

    for w_idx in conv_weight_indices {
        let prefix: String = {
            let path = &snaps[w_idx].0;
            let key = path.join(".");
            let key = key.strip_suffix("conv0.weight").unwrap_or(&key);
            format!("{key}conv1")
        };

        let gamma_idx = key_map.get(&format!("{prefix}.gamma"));
        let beta_idx = key_map.get(&format!("{prefix}.beta"));
        let mean_idx = key_map.get(&format!("{prefix}.running_mean"));
        let var_idx = key_map.get(&format!("{prefix}.running_var"));

        let (Some(&gi), Some(&bi), Some(&mi), Some(&vi)) = (gamma_idx, beta_idx, mean_idx, var_idx)
        else {
            continue;
        };

        let gamma = tensor_data_to_f32_vec(&snaps[gi].1);
        let beta = tensor_data_to_f32_vec(&snaps[bi].1);
        let mean = tensor_data_to_f32_vec(&snaps[mi].1);
        let var = tensor_data_to_f32_vec(&snaps[vi].1);

        let out_ch = gamma.len();
        if beta.len() != out_ch || mean.len() != out_ch || var.len() != out_ch {
            continue;
        }

        let scale: Vec<f32> = (0..out_ch)
            .map(|i| gamma[i] / (var[i] + eps).sqrt())
            .collect();
        let folded_bias: Vec<f32> = (0..out_ch)
            .map(|i| beta[i] - mean[i] * scale[i])
            .collect();

        let weight_data = &snaps[w_idx].1;
        let weight_dims: Vec<usize> = weight_data.shape.as_slice().to_vec();
        let weight_f32 = tensor_data_to_f32_vec(weight_data);
        let total_elems = weight_f32.len();
        let elems_per_ch = total_elems / out_ch;
        let folded_weight: Vec<f32> = (0..total_elems)
            .map(|i| {
                let ch = i / elems_per_ch;
                weight_f32[i] * scale[ch]
            })
            .collect();

        let w_dtype = snaps[w_idx].1.dtype;
        snaps[w_idx].1 = f32_vec_to_tensor_data(&folded_weight, &weight_dims, w_dtype);

        let bn_dtype = snaps[gi].1.dtype;
        let ones: Vec<f32> = vec![1.0; out_ch];
        let zeros: Vec<f32> = vec![0.0; out_ch];
        snaps[gi].1 = f32_vec_to_tensor_data(&ones, &[out_ch], bn_dtype);
        snaps[bi].1 = f32_vec_to_tensor_data(&folded_bias, &[out_ch], bn_dtype);
        snaps[mi].1 = f32_vec_to_tensor_data(&zeros, &[out_ch], bn_dtype);
        snaps[vi].1 = f32_vec_to_tensor_data(&ones, &[out_ch], bn_dtype);

        folded_count += 1;
    }

    tracing::info!("folded {} BN layers into conv weights", folded_count);
    snaps
}

fn tensor_data_to_f32_vec(data: &burn::tensor::TensorData) -> Vec<f32> {
    use burn::tensor::ElementConversion;
    use burn::tensor::DType;
    match data.dtype {
        DType::F32 | DType::Flex32 => data
            .as_slice::<f32>()
            .unwrap_or(&[])
            .iter()
            .map(|&v| v.elem())
            .collect(),
        DType::F16 => data
            .as_slice::<burn::tensor::f16>()
            .unwrap_or(&[])
            .iter()
            .map(|&v| v.elem())
            .collect(),
        DType::BF16 => data
            .as_slice::<burn::tensor::bf16>()
            .unwrap_or(&[])
            .iter()
            .map(|&v| v.elem())
            .collect(),
        DType::F64 => data
            .as_slice::<f64>()
            .unwrap_or(&[])
            .iter()
            .map(|&v| v.elem())
            .collect(),
        _ => Vec::new(),
    }
}

fn f32_vec_to_tensor_data(
    vals: &[f32],
    dims: &[usize],
    dtype: burn::tensor::DType,
) -> burn::tensor::TensorData {
    use burn::tensor::ElementConversion;
    use burn::tensor::DType;
    match dtype {
        DType::F16 => {
            let converted: Vec<burn::tensor::f16> = vals.iter().map(|&v| v.elem()).collect();
            burn::tensor::TensorData::new(converted, dims.to_vec())
        }
        DType::BF16 => {
            let converted: Vec<burn::tensor::bf16> = vals.iter().map(|&v| v.elem()).collect();
            burn::tensor::TensorData::new(converted, dims.to_vec())
        }
        DType::F64 => {
            let converted: Vec<f64> = vals.iter().map(|&v| v.elem()).collect();
            burn::tensor::TensorData::new(converted, dims.to_vec())
        }
        _ => burn::tensor::TensorData::new(vals.to_vec(), dims.to_vec()),
    }
}
