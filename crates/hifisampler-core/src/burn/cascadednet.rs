use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::{Tensor, backend::Backend};

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

        let l1 = {
            unsafe { std::env::set_var("BURN_DUMP_PREFIX", "stg1l_"); }
            let r = self.stg1_low_band_net.forward(l1_in.clone());
            unsafe { std::env::remove_var("BURN_DUMP_PREFIX"); }
            r
        };
        let h1 = {
            unsafe { std::env::set_var("BURN_DUMP_PREFIX", "stg1h_"); }
            let r = self.stg1_high_band_net.forward(h1_in.clone());
            unsafe { std::env::remove_var("BURN_DUMP_PREFIX"); }
            r
        };
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
        let eps = 1e-8;
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
        if let Ok(dir) = std::env::var("BURN_DUMP_LAYERS") {
            use burn::tensor::ElementConversion;
            let path = std::path::Path::new(&dir).join(format!("{name}.raw"));
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
}
