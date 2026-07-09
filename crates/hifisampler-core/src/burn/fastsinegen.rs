use burn::tensor::{Int, Tensor, backend::Backend};

/// 复刻 `nsf_hifigan.py::Generator.fastsinegen`（mini_nsf=true 路径）。
///
/// - `f0`: `[B, T]` 基频（Hz）
/// - 返回 `[B, 1, T*upp]` 正弦谐波源
///
/// PyTorch 原始逻辑:
/// ```python
/// n = torch.arange(1, upp + 1)
/// s0 = f0.unsqueeze(-1) / source_sr              # [B, T, 1]
/// ds0 = F.pad(s0[:,1:,:] - s0[:,:-1,:], (0,0,0,1))  # [B, T, 1]
/// rad = s0 * n + 0.5 * ds0 * n * (n-1) / upp       # [B, T, 64]
/// rad2 = fmod(rad[...,-1:] + 0.5, 1.0) - 0.5       # [B, T, 1]
/// rad_acc = fmod(cumsum(rad2, dim=1), 1.0)         # [B, T, 1]
/// rad += F.pad(rad_acc[:,:-1,:], (0,0,1,0))        # [B, T, 64]
/// rad = rad.reshape(B, 1, -1)                      # [B, 1, T*64]
/// sines = sin(2 * pi * rad)
/// ```
pub fn fastsinegen<B: Backend>(
    f0: Tensor<B, 2>,
    upp: usize,
    source_sr: f64,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch, t] = f0.dims();

    // n = arange(1, upp+1) as float  -> [1, 1, upp]（reshape 以便与 [B,T,1] 广播）
    let n = Tensor::<B, 1, Int>::arange(1..(upp as i64 + 1), device)
        .float()
        .reshape([1, 1, upp]);

    // s0 = f0.unsqueeze(-1) / source_sr  -> [B, T, 1]
    let s0: Tensor<B, 3> = f0.unsqueeze_dims(&[-1]).div_scalar(source_sr);

    // ds0 = pad(s0[:,1:,:] - s0[:,:-1,:], (0,0,0,1))  -> [B, T, 1]
    let diff = s0.clone().slice([0..batch, 1..t, 0..1]) - s0.clone().slice([0..batch, 0..(t - 1), 0..1]);
    let ds0 = diff.pad([(0, 0), (0, 1), (0, 0)], 0.0);

    // rad = s0 * n + 0.5 * ds0 * n * (n-1) / upp  -> [B, T, upp]
    let n_minus_1 = n.clone() - 1.0;
    let rad = s0.clone() * n.clone() + (ds0 * n.clone() * n_minus_1).mul_scalar(0.5) / (upp as f64);

    // rad2 = fmod(rad[...,-1:] + 0.5, 1.0) - 0.5  -> [B, T, 1]
    let rad_last = rad.clone().slice([0..batch, 0..t, (upp - 1)..upp]);
    let rad2 = rad_last.add_scalar(0.5).fmod_scalar(1.0).sub_scalar(0.5);

    // rad_acc = fmod(cumsum(rad2, dim=1), 1.0)  -> [B, T, 1]
    let rad_acc = rad2.cumsum(1).fmod_scalar(1.0);

    // rad += pad(rad_acc[:,:-1,:], (0,0,1,0))  -> [B, T, upp]
    let rad_acc_shifted = rad_acc
        .clone()
        .slice([0..batch, 0..(t - 1), 0..1])
        .pad([(0, 0), (1, 0), (0, 0)], 0.0);
    let rad = rad + rad_acc_shifted;

    // rad = rad.reshape(B, 1, -1)  -> [B, 1, T*upp]
    let rad = rad.reshape([batch, 1, t * upp]);

    // sines = sin(2 * pi * rad)
    rad.mul_scalar(2.0 * std::f64::consts::PI).sin()
}
