use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

use super::conv2dbnactiv::Conv2dBnActiv;

/// 复刻 `layers.py::Decoder`：bilinear upsample (×2) + crop_center(skip) + concat + Conv2dBNActiv (ReLU).
///
/// `fixed_length=true` 时：`F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)`
/// `fixed_length=false` 时有 pad+interpolate+slice 的奇数尺寸路径，但 CascadedNet 默认 fixed_length=true。
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub conv1: Conv2dBnActiv<B>,
    pub fixed_length: bool,
}

impl<B: Backend> Decoder<B> {
    /// `nin, nout, ksize, stride=1, pad`.
    pub fn new(
        nin: usize,
        nout: usize,
        ksize: usize,
        _stride: usize,
        pad: usize,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv2dBnActiv::new(nin, nout, ksize, 1, [pad, pad], [1, 1], true, device);
        Self {
            conv1,
            fixed_length: true,
        }
    }

    /// `x`: `[B, C, H, W]`, `skip`: 可选 `[B, C2, H, W]`（与 upsampled x 对齐）.
    pub fn forward(&self, x: Tensor<B, 4>, skip: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        let [_batch, _c, h, w] = x.dims();

        let x = if self.fixed_length {
            // F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            burn::tensor::module::interpolate(
                x,
                [h * 2, w * 2],
                InterpolateOptions::new(InterpolateMode::Bilinear)
                    .with_align_corners(true),
            )
        } else {
            // fixed_length=false 路径（CascadedNet 不用，简化处理）
            burn::tensor::module::interpolate(
                x,
                [h * 2 + 1, w * 2 + 1],
                InterpolateOptions::new(InterpolateMode::Bilinear)
                    .with_align_corners(true),
            )
        };

        let x = match skip {
            Some(s) => {
                let skip = crop_center(s, &x);
                Tensor::cat(vec![x, skip], 1)
            }
            None => x,
        };

        self.conv1.forward(x)
    }
}

/// 复刻 `layers.py::crop_center`：把 h1 裁剪到与 h2 相同的时间维度（dim=3）.
/// h1 和 h2 的频维（dim=2）应相同；只裁时间维。
fn crop_center<B: Backend>(h1: Tensor<B, 4>, h2: &Tensor<B, 4>) -> Tensor<B, 4> {
    let h1_dims = h1.dims();
    let h2_dims = h2.dims();
    if h1_dims[3] == h2_dims[3] {
        return h1;
    }
    debug_assert!(h1_dims[3] > h2_dims[3], "h1 width must be > h2 width");
    let s_time = (h1_dims[3] - h2_dims[3]) / 2;
    let e_time = s_time + h2_dims[3];
    h1.slice([0..h1_dims[0], 0..h1_dims[1], 0..h1_dims[2], s_time..e_time])
}
