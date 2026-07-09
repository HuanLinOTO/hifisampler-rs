use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::tensor::{Tensor, backend::Backend};
use burn::tensor::activation::{leaky_relu, relu};

/// 复刻 `layers.py::Conv2DBNActiv`：Conv2d (no bias) + BatchNorm2d + activation.
///
/// `activ` 为 true 时用 ReLU（默认），false 时用 LeakyReLU（Encoder 用）。
#[derive(Module, Debug)]
pub struct Conv2dBnActiv<B: Backend> {
    pub conv: Conv2dBnActivInner<B>,
    pub use_relu: bool,
}

/// 内部结构：conv (no bias) + batchnorm，匹配 PyTorch 的 `nn.Sequential(Conv2d, BatchNorm2d, Activ)`.
/// burn-store 会把 `conv.0.weight` 映射到 Conv2d，`conv.1.{weight,bias,running_mean,running_var}` 映射到 BatchNorm。
#[derive(Module, Debug)]
pub struct Conv2dBnActivInner<B: Backend> {
    pub conv0: Conv2d<B>,
    pub conv1: BatchNorm<B>,
}

impl<B: Backend> Conv2dBnActiv<B> {
    /// `nin, nout, ksize, stride, pad=[ph,pw], dilation=[dh,dw]`.
    pub fn new(
        nin: usize,
        nout: usize,
        ksize: usize,
        stride: usize,
        pad: [usize; 2],
        dilation: [usize; 2],
        use_relu: bool,
        device: &B::Device,
    ) -> Self {
        let conv0 = Conv2dConfig::new([nin, nout], [ksize, ksize])
            .with_stride([stride, stride])
            .with_dilation(dilation)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(
                pad[0], pad[1], pad[0], pad[1],  // (top, left, bottom, right)
            ))
            .with_bias(false)
            .init(device);
        let conv1 = BatchNormConfig::new(nout).init(device);
        Self {
            conv: Conv2dBnActivInner { conv0, conv1 },
            use_relu,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv.conv0.forward(x);
        let h = self.conv.conv1.forward(h);
        if self.use_relu {
            relu(h)
        } else {
            leaky_relu(h, 0.01)  // nn.LeakyReLU default negative_slope=0.01
        }
    }
}
