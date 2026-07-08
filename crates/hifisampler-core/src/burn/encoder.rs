use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use super::conv2dbnactiv::Conv2dBnActiv;

/// 复刻 `layers.py::Encoder`：两个 Conv2DBNActiv（LeakyReLU），第二个 stride=1.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv1: Conv2dBnActiv<B>,
    pub conv2: Conv2dBnActiv<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(
        nin: usize,
        nout: usize,
        ksize: usize,
        stride: usize,
        pad: usize,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv2dBnActiv::new(nin, nout, ksize, stride, [pad, pad], [1, 1], false, device);
        let conv2 = Conv2dBnActiv::new(nout, nout, ksize, 1, [pad, pad], [1, 1], false, device);
        Self { conv1, conv2 }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(x);
        self.conv2.forward(h)
    }
}
