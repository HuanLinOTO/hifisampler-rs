use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::leaky_relu;
use burn::tensor::{Tensor, backend::Backend};

/// HiFi-GAN MRF 中的 ResBlock1（resblock="1"）。
/// 3 组并行 dilation 的 Conv1d 链，每组 convs1(dilation=d) -> convs2(dilation=1)，带残差。
#[derive(Module, Debug)]
pub struct ResBlock1<B: Backend> {
    convs1: Vec<Conv1d<B>>,
    convs2: Vec<Conv1d<B>>,
}

impl<B: Backend> ResBlock1<B> {
    /// `dilations` 长度决定并行分支数（vocoder 中为 3：[1, 3, 5]）。
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: [usize; 3],
        device: &B::Device,
    ) -> Self {
        let convs1 = dilations
            .iter()
            .map(|&d| {
                let pad = d * (kernel_size - 1) / 2;
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_dilation(d)
                    .with_padding(PaddingConfig1d::Explicit(pad, pad))
                    .init(device)
            })
            .collect();
        let convs2 = (0..3)
            .map(|_| {
                let pad = (kernel_size - 1) / 2;
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_padding(PaddingConfig1d::Explicit(pad, pad))
                    .init(device)
            })
            .collect();
        Self { convs1, convs2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = leaky_relu(x.clone(), 0.1);
            let xt = c1.forward(xt);
            let xt = leaky_relu(xt, 0.1);
            let xt = c2.forward(xt);
            x = xt + x;
        }
        x
    }
}
