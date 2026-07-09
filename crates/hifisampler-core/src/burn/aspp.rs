use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use super::conv2dbnactiv::Conv2dBnActiv;

/// 复刻 `layers.py::ASPPModule`：5 分支（mean+conv, 1×1conv, 3×dilated×3）+ bottleneck.
///
/// 注意 PyTorch 的 `Mean(dim=-2, keepdims=True)` 是对频维（H）求均值，保留维度，
/// 然后 `repeat(1, 1, h, 1)` 复制回原尺寸。等价于对 H 维广播均值。
#[derive(Module, Debug)]
pub struct AsppModule<B: Backend> {
    /// 分支1：mean(dim=-2) → Conv2dBnActiv(1×1)
    pub conv1: AsppBranch1<B>,
    /// 分支2：Conv2dBnActiv(1×1)
    pub conv2: Conv2dBnActiv<B>,
    /// 分支3：Conv2dBnActiv(3×3, dilation=dilations[0])
    pub conv3: Conv2dBnActiv<B>,
    /// 分支4：Conv2dBnActiv(3×3, dilation=dilations[1])
    pub conv4: Conv2dBnActiv<B>,
    /// 分支5：Conv2dBnActiv(3×3, dilation=dilations[2])
    pub conv5: Conv2dBnActiv<B>,
    /// bottleneck：Conv2dBnActiv(1×1, nin=5*nout)
    pub bottleneck: Conv2dBnActiv<B>,
}

/// 分支1：Mean(dim=-2, keepdims) + Conv2dBnActiv.
/// burn-store 映射：PyTorch `conv1.0` 是 Mean（无参数），`conv1.1` 是 Conv2dBnActiv.
/// 所以 inner 结构需要 `conv1.1.conv.0` 和 `conv1.1.conv.1` 对应。
#[derive(Module, Debug)]
pub struct AsppBranch1<B: Backend> {
    /// 对应 PyTorch conv1.1（Conv2dBnActiv），命名为 inner1 以匹配 key `conv1.1.*`
    pub inner1: Conv2dBnActiv<B>,
}

impl<B: Backend> AsppModule<B> {
    /// `nin, nout, dilations=[d0,d1,d2]`.
    pub fn new(
        nin: usize,
        nout: usize,
        dilations: [usize; 2],
        device: &B::Device,
    ) -> Self {
        // PyTorch default dilations for CascadedNet BaseNet: ((4,2),(8,4),(12,6))
        // 即 3 个分支，dilation 分别是 4, 8, 12。但 layers.py ASPPModule 默认 (4,8,12)。
        // BaseNet 传入 dilations=((4,2),(8,4),(12,6))，但 ASPPModule 只用 [0] of each:
        //   dilations[0]=4, dilations[1]=8, dilations[2]=12
        // 这里参数 dilations 已是 [d0, d1]... 实际需要 3 个。修正签名。
        // 实际调用：ASPPModule(nout*8, nout*8, dilations=((4,2),(8,4),(12,6)))
        // layers.py 用 dilations[0], dilations[1], dilations[2] 即 (4,2),(8,4),(12,6)
        // 但 Conv2dBnActiv 的 dilation 参数是单个 usize... 
        // 看 layers.py:101: dilations=(4, 8, 12) 默认，BaseNet 传 ((4,2),(8,4),(12,6))
        // layers.py:111: Conv2dBnActiv(nin, nout, 3, 1, dilations[0], dilations[0])
        // 即 pad=dilations[0], dilation=dilations[0]。对 ((4,2),(8,4),(12,6))：
        //   conv3: pad=dilation=(4,2)[0]=4? 不，dilations[0] 是 (4,2) 元组...
        // 实际 layers.py:101 签名 dilations=(4,8,12) 是 3 个 int。
        // BaseNet: dilations=((4, 2), (8, 4), (12, 6)) 传给 ASPPModule
        // layers.py:111: dilations[0] = (4,2) ... 但 Conv2dBnActiv 的 dilation 参数是 int
        // 矛盾。重读 layers.py。
        // 实际 layers.py:99-118: ASPPModule(dilations=(4,8,12)) 默认 3 个 int
        // BaseNet 传 dilations=((4,2),(8,4),(12,6)) — 这是 3 个元组
        // layers.py:111: Conv2dBnActiv(..., dilations[0], dilations[0])
        //   dilations[0] 对默认=(4,) 的第一个元素 4
        //   对 BaseNet 传参 = (4,2) 元组... 但 Conv2dBnActiv dilation 参数是 int
        // 所以 BaseNet 的 dilations 实际是 [[4,2],[8,4],[12,6]] 但只用 [0]=4? 
        // 不，Python dilations[0] 对 ((4,2),(8,4),(12,6)) 是 (4,2)。
        // Conv2dBnActiv(dilation=(4,2)) — 但 Conv2dBnActiv 的 dilation 参数是 int...
        // 看 layers.py:26: dilation=1 默认 int。但可以接受元组吗？nn.Conv2d 可以。
        // 所以 Conv2dBnActiv 的 dilation 实际传的是 (4,2) 元组给 nn.Conv2d。
        // nn.Conv2d 的 dilation 接受 int 或 tuple。(4,2) 表示 H 方向 dilation=4, W 方向=2。
        // 但 Burn 的 Conv2dConfig dilation 是 [usize;2]...
        // 简化：CascadedNet 实际用对称 dilation（4,4),(8,8),(12,12)？
        // 不，BaseNet 传 ((4,2),(8,4),(12,6)) 明确是非对称。
        // 但看 ASPPModule conv3/4/5 的 pad 也用 dilations[i]:
        //   pad=dilations[0]=(4,2) — nn.Conv2d padding 接受 int 或 tuple
        // 所以 conv3: kernel=3, dilation=(4,2), padding=(4,2)
        // 这对 H 方向 pad=4, W 方向 pad=2。
        // Burn Conv2dConfig: dilation=[dh, dw], padding=Explicit(top,bottom,left,right)
        // 对于对称 pad: Explicit(4,4,2,2)
        // 这里我们简化处理——需要完整元组。重写 new 签名。
        let _ = (nin, nout, dilations, device);
        unimplemented!("see full impl below")
    }
}

impl<B: Backend> AsppModule<B> {
    pub fn new_full(
        nin: usize,
        nout: usize,
        dilations: [[usize; 2]; 3],
        device: &B::Device,
    ) -> Self {
        // conv1: Mean + Conv2dBnActiv(1×1, pad=0, dilation=1, ReLU)
        let inner1 = Conv2dBnActiv::new(nin, nout, 1, 1, [0, 0], [1, 1], true, device);
        let conv1 = AsppBranch1 { inner1 };
        // conv2: Conv2dBnActiv(1×1, pad=0, dilation=1, ReLU)
        let conv2 = Conv2dBnActiv::new(nin, nout, 1, 1, [0, 0], [1, 1], true, device);
        // conv3/4/5: Conv2dBnActiv(3×3, pad=d[i], dilation=d[i], ReLU)
        // PyTorch dilations=((4,2),(8,4),(12,6)): pad/dilation = (d_h, d_w)
        let conv3 = Conv2dBnActiv::new(nin, nout, 3, 1, dilations[0], dilations[0], true, device);
        let conv4 = Conv2dBnActiv::new(nin, nout, 3, 1, dilations[1], dilations[1], true, device);
        let conv5 = Conv2dBnActiv::new(nin, nout, 3, 1, dilations[2], dilations[2], true, device);
        // bottleneck: Conv2dBnActiv(1×1, nin=5*nout, nout, ReLU)
        let bottleneck = Conv2dBnActiv::new(nout * 5, nout, 1, 1, [0, 0], [1, 1], true, device);
        Self {
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            bottleneck,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _c, h, w] = x.dims();
        // feat1 = conv1(x): mean(dim=-2, keepdim) then conv, then repeat(1,1,h,1)
        let feat1 = {
            // Burn mean_dim returns keepdim shape [B,C,1,W]
            let meaned = x.clone().mean_dim(2);
            let conv_out = self.conv1.inner1.forward(meaned);
            // repeat(1, 1, h, 1) — repeat along H dim h times
            conv_out.repeat(&[1, 1, h, 1])
        };
        let feat2 = self.conv2.forward(x.clone());
        let feat3 = self.conv3.forward(x.clone());
        let feat4 = self.conv4.forward(x.clone());
        let feat5 = self.conv5.forward(x.clone());
        let out = Tensor::cat(vec![feat1, feat2, feat3, feat4, feat5], 1);
        self.bottleneck.forward(out)
    }
}
