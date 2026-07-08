use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

use super::conv2dbnactiv::Conv2dBnActiv;

/// 复刻 `layers.py::LSTMModule`：Conv2dBnActiv(→1ch) + 双向 LSTM + Linear+BN1d+ReLU.
///
/// PyTorch:
///   conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
///   lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm//2, bidirectional=True)
///   dense = nn.Sequential(Linear(nout_lstm, nin_lstm), BatchNorm1d(nin_lstm), ReLU)
///
/// forward:
///   h = conv(x)[:, 0]              # [N, nbins, nframes]
///   h = h.permute(2, 0, 1)         # [nframes, N, nbins]  (nin_lstm=nbins)
///   h, _ = lstm(h)                 # [nframes, N, nout_lstm]
///   h = dense(h.reshape(-1, nout_lstm))  # [nframes*N, nin_lstm]
///   h = h.reshape(nframes, N, 1, nin_lstm).permute(1, 2, 3, 0)  # [N, 1, nin_lstm, nframes]
#[derive(Module, Debug)]
pub struct LstmModule<B: Backend> {
    pub conv: Conv2dBnActiv<B>,
    /// PyTorch LSTM 权重（手动加载，非 Burn BiLstm）。
    /// weight_ih_l0: [4*hidden, input], weight_hh_l0: [4*hidden, hidden]
    /// bias_ih_l0: [4*hidden], bias_hh_l0: [4*hidden]
    /// _reverse 同理。
    pub lstm: ManualBiLstm<B>,
    /// dense: Linear(nout_lstm, nin_lstm) + BatchNorm1d(nin_lstm)
    /// PyTorch key: dense.0.weight/bias, dense.1.weight/bias/running_mean/running_var
    pub dense0: Linear<B>,
    pub dense1: BatchNorm<B>,
}

/// 手动双向 LSTM，直接加载 PyTorch nn.LSTM 权重。
/// 4 个 gate: input(i), forget(f), cell(g), output(o)
/// PyTorch weight_ih 顺序: [i, f, g, o]，每个 hidden 大小。
#[derive(Module, Debug)]
pub struct ManualBiLstm<B: Backend> {
    // forward direction
    pub weight_ih_l0: burn::module::Param<Tensor<B, 2>>,
    pub weight_hh_l0: burn::module::Param<Tensor<B, 2>>,
    pub bias_ih_l0: burn::module::Param<Tensor<B, 1>>,
    pub bias_hh_l0: burn::module::Param<Tensor<B, 1>>,
    // reverse direction
    pub weight_ih_l0_reverse: burn::module::Param<Tensor<B, 2>>,
    pub weight_hh_l0_reverse: burn::module::Param<Tensor<B, 2>>,
    pub bias_ih_l0_reverse: burn::module::Param<Tensor<B, 1>>,
    pub bias_hh_l0_reverse: burn::module::Param<Tensor<B, 1>>,
    pub input_size: usize,
    pub hidden_size: usize,
    /// CPU 权重缓存：首次 forward_cpu 时从 GPU 搬过来，后续复用。
    /// 避免每次 forward 都搬 8 个权重 tensor（省 ~800ms GPU→CPU sync）。
    #[module(skip)]
    pub cpu_cache: std::sync::OnceLock<CpuLstmWeights>,
}

/// 预搬到 CPU 的 LSTM 权重（ndarray 格式）。
#[derive(Clone, Debug)]
pub struct CpuLstmWeights {
    pub w_ih_f: ndarray::Array2<f32>,
    pub w_hh_f: ndarray::Array2<f32>,
    pub b_ih_f: ndarray::Array1<f32>,
    pub b_hh_f: ndarray::Array1<f32>,
    pub w_ih_r: ndarray::Array2<f32>,
    pub w_hh_r: ndarray::Array2<f32>,
    pub b_ih_r: ndarray::Array1<f32>,
    pub b_hh_r: ndarray::Array1<f32>,
}

impl<B: Backend> ManualBiLstm<B> {
    /// 输入: [seq_len, batch, input_size]
    /// 输出: [seq_len, batch, hidden_size*2]
    ///
    /// 默认走 CPU (ndarray) 路径以避免 GPU 逐时间步同步开销。
    /// 设置环境变量 `HIFISAMPLER_LSTM_BACKEND=gpu` 可切回 GPU 路径（用于对比/调试）。
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match std::env::var("HIFISAMPLER_LSTM_BACKEND").as_deref() {
            Ok("gpu") => self.forward_gpu(x),
            _ => self.forward_cpu(x),
        }
    }

    /// CPU LSTM via ndarray。一次 GPU→CPU 同步读入输入+权重，串行循环在 CPU 跑，
    /// 最后一次 CPU→GPU 同步写回输出。避免 GPU 上每步 matmul+激活的 kernel launch +
    /// sync 开销。
    fn forward_cpu(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();

        // 1. 输入搬到 CPU（仅 input tensor，权重用缓存）
        let x_arr = burn3_to_ndarray(&x); // [seq, batch, input]

        // 2. 获取或初始化 CPU 权重缓存（OnceLock，首次调用从 GPU 搬运）
        let weights = self.cpu_cache.get_or_init(|| CpuLstmWeights {
            w_ih_f: burn2_to_ndarray(&self.weight_ih_l0.val()),
            w_hh_f: burn2_to_ndarray(&self.weight_hh_l0.val()),
            b_ih_f: burn1_to_ndarray(&self.bias_ih_l0.val()),
            b_hh_f: burn1_to_ndarray(&self.bias_hh_l0.val()),
            w_ih_r: burn2_to_ndarray(&self.weight_ih_l0_reverse.val()),
            w_hh_r: burn2_to_ndarray(&self.weight_hh_l0_reverse.val()),
            b_ih_r: burn1_to_ndarray(&self.bias_ih_l0_reverse.val()),
            b_hh_r: burn1_to_ndarray(&self.bias_hh_l0_reverse.val()),
        });

        // 3. CPU 上跑双向 LSTM
        let fwd = run_lstm_direction_cpu(
            &x_arr, &weights.w_ih_f, &weights.w_hh_f, &weights.b_ih_f, &weights.b_hh_f, false, self.hidden_size,
        );
        let rev = run_lstm_direction_cpu(
            &x_arr, &weights.w_ih_r, &weights.w_hh_r, &weights.b_ih_r, &weights.b_hh_r, true, self.hidden_size,
        );

        // 4. 拼接 forward + reverse: [seq, batch, hidden*2]
        let out_arr =
            ndarray::concatenate(ndarray::Axis(2), &[fwd.view(), rev.view()]).unwrap();

        // 5. 一次性搬回 GPU
        ndarray3_to_burn(out_arr, &device)
    }

    /// GPU LSTM（原实现，保留用于对比/调试）。
    #[allow(dead_code)]
    fn forward_gpu(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [seq_len, batch, _input_size] = x.dims();
        let device = x.device();

        let fwd = self.run_direction(
            x.clone(),
            &self.weight_ih_l0.val(),
            &self.weight_hh_l0.val(),
            &self.bias_ih_l0.val(),
            &self.bias_hh_l0.val(),
            false,
            &device,
            seq_len,
            batch,
        );
        let rev = self.run_direction(
            x,
            &self.weight_ih_l0_reverse.val(),
            &self.weight_hh_l0_reverse.val(),
            &self.bias_ih_l0_reverse.val(),
            &self.bias_hh_l0_reverse.val(),
            true,
            &device,
            seq_len,
            batch,
        );
        // concat forward and reverse: [seq_len, batch, hidden*2]
        Tensor::cat(vec![fwd, rev], 2)
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    fn run_direction(
        &self,
        x: Tensor<B, 3>,           // [seq, batch, input]
        w_ih: &Tensor<B, 2>,        // [4*hidden, input]
        w_hh: &Tensor<B, 2>,        // [4*hidden, hidden]
        b_ih: &Tensor<B, 1>,        // [4*hidden]
        b_hh: &Tensor<B, 1>,        // [4*hidden]
        reverse: bool,
        device: &B::Device,
        seq_len: usize,
        batch: usize,
    ) -> Tensor<B, 3> {
        let hidden = self.hidden_size;
        let four_hidden = 4 * hidden;

        // 优化 1：权重转置预算一次（原代码每步都 swap_dims 一次）
        let w_ih_t = w_ih.clone().swap_dims(0, 1); // [input, 4*hidden]
        let w_hh_t = w_hh.clone().swap_dims(0, 1); // [hidden, 4*hidden]

        // 优化 2：bias 合并预算一次（原代码每步两次 unsqueeze+add）
        // b_ih + b_hh → [4*hidden] → [1, 4*hidden] 用于广播
        let bias_sum = b_ih.clone().add(b_hh.clone()).unsqueeze_dim(0);

        // 优化 3：批量 input projection —— 1 个大 matmul 替代 seq_len 个小 matmul
        // input projection 不依赖 h_prev，可一次性算完整个序列
        // x: [seq, batch, input] → reshape [seq*batch, input] @ [input, 4*hidden] → [seq*batch, 4*hidden]
        let x_2d = x.reshape([seq_len * batch, self.input_size]);
        let gi_all_2d = x_2d.matmul(w_ih_t); // [seq*batch, 4*hidden]
        let gi_all = gi_all_2d.reshape([seq_len, batch, four_hidden]);

        // 优化 4：预分配输出 buffer，用 slice_assign 写入（替代 Vec::push + 末尾 cat）
        let mut output = Tensor::<B, 3>::empty([seq_len, batch, hidden], device);

        let mut h_prev = Tensor::<B, 2>::zeros([batch, hidden], device);
        let mut c_prev = Tensor::<B, 2>::zeros([batch, hidden], device);

        // 顺序遍历：out_t 是输出位置，in_t 是输入位置（reverse 时反向）
        let in_indices: Vec<usize> = if reverse {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for (out_t, in_t) in in_indices.iter().enumerate() {
            let in_t = *in_t;

            // 取本时间步的 input projection（已预算）：[batch, 4*hidden]
            let gi_t = gi_all
                .clone()
                .slice([in_t..in_t + 1, 0..batch, 0..four_hidden])
                .squeeze_dim(0);

            // hidden projection（依赖 h_prev，必须串行）：[batch, 4*hidden]
            let gh = h_prev.clone().matmul(w_hh_t.clone());

            // gates = gi_t + gh + bias_sum (广播)
            let gates = gi_t.add(gh).add(bias_sum.clone());

            // Split gates: [batch, 4*hidden] → 4 × [batch, hidden]
            // PyTorch order: i, f, g, o
            let i = gates.clone().slice([0..batch, 0..hidden]);
            let f = gates.clone().slice([0..batch, hidden..2 * hidden]);
            let g = gates.clone().slice([0..batch, 2 * hidden..3 * hidden]);
            let o = gates.clone().slice([0..batch, 3 * hidden..four_hidden]);

            let i_sig = burn::tensor::activation::sigmoid(i);
            let f_sig = burn::tensor::activation::sigmoid(f);
            let g_tanh = g.tanh();
            let o_sig = burn::tensor::activation::sigmoid(o);

            c_prev = f_sig * c_prev + i_sig * g_tanh;
            h_prev = o_sig * c_prev.clone().tanh();

            // 写入预分配的输出 buffer
            output = output.slice_assign(
                [out_t..out_t + 1, 0..batch, 0..hidden],
                h_prev.clone().unsqueeze_dim(0),
            );
        }

        output
    }
}

impl<B: Backend> LstmModule<B> {
    pub fn new(
        nin_conv: usize,
        nin_lstm: usize,
        nout_lstm: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dBnActiv::new(nin_conv, 1, 1, 1, [0, 0], [1, 1], true, device);
        // LSTM: input_size=nin_lstm, hidden_size=nout_lstm//2
        let hidden = nout_lstm / 2;
        let lstm = ManualBiLstm {
            weight_ih_l0: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden, nin_lstm], device)),
            weight_hh_l0: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden, hidden], device)),
            bias_ih_l0: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden], device)),
            bias_hh_l0: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden], device)),
            weight_ih_l0_reverse: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden, nin_lstm], device)),
            weight_hh_l0_reverse: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden, hidden], device)),
            bias_ih_l0_reverse: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden], device)),
            bias_hh_l0_reverse: burn::module::Param::from_tensor(Tensor::zeros([4 * hidden], device)),
            input_size: nin_lstm,
            hidden_size: hidden,
            cpu_cache: std::sync::OnceLock::new(),
        };
        let dense0 = LinearConfig::new(nout_lstm, nin_lstm).init(device);
        let dense1 = BatchNormConfig::new(nin_lstm).init(device);
        Self {
            conv,
            lstm,
            dense0,
            dense1,
        }
    }

    /// x: [N, C, nbins, nframes]
    /// 输出: [N, 1, nin_lstm, nframes]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, _c, nbins, nframes] = x.dims();
        let device = x.device();

        // h = conv(x)[:, 0]  → [N, nbins, nframes]
        let h = self.conv.forward(x).slice([0..n, 0..1, 0..nbins, 0..nframes]).squeeze_dim(1);
        // h.permute(2, 0, 1) → [nframes, N, nbins]
        let h = h.swap_dims(0, 2).swap_dims(1, 2);
        // h, _ = lstm(h) → [nframes, N, nout_lstm]
        let h = self.lstm.forward(h);
        // dense(h.reshape(-1, nout_lstm)) → [nframes*N, nin_lstm]
        let nout_lstm = self.lstm.hidden_size * 2;
        let h_flat = h.reshape([nframes * n, nout_lstm]);
        let h_flat = self.dense0.forward(h_flat);
        let h_flat = self.dense1.forward(h_flat);
        let h_flat = burn::tensor::activation::relu(h_flat);  // nn.Sequential(Linear, BN, ReLU)
        // h.reshape(nframes, N, 1, nbins).permute(1, 2, 3, 0) → [N, 1, nbins, nframes]
        // permute(1,2,3,0) on [nframes, N, 1, nbins]:
        //   swap(0,1)→[N,nframes,1,nbins], swap(1,2)→[N,1,nframes,nbins], swap(2,3)→[N,1,nbins,nframes]
        let h = h_flat.reshape([nframes, n, 1, nbins]);
        h.swap_dims(0, 1).swap_dims(1, 2).swap_dims(2, 3)
    }
}

// =============================================================================
// CPU LSTM 实现（ndarray）
// =============================================================================

/// Burn Tensor<B, 3> → ndarray Array3<f32>
/// 调用 to_data() 触发一次 GPU→CPU 同步。
fn burn3_to_ndarray<B: Backend>(t: &Tensor<B, 3>) -> ndarray::Array3<f32> {
    use burn::tensor::ElementConversion;
    let data = t.to_data();
    let shape = data.shape.clone();
    let vec: Vec<f32> = data
        .as_slice::<<B as burn::tensor::backend::BackendTypes>::FloatElem>()
        .unwrap_or(&[])
        .iter()
        .map(|&v| v.elem::<f32>())
        .collect();
    ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), vec)
        .expect("burn3_to_ndarray: shape mismatch")
}

fn burn2_to_ndarray<B: Backend>(t: &Tensor<B, 2>) -> ndarray::Array2<f32> {
    use burn::tensor::ElementConversion;
    let data = t.to_data();
    let shape = data.shape.clone();
    let vec: Vec<f32> = data
        .as_slice::<<B as burn::tensor::backend::BackendTypes>::FloatElem>()
        .unwrap_or(&[])
        .iter()
        .map(|&v| v.elem::<f32>())
        .collect();
    ndarray::Array2::from_shape_vec((shape[0], shape[1]), vec)
        .expect("burn2_to_ndarray: shape mismatch")
}

fn burn1_to_ndarray<B: Backend>(t: &Tensor<B, 1>) -> ndarray::Array1<f32> {
    use burn::tensor::ElementConversion;
    let data = t.to_data();
    let shape = data.shape.clone();
    let vec: Vec<f32> = data
        .as_slice::<<B as burn::tensor::backend::BackendTypes>::FloatElem>()
        .unwrap_or(&[])
        .iter()
        .map(|&v| v.elem::<f32>())
        .collect();
    ndarray::Array1::from_shape_vec(shape[0], vec).expect("burn1_to_ndarray: shape mismatch")
}

/// ndarray Array3<f32> → Burn Tensor<B, 3>
/// 调用 from_data() 触发一次 CPU→GPU 同步。
fn ndarray3_to_burn<B: Backend>(
    arr: ndarray::Array3<f32>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let dims = arr.shape();
    let data_vec: Vec<f32> = arr.iter().cloned().collect();
    let tensor_data = burn::tensor::TensorData::new(data_vec, [dims[0], dims[1], dims[2]]);
    Tensor::<B, 3>::from_data(tensor_data, device)
}

/// CPU 上跑单方向 LSTM。
///
/// - `x`: [seq, batch, input]
/// - `w_ih`: [4*hidden, input]
/// - `w_hh`: [4*hidden, hidden]
/// - `b_ih`, `b_hh`: [4*hidden]
/// - 返回: [seq, batch, hidden]
///
/// PyTorch gate 顺序: i, f, g, o
///
/// 优化：fused loop — 每个时间步用单个循环算完所有 gate 激活 + 状态更新，
/// 避免原版 8+ 次 mapv/元素运算的数组分配。
fn run_lstm_direction_cpu(
    x: &ndarray::Array3<f32>,
    w_ih: &ndarray::Array2<f32>,
    w_hh: &ndarray::Array2<f32>,
    b_ih: &ndarray::Array1<f32>,
    b_hh: &ndarray::Array1<f32>,
    reverse: bool,
    hidden: usize,
) -> ndarray::Array3<f32> {
    let seq = x.shape()[0];
    let batch = x.shape()[1];
    let _four_h = 4 * hidden;
    let four_h = 4 * hidden;

    // bias_sum = b_ih + b_hh
    let bias_sum = b_ih + b_hh; // [4h]

    // 预分配输出
    let mut output = ndarray::Array3::<f32>::zeros((seq, batch, hidden));

    // h_prev, c_prev: [batch, hidden]
    let mut h_prev = ndarray::Array2::<f32>::zeros((batch, hidden));
    let mut c_prev = ndarray::Array2::<f32>::zeros((batch, hidden));

    // 转置权重为 owned（[input, 4h] 和 [hidden, 4h]），便于 matmul
    let w_ih_t = w_ih.t().to_owned();
    let w_hh_t = w_hh.t().to_owned();

    // 批量 input projection：gi_all = x_2d @ w_ih_t → [seq*batch, 4h]
    // 一次性算完所有时间步的 input projection（不依赖 h_prev）
    // x: [seq, batch, input] → reshape [seq*batch, input]
    let x_2d = x.to_shape((seq * batch, x.shape()[2])).unwrap();
    let gi_all_2d = x_2d.dot(&w_ih_t); // [seq*batch, 4h]
    let gi_all = gi_all_2d
        .to_shape((seq, batch, four_h))
        .unwrap()
        .to_owned();

    // 遍历顺序：forward 时 0..seq，reverse 时 seq-1..0
    let indices: Vec<usize> = if reverse {
        (0..seq).rev().collect()
    } else {
        (0..seq).collect()
    };

    for in_t in indices {
        // gi_t: [batch, 4h] — 从预算的 gi_all 中取（view，无拷贝）
        let gi_t = gi_all.index_axis(ndarray::Axis(0), in_t);

        // gh = h_prev @ w_hh.T: [batch, 4h]（依赖 h_prev，必须每步算）
        let gh = h_prev.dot(&w_hh_t);

        // FUSED: gate 激活 + 状态更新（单循环，避免 8+ 次 mapv 分配）
        for b in 0..batch {
            let gi_row = gi_t.row(b);
            let gh_row = gh.row(b);
            let bias = &bias_sum;

            let base_f = hidden;
            let base_g = 2 * hidden;
            let base_o = 3 * hidden;

            for k in 0..hidden {
                let i_val = sigmoid_f32(gi_row[k] + gh_row[k] + bias[k]);
                let f_val = sigmoid_f32(gi_row[base_f + k] + gh_row[base_f + k] + bias[base_f + k]);
                let g_val = (gi_row[base_g + k] + gh_row[base_g + k] + bias[base_g + k]).tanh();
                let o_val = sigmoid_f32(gi_row[base_o + k] + gh_row[base_o + k] + bias[base_o + k]);

                let c_old = c_prev[(b, k)];
                let c_new = f_val * c_old + i_val * g_val;
                c_prev[(b, k)] = c_new;

                let h_new = o_val * c_new.tanh();
                h_prev[(b, k)] = h_new;
                output[(in_t, b, k)] = h_new;
            }
        }
    }

    output
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    // 数值稳定 sigmoid
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
