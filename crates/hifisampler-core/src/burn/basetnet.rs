use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use super::aspp::AsppModule;
use super::conv2dbnactiv::Conv2dBnActiv;
use super::decoder::Decoder;
use super::encoder::Encoder;
use super::lstm_module::LstmModule;

/// 复刻 `nets.py::BaseNet`：4级 encoder + ASPP + 4级 decoder + LSTM.
#[derive(Module, Debug)]
pub struct BaseNet<B: Backend> {
    pub enc1: Conv2dBnActiv<B>,
    pub enc2: Encoder<B>,
    pub enc3: Encoder<B>,
    pub enc4: Encoder<B>,
    pub enc5: Encoder<B>,
    pub aspp: AsppModule<B>,
    pub dec4: Decoder<B>,
    pub dec3: Decoder<B>,
    pub dec2: Decoder<B>,
    pub lstm_dec2: LstmModule<B>,
    pub dec1: Decoder<B>,
}

#[derive(Config, Debug)]
pub struct BaseNetConfig {
    pub nin: usize,
    pub nout: usize,
    pub nin_lstm: usize,
    pub nout_lstm: usize,
    #[config(default = "[[4, 2], [8, 4], [12, 6]]")]
    pub dilations: [[usize; 2]; 3],
}

impl BaseNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BaseNet<B> {
        let n = self.nout;
        let enc1 = Conv2dBnActiv::new(self.nin, n, 3, 1, [1, 1], [1, 1], true, device);
        let enc2 = Encoder::new(n, n * 2, 3, 2, 1, device);
        let enc3 = Encoder::new(n * 2, n * 4, 3, 2, 1, device);
        let enc4 = Encoder::new(n * 4, n * 6, 3, 2, 1, device);
        let enc5 = Encoder::new(n * 6, n * 8, 3, 2, 1, device);

        let aspp = AsppModule::new_full(n * 8, n * 8, self.dilations, device);

        let dec4 = Decoder::new(n * (6 + 8), n * 6, 3, 1, 1, device);
        let dec3 = Decoder::new(n * (4 + 6), n * 4, 3, 1, 1, device);
        let dec2 = Decoder::new(n * (2 + 4), n * 2, 3, 1, 1, device);
        let lstm_dec2 = LstmModule::new(n * 2, self.nin_lstm, self.nout_lstm, device);
        let dec1 = Decoder::new(n * (1 + 2) + 1, n * 1, 3, 1, 1, device);

        BaseNet {
            enc1,
            enc2,
            enc3,
            enc4,
            enc5,
            aspp,
            dec4,
            dec3,
            dec2,
            lstm_dec2,
            dec1,
        }
    }
}

impl<B: Backend> BaseNet<B> {
    /// x: [B, nin, H, W] → [B, nout, H, W]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let e1 = self.enc1.forward(x);
        let e2 = self.enc2.forward(e1.clone());
        let e3 = self.enc3.forward(e2.clone());
        let e4 = self.enc4.forward(e3.clone());
        let e5 = self.enc5.forward(e4.clone());

        let h = self.aspp.forward(e5);

        let h = self.dec4.forward(h, Some(e4));
        let h = self.dec3.forward(h, Some(e3));
        let h = self.dec2.forward(h, Some(e2));
        // h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        let lstm_out = self.lstm_dec2.forward(h.clone());
        let h = Tensor::cat(vec![h, lstm_out], 1);
        let h = self.dec1.forward(h, Some(e1));

        h
    }
}
