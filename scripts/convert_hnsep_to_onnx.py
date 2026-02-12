#!/usr/bin/env python3
"""
Convert HN-SEP PyTorch model to ONNX format.
Run from hifisampler/ directory with its .venv activated.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hifisampler"
    ),
)

import torch
import torch.nn as nn
import yaml
import numpy as np
import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Import from hifisampler codebase
from hnsep.nets import CascadedNet


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class OnnxCompatibleCascadedNet(nn.Module):
    """ONNX-compatible version that handles complex numbers as separate real/imag channels."""

    def __init__(self, original_model):
        super().__init__()
        self.n_fft = original_model.n_fft
        self.hop_length = original_model.hop_length
        self.max_bin = original_model.max_bin
        self.output_bin = original_model.output_bin
        self.offset = original_model.offset

        self.stg1_low_band_net = original_model.stg1_low_band_net
        self.stg1_high_band_net = original_model.stg1_high_band_net
        self.stg2_low_band_net = original_model.stg2_low_band_net
        self.stg2_high_band_net = original_model.stg2_high_band_net
        self.stg3_full_band_net = original_model.stg3_full_band_net
        self.out = original_model.out

    def forward(self, x):
        """
        x: [batch, 2, freq, time] where channel 0=real, channel 1=imag
        Returns: [batch, 2, output_bin, time] bounded complex mask
        """
        x = x[:, :, : self.max_bin]

        bandw = x.size(2) // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = self.out(f3)

        real_part = mask[:, :1]
        imag_part = mask[:, 1:]

        mask_mag = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        tanh_mag = torch.tanh(mask_mag)

        real_normalized = tanh_mag * real_part / (mask_mag + 1e-8)
        imag_normalized = tanh_mag * imag_part / (mask_mag + 1e-8)

        mask = torch.cat([real_normalized, imag_normalized], dim=1)

        mask = torch.nn.functional.pad(
            input=mask, pad=(0, 0, 0, self.output_bin - mask.size(2)), mode="replicate"
        )

        return mask


def main():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "hifisampler",
        "hnsep",
        "vr",
        "model.pt",
    )
    config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "hifisampler",
        "hnsep",
        "vr",
        "model.onnx",
    )

    logging.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    logging.info(f"Loading model from: {model_path}")
    model = CascadedNet(
        args_dict["n_fft"],
        args_dict["hop_length"],
        args_dict["n_out"],
        args_dict["n_out_lstm"],
        is_complex=True,
        is_mono=args_dict["is_mono"],
        fixed_length=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    onnx_model = OnnxCompatibleCascadedNet(model)
    onnx_model.eval()

    freq_bins = args_dict["n_fft"] // 2 + 1  # 1025
    dummy_input = torch.randn(1, 2, freq_bins, 256)

    logging.info(f"Exporting ONNX to: {output_path}")
    logging.info(f"Input shape: {dummy_input.shape}")

    torch.onnx.export(
        onnx_model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {3: "time_frames"}, "output": {3: "time_frames"}},
        verbose=False,
    )

    logging.info("Simplifying with OnnxSlim...")
    try:
        import onnx
        import onnxslim

        model_onnx = onnxslim.slim(output_path)
        onnx.save(model_onnx, output_path)
        logging.info("OnnxSlim completed.")
    except Exception as e:
        logging.warning(f"OnnxSlim failed: {e}")

    # Verify
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = sess.run(["output"], {"input": dummy_input.numpy()})
    logging.info(f"Verification output shape: {out[0].shape}")

    # Compare with PyTorch
    with torch.no_grad():
        pt_out = onnx_model(dummy_input)

    diff = np.abs(pt_out.numpy() - out[0])
    logging.info(f"Max abs diff (ONNX vs PyTorch): {diff.max():.8f}")
    logging.info(f"Mean abs diff: {diff.mean():.8f}")

    logging.info("HN-SEP ONNX conversion done!")


if __name__ == "__main__":
    main()
