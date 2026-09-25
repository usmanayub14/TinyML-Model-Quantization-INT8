"""Small offline checks for deterministic inputs and real static conversion."""

import unittest

import torch
from torch import nn
from torch.ao.quantization import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
from torchvision.models.quantization import mobilenet_v2

from calibration_dataloader import SyntheticCalibrationDataset, get_calibration_dataloader
from ptq_quantizer import build_qconfig, minmax_qconfig, quantize_static


class PipelineTests(unittest.TestCase):
    def test_synthetic_data_is_deterministic_and_rng_independent(self):
        dataset = SyntheticCalibrationDataset(3, seed=42)
        rng_state = torch.get_rng_state().clone()
        image = dataset[1]
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        torch.rand(11)  # Global RNG draws cannot affect the private sample RNG.
        self.assertTrue(torch.equal(image, dataset[1]))
        self.assertFalse(torch.equal(image, dataset[0]))
        batches = list(get_calibration_dataloader(3, batch_size=2, seed=42))
        self.assertEqual([batch.shape[0] for batch in batches], [2, 1])
        self.assertTrue(torch.equal(batches[0][1], image))

    def test_minmax_policy(self):
        config = minmax_qconfig("x86")
        self.assertIsInstance(config.activation(), MinMaxObserver)
        self.assertEqual(config.activation().quant_max, 127)
        self.assertIsInstance(config.weight(), PerChannelMinMaxObserver)
        self.assertEqual(config.weight().dtype, torch.qint8)

    def test_histogram_policy_and_invalid_options(self):
        config = build_qconfig("x86", "histogram", histogram_bins=256)
        observer = config.activation()
        self.assertIsInstance(observer, HistogramObserver)
        self.assertEqual(observer.bins, 256)
        self.assertEqual(observer.quant_max, 127)
        self.assertIsInstance(config.weight(), PerChannelMinMaxObserver)
        observer(torch.tensor([-3.0, 0.1, 0.2, 0.3, 5.0]))
        scale, zero_point = observer.calculate_qparams()
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue((scale > 0).all())
        self.assertTrue(((zero_point >= 0) & (zero_point <= 127)).all())
        with self.assertRaises(ValueError):
            build_qconfig("x86", "kl")
        with self.assertRaises(ValueError):
            build_qconfig("x86", histogram_bins=1)

    def test_static_conversion_preserves_fp32_and_executes_int8(self):
        # Random weights keep this structural test offline. main.py separately
        # validates conversion and serialization of the real pretrained model.
        torch.set_num_threads(1)
        model = mobilenet_v2(weights=None, quantize=False).eval()
        loader = get_calibration_dataloader(2, batch_size=2)
        example = SyntheticCalibrationDataset(1, seed=200)[0].unsqueeze(0)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            baseline_output = model(example)
        previous_backend = torch.backends.quantized.engine
        try:
            quantized, metadata = quantize_static(model, loader, calibration_method="histogram", histogram_bins=128)
            self.assertEqual(metadata["activation_observer"], "HistogramObserver")
            self.assertEqual(metadata["calibration_samples"], 2)
            self.assertEqual(metadata["quantized_conv_layers"], 52)
            self.assertEqual(metadata["quantized_linear_layers"], 1)
            self.assertFalse(metadata["remaining_fp32_compute_layers"])
            self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[name], value), name)
            with torch.inference_mode():
                output = quantized(example)
                self.assertTrue(torch.equal(baseline_output, model(example)))
            self.assertEqual(tuple(output.shape), (1, 1000))
            self.assertTrue(torch.isfinite(output).all())
        finally:
            torch.backends.quantized.engine = previous_backend


if __name__ == "__main__":
    unittest.main()
