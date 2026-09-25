"""Offline arithmetic and hook-lifecycle regression tests."""

import json
import math
import unittest

import torch
from torch import nn

from evaluator import evaluate_layerwise


class Affine(nn.Module):
    def __init__(self, scale=1.0, offset=0.0):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def forward(self, value):
        return value * self.scale + self.offset


class QuantizedOutput(nn.Module):
    def forward(self, value):
        return torch.quantize_per_tensor(value, scale=0.5, zero_point=0, dtype=torch.qint8)


class RaiseError(nn.Module):
    def forward(self, value):
        raise RuntimeError("intentional forward failure")


class DiagnosticTests(unittest.TestCase):
    @staticmethod
    def assert_no_hooks(*models):
        for model in models:
            for module in model.modules():
                if module._forward_hooks:
                    raise AssertionError("Diagnostic forward hooks leaked")

    def test_element_weighting_and_known_snr_across_unequal_batches(self):
        fp32, int8 = nn.Sequential(Affine()), nn.Sequential(Affine(offset=1.0))
        # Signal = 1 + 4 + 9; squared error = 1 + 1 + 1.
        result = evaluate_layerwise(fp32, int8, [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])], ["0"])
        layer = result["layers"][0]
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["batch_count"], 2)
        self.assertEqual(layer["element_count"], 3)
        self.assertAlmostEqual(layer["mse"], 1.0)
        self.assertAlmostEqual(layer["signal_power"], 14.0 / 3.0)
        self.assertAlmostEqual(layer["nmse"], 3.0 / 14.0)
        self.assertAlmostEqual(layer["snr_db"], 10.0 * math.log10(14.0 / 3.0))
        self.assertEqual(layer["snr_status"], "finite")
        json.dumps(result, allow_nan=False)
        self.assert_no_hooks(fp32, int8)

    def test_quantized_outputs_are_dequantized(self):
        fp32, int8 = nn.Sequential(nn.Identity()), nn.Sequential(QuantizedOutput())
        result = evaluate_layerwise(fp32, int8, [torch.tensor([[0.25, 0.75]])], ["0"])
        self.assertAlmostEqual(result["layers"][0]["mse"], 0.0625)
        self.assert_no_hooks(fp32, int8)

    def test_clone_protects_capture_from_downstream_inplace_mutation(self):
        fp32 = nn.Sequential(nn.Identity(), nn.ReLU(inplace=True))
        int8 = nn.Sequential(nn.Identity(), nn.Identity())
        result = evaluate_layerwise(fp32, int8, [torch.tensor([[-2.0, 1.0]])], ["0"])
        self.assertEqual(result["layers"][0]["mse"], 0.0)
        self.assertEqual(result["layers"][0]["signal_power"], 2.5)
        self.assert_no_hooks(fp32, int8)

    def test_zero_noise_and_zero_signal_are_explicit_and_json_safe(self):
        for reference, actual, status, nmse in (
            (1.0, 1.0, "zero_noise", 0.0),
            (0.0, 0.0, "zero_signal_and_noise", None),
            (0.0, 1.0, "zero_signal", None),
        ):
            with self.subTest(status=status):
                fp32 = nn.Sequential(Affine(scale=0.0, offset=reference))
                int8 = nn.Sequential(Affine(scale=0.0, offset=actual))
                result = evaluate_layerwise(fp32, int8, [torch.ones(1, 2)], ["0"])
                layer = result["layers"][0]
                self.assertEqual(layer["snr_status"], status)
                self.assertIsNone(layer["snr_db"])
                self.assertEqual(layer["nmse"], nmse)
                json.dumps(result, allow_nan=False)
                self.assert_no_hooks(fp32, int8)

    def test_hook_cleanup_and_training_state_restoration_on_forward_error(self):
        fp32 = nn.Sequential(nn.Identity(), nn.Identity()).train()
        fp32[0].eval()  # Preserve mixed per-module states, too.
        int8 = nn.Sequential(nn.Identity(), RaiseError()).eval()
        original_states = [[module.training for module in model.modules()] for model in (fp32, int8)]
        with self.assertRaisesRegex(RuntimeError, "intentional forward failure"):
            evaluate_layerwise(fp32, int8, [torch.ones(1, 2)], ["0"])
        self.assert_no_hooks(fp32, int8)
        self.assertEqual(original_states, [[module.training for module in model.modules()] for model in (fp32, int8)])

    def test_success_restores_training_flags_and_preserves_existing_hooks(self):
        fp32, int8 = nn.Sequential(nn.Dropout()), nn.Sequential(nn.Dropout())
        calls = []
        existing = fp32[0].register_forward_hook(lambda *_args: calls.append(True))
        try:
            result = evaluate_layerwise(fp32, int8, [torch.ones(1, 2)], ["0"])
            self.assertEqual(result["layers"][0]["mse"], 0.0)
            self.assertTrue(fp32.training and fp32[0].training and int8.training)
            self.assertEqual(len(fp32[0]._forward_hooks), 1)
            self.assertEqual(len(calls), 1)
        finally:
            existing.remove()
        self.assert_no_hooks(fp32, int8)

    def test_empty_nonfinite_and_missing_layers_are_rejected_without_hooks(self):
        fp32, int8 = nn.Sequential(nn.Identity()), nn.Sequential(nn.Identity())
        cases = (
            ([], ["0"], "at least one batch"),
            ([torch.tensor([[float("nan")]])], ["0"], "finite"),
            ([torch.ones(1, 2)], ["missing"], "has no layer"),
        )
        for inputs, names, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_layerwise(fp32, int8, inputs, names)
                self.assert_no_hooks(fp32, int8)

    def test_mismatched_shapes_cleanup_hooks(self):
        fp32, int8 = nn.Sequential(nn.Identity()), nn.Sequential(nn.Flatten())
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            evaluate_layerwise(fp32, int8, [torch.ones(1, 2, 3)], ["0"])
        self.assert_no_hooks(fp32, int8)


if __name__ == "__main__":
    unittest.main()
