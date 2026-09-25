"""Load pretrained, quantization-ready FP32 MobileNetV2 on the CPU."""

from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.quantization import mobilenet_v2
from torchvision.models.quantization.mobilenetv2 import QuantizableMobileNetV2


WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V2


def load_fp32_model(cache_dir: Path | None = None) -> QuantizableMobileNetV2:
    """Download/cache ImageNet weights and return an unfused FP32 eval model.

    The torchvision quantizable architecture uses ReLU in place of ReLU6,
    FloatFunctional residual additions, and Quant/DeQuant stubs. With
    quantize=False all computation is still FP32. Using this architecture for
    BOTH baselines holds that ReLU adaptation fixed across the comparison.
    No randomly initialized fallback is used when the download fails.
    """
    cache_dir = cache_dir or Path(__file__).resolve().parent / ".cache" / "torch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))
    model = mobilenet_v2(weights=WEIGHTS, quantize=False, progress=True)
    return model.cpu().eval()
