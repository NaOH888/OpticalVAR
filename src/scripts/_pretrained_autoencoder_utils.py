from __future__ import annotations

import math

import torch


def load_autoencoder_cls():
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "diffusers is required for pretrained autoencoder workflows. "
            "Install it with `pip install diffusers transformers accelerate safetensors`."
        ) from exc
    return AutoencoderKL


def tensor_gray_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 4:
        raise ValueError(f"expected BCHW tensor, got shape {tuple(image.shape)}")
    if int(image.shape[1]) == 3:
        return image
    if int(image.shape[1]) != 1:
        raise ValueError(f"expected 1 or 3 channels, got {tuple(image.shape)}")
    return image.repeat(1, 3, 1, 1)


def tensor_rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 4 or int(image.shape[1]) != 3:
        raise ValueError(f"expected RGB BCHW tensor, got shape {tuple(image.shape)}")
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())
