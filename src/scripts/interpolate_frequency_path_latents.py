from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import MultiScaleFrequencyTargetTransform, NpzImageDataset
from scripts._pretrained_autoencoder_utils import (
    load_autoencoder_cls,
    tensor_gray_to_rgb,
    tensor_rgb_to_gray,
)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def _save_single_image(path: Path, image: np.ndarray, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(3, 3))
    axis.imshow(_normalize_for_display(image), cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _save_grid(images: list[np.ndarray], titles: list[str], path: Path, *, cols: int = 4) -> None:
    rows = int(math.ceil(len(images) / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    axes = axes.reshape(rows, cols)
    for axis in axes.flat:
        axis.axis("off")
    for index, (image, title) in enumerate(zip(images, titles)):
        axis = axes.flat[index]
        axis.imshow(_normalize_for_display(image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _encode_grayscale_image(model: Any, image: torch.Tensor, *, device: str, latent_mode: str) -> torch.Tensor:
    if image.dim() != 3 or int(image.shape[0]) != 1:
        raise ValueError(f"expected grayscale CHW image with one channel, got {tuple(image.shape)}")
    batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    batch_rgb = tensor_gray_to_rgb(batch)
    vae_input = batch_rgb * 2.0 - 1.0
    encoded = model.encode(vae_input)
    if latent_mode == "sample":
        latent = encoded.latent_dist.sample()
    elif latent_mode == "mode":
        latent = encoded.latent_dist.mode()
    else:
        raise ValueError(f"latent_mode must be 'mode' or 'sample', got {latent_mode!r}")
    return latent.detach()


@torch.no_grad()
def _decode_latent(model: Any, latent: torch.Tensor) -> torch.Tensor:
    decoded = model.decode(latent).sample
    recon_rgb = ((decoded.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    recon_gray = tensor_rgb_to_gray(recon_rgb)
    return recon_gray[0].detach().cpu()


def run_interpolation(
    *,
    manifest_path: Path,
    sample_index: int,
    model_id: str,
    output_dir: Path,
    num_levels: int,
    cutoffs: tuple[float, ...],
    transition_width: float,
    interpolation_count: int,
    latent_mode: str,
    device: str,
) -> dict[str, Any]:
    if interpolation_count <= 0:
        raise ValueError(f"interpolation_count must be positive, got {interpolation_count!r}")

    dataset = NpzImageDataset.from_manifest(manifest_path, channel_mode="keep")
    try:
        sample = dataset[int(sample_index)]
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        transform = MultiScaleFrequencyTargetTransform(
            num_levels=int(num_levels),
            cutoffs=cutoffs,
            transition_width=float(transition_width),
        )
        transformed = transform(image)
        target_scales = list(transformed["target_scales"])
        target_bands = list(transformed["target_bands"])

        AutoencoderKL = load_autoencoder_cls()
        model = AutoencoderKL.from_pretrained(model_id).to(device)
        model.eval()

        output_dir.mkdir(parents=True, exist_ok=True)

        latent_entries: list[dict[str, Any]] = []
        for level_index, scale_image in enumerate(target_scales, start=1):
            latent = _encode_grayscale_image(
                model,
                scale_image,
                device=device,
                latent_mode=latent_mode,
            )
            latent_entries.append(
                {
                    "name": f"scale_{level_index}",
                    "kind": "original",
                    "level_index": int(level_index),
                    "alpha": None,
                    "latent": latent,
                }
            )

        interpolated_entries: list[dict[str, Any]] = []
        for left_index in range(len(latent_entries) - 1):
            left = latent_entries[left_index]
            right = latent_entries[left_index + 1]
            for interp_index in range(1, interpolation_count + 1):
                alpha = float(interp_index) / float(interpolation_count + 1)
                latent = (1.0 - alpha) * left["latent"] + alpha * right["latent"]
                interpolated_entries.append(
                    {
                        "name": f"{left['name']}_to_{right['name']}_interp_{interp_index}",
                        "kind": "interpolated",
                        "left_level_index": int(left["level_index"]),
                        "right_level_index": int(right["level_index"]),
                        "alpha": alpha,
                        "latent": latent,
                    }
                )

        all_entries = latent_entries + interpolated_entries

        scale_images_for_grid: list[np.ndarray] = []
        scale_titles_for_grid: list[str] = []
        for level_index, scale_image in enumerate(target_scales, start=1):
            image_np = scale_image[0].detach().cpu().numpy()
            scale_images_for_grid.append(image_np)
            scale_titles_for_grid.append(f"target_scale_{level_index}")
            _save_single_image(
                output_dir / f"target_scale_{level_index:02d}.png",
                image_np,
                title=f"target_scale_{level_index}",
            )
        for level_index, band_image in enumerate(target_bands, start=1):
            _save_single_image(
                output_dir / f"target_band_{level_index:02d}.png",
                band_image[0].detach().cpu().numpy(),
                title=f"target_band_{level_index}",
            )

        decoded_images_for_grid: list[np.ndarray] = []
        decoded_titles_for_grid: list[str] = []
        summaries: list[dict[str, Any]] = []
        for entry in all_entries:
            decoded = _decode_latent(model, entry["latent"])
            decoded_np = decoded[0].numpy()
            decoded_images_for_grid.append(decoded_np)
            decoded_titles_for_grid.append(str(entry["name"]))
            _save_single_image(
                output_dir / f"{entry['name']}.png",
                decoded_np,
                title=str(entry["name"]),
            )
            latent_tensor = entry["latent"].detach().cpu()
            summaries.append(
                {
                    "name": str(entry["name"]),
                    "kind": str(entry["kind"]),
                    "alpha": entry["alpha"],
                    "latent_shape": list(latent_tensor.shape),
                    "latent_mean": float(latent_tensor.mean().item()),
                    "latent_std": float(latent_tensor.std().item()),
                    "decoded_min": float(decoded.min().item()),
                    "decoded_max": float(decoded.max().item()),
                }
            )

        _save_single_image(
            output_dir / "input_image.png",
            image[0].detach().cpu().numpy(),
            title=f"input#{sample_index}",
        )
        _save_grid(
            [image[0].detach().cpu().numpy(), *scale_images_for_grid],
            [f"input#{sample_index}", *scale_titles_for_grid],
            output_dir / "frequency_path_grid.png",
            cols=3,
        )
        _save_grid(
            decoded_images_for_grid,
            decoded_titles_for_grid,
            output_dir / "decoded_latents_grid.png",
            cols=3,
        )

        summary = {
            "manifest_path": str(manifest_path),
            "sample_index": int(sample_index),
            "sample_id": int(sample["sample_id"]) if "sample_id" in sample else None,
            "model_id": str(model_id),
            "latent_mode": str(latent_mode),
            "num_levels": int(num_levels),
            "cutoffs": [float(value) for value in cutoffs],
            "transition_width": float(transition_width),
            "interpolation_count": int(interpolation_count),
            "entries": summaries,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    finally:
        dataset.close()


def _parse_cutoffs(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(float(part) for part in parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one image into cumulative frequency-path targets, encode each path image with a pretrained "
            "AutoencoderKL, interpolate between adjacent path latents, and decode all original/interpolated latents."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/celeba/celeba_train_gray_176_lm.json"),
        help="Path to the image manifest.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        required=True,
        help="Dataset index of the image to process.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers AutoencoderKL model id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output images and summary JSON.",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=5,
        help="Number of cumulative frequency levels.",
    )
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="0.04,0.08,0.16,0.32",
        help="Comma-separated explicit cutoffs for the cumulative frequency path.",
    )
    parser.add_argument(
        "--transition-width",
        type=float,
        default=0.05,
        help="Smooth radial transition width used by the frequency transform.",
    )
    parser.add_argument(
        "--interpolation-count",
        type=int,
        default=2,
        help="How many latent interpolation points to place between each adjacent path latent.",
    )
    parser.add_argument(
        "--latent-mode",
        type=str,
        default="mode",
        choices=("mode", "sample"),
        help="Whether to use the latent posterior mode or sample.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_interpolation(
        manifest_path=args.manifest.resolve(),
        sample_index=int(args.sample_index),
        model_id=str(args.model_id),
        output_dir=args.output_dir.resolve(),
        num_levels=int(args.num_levels),
        cutoffs=_parse_cutoffs(str(args.cutoffs)),
        transition_width=float(args.transition_width),
        interpolation_count=int(args.interpolation_count),
        latent_mode=str(args.latent_mode),
        device=str(args.device),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
