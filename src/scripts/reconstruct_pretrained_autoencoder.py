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

from optical.data import NpzImageDataset
from scripts._pretrained_autoencoder_utils import l2_loss as _l2_loss
from scripts._pretrained_autoencoder_utils import load_autoencoder_cls as _load_autoencoder_cls
from scripts._pretrained_autoencoder_utils import psnr as _psnr
from scripts._pretrained_autoencoder_utils import tensor_gray_to_rgb as _tensor_gray_to_rgb
from scripts._pretrained_autoencoder_utils import tensor_rgb_to_gray as _tensor_rgb_to_gray


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)

def _save_grid(
    images: list[np.ndarray],
    titles: list[str],
    path: Path,
    *,
    cols: int = 4,
) -> None:
    rows = int(math.ceil(len(images) / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    axes = axes.reshape(rows, cols)
    for axis in axes.flat:
        axis.axis("off")
    for index, (image, title) in enumerate(zip(images, titles)):
        axis = axes.flat[index]
        axis.imshow(_normalize_image(image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _parse_indices(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    return result


@torch.no_grad()
def reconstruct_samples(
    *,
    manifest_path: Path,
    model_id: str,
    output_dir: Path,
    sample_indices: list[int],
    metric_max_items: int,
    metric_fraction: float | None,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    AutoencoderKL = _load_autoencoder_cls()
    dataset = NpzImageDataset.from_manifest(manifest_path, channel_mode="keep")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = AutoencoderKL.from_pretrained(model_id)
        model = model.to(device)
        model.eval()

        chosen_indices = sample_indices if sample_indices else list(range(min(8, len(dataset))))
        grid_images: list[np.ndarray] = []
        grid_titles: list[str] = []
        per_sample: list[dict[str, Any]] = []
        l1_values: list[float] = []
        l2_values: list[float] = []
        psnr_values: list[float] = []

        if metric_fraction is not None:
            if not 0.0 < float(metric_fraction) <= 1.0:
                raise ValueError(f"metric_fraction must be in (0,1], got {metric_fraction!r}")
            metric_limit = max(1, int(round(len(dataset) * float(metric_fraction))))
        else:
            metric_limit = min(int(metric_max_items), len(dataset))
        current_batch: list[torch.Tensor] = []
        current_indices: list[int] = []

        def flush_batch() -> None:
            if not current_batch:
                return
            batch = torch.stack(current_batch, dim=0).to(device=device, dtype=torch.float32)
            batch_rgb = _tensor_gray_to_rgb(batch)
            vae_input = batch_rgb * 2.0 - 1.0
            encoded = model.encode(vae_input)
            latents = encoded.latent_dist.mode()
            decoded = model.decode(latents).sample
            recon_rgb = ((decoded.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
            recon_gray = _tensor_rgb_to_gray(recon_rgb).cpu()
            originals = batch.cpu()
            for batch_offset, dataset_index in enumerate(current_indices):
                original = originals[batch_offset]
                recon = recon_gray[batch_offset]
                l1_value = float(torch.mean(torch.abs(recon - original)).item())
                l2_value = _l2_loss(recon, original)
                psnr_value = _psnr(recon, original)
                if dataset_index in chosen_indices:
                    original_np = original[0].numpy()
                    recon_np = recon[0].numpy()
                    grid_images.extend([original_np, recon_np])
                    grid_titles.extend([f"orig#{dataset_index}", f"recon#{dataset_index}"])
                    per_sample.append(
                        {
                            "sample_index": int(dataset_index),
                            "sample_id": int(dataset[dataset_index]["sample_id"]),
                            "l1": l1_value,
                            "l2": l2_value,
                            "psnr": psnr_value,
                        }
                    )
                l1_values.append(l1_value)
                l2_values.append(l2_value)
                psnr_values.append(psnr_value)
            current_batch.clear()
            current_indices.clear()

        for dataset_index in range(metric_limit):
            sample = dataset[dataset_index]
            image = sample["image"]
            if image.dim() != 3 or int(image.shape[0]) != 1:
                raise ValueError(f"expected grayscale image [1,H,W], got {tuple(image.shape)}")
            current_batch.append(image)
            current_indices.append(dataset_index)
            if len(current_batch) >= int(batch_size):
                flush_batch()
        flush_batch()

        _save_grid(grid_images, grid_titles, output_dir / "reconstruction_grid.png", cols=4)
        summary = {
            "manifest_path": str(manifest_path),
            "model_id": str(model_id),
            "device": str(device),
            "metric_items": int(metric_limit),
            "metric_fraction": None if metric_fraction is None else float(metric_fraction),
            "sample_indices": [int(index) for index in chosen_indices],
            "mean_l1": float(np.mean(l1_values)) if l1_values else None,
            "std_l1": float(np.std(l1_values)) if l1_values else None,
            "mean_l2": float(np.mean(l2_values)) if l2_values else None,
            "std_l2": float(np.std(l2_values)) if l2_values else None,
            "mean_psnr": float(np.mean(psnr_values)) if psnr_values else None,
            "std_psnr": float(np.std(psnr_values)) if psnr_values else None,
            "samples": per_sample,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    finally:
        dataset.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct dataset samples with a pretrained autoencoder baseline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/celeba/celeba_train_gray_176_lm.json"),
        help="Path to the image manifest.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Hugging Face diffusers AutoencoderKL model id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for reconstruction outputs.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated dataset indices to render in the grid.",
    )
    parser.add_argument(
        "--metric-max-items",
        type=int,
        default=64,
        help="How many dataset items to use when averaging L1 / L2 / PSNR when metric_fraction is not set.",
    )
    parser.add_argument(
        "--metric-fraction",
        type=float,
        default=None,
        help="Optional dataset fraction in (0,1] used for averaging metrics. Overrides metric_max_items.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for VAE reconstruction.",
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
    summary = reconstruct_samples(
        manifest_path=args.manifest.resolve(),
        model_id=args.model_id,
        output_dir=args.output_dir.resolve(),
        sample_indices=_parse_indices(args.sample_indices),
        metric_max_items=int(args.metric_max_items),
        metric_fraction=args.metric_fraction,
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
