from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import NpzImageDataset, ReferencedImageLatentDataset


def _resolve_path(path_value: str, *, cwd: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (cwd / candidate).resolve()


def _load_image_dataset(
    manifest_path: Path,
    *,
    channel_mode: str,
    max_items: int | None,
) -> Any:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "image_manifest_path" in payload:
        return ReferencedImageLatentDataset.from_latent_manifest(
            manifest_path,
            max_items=max_items,
            channel_mode=channel_mode,
        )
    return NpzImageDataset.from_manifest(
        manifest_path,
        max_items=max_items,
        channel_mode=channel_mode,
    )


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.dim() != 3:
        raise ValueError(f"image must be [H,W] or [C,H,W], got {tuple(image.shape)}")
    return image.to(dtype=torch.float32)


def _build_normalized_radius(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square())
    radius = torch.fft.fftshift(radius)
    return radius / torch.clamp_min(radius.max(), 1.0e-12)


def _compute_average_power_map(dataset: Any) -> torch.Tensor:
    accumulated_power: torch.Tensor | None = None
    sample_count = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        normalized = _normalize_image(image)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(normalized, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )
        power_map = spectrum.abs().square().mean(dim=0)
        if accumulated_power is None:
            accumulated_power = torch.zeros_like(power_map)
        accumulated_power += power_map
        sample_count += 1
    if accumulated_power is None or sample_count <= 0:
        raise RuntimeError("dataset is empty, cannot compute frequency cutoffs")
    return accumulated_power / float(sample_count)


def _compute_equal_power_cutoffs(power_map: torch.Tensor, *, num_levels: int) -> list[float]:
    if num_levels <= 1:
        return []
    height, width = int(power_map.shape[-2]), int(power_map.shape[-1])
    radius = _build_normalized_radius(height, width, device=power_map.device, dtype=power_map.dtype)
    flat_radius = radius.reshape(-1)
    flat_power = power_map.reshape(-1)
    order = torch.argsort(flat_radius)
    sorted_radius = flat_radius[order]
    sorted_power = flat_power[order]
    cumulative_power = torch.cumsum(sorted_power, dim=0)
    total_power = float(cumulative_power[-1].item())
    if total_power <= 0.0:
        raise RuntimeError("total spectral power must be positive")

    cutoffs: list[float] = []
    for level_idx in range(1, num_levels):
        target_power = total_power * float(level_idx) / float(num_levels)
        cutoff_index = int(torch.searchsorted(cumulative_power, torch.tensor(target_power, dtype=cumulative_power.dtype)))
        cutoff_index = min(max(cutoff_index, 0), int(sorted_radius.numel()) - 1)
        cutoffs.append(float(sorted_radius[cutoff_index].item()))
    return cutoffs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute equal-power radial frequency cutoffs from an image manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the source image manifest.")
    parser.add_argument("--num-levels", type=int, required=True, help="Number of multiscale levels.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output cutoff JSON.")
    parser.add_argument("--channel-mode", type=str, default="keep", help="Dataset channel mode for manifest loading.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional dataset subset size for quick estimation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = _resolve_path(str(args.manifest), cwd=Path.cwd())
    output_path = _resolve_path(str(args.output), cwd=Path.cwd())
    dataset = _load_image_dataset(
        manifest_path,
        channel_mode=str(args.channel_mode),
        max_items=args.max_items,
    )
    average_power = _compute_average_power_map(dataset)
    cutoffs = _compute_equal_power_cutoffs(average_power, num_levels=int(args.num_levels))

    payload = {
        "mode": "equal_power",
        "manifest_path": str(manifest_path),
        "num_levels": int(args.num_levels),
        "num_items": int(len(dataset)),
        "cutoffs": cutoffs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
