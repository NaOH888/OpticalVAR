from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import NpzImageDataset
from vae import build_cvae


def _resolve_path(path_value: str, *, config_dir: Path, repo_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return (repo_root / candidate).resolve()


def _tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    image = tensor.detach().cpu().float()
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3 and int(image.shape[0]) == 1:
        image = image[0]
    return image.clamp(0.0, 1.0)


def _save_panel(path: Path, title: str, image: torch.Tensor) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(_tensor_to_image(image), cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_grid(path: Path, title: str, images: list[torch.Tensor]) -> None:
    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = list(axes)
    else:
        axes_list = list(axes.reshape(-1))
    for idx, ax in enumerate(axes_list):
        if idx < len(images):
            ax.imshow(_tensor_to_image(images[idx]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"sample {idx}")
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample reconstructions or prior generations from a trained cVAE.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to cVAE checkpoint.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Optional dataset manifest override.")
    parser.add_argument("--sample-index", type=int, default=None, help="Dataset sample index for reconstruction mode.")
    parser.add_argument("--random-prior", action="store_true", help="Enable prior sampling mode.")
    parser.add_argument("--label", type=int, default=None, help="Class label for class-index prior sampling.")
    parser.add_argument("--condition-index", type=int, default=None, help="Dataset index whose condition vector is reused in attribute mode.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of prior samples.")
    parser.add_argument("--latent-seed", type=int, default=42, help="Random seed for prior z.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fixed_mode = args.sample_index is not None
    random_mode = bool(args.random_prior)
    if fixed_mode == random_mode:
        raise ValueError("Choose exactly one mode: --sample-index for reconstruction or --random-prior for prior sampling.")

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    config = checkpoint["config"]
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = args.checkpoint.resolve().parent
    manifest_path = (
        args.data_manifest.resolve()
        if args.data_manifest is not None
        else _resolve_path(config["dataset"]["manifest_path"], config_dir=config_dir, repo_root=repo_root)
    )
    dataset = NpzImageDataset.from_manifest(manifest_path, channel_mode=str(config["dataset"]["channel_mode"]))
    model = build_cvae(dict(config["model"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_mode = str(config["model"].get("condition_mode", "class_index"))
    latent_dim = int(config["model"]["latent_height"]) * int(config["model"]["latent_width"])

    try:
        if fixed_mode:
            sample = dataset[int(args.sample_index)]
            image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
            label = sample["label"].unsqueeze(0) if sample["label"].dim() == 0 else sample["label"].unsqueeze(0)
            if condition_mode == "attribute_vector":
                label = label.to(device=device, dtype=torch.float32)
            else:
                label = label.to(device=device, dtype=torch.long).reshape(-1)
            with torch.no_grad():
                encoded = model.encode(image, label, sample_posterior=False)
                recon = model.decode(encoded.z, label)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(args.latent_seed))
                z = torch.randn((1, latent_dim), generator=generator, dtype=torch.float32).to(device=device)
                prior = model.decode(z, label)
            _save_panel(args.output_dir / f"sample_{int(args.sample_index):04d}_target.png", "target", image)
            _save_panel(args.output_dir / f"sample_{int(args.sample_index):04d}_recon.png", "reconstruction", recon)
            _save_panel(args.output_dir / f"sample_{int(args.sample_index):04d}_prior.png", "prior_same_condition", prior)
            print(
                json.dumps(
                    {
                        "mode": "reconstruct",
                        "sample_index": int(args.sample_index),
                        "sample_id": int(sample["sample_id"]),
                        "output_dir": str(args.output_dir.resolve()),
                        "condition_mode": condition_mode,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.latent_seed))
        z = torch.randn((int(args.num_samples), latent_dim), generator=generator, dtype=torch.float32).to(device=device)
        if condition_mode == "attribute_vector":
            if args.condition_index is None:
                raise ValueError("--condition-index is required for attribute_vector prior sampling")
            condition_sample = dataset[int(args.condition_index)]
            condition = condition_sample["label"].to(device=device, dtype=torch.float32).unsqueeze(0).repeat(int(args.num_samples), 1)
            title = f"prior condition-index {int(args.condition_index)}"
            prefix = f"condition_{int(args.condition_index):04d}"
        else:
            if args.label is None:
                raise ValueError("--label is required for class_index prior sampling")
            condition = torch.full((int(args.num_samples),), int(args.label), device=device, dtype=torch.long)
            title = f"prior label {int(args.label)}"
            prefix = f"label_{int(args.label):02d}"
        with torch.no_grad():
            decoded = model.decode(z, condition)
        images = [decoded[index : index + 1] for index in range(int(args.num_samples))]
        _save_grid(args.output_dir / f"{prefix}_grid.png", title, images)
        print(
            json.dumps(
                {
                    "mode": "random_prior",
                    "output_dir": str(args.output_dir.resolve()),
                    "condition_mode": condition_mode,
                    "num_samples": int(args.num_samples),
                    "latent_seed": int(args.latent_seed),
                    "label": None if args.label is None else int(args.label),
                    "condition_index": None if args.condition_index is None else int(args.condition_index),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
