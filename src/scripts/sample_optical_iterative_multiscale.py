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

from scripts.train_optical_iterative_multiscale import (
    _build_condition_batch,
    _build_dataset_and_loader,
    _build_model,
    _load_config,
    _move_batch_to_device,
)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _tensor_to_image(x: torch.Tensor) -> torch.Tensor:
    image = x.detach().cpu()
    if image.is_complex():
        image = image.abs()
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3 and int(image.shape[0]) == 1:
        image = image[0]
    return image.to(dtype=torch.float32)


def _normalize_mean_power(image: torch.Tensor) -> torch.Tensor:
    mean_power = image.mean().clamp_min(1.0e-8)
    return image / mean_power


def _save_panel(path: Path, image: torch.Tensor, *, cmap: str, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(_normalize_mean_power(_tensor_to_image(image)), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_step_grid(
    path: Path,
    *,
    target: torch.Tensor,
    predictions: tuple[torch.Tensor, ...],
    states: tuple[torch.Tensor, ...],
) -> None:
    total_items = 1 + len(predictions) + len(states)
    cols = 3
    rows = math.ceil(total_items / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_list = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]

    panels: list[tuple[str, torch.Tensor]] = [("target", target)]
    panels.extend((f"pred_{index:02d}", tensor) for index, tensor in enumerate(predictions, start=1))
    panels.extend((f"state_{index:02d}", tensor) for index, tensor in enumerate(states, start=1))

    for axis, (title, tensor) in zip(axes_list, panels):
        axis.imshow(_normalize_mean_power(_tensor_to_image(tensor)), cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    for axis in axes_list[len(panels) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _summarize_label(label_value: torch.Tensor | None) -> int | list[float] | None:
    if label_value is None:
        return None
    label_tensor = label_value.detach().cpu()
    if label_tensor.numel() == 1:
        return int(label_tensor.reshape(-1)[0])
    return label_tensor.reshape(-1).tolist()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample iterative multiscale optical model predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the training checkpoint.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Optional dataset manifest override.")
    parser.add_argument("--sample-index", type=int, required=True, help="Dataset sample index.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save sample images.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(args.checkpoint.resolve(), device)
    config = checkpoint["config"]
    if args.data_manifest is not None:
        config["dataset"]["manifest_path"] = str(args.data_manifest.resolve())
    config["dataset"]["batch_size"] = 1
    config["dataset"]["shuffle"] = False
    config["dataset"]["num_workers"] = 0
    config["dataset"]["drop_last"] = False

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root
    dataset, _ = _build_dataset_and_loader(config, config_dir=config_dir, repo_root=repo_root)
    sample = dataset[int(args.sample_index)]
    batch = next(iter(torch.utils.data.DataLoader([sample], batch_size=1)))
    batch = _move_batch_to_device(batch, device)

    model = _build_model(config, sample_item=sample).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    latent = batch["latent"].to(device=device, dtype=torch.float32)
    condition = _build_condition_batch(batch, config=config, device=device)
    condition_mode = config["encoder"].get("condition_mode")
    iterative_cfg = dict(config["iterative"])
    num_steps = int(iterative_cfg["num_steps"])

    with torch.no_grad():
        output = model(
            latent=latent,
            condition=condition if condition_mode == "attribute_vector" else None,
            class_labels=condition if condition_mode != "attribute_vector" else None,
            num_steps=num_steps,
            detach_prev_state=bool(iterative_cfg.get("detach_prev_state", False)),
        )

    predictions = output["predictions"]
    states = output["states"]
    target = batch[f"target_scale_{num_steps}"]
    prefix_base = f"sample_{int(args.sample_index):04d}"

    _save_panel(
        args.output_dir / f"{prefix_base}_target.png",
        target,
        cmap="gray",
        title="target_final",
    )
    for index, prediction in enumerate(predictions, start=1):
        _save_panel(
            args.output_dir / f"{prefix_base}_step_{index:02d}.png",
            prediction,
            cmap="gray",
            title=f"prediction_step_{index}",
        )
    for index, state in enumerate(states, start=1):
        _save_panel(
            args.output_dir / f"{prefix_base}_state_{index:02d}.png",
            state,
            cmap="gray",
            title=f"state_step_{index}",
        )
    _save_step_grid(
        args.output_dir / f"{prefix_base}_overview.png",
        target=target,
        predictions=predictions,
        states=states,
    )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sample_index": int(args.sample_index),
        "sample_id": int(batch["sample_id"][0]),
        "output_dir": str(args.output_dir.resolve()),
        "label": _summarize_label(batch["label"][0] if "label" in batch else None),
        "num_steps": num_steps,
    }
    (args.output_dir / f"{prefix_base}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
