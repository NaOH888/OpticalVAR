from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from scripts.train_optical_iterative_multiscale import _build_condition_batch, _build_dataset_and_loader, _build_model


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _close_dataset(dataset: Any) -> None:
    close_fn = getattr(dataset, "close", None)
    if callable(close_fn):
        close_fn()
        return
    base_dataset = getattr(dataset, "base_dataset", None)
    if base_dataset is not None:
        base_close = getattr(base_dataset, "close", None)
        if callable(base_close):
            base_close()


def _tensor_to_display_image(tensor: torch.Tensor, *, mode: str) -> torch.Tensor:
    image = tensor.detach().cpu().float()
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3:
        if int(image.shape[0]) == 1:
            image = image[0]
        else:
            image = image.mean(dim=0)
    if mode == "phase":
        phase_period = float(2.0 * math.pi)
        return torch.remainder(image, phase_period) / phase_period
    mean_power = image.mean().clamp_min(1.0e-8)
    return image / mean_power


def _save_panel(path: Path, title: str, image: torch.Tensor, *, mode: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(_tensor_to_display_image(image, mode=mode), cmap="twilight" if mode == "phase" else "gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_grid(path: Path, title: str, images: list[torch.Tensor], *, mode: str) -> None:
    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = list(axes)
    else:
        axes_list = list(axes.reshape(-1))
    cmap = "twilight" if mode == "phase" else "gray"
    for idx, axis in enumerate(axes_list):
        if idx < len(images):
            axis.imshow(_tensor_to_display_image(images[idx], mode=mode), cmap=cmap)
            axis.set_title(f"sample {idx}")
        axis.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).item())


def _pick_candidate_indices(
    *,
    dataset_length: int,
    anchor_index: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    population = [index for index in range(dataset_length) if index != anchor_index]
    if not population:
        raise ValueError("dataset must contain at least two samples for iterative diagnosis")
    rng = random.Random(int(seed))
    if len(population) <= int(num_samples):
        return population
    return rng.sample(population, k=int(num_samples))


def _summarize_label(label: torch.Tensor) -> int | list[float]:
    label_cpu = label.detach().cpu()
    if label_cpu.numel() == 1:
        return int(label_cpu.reshape(-1)[0])
    return label_cpu.reshape(-1).tolist()


def _build_condition_from_sample(sample: dict[str, Any], *, config: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if "label" not in sample:
        return None
    label = sample["label"] if isinstance(sample["label"], torch.Tensor) else torch.as_tensor(sample["label"])
    batch = {"label": label.unsqueeze(0).to(device)}
    return _build_condition_batch(batch, config=config, device=device)


def _run_model(
    model: Any,
    *,
    latent: torch.Tensor,
    condition: torch.Tensor | None,
    condition_mode: str | None,
    num_steps: int,
    detach_prev_state: bool,
) -> dict[str, tuple[torch.Tensor, ...]]:
    with torch.no_grad():
        output = model(
            latent=latent,
            condition=condition if condition_mode == "attribute_vector" else None,
            class_labels=condition if condition_mode != "attribute_vector" else None,
            num_steps=num_steps,
            detach_prev_state=detach_prev_state,
        )
    return {
        "control_maps": tuple(item.detach().cpu() for item in output["control_maps"]),
        "states": tuple(item.detach().cpu() for item in output["states"]),
        "predictions": tuple(item.detach().cpu() for item in output["predictions"]),
    }


def _per_step_mads(anchor_items: tuple[torch.Tensor, ...], sample_items: tuple[torch.Tensor, ...]) -> list[float]:
    if len(anchor_items) != len(sample_items):
        raise ValueError("anchor_items and sample_items must have same length")
    return [_mean_abs_diff(anchor, sample) for anchor, sample in zip(anchor_items, sample_items)]


def _mean_per_step(records: list[list[float]]) -> list[float]:
    if not records:
        return []
    num_steps = len(records[0])
    return [
        float(sum(record[step_idx] for record in records) / len(records))
        for step_idx in range(num_steps)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose latent/condition usage for iterative optical model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to iterative training checkpoint.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Optional manifest override.")
    parser.add_argument("--anchor-index", type=int, default=0, help="Anchor sample index.")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of comparison samples.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for candidate indices.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary and grids.")
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
    source_light_mode = str(config["optical"]["source"]["light_mode"]).lower()
    repo_root = Path(__file__).resolve().parents[2]
    dataset, _ = _build_dataset_and_loader(config, config_dir=repo_root, repo_root=repo_root)

    try:
        anchor_index = int(args.anchor_index)
        anchor_sample = dataset[anchor_index]
        candidate_indices = _pick_candidate_indices(
            dataset_length=len(dataset),
            anchor_index=anchor_index,
            num_samples=int(args.num_samples),
            seed=int(args.seed),
        )
        candidate_samples = [dataset[index] for index in candidate_indices]

        model = _build_model(config, sample_item=anchor_sample).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        iterative_cfg = dict(config["iterative"])
        num_steps = int(iterative_cfg["num_steps"])
        detach_prev_state = bool(iterative_cfg.get("detach_prev_state", False))
        condition_mode = config["encoder"].get("condition_mode")

        anchor_latent = anchor_sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
        anchor_condition = _build_condition_from_sample(anchor_sample, config=config, device=device)
        anchor_outputs = _run_model(
            model,
            latent=anchor_latent,
            condition=anchor_condition,
            condition_mode=condition_mode,
            num_steps=num_steps,
            detach_prev_state=detach_prev_state,
        )

        latent_only_results: list[dict[str, Any]] = []
        condition_only_results: list[dict[str, Any]] = []
        latent_prediction_grids: list[list[torch.Tensor]] = [[anchor_outputs["predictions"][step]] for step in range(num_steps)]
        latent_state_grids: list[list[torch.Tensor]] = [[anchor_outputs["states"][step]] for step in range(num_steps)]
        latent_control_grids: list[list[torch.Tensor]] = [[anchor_outputs["control_maps"][step]] for step in range(num_steps)]
        condition_prediction_grids: list[list[torch.Tensor]] = [[anchor_outputs["predictions"][step]] for step in range(num_steps)]
        condition_state_grids: list[list[torch.Tensor]] = [[anchor_outputs["states"][step]] for step in range(num_steps)]
        condition_control_grids: list[list[torch.Tensor]] = [[anchor_outputs["control_maps"][step]] for step in range(num_steps)]

        for sample_index, sample in zip(candidate_indices, candidate_samples):
            sample_latent = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
            sample_condition = _build_condition_from_sample(sample, config=config, device=device)

            latent_outputs = _run_model(
                model,
                latent=sample_latent,
                condition=anchor_condition,
                condition_mode=condition_mode,
                num_steps=num_steps,
                detach_prev_state=detach_prev_state,
            )
            latent_control_mads = _per_step_mads(anchor_outputs["control_maps"], latent_outputs["control_maps"])
            latent_state_mads = _per_step_mads(anchor_outputs["states"], latent_outputs["states"])
            latent_prediction_mads = _per_step_mads(anchor_outputs["predictions"], latent_outputs["predictions"])
            latent_only_results.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": int(sample["sample_id"]),
                    "control_map_mad_per_step": latent_control_mads,
                    "state_mad_per_step": latent_state_mads,
                    "prediction_mad_per_step": latent_prediction_mads,
                    "final_detector_mad": latent_prediction_mads[-1],
                }
            )
            for step in range(num_steps):
                latent_prediction_grids[step].append(latent_outputs["predictions"][step])
                latent_state_grids[step].append(latent_outputs["states"][step])
                latent_control_grids[step].append(latent_outputs["control_maps"][step])

            condition_outputs = _run_model(
                model,
                latent=anchor_latent,
                condition=sample_condition,
                condition_mode=condition_mode,
                num_steps=num_steps,
                detach_prev_state=detach_prev_state,
            )
            condition_control_mads = _per_step_mads(anchor_outputs["control_maps"], condition_outputs["control_maps"])
            condition_state_mads = _per_step_mads(anchor_outputs["states"], condition_outputs["states"])
            condition_prediction_mads = _per_step_mads(anchor_outputs["predictions"], condition_outputs["predictions"])
            condition_only_results.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": int(sample["sample_id"]),
                    "label": _summarize_label(sample["label"]),
                    "control_map_mad_per_step": condition_control_mads,
                    "state_mad_per_step": condition_state_mads,
                    "prediction_mad_per_step": condition_prediction_mads,
                    "final_detector_mad": condition_prediction_mads[-1],
                }
            )
            for step in range(num_steps):
                condition_prediction_grids[step].append(condition_outputs["predictions"][step])
                condition_state_grids[step].append(condition_outputs["states"][step])
                condition_control_grids[step].append(condition_outputs["control_maps"][step])

        _save_panel(args.output_dir / "anchor_target.png", "anchor_target", anchor_sample["target_final"], mode="intensity")
        for step_idx in range(1, num_steps + 1):
            _save_panel(
                args.output_dir / f"anchor_target_scale_{step_idx:02d}.png",
                f"anchor_target_scale_{step_idx}",
                anchor_sample[f"target_scale_{step_idx}"],
                mode="intensity",
            )
            _save_panel(
                args.output_dir / f"anchor_control_step_{step_idx:02d}.png",
                f"anchor_control_step_{step_idx}",
                anchor_outputs["control_maps"][step_idx - 1],
                mode="phase" if source_light_mode == "phase" else "intensity",
            )
            _save_panel(
                args.output_dir / f"anchor_state_step_{step_idx:02d}.png",
                f"anchor_state_step_{step_idx}",
                anchor_outputs["states"][step_idx - 1],
                mode="intensity",
            )
            _save_panel(
                args.output_dir / f"anchor_prediction_step_{step_idx:02d}.png",
                f"anchor_prediction_step_{step_idx}",
                anchor_outputs["predictions"][step_idx - 1],
                mode="intensity",
            )
            _save_grid(
                args.output_dir / f"latent_vary_control_step_{step_idx:02d}.png",
                f"fixed condition, varying latent: control step {step_idx}",
                latent_control_grids[step_idx - 1],
                mode="phase" if source_light_mode == "phase" else "intensity",
            )
            _save_grid(
                args.output_dir / f"latent_vary_state_step_{step_idx:02d}.png",
                f"fixed condition, varying latent: state step {step_idx}",
                latent_state_grids[step_idx - 1],
                mode="intensity",
            )
            _save_grid(
                args.output_dir / f"latent_vary_prediction_step_{step_idx:02d}.png",
                f"fixed condition, varying latent: prediction step {step_idx}",
                latent_prediction_grids[step_idx - 1],
                mode="intensity",
            )
            _save_grid(
                args.output_dir / f"condition_vary_control_step_{step_idx:02d}.png",
                f"fixed latent, varying condition: control step {step_idx}",
                condition_control_grids[step_idx - 1],
                mode="phase" if source_light_mode == "phase" else "intensity",
            )
            _save_grid(
                args.output_dir / f"condition_vary_state_step_{step_idx:02d}.png",
                f"fixed latent, varying condition: state step {step_idx}",
                condition_state_grids[step_idx - 1],
                mode="intensity",
            )
            _save_grid(
                args.output_dir / f"condition_vary_prediction_step_{step_idx:02d}.png",
                f"fixed latent, varying condition: prediction step {step_idx}",
                condition_prediction_grids[step_idx - 1],
                mode="intensity",
            )

        latent_control_records = [item["control_map_mad_per_step"] for item in latent_only_results]
        latent_state_records = [item["state_mad_per_step"] for item in latent_only_results]
        latent_prediction_records = [item["prediction_mad_per_step"] for item in latent_only_results]
        condition_control_records = [item["control_map_mad_per_step"] for item in condition_only_results]
        condition_state_records = [item["state_mad_per_step"] for item in condition_only_results]
        condition_prediction_records = [item["prediction_mad_per_step"] for item in condition_only_results]

        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest_path": str(config["dataset"]["manifest_path"]),
            "anchor_index": anchor_index,
            "anchor_sample_id": int(anchor_sample["sample_id"]),
            "anchor_label": _summarize_label(anchor_sample["label"]),
            "candidate_indices": candidate_indices,
            "latent_only_mean": {
                "control_map_mad_per_step": _mean_per_step(latent_control_records),
                "state_mad_per_step": _mean_per_step(latent_state_records),
                "prediction_mad_per_step": _mean_per_step(latent_prediction_records),
                "final_detector_mad": float(sum(item["final_detector_mad"] for item in latent_only_results) / len(latent_only_results)),
            },
            "condition_only_mean": {
                "control_map_mad_per_step": _mean_per_step(condition_control_records),
                "state_mad_per_step": _mean_per_step(condition_state_records),
                "prediction_mad_per_step": _mean_per_step(condition_prediction_records),
                "final_detector_mad": float(sum(item["final_detector_mad"] for item in condition_only_results) / len(condition_only_results)),
            },
            "latent_only": latent_only_results,
            "condition_only": condition_only_results,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        _close_dataset(dataset)


if __name__ == "__main__":
    main()
