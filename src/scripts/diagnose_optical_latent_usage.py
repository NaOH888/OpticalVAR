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

from scripts.train_optical_multiscale import _build_dataset_and_loader, _build_model


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
        image = torch.remainder(image, phase_period) / phase_period
        return image
    mean_power = image.mean().clamp_min(1.0e-8)
    return image / mean_power


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
    for idx, ax in enumerate(axes_list):
        if idx < len(images):
            ax.imshow(_tensor_to_display_image(images[idx], mode=mode), cmap=cmap)
            ax.set_title(f"sample {idx}")
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_panel(path: Path, title: str, image: torch.Tensor, *, mode: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(_tensor_to_display_image(image, mode=mode), cmap="twilight" if mode == "phase" else "gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).item())


def _run_model(
    model: Any,
    *,
    latent: torch.Tensor,
    condition: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model(latent.unsqueeze(0), class_labels=condition.unsqueeze(0))
    return {
        "encoder_output": output["encoder_output"][0].detach().cpu(),
        "slm_input": output["slm_input"][0].detach().cpu(),
        "final_detector": output["final_detector"][0].detach().cpu(),
    }


def _pick_candidate_indices(
    *,
    dataset_length: int,
    anchor_index: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    population = [index for index in range(dataset_length) if index != anchor_index]
    if not population:
        raise ValueError("dataset must contain at least two samples for latent/condition diagnosis")
    rng = random.Random(int(seed))
    if len(population) <= int(num_samples):
        return population
    return rng.sample(population, k=int(num_samples))


def _summarize_label(label: torch.Tensor) -> int | list[float]:
    label_cpu = label.detach().cpu()
    if label_cpu.numel() == 1:
        return int(label_cpu.reshape(-1)[0])
    return label_cpu.reshape(-1).tolist()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether latent and condition pathways are being used by an optical model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the optical training checkpoint.")
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

    repo_root = Path(__file__).resolve().parents[2]
    dataset, _, _ = _build_dataset_and_loader(
        config,
        config_dir=repo_root,
        repo_root=repo_root,
    )

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

        anchor_latent = anchor_sample["latent"].to(device=device)
        anchor_condition = anchor_sample["label"].to(device=device)
        anchor_outputs = _run_model(model, latent=anchor_latent, condition=anchor_condition)

        latent_only_results: list[dict[str, Any]] = []
        condition_only_results: list[dict[str, Any]] = []
        latent_final_images = [anchor_outputs["final_detector"]]
        latent_phase_images = [anchor_outputs["encoder_output"]]
        condition_final_images = [anchor_outputs["final_detector"]]
        condition_phase_images = [anchor_outputs["encoder_output"]]

        for sample_index, sample in zip(candidate_indices, candidate_samples):
            latent_outputs = _run_model(
                model,
                latent=sample["latent"].to(device=device),
                condition=anchor_condition,
            )
            latent_only_results.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": int(sample["sample_id"]),
                    "encoder_output_mad": _mean_abs_diff(anchor_outputs["encoder_output"], latent_outputs["encoder_output"]),
                    "slm_input_mad": _mean_abs_diff(anchor_outputs["slm_input"], latent_outputs["slm_input"]),
                    "final_detector_mad": _mean_abs_diff(anchor_outputs["final_detector"], latent_outputs["final_detector"]),
                }
            )
            latent_final_images.append(latent_outputs["final_detector"])
            latent_phase_images.append(latent_outputs["encoder_output"])

            condition_outputs = _run_model(
                model,
                latent=anchor_latent,
                condition=sample["label"].to(device=device),
            )
            condition_only_results.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": int(sample["sample_id"]),
                    "label": _summarize_label(sample["label"]),
                    "encoder_output_mad": _mean_abs_diff(anchor_outputs["encoder_output"], condition_outputs["encoder_output"]),
                    "slm_input_mad": _mean_abs_diff(anchor_outputs["slm_input"], condition_outputs["slm_input"]),
                    "final_detector_mad": _mean_abs_diff(anchor_outputs["final_detector"], condition_outputs["final_detector"]),
                }
            )
            condition_final_images.append(condition_outputs["final_detector"])
            condition_phase_images.append(condition_outputs["encoder_output"])

        stage_sensitivity: list[dict[str, Any]] = []
        if anchor_latent.dtype == torch.long and anchor_latent.dim() == 1:
            for stage_idx in range(int(anchor_latent.shape[0])):
                per_stage_results: list[dict[str, Any]] = []
                for sample_index, sample in zip(candidate_indices, candidate_samples):
                    mutated_latent = anchor_latent.clone()
                    mutated_latent[stage_idx] = sample["latent"][stage_idx].to(device=device)
                    mutated_outputs = _run_model(
                        model,
                        latent=mutated_latent,
                        condition=anchor_condition,
                    )
                    per_stage_results.append(
                        {
                            "sample_index": int(sample_index),
                            "sample_id": int(sample["sample_id"]),
                            "encoder_output_mad": _mean_abs_diff(anchor_outputs["encoder_output"], mutated_outputs["encoder_output"]),
                            "slm_input_mad": _mean_abs_diff(anchor_outputs["slm_input"], mutated_outputs["slm_input"]),
                            "final_detector_mad": _mean_abs_diff(anchor_outputs["final_detector"], mutated_outputs["final_detector"]),
                        }
                    )
                stage_sensitivity.append(
                    {
                        "stage_index": stage_idx,
                        "mean_encoder_output_mad": float(sum(item["encoder_output_mad"] for item in per_stage_results) / len(per_stage_results)),
                        "mean_slm_input_mad": float(sum(item["slm_input_mad"] for item in per_stage_results) / len(per_stage_results)),
                        "mean_final_detector_mad": float(sum(item["final_detector_mad"] for item in per_stage_results) / len(per_stage_results)),
                        "samples": per_stage_results,
                    }
                )

        _save_panel(
            args.output_dir / "anchor_target.png",
            title="anchor_target",
            image=anchor_sample["target_final"],
            mode="intensity",
        )
        _save_panel(
            args.output_dir / "anchor_phase.png",
            title="anchor_phase",
            image=anchor_outputs["encoder_output"],
            mode="phase",
        )
        _save_panel(
            args.output_dir / "anchor_final_detector.png",
            title="anchor_final_detector",
            image=anchor_outputs["final_detector"],
            mode="intensity",
        )
        _save_grid(
            args.output_dir / "latent_vary_phase_grid.png",
            title="fixed condition, varying latent: encoder phase",
            images=latent_phase_images,
            mode="phase",
        )
        _save_grid(
            args.output_dir / "latent_vary_final_grid.png",
            title="fixed condition, varying latent: final detector",
            images=latent_final_images,
            mode="intensity",
        )
        _save_grid(
            args.output_dir / "condition_vary_phase_grid.png",
            title="fixed latent, varying condition: encoder phase",
            images=condition_phase_images,
            mode="phase",
        )
        _save_grid(
            args.output_dir / "condition_vary_final_grid.png",
            title="fixed latent, varying condition: final detector",
            images=condition_final_images,
            mode="intensity",
        )

        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest_path": str(config["dataset"]["manifest_path"]),
            "anchor_index": anchor_index,
            "anchor_sample_id": int(anchor_sample["sample_id"]),
            "anchor_label": _summarize_label(anchor_sample["label"]),
            "candidate_indices": candidate_indices,
            "latent_only_mean": {
                "encoder_output_mad": float(sum(item["encoder_output_mad"] for item in latent_only_results) / len(latent_only_results)),
                "slm_input_mad": float(sum(item["slm_input_mad"] for item in latent_only_results) / len(latent_only_results)),
                "final_detector_mad": float(sum(item["final_detector_mad"] for item in latent_only_results) / len(latent_only_results)),
            },
            "condition_only_mean": {
                "encoder_output_mad": float(sum(item["encoder_output_mad"] for item in condition_only_results) / len(condition_only_results)),
                "slm_input_mad": float(sum(item["slm_input_mad"] for item in condition_only_results) / len(condition_only_results)),
                "final_detector_mad": float(sum(item["final_detector_mad"] for item in condition_only_results) / len(condition_only_results)),
            },
            "latent_only": latent_only_results,
            "condition_only": condition_only_results,
            "rvq_stage_sensitivity": stage_sensitivity,
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
