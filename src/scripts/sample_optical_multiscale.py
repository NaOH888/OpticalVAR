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

from scripts.train_optical_multiscale import (
    _build_dataset_and_loader,
    _build_model,
    _build_model_inputs,
    _move_batch_to_device,
)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def tensor_to_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    if torch.is_complex(x):
        x = x.abs()
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3:
        if int(x.shape[0]) == 1:
            x = x[0]
        else:
            x = x.mean(dim=0)
    x = x.float()
    mean_power = x.mean().clamp_min(1.0e-8)
    return x / mean_power


def tensor_to_band_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    if torch.is_complex(x):
        x = x.real
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3:
        if int(x.shape[0]) == 1:
            x = x[0]
        else:
            x = x.mean(dim=0)
    x = x.float()
    max_abs = x.abs().max().clamp_min(1.0e-8)
    return x / max_abs


def _save_panel(path: Path, image: torch.Tensor, *, cmap: str, title: str | None = None) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(tensor_to_image(image), cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_band_panel(path: Path, image: torch.Tensor, *, title: str | None = None) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(tensor_to_band_image(image), cmap="bwr", vmin=-1.0, vmax=1.0)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_prefix_band_compare_panel(
    path: Path,
    *,
    prefix_readout: torch.Tensor,
    target_band: torch.Tensor,
    index: int,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_image(prefix_readout), cmap="gray")
    plt.title(f"prefix_readout_{index}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_band_image(target_band), cmap="bwr", vmin=-1.0, vmax=1.0)
    plt.title(f"target_band_{index}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _summarize_label(label_value: torch.Tensor | None) -> int | list[float] | None:
    if label_value is None:
        return None
    label_tensor = label_value.detach().cpu()
    if label_tensor.numel() == 1:
        return int(label_tensor.reshape(-1)[0])
    return label_tensor.reshape(-1).tolist()


def _validate_mode(args: argparse.Namespace) -> str:
    fixed_mode = args.sample_index is not None
    random_mode = bool(args.random_latent)
    if fixed_mode == random_mode:
        raise ValueError(
            "Choose exactly one sampling mode: either set --sample-index for fixed latent sampling, "
            "or set --random-latent together with --label."
        )
    if random_mode and args.label is None:
        raise ValueError("--label is required when --random-latent is enabled")
    if random_mode and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive in random latent mode")
    return "fixed" if fixed_mode else "random"


def _make_dummy_target_from_config(config: dict[str, Any]) -> torch.Tensor:
    detector_cfg = dict(config["optical"]["detector"])
    height = int(detector_cfg["height_num"])
    width = int(detector_cfg["width_num"])
    return torch.zeros((1, height, width), dtype=torch.float32)


def _build_fixed_mode_batch(
    config: dict[str, Any],
    *,
    sample_index: int,
    device: torch.device,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fake_config_dir = repo_root
    dataset, _, _ = _build_dataset_and_loader(
        config,
        config_dir=fake_config_dir,
        repo_root=repo_root,
    )
    sample = dataset[int(sample_index)]
    batch = next(iter(torch.utils.data.DataLoader([sample], batch_size=1)))
    batch = _move_batch_to_device(batch, device)
    return batch, sample


def _build_random_mode_inputs(
    config: dict[str, Any],
    *,
    label: int,
    num_samples: int,
    latent_seed: int,
    device: torch.device,
    model: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(latent_seed))
    encoder_cfg = dict(config["encoder"])
    latent = torch.randn(
        (
            int(num_samples),
            int(encoder_cfg.get("noise_channels", 1)),
            int(encoder_cfg["input_height"]),
            int(encoder_cfg["input_width"]),
        ),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    class_labels = torch.full(
        (int(num_samples),),
        fill_value=int(label),
        device=device,
        dtype=torch.long,
    )
    return latent, class_labels


def _save_output_panels(
    *,
    output_dir: Path,
    prefix_base: str,
    latent_noise: torch.Tensor,
    encoder_phase: torch.Tensor,
    slm_phase: torch.Tensor,
    prefix_readouts: tuple[torch.Tensor, ...],
    final_prediction: torch.Tensor,
    label: int | list[float] | None,
    target: torch.Tensor | None = None,
    target_bands: tuple[torch.Tensor, ...] | None = None,
) -> None:
    if target is not None:
        _save_panel(
            output_dir / f"{prefix_base}_target.png",
            target,
            cmap="gray",
            title="target_final",
        )
    _save_panel(
        output_dir / f"{prefix_base}_noise.png",
        latent_noise,
        cmap="gray",
        title="latent_noise",
    )
    _save_panel(
        output_dir / f"{prefix_base}_encoder_phase.png",
        encoder_phase,
        cmap="twilight",
        title="encoder_phase",
    )
    _save_panel(
        output_dir / f"{prefix_base}_slm.png",
        slm_phase,
        cmap="twilight",
        title="slm_phase",
    )
    for index, readout in enumerate(prefix_readouts, start=1):
        _save_panel(
            output_dir / f"{prefix_base}_prefix_{index:02d}.png",
            readout,
            cmap="gray",
            title=f"prefix_readout_{index}",
        )
        if target_bands is not None and index <= len(target_bands):
            target_band = target_bands[index - 1]
            _save_band_panel(
                output_dir / f"{prefix_base}_target_band_{index:02d}.png",
                target_band,
                title=f"target_band_{index}",
            )
            _save_prefix_band_compare_panel(
                output_dir / f"{prefix_base}_prefix_band_compare_{index:02d}.png",
                prefix_readout=readout,
                target_band=target_band,
                index=index,
            )
    _save_panel(
        output_dir / f"{prefix_base}_final_detector.png",
        final_prediction,
        cmap="gray",
        title="final_detector",
    )

    panels: list[tuple[str, torch.Tensor]] = []
    if target is not None:
        panels.append(("target", target))
    panels.extend(
        [
            ("noise", latent_noise),
            ("encoder_phase", encoder_phase),
            ("slm_phase", slm_phase),
            ("prediction", final_prediction),
        ]
    )
    plt.figure(figsize=(4 * len(panels), 4))
    for idx, (title, tensor) in enumerate(panels, start=1):
        plt.subplot(1, len(panels), idx)
        plt.imshow(tensor_to_image(tensor), cmap="gray" if "phase" not in title else "twilight")
        if title == "prediction" and label is not None:
            if isinstance(label, list):
                plt.title(f"{title}\nattr_dim={len(label)}")
            else:
                plt.title(f"{title}\nlabel={label}")
        else:
            plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix_base}_overview.png", dpi=200)
    plt.close()


def _save_random_label_grid(
    *,
    output_dir: Path,
    label: int,
    predictions: list[torch.Tensor],
) -> None:
    if not predictions:
        return
    num_samples = len(predictions)
    cols = min(4, num_samples)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = list(axes)
    else:
        axes_list = list(axes.reshape(-1))
    for idx, ax in enumerate(axes_list):
        if idx < num_samples:
            ax.imshow(tensor_to_image(predictions[idx]), cmap="gray")
            ax.set_title(f"sample {idx}")
        ax.axis("off")
    fig.suptitle(f"random label {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"random_label_{label:02d}_grid.png", dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample detector-plane readouts from a trained optical multiscale model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the training checkpoint.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Optional dataset manifest override.")
    parser.add_argument("--sample-index", type=int, default=None, help="Dataset sample index for fixed latent sampling mode.")
    parser.add_argument("--label", type=int, default=None, help="Class label used in random latent sampling mode.")
    parser.add_argument("--random-latent", action="store_true", help="Enable random latent + label sampling mode.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of random latent samples to draw in random mode.")
    parser.add_argument(
        "--detail-sample",
        action="store_true",
        help="In random latent mode, also save per-sample detail panels instead of only the label grid.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save sample images.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument(
        "--latent-seed",
        type=int,
        default=None,
        help="Optional override for the fixed latent seed. Defaults to encoder.latent_seed in the checkpoint config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = _validate_mode(args)
    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(args.checkpoint.resolve(), device)
    config = checkpoint["config"]
    if args.data_manifest is not None:
        config["dataset"]["manifest_path"] = str(args.data_manifest.resolve())

    repo_root = Path(__file__).resolve().parents[2]
    if mode == "fixed":
        config["dataset"]["batch_size"] = 1
        config["dataset"]["shuffle"] = False
        config["dataset"]["num_workers"] = 0
        config["dataset"]["drop_last"] = False
        batch, sample_item = _build_fixed_mode_batch(
            config,
            sample_index=int(args.sample_index),
            device=device,
            repo_root=repo_root,
        )
    else:
        batch = {}
        sample_item = {
            "target_final": _make_dummy_target_from_config(config),
        }

    model = _build_model(config, sample_item=sample_item).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    if mode == "fixed":
        model_input, class_labels = _build_model_inputs(
            batch,
            model=model,
            config=config,
            device=device,
            latent_seed_override=args.latent_seed,
        )
        with torch.no_grad():
            output = model(model_input, class_labels=class_labels)

        prefix_readouts = output["prefix_readouts"]
        final_prediction = output["final_detector"]
        target = batch["target_final"]
        target_bands = tuple(
            batch[f"target_band_{index}"]
            for index in range(1, len(prefix_readouts) + 1)
            if f"target_band_{index}" in batch
        )
        latent_noise = model_input
        slm_phase = output["slm_input"]
        encoder_phase = output["encoder_output"]
        prefix_base = f"sample_{int(args.sample_index):04d}"
        label_value = _summarize_label(batch["label"][0] if "label" in batch else None)

        _save_output_panels(
            output_dir=args.output_dir,
            prefix_base=prefix_base,
            latent_noise=latent_noise,
            encoder_phase=encoder_phase,
            slm_phase=slm_phase,
            prefix_readouts=prefix_readouts,
            final_prediction=final_prediction,
            label=label_value,
            target=target,
            target_bands=target_bands,
        )
        summary: dict[str, Any] = {
            "mode": mode,
            "checkpoint": str(args.checkpoint.resolve()),
            "sample_index": int(args.sample_index),
            "sample_id": int(batch["sample_id"][0]),
            "output_dir": str(args.output_dir.resolve()),
            "label": label_value,
            "prefix_count": len(prefix_readouts),
            "latent_seed": int(config["encoder"].get("latent_seed", 0)) if args.latent_seed is None else int(args.latent_seed),
        }
    else:
        effective_latent_seed = int(config["encoder"].get("latent_seed", 0)) if args.latent_seed is None else int(args.latent_seed)
        model_input, class_labels = _build_random_mode_inputs(
            config,
            label=int(args.label),
            num_samples=int(args.num_samples),
            latent_seed=effective_latent_seed,
            device=device,
            model=model,
        )
        generated: list[dict[str, Any]] = []
        final_predictions: list[torch.Tensor] = []
        for sample_offset in range(int(args.num_samples)):
            with torch.no_grad():
                output = model(
                    model_input[sample_offset : sample_offset + 1],
                    class_labels=class_labels[sample_offset : sample_offset + 1],
                )
            prefix_base = f"random_label_{int(args.label):02d}_{sample_offset:04d}"
            prefix_readouts = output["prefix_readouts"]
            final_prediction = output["final_detector"]
            final_predictions.append(final_prediction)
            latent_noise = model_input[sample_offset : sample_offset + 1]
            slm_phase = output["slm_input"]
            encoder_phase = output["encoder_output"]
            if bool(args.detail_sample):
                _save_output_panels(
                    output_dir=args.output_dir,
                    prefix_base=prefix_base,
                    latent_noise=latent_noise,
                    encoder_phase=encoder_phase,
                    slm_phase=slm_phase,
                    prefix_readouts=prefix_readouts,
                    final_prediction=final_prediction,
                    label=int(args.label),
                    target=None,
                )
            generated.append(
                {
                    "sample_offset": sample_offset,
                    "label": int(args.label),
                    "prefix_count": len(prefix_readouts),
                }
            )
        _save_random_label_grid(
            output_dir=args.output_dir,
            label=int(args.label),
            predictions=final_predictions,
        )
        summary = {
            "mode": mode,
            "checkpoint": str(args.checkpoint.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "label": int(args.label),
            "num_samples": int(args.num_samples),
            "detail_sample": bool(args.detail_sample),
            "latent_seed": effective_latent_seed,
            "samples": generated,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
