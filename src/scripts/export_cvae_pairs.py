from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import NpzImageDataset
from vae import build_cvae


def _load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_path(path_value: str, *, config_dir: Path, repo_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return (repo_root / candidate).resolve()


def export_pairs(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = config_path.resolve().parent
    runtime_cfg = dict(config["runtime"])
    dataset_cfg = dict(config["dataset"])
    export_cfg = dict(config["export"])
    checkpoint_path = _resolve_path(config["checkpoint_path"], config_dir=config_dir, repo_root=repo_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_config = checkpoint["config"]
    model = build_cvae(dict(teacher_config["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(str(runtime_cfg["device"]))
    model.to(device).eval()

    dataset = NpzImageDataset.from_manifest(
        _resolve_path(dataset_cfg["manifest_path"], config_dir=config_dir, repo_root=repo_root),
        max_items=dataset_cfg.get("max_items"),
        channel_mode=str(dataset_cfg["channel_mode"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(dataset_cfg["num_workers"]),
        drop_last=False,
    )

    latent_height = int(teacher_config["model"]["latent_height"])
    latent_width = int(teacher_config["model"]["latent_width"])
    sample_posterior = str(export_cfg["latent_mode"]) == "posterior_sample"
    latents: list[np.ndarray] = []
    teacher_images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    condition_mode = str(teacher_config["model"].get("condition_mode", "class_index"))

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            if condition_mode == "attribute_vector":
                batch_labels = batch["label"].to(device=device, dtype=torch.float32)
            else:
                batch_labels = batch["label"].to(device=device, dtype=torch.long).reshape(-1)
            encoded = model.encode(images, batch_labels, sample_posterior=sample_posterior)
            decoded = model.decode(encoded.z, batch_labels)
            latents.append(
                encoded.z.reshape(encoded.z.shape[0], 1, latent_height, latent_width).detach().cpu().numpy().astype(np.float32)
            )
            teacher_images.append(decoded.detach().cpu().numpy().astype(np.float32))
            if condition_mode == "attribute_vector":
                labels.append(batch["label"].cpu().numpy().astype(np.float32))
            else:
                labels.append(batch["label"].cpu().numpy().astype(np.int64))
            sample_ids.append(batch["sample_id"].cpu().numpy().astype(np.int64))

    output_manifest = _resolve_path(export_cfg["output_manifest_path"], config_dir=config_dir, repo_root=repo_root)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_npz = output_manifest.with_suffix(".npz")
    np.savez(
        output_npz,
        latents=np.concatenate(latents, axis=0),
        teacher_images=np.concatenate(teacher_images, axis=0),
        labels=np.concatenate(labels, axis=0),
        sample_ids=np.concatenate(sample_ids, axis=0),
    )
    manifest = {
        "dataset_name": "fashionmnist_cvae_pairs",
        "image_key": "teacher_images",
        "label_key": "labels",
        "latent_key": "latents",
        "sample_id_key": "sample_ids",
        "num_items": int(sum(item.shape[0] for item in labels)),
        "image_shape_nchw": list(np.concatenate(teacher_images, axis=0).shape),
        "latent_shape_nchw": list(np.concatenate(latents, axis=0).shape),
        "checkpoint_path": str(checkpoint_path),
        "latent_mode": str(export_cfg["latent_mode"]),
        "config": config,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_manifest": str(output_manifest),
        "output_npz": str(output_npz),
        "num_items": manifest["num_items"],
        "latent_shape_nchw": manifest["latent_shape_nchw"],
        "image_shape_nchw": manifest["image_shape_nchw"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latent-label-teacher_image pairs from a trained cVAE teacher.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = export_pairs(_load_config(args.config.resolve()), config_path=args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
