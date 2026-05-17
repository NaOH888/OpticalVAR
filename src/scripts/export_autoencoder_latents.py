from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import NpzImageDataset
from scripts._pretrained_autoencoder_utils import load_autoencoder_cls, tensor_gray_to_rgb, tensor_rgb_to_gray


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


def _iter_batches(dataset: NpzImageDataset, batch_size: int):
    batch_images: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    batch_sample_ids: list[torch.Tensor] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"]
        label = sample["label"]
        sample_id = sample["sample_id"]
        batch_images.append(image)
        batch_labels.append(label)
        batch_sample_ids.append(sample_id)
        if len(batch_images) >= int(batch_size):
            yield (
                torch.stack(batch_images, dim=0),
                torch.stack(batch_labels, dim=0),
                torch.stack(batch_sample_ids, dim=0),
            )
            batch_images.clear()
            batch_labels.clear()
            batch_sample_ids.clear()
    if batch_images:
        yield (
            torch.stack(batch_images, dim=0),
            torch.stack(batch_labels, dim=0),
            torch.stack(batch_sample_ids, dim=0),
        )


@torch.no_grad()
def export_latents(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = config_path.resolve().parent
    runtime_cfg = dict(config["runtime"])
    dataset_cfg = dict(config["dataset"])
    autoencoder_cfg = dict(config["autoencoder"])
    export_cfg = dict(config["export"])

    device = torch.device(str(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    manifest_path = _resolve_path(dataset_cfg["manifest_path"], config_dir=config_dir, repo_root=repo_root)
    dataset = NpzImageDataset.from_manifest(
        manifest_path,
        max_items=dataset_cfg.get("max_items"),
        channel_mode=str(dataset_cfg.get("channel_mode", "keep")),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    AutoencoderKL = load_autoencoder_cls()
    model_id = str(autoencoder_cfg["model_id"])
    model = AutoencoderKL.from_pretrained(model_id).to(device)
    model.eval()
    scaling_factor = float(getattr(model.config, "scaling_factor", 0.18215))

    latents: list[np.ndarray] = []
    teacher_images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    batch_size = int(autoencoder_cfg.get("batch_size", 8))

    try:
        for images, batch_labels, batch_sample_ids in _iter_batches(dataset, batch_size=batch_size):
            if images.dim() != 4 or int(images.shape[1]) != 1:
                raise ValueError(f"expected grayscale images [B,1,H,W], got {tuple(images.shape)}")
            images = images.to(device=device, dtype=torch.float32)
            batch_rgb = tensor_gray_to_rgb(images)
            vae_input = batch_rgb * 2.0 - 1.0
            encoded = model.encode(vae_input)
            latent_mode = str(autoencoder_cfg.get("latent_mode", "mode"))
            if latent_mode == "sample":
                latent = encoded.latent_dist.sample()
            elif latent_mode == "mode":
                latent = encoded.latent_dist.mode()
            else:
                raise ValueError(f"unsupported latent_mode: {latent_mode!r}")
            decoded = model.decode(latent).sample
            recon_rgb = ((decoded.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
            recon_gray = tensor_rgb_to_gray(recon_rgb).cpu().numpy().astype(np.float32, copy=False)

            latents.append(latent.cpu().numpy().astype(np.float32, copy=False))
            teacher_images.append(recon_gray)
            label_np = batch_labels.cpu().numpy()
            if label_np.ndim == 1:
                labels.append(label_np.astype(np.int64, copy=False))
            else:
                labels.append(label_np.astype(np.float32, copy=False))
            sample_ids.append(batch_sample_ids.cpu().numpy().astype(np.int64, copy=False))

        output_manifest = _resolve_path(export_cfg["output_manifest_path"], config_dir=config_dir, repo_root=repo_root)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_prefix = output_manifest.stem
        shard_size = int(export_cfg.get("shard_size", 4096))

        all_latents = np.concatenate(latents, axis=0)
        all_teacher_images = np.concatenate(teacher_images, axis=0)
        all_labels = np.concatenate(labels, axis=0)
        all_sample_ids = np.concatenate(sample_ids, axis=0)

        npz_files: list[str] = []
        shards: list[dict[str, Any]] = []
        for shard_index, start in enumerate(range(0, all_latents.shape[0], shard_size)):
            end = min(start + shard_size, all_latents.shape[0])
            shard_name = f"{output_prefix}_part{shard_index:04d}.npz"
            shard_path = output_manifest.parent / shard_name
            np.savez(
                shard_path,
                teacher_images=all_teacher_images[start:end].astype(np.float32, copy=False),
                latents=all_latents[start:end].astype(np.float32, copy=False),
                labels=all_labels[start:end],
                sample_ids=all_sample_ids[start:end],
            )
            npz_files.append(shard_name)
            shards.append(
                {
                    "filename": shard_name,
                    "num_items": int(end - start),
                    "sample_id_start": int(all_sample_ids[start]),
                    "sample_id_end": int(all_sample_ids[end - 1]),
                }
            )

        manifest = {
            "dataset_name": f"{dataset_cfg.get('dataset_name', 'dataset')}_autoencoder_pairs",
            "split": str(dataset_cfg.get("split", payload.get("split", "train"))),
            "num_items": int(all_latents.shape[0]),
            "image_key": "teacher_images",
            "label_key": "labels",
            "latent_key": "latents",
            "latent_source": "autoencoderkl",
            "latent_type": "continuous_map",
            "latent_spec": {
                "shape": list(all_latents.shape[1:]),
                "model_id": model_id,
                "latent_mode": str(autoencoder_cfg.get("latent_mode", "mode")),
                "scaling_factor": scaling_factor,
            },
            "sample_id_key": "sample_ids",
            "label_names": payload.get("label_names"),
            "label_semantics": payload.get("label_semantics"),
            "condition_components": payload.get("condition_components"),
            "npz_files": npz_files,
            "shards": shards,
            "config": config,
        }
        output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "output_manifest": str(output_manifest),
            "num_items": int(all_latents.shape[0]),
            "latent_shape_nchw": list(all_latents.shape),
            "image_shape_nchw": list(all_teacher_images.shape),
        }
    finally:
        dataset.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pretrained AutoencoderKL latents and teacher reconstructions.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = export_latents(_load_config(args.config.resolve()), config_path=args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
