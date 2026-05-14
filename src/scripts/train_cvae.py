from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import NpzImageDataset
from vae import build_cvae, build_perceptual_loss


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_condition_batch(batch: dict[str, Any], *, model_cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    condition_mode = str(model_cfg.get("condition_mode", "class_index"))
    if "label" not in batch:
        raise KeyError("dataset batch must contain 'label' for cVAE conditioning")
    if condition_mode == "attribute_vector":
        return batch["label"].to(device=device, dtype=torch.float32)
    return batch["label"].to(device=device, dtype=torch.long).reshape(-1)


def train(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = config_path.resolve().parent
    runtime_cfg = dict(config["runtime"])
    dataset_cfg = dict(config["dataset"])
    model_cfg = dict(config["model"])
    train_cfg = dict(config["training"])

    _seed_everything(int(runtime_cfg["seed"]))
    device = torch.device(str(runtime_cfg["device"]))

    dataset = NpzImageDataset.from_manifest(
        _resolve_path(dataset_cfg["manifest_path"], config_dir=config_dir, repo_root=repo_root),
        max_items=dataset_cfg.get("max_items"),
        channel_mode=str(dataset_cfg["channel_mode"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_cfg["batch_size"]),
        shuffle=bool(dataset_cfg["shuffle"]),
        num_workers=int(dataset_cfg["num_workers"]),
        drop_last=bool(dataset_cfg.get("drop_last", False)),
    )

    model = build_cvae(model_cfg).to(device)
    perceptual_loss_fn = build_perceptual_loss(train_cfg)
    if perceptual_loss_fn is not None:
        perceptual_loss_fn = perceptual_loss_fn.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    kl_weight = float(train_cfg["kl_weight"])
    perceptual_weight = float(train_cfg.get("perceptual_weight", 0.0))
    output_dir = _resolve_path(str(train_cfg["output_dir"]), config_dir=config_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    history_path = output_dir / "history.jsonl"

    latest_metrics: dict[str, Any] = {}
    for epoch_idx in range(int(train_cfg["epochs"])):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0
        running_perceptual = 0.0
        step_count = 0
        for step_idx, batch in enumerate(loader, start=1):
            inputs = {
                "data": batch["image"].to(device=device, dtype=torch.float32),
                "labels": _resolve_condition_batch(batch, model_cfg=model_cfg, device=device),
            }
            output = model(inputs)
            if perceptual_loss_fn is None:
                perceptual_loss = torch.zeros((), device=device, dtype=output.recon_loss.dtype)
            else:
                perceptual_loss = perceptual_loss_fn(output.recon_x, inputs["data"])
            total_loss = output.recon_loss + kl_weight * output.reg_loss + perceptual_weight * perceptual_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            running_total += float(total_loss.detach().cpu())
            running_recon += float(output.recon_loss.detach().cpu())
            running_kl += float(output.reg_loss.detach().cpu())
            running_perceptual += float(perceptual_loss.detach().cpu())
            step_count += 1

            if step_idx % int(train_cfg["log_interval"]) == 0:
                print(
                    f"[epoch {epoch_idx + 1}/{int(train_cfg['epochs'])}] "
                    f"step={step_idx} total={running_total / step_count:.6f} "
                    f"recon={running_recon / step_count:.6f} "
                    f"kl={running_kl / step_count:.6f} "
                    f"perc={running_perceptual / step_count:.6f}"
                )
            if train_cfg.get("max_steps_per_epoch") is not None and step_idx >= int(train_cfg["max_steps_per_epoch"]):
                break

        latest_metrics = {
            "epoch": epoch_idx + 1,
            "total_loss": running_total / step_count,
            "recon_loss": running_recon / step_count,
            "kl_loss": running_kl / step_count,
            "perceptual_loss": running_perceptual / step_count,
        }
        _append_jsonl(history_path, latest_metrics)
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "metrics": latest_metrics,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if bool(train_cfg.get("save_every_epoch", False)):
            torch.save(checkpoint, output_dir / f"epoch_{epoch_idx + 1:03d}.pt")
        print(
            f"[epoch {epoch_idx + 1}/{int(train_cfg['epochs'])}] "
            f"total={latest_metrics['total_loss']:.6f} "
            f"recon={latest_metrics['recon_loss']:.6f} "
            f"kl={latest_metrics['kl_loss']:.6f} "
            f"perc={latest_metrics['perceptual_loss']:.6f}"
        )

    return {
        "output_dir": str(output_dir),
        "latest_checkpoint": str(output_dir / "latest.pt"),
        "history_path": str(history_path),
        "metrics": latest_metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FashionMNIST conditional VAE teacher.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = train(_load_config(args.config.resolve()), config_path=args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
