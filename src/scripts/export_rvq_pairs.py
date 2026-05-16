from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


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


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _count_manifest_items(manifest_path: Path) -> int:
    payload = _load_manifest(manifest_path)
    num_items = payload.get("num_items")
    if num_items is not None:
        return int(num_items)

    sample_id_key = str(payload.get("sample_id_key", "sample_ids"))
    npz_files = payload.get("npz_files")
    if npz_files is None:
        npz_files = [manifest_path.name.replace(".json", ".npz")]
    total_items = 0
    for filename in npz_files:
        shard_path = (manifest_path.parent / str(filename)).resolve()
        archive = np.load(shard_path, allow_pickle=False)
        try:
            if sample_id_key in archive.files:
                total_items += int(archive[sample_id_key].shape[0])
            else:
                image_key = str(payload.get("image_key", "images"))
                total_items += int(archive[image_key].shape[0])
        finally:
            archive.close()
    return total_items


def _format_seconds(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _iter_source_shards(
    manifest_path: Path,
    *,
    max_items: int | None,
    channel_mode: str,
):
    payload = _load_manifest(manifest_path)
    image_key = str(payload["image_key"])
    label_key = str(payload["label_key"])
    sample_id_key = str(payload["sample_id_key"])
    remaining = None if max_items is None else int(max_items)

    npz_files = payload.get("npz_files")
    if npz_files is None:
        npz_files = [manifest_path.name.replace(".json", ".npz")]
    for filename in npz_files:
        shard_path = (manifest_path.parent / str(filename)).resolve()
        archive = np.load(shard_path, allow_pickle=False)
        try:
            images = archive[image_key]
            labels = archive[label_key]
            sample_ids = archive[sample_id_key]
            if channel_mode == "first" and images.ndim == 4:
                images = images[:, :1]
            elif channel_mode == "mean" and images.ndim == 4:
                images = images.mean(axis=1, keepdims=True)
            if remaining is not None:
                if remaining <= 0:
                    break
                take = min(int(images.shape[0]), remaining)
                images = images[:take]
                labels = labels[:take]
                sample_ids = sample_ids[:take]
                remaining -= take
            yield images.astype(np.float32, copy=False), labels, sample_ids
        finally:
            archive.close()


def _fit_incremental_pca(
    manifest_path: Path,
    *,
    pca_dim: int,
    batch_size: int,
    max_items: int | None,
    channel_mode: str,
) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=int(pca_dim), batch_size=int(batch_size))
    manifest_items = _count_manifest_items(manifest_path)
    total_items = min(
        manifest_items,
        int(max_items) if max_items is not None else manifest_items,
    )
    processed_items = 0
    start_time = time.perf_counter()
    print(
        f"[export_rvq][pca_fit] start total_items={total_items} pca_dim={int(pca_dim)} batch_size={int(batch_size)}"
    )
    for images, _, _ in _iter_source_shards(
        manifest_path,
        max_items=max_items,
        channel_mode=channel_mode,
    ):
        flat = images.reshape(images.shape[0], -1)
        for start in range(0, flat.shape[0], int(batch_size)):
            batch = flat[start : start + int(batch_size)]
            ipca.partial_fit(batch)
            processed_items += int(batch.shape[0])
            if processed_items % max(int(batch_size) * 20, 1) == 0 or processed_items >= total_items:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[export_rvq][pca_fit] processed={processed_items}/{total_items} "
                    f"elapsed={_format_seconds(elapsed)}"
                )
    print(
        f"[export_rvq][pca_fit] done processed={processed_items}/{total_items} "
        f"elapsed={_format_seconds(time.perf_counter() - start_time)}"
    )
    return ipca


def _project_dataset(
    manifest_path: Path,
    *,
    ipca: IncrementalPCA,
    batch_size: int,
    max_items: int | None,
    channel_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    sample_ids_list: list[np.ndarray] = []
    total_items = min(
        _count_manifest_items(manifest_path),
        int(max_items) if max_items is not None else _count_manifest_items(manifest_path),
    )
    processed_items = 0
    start_time = time.perf_counter()
    print(f"[export_rvq][project] start total_items={total_items} batch_size={int(batch_size)}")
    for images, labels, sample_ids in _iter_source_shards(
        manifest_path,
        max_items=max_items,
        channel_mode=channel_mode,
    ):
        flat = images.reshape(images.shape[0], -1)
        projected_batches: list[np.ndarray] = []
        for start in range(0, flat.shape[0], int(batch_size)):
            batch = flat[start : start + int(batch_size)]
            projected_batches.append(ipca.transform(batch).astype(np.float32, copy=False))
            processed_items += int(batch.shape[0])
            if processed_items % max(int(batch_size) * 20, 1) == 0 or processed_items >= total_items:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[export_rvq][project] processed={processed_items}/{total_items} "
                    f"elapsed={_format_seconds(elapsed)}"
                )
        features_list.append(np.concatenate(projected_batches, axis=0))
        labels_list.append(labels)
        sample_ids_list.append(sample_ids.astype(np.int64, copy=False))
    print(
        f"[export_rvq][project] done processed={processed_items}/{total_items} "
        f"elapsed={_format_seconds(time.perf_counter() - start_time)}"
    )
    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(sample_ids_list, axis=0),
    )


def _perform_rvq(
    features: np.ndarray,
    *,
    num_stages: int,
    codebook_size: int,
    batch_size: int,
    seed: int,
    fit_max_items: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    residual = features.astype(np.float32, copy=True)
    codes = np.empty((features.shape[0], int(num_stages)), dtype=np.int32)
    codebooks: list[np.ndarray] = []
    total_items = int(features.shape[0])
    rng = np.random.default_rng(int(seed))
    fit_indices: np.ndarray | None = None
    fit_items = total_items
    if fit_max_items is not None and int(fit_max_items) > 0 and int(fit_max_items) < total_items:
        fit_items = int(fit_max_items)
        fit_indices = np.sort(rng.choice(total_items, size=fit_items, replace=False))
    print(
        f"[export_rvq][rvq] start total_items={total_items} stages={int(num_stages)} "
        f"codebook_size={int(codebook_size)} batch_size={int(batch_size)} fit_items={fit_items}"
    )
    fit_residual = residual if fit_indices is None else residual[fit_indices].copy()
    for stage_idx in range(int(num_stages)):
        stage_start = time.perf_counter()
        kmeans = MiniBatchKMeans(
            n_clusters=int(codebook_size),
            batch_size=int(batch_size),
            random_state=int(seed) + stage_idx,
            n_init="auto",
        )
        print(f"[export_rvq][rvq] stage={stage_idx + 1}/{int(num_stages)} fit fit_items={fit_items}")
        kmeans.fit(fit_residual)
        print(f"[export_rvq][rvq] stage={stage_idx + 1}/{int(num_stages)} assign")
        stage_codes = kmeans.predict(residual).astype(np.int32, copy=False)
        stage_centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
        quantized = stage_centers[stage_codes]
        codes[:, stage_idx] = stage_codes
        codebooks.append(stage_centers)
        residual = residual - quantized
        if fit_indices is not None:
            fit_residual = residual[fit_indices].copy()
        else:
            fit_residual = residual
        residual_norm = float(np.linalg.norm(residual) / max(total_items, 1))
        print(
            f"[export_rvq][rvq] stage={stage_idx + 1}/{int(num_stages)} done "
            f"elapsed={_format_seconds(time.perf_counter() - stage_start)} "
            f"residual_norm_per_item={residual_norm:.6f}"
        )
    return codes, np.stack(codebooks, axis=0)


def _save_rvq_shards(
    *,
    output_manifest: Path,
    image_manifest: Path,
    image_manifest_payload: dict[str, Any],
    labels: np.ndarray,
    sample_ids: np.ndarray,
    rvq_codes: np.ndarray,
    rvq_model: np.ndarray,
    pca: IncrementalPCA,
    config: dict[str, Any],
    shard_size: int,
) -> dict[str, Any]:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = output_manifest.stem
    shards: list[dict[str, Any]] = []
    npz_files: list[str] = []
    code_dtype = np.int16 if int(config["rvq"]["codebook_size"]) <= np.iinfo(np.int16).max else np.int32

    for shard_index, start in enumerate(range(0, rvq_codes.shape[0], int(shard_size))):
        end = min(start + int(shard_size), rvq_codes.shape[0])
        shard_name = f"{output_prefix}_part{shard_index:04d}.npz"
        shard_path = output_manifest.parent / shard_name
        print(
            f"[export_rvq][save] shard={shard_index + 1} items={end - start} "
            f"range=[{start},{end}) path={shard_name}"
        )
        np.savez(
            shard_path,
            rvq_codes=rvq_codes[start:end].astype(code_dtype, copy=False),
            labels=labels[start:end],
            sample_ids=sample_ids[start:end],
        )
        npz_files.append(shard_name)
        shards.append(
            {
                "filename": shard_name,
                "num_items": int(end - start),
                "sample_id_start": int(sample_ids[start]),
                "sample_id_end": int(sample_ids[end - 1]),
            }
        )

    model_name = f"{output_prefix}_model.npz"
    np.savez(
        output_manifest.parent / model_name,
        pca_components=pca.components_.astype(np.float32, copy=False),
        pca_mean=pca.mean_.astype(np.float32, copy=False),
        rvq_codebooks=rvq_model.astype(np.float32, copy=False),
    )

    manifest = {
        "dataset_name": f"{config['dataset'].get('dataset_name', 'dataset')}_rvq_pairs",
        "split": str(config["dataset"].get("split", "train")),
        "num_items": int(rvq_codes.shape[0]),
        "image_manifest_path": os.path.relpath(
            str(image_manifest.resolve()),
            start=str(output_manifest.parent.resolve()),
        ),
        "image_key": None,
        "label_key": "labels",
        "sample_id_key": "sample_ids",
        "latent_source": "rvq",
        "latent_type": "discrete_code",
        "latent_key": "rvq_codes",
        "latent_spec": {
            "num_stages": int(config["rvq"]["num_stages"]),
            "codebook_size": int(config["rvq"]["codebook_size"]),
            "shape": [int(config["rvq"]["num_stages"])],
            "pca_dim": int(config["rvq"]["pca_dim"]),
            "rvq_model_file": model_name,
        },
        "label_names": image_manifest_payload.get("label_names"),
        "label_semantics": image_manifest_payload.get("label_semantics"),
        "condition_components": image_manifest_payload.get("condition_components"),
        "npz_files": npz_files,
        "shards": shards,
        "config": config,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_manifest": str(output_manifest),
        "num_items": int(rvq_codes.shape[0]),
        "latent_shape": [int(rvq_codes.shape[0]), int(rvq_codes.shape[1])],
        "rvq_model_file": str(output_manifest.parent / model_name),
    }


def export_pairs(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = config_path.resolve().parent
    runtime_cfg = dict(config["runtime"])
    dataset_cfg = dict(config["dataset"])
    rvq_cfg = dict(config["rvq"])
    export_cfg = dict(config["export"])

    image_manifest = _resolve_path(dataset_cfg["manifest_path"], config_dir=config_dir, repo_root=repo_root)
    image_manifest_payload = _load_manifest(image_manifest)
    overall_start = time.perf_counter()
    print(
        f"[export_rvq] start manifest={image_manifest} num_items={image_manifest_payload.get('num_items')} "
        f"max_items={dataset_cfg.get('max_items')} pca_dim={int(rvq_cfg['pca_dim'])} "
        f"stages={int(rvq_cfg['num_stages'])} codebook_size={int(rvq_cfg['codebook_size'])}"
    )
    pca_fit_max_items = rvq_cfg.get("pca_fit_max_items")
    rvq_fit_max_items = rvq_cfg.get("rvq_fit_max_items")
    ipca = _fit_incremental_pca(
        image_manifest,
        pca_dim=int(rvq_cfg["pca_dim"]),
        batch_size=int(rvq_cfg.get("batch_size", 512)),
        max_items=(
            int(pca_fit_max_items)
            if pca_fit_max_items is not None
            else dataset_cfg.get("max_items")
        ),
        channel_mode=str(dataset_cfg.get("channel_mode", "keep")),
    )
    features, labels, sample_ids = _project_dataset(
        image_manifest,
        ipca=ipca,
        batch_size=int(rvq_cfg.get("batch_size", 512)),
        max_items=dataset_cfg.get("max_items"),
        channel_mode=str(dataset_cfg.get("channel_mode", "keep")),
    )
    rvq_codes, rvq_codebooks = _perform_rvq(
        features,
        num_stages=int(rvq_cfg["num_stages"]),
        codebook_size=int(rvq_cfg["codebook_size"]),
        batch_size=int(rvq_cfg.get("batch_size", 512)),
        seed=int(runtime_cfg.get("seed", 42)),
        fit_max_items=None if rvq_fit_max_items is None else int(rvq_fit_max_items),
    )
    output_manifest = _resolve_path(export_cfg["output_manifest_path"], config_dir=config_dir, repo_root=repo_root)
    result = _save_rvq_shards(
        output_manifest=output_manifest,
        image_manifest=image_manifest,
        image_manifest_payload=image_manifest_payload,
        labels=labels,
        sample_ids=sample_ids,
        rvq_codes=rvq_codes,
        rvq_model=rvq_codebooks,
        pca=ipca,
        config=config,
        shard_size=int(export_cfg.get("shard_size", 4096)),
    )
    print(
        f"[export_rvq] done output_manifest={result['output_manifest']} "
        f"elapsed={_format_seconds(time.perf_counter() - overall_start)}"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RVQ latent pairs that reference an existing image manifest.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = export_pairs(_load_config(args.config.resolve()), config_path=args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
