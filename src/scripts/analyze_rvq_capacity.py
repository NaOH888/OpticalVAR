from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import ReferencedImageLatentDataset


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_model_path(manifest_path: Path, payload: dict[str, Any]) -> Path:
    latent_spec = dict(payload.get("latent_spec", {}))
    model_name = latent_spec.get("rvq_model_file")
    if model_name is None:
        raise KeyError("latent manifest must contain latent_spec.rvq_model_file")
    return (manifest_path.parent / str(model_name)).resolve()


def _parse_index_list(value: str | None, *, max_value: int, one_indexed: bool = False) -> list[int]:
    if value is None or value.strip() == "":
        return []
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if one_indexed:
            index -= 1
        if index < 0 or index >= max_value:
            raise ValueError(f"index {index} out of range for max_value={max_value}")
        indices.append(index)
    deduped = sorted(set(indices))
    return deduped


def _parse_positive_counts(value: str | None, *, max_value: int) -> list[int]:
    if value is None or value.strip() == "":
        return []
    counts: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        count = int(part)
        if count <= 0 or count > max_value:
            raise ValueError(f"count {count} out of range for max_value={max_value}")
        counts.append(count)
    return sorted(set(counts))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def _save_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    path: Path,
    *,
    cols: int = 4,
    cmap: str = "gray",
    signed: bool = False,
) -> None:
    if len(images) != len(titles):
        raise ValueError("images and titles must have identical lengths")
    rows = int(math.ceil(len(images) / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    axes = axes.reshape(rows, cols)
    for axis in axes.flat:
        axis.axis("off")
    for index, (image, title) in enumerate(zip(images, titles)):
        axis = axes.flat[index]
        if signed:
            scale = max(float(np.abs(image).max()), 1e-8)
            axis.imshow(image / scale, cmap=cmap, vmin=-1.0, vmax=1.0)
        else:
            axis.imshow(_normalize_image(image), cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _save_curve(values: list[float], path: Path, *, title: str, ylabel: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(np.arange(len(values)), values, marker="o")
    axis.set_title(title)
    axis.set_xlabel("stage")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _project_image(flat_image: np.ndarray, *, pca_components: np.ndarray, pca_mean: np.ndarray) -> np.ndarray:
    centered = flat_image.astype(np.float32, copy=False) - pca_mean.astype(np.float32, copy=False)
    return centered @ pca_components.astype(np.float32, copy=False).T


def _decode_codes_to_feature(codes: np.ndarray, rvq_codebooks: np.ndarray, *, upto_stage: int | None = None) -> np.ndarray:
    stage_count = rvq_codebooks.shape[0] if upto_stage is None else int(upto_stage)
    stage_count = max(min(stage_count, rvq_codebooks.shape[0]), 0)
    if stage_count == 0:
        return np.zeros((rvq_codebooks.shape[-1],), dtype=np.float32)
    feature = np.zeros((rvq_codebooks.shape[-1],), dtype=np.float32)
    for stage_index in range(stage_count):
        feature += rvq_codebooks[stage_index, int(codes[stage_index])].astype(np.float32, copy=False)
    return feature


def _inverse_project_feature(feature: np.ndarray, *, pca_components: np.ndarray, pca_mean: np.ndarray) -> np.ndarray:
    return feature.astype(np.float32, copy=False) @ pca_components.astype(np.float32, copy=False) + pca_mean.astype(
        np.float32, copy=False
    )


def _reconstruct_image(
    codes: np.ndarray,
    *,
    rvq_codebooks: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    image_shape: tuple[int, ...],
    upto_stage: int | None = None,
) -> np.ndarray:
    feature = _decode_codes_to_feature(codes, rvq_codebooks, upto_stage=upto_stage)
    flat = _inverse_project_feature(feature, pca_components=pca_components, pca_mean=pca_mean)
    image = flat.reshape(image_shape).astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    return image


def analyze_rvq_capacity(
    *,
    manifest_path: Path,
    output_dir: Path,
    sample_index: int,
    donor_index: int,
    cumulative_stage_counts: list[int],
    swap_stages: list[int],
) -> dict[str, Any]:
    payload = _load_manifest(manifest_path)
    model_path = _resolve_model_path(manifest_path, payload)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReferencedImageLatentDataset.from_latent_manifest(manifest_path)
    try:
        sample = dataset[sample_index]
        donor = dataset[donor_index]
        image = sample["image"].detach().cpu().numpy().astype(np.float32, copy=False)
        donor_image = donor["image"].detach().cpu().numpy().astype(np.float32, copy=False)
        codes = sample["latent"].detach().cpu().numpy().astype(np.int64, copy=False)
        donor_codes = donor["latent"].detach().cpu().numpy().astype(np.int64, copy=False)
        image_shape = tuple(int(dim) for dim in image.shape)
        flat_image = image.reshape(-1)

        with np.load(model_path, allow_pickle=False) as model_data:
            pca_components = model_data["pca_components"].astype(np.float32, copy=False)
            pca_mean = model_data["pca_mean"].astype(np.float32, copy=False)
            rvq_codebooks = model_data["rvq_codebooks"].astype(np.float32, copy=False)

        num_stages = int(rvq_codebooks.shape[0])
        cumulative_stage_counts = [count for count in cumulative_stage_counts if 1 <= count <= num_stages]
        if not cumulative_stage_counts:
            cumulative_stage_counts = sorted(set([1, min(2, num_stages), num_stages]))
        swap_stages = [stage for stage in swap_stages if 0 <= stage < num_stages]
        if not swap_stages:
            swap_stages = sorted(set([0, min(1, num_stages - 1), num_stages - 1]))

        projected_image = _project_image(flat_image, pca_components=pca_components, pca_mean=pca_mean)

        cumulative_images: list[np.ndarray] = []
        cumulative_titles: list[str] = []
        residual_norms: list[float] = []
        previous_reconstruction = np.zeros_like(flat_image, dtype=np.float32)
        increment_images: list[np.ndarray] = []
        increment_titles: list[str] = []
        for stage_count in range(1, num_stages + 1):
            feature = _decode_codes_to_feature(codes, rvq_codebooks, upto_stage=stage_count)
            residual_norms.append(float(np.linalg.norm(projected_image - feature)))
            if stage_count in cumulative_stage_counts:
                reconstructed_flat = _inverse_project_feature(feature, pca_components=pca_components, pca_mean=pca_mean)
                reconstructed_image = reconstructed_flat.reshape(image_shape).astype(np.float32, copy=False)
                if reconstructed_image.ndim == 3 and reconstructed_image.shape[0] == 1:
                    reconstructed_image = reconstructed_image[0]
                cumulative_images.append(reconstructed_image)
                cumulative_titles.append(f"stages 1-{stage_count}")
                increment = (reconstructed_flat - previous_reconstruction).reshape(image_shape).astype(np.float32, copy=False)
                previous_reconstruction = reconstructed_flat
                if increment.ndim == 3 and increment.shape[0] == 1:
                    increment = increment[0]
                increment_images.append(increment)
                increment_titles.append(f"delta@{stage_count}")

        original_image = image[0] if image.ndim == 3 and image.shape[0] == 1 else image
        donor_display = donor_image[0] if donor_image.ndim == 3 and donor_image.shape[0] == 1 else donor_image
        _save_image_grid(
            [original_image, donor_display] + cumulative_images,
            ["anchor_original", "donor_original"] + cumulative_titles,
            output_dir / "cumulative_reconstructions.png",
            cols=4,
        )
        _save_image_grid(
            increment_images,
            increment_titles,
            output_dir / "increment_reconstructions.png",
            cols=4,
            cmap="bwr",
            signed=True,
        )
        _save_curve(residual_norms, output_dir / "residual_curve.png", title="RVQ Residual Norm", ylabel="L2 norm")

        full_reconstruction = _reconstruct_image(
            codes,
            rvq_codebooks=rvq_codebooks,
            pca_components=pca_components,
            pca_mean=pca_mean,
            image_shape=image_shape,
        )
        full_reconstruction_flat = full_reconstruction.reshape(-1)
        swap_images: list[np.ndarray] = []
        swap_titles: list[str] = []
        swap_diffs: list[np.ndarray] = []
        swap_summary: list[dict[str, Any]] = []
        for stage_index in swap_stages:
            swapped_codes = codes.copy()
            swapped_codes[stage_index] = donor_codes[stage_index]
            swapped_image = _reconstruct_image(
                swapped_codes,
                rvq_codebooks=rvq_codebooks,
                pca_components=pca_components,
                pca_mean=pca_mean,
                image_shape=image_shape,
            )
            swapped_flat = swapped_image.reshape(-1)
            abs_diff = np.abs(swapped_flat - full_reconstruction_flat)
            diff_image = (swapped_flat - full_reconstruction_flat).reshape(swapped_image.shape).astype(np.float32, copy=False)
            if diff_image.ndim == 3 and diff_image.shape[0] == 1:
                diff_image = diff_image[0]
            if swapped_image.ndim == 3 and swapped_image.shape[0] == 1:
                swapped_image = swapped_image[0]
            swap_images.append(swapped_image)
            swap_titles.append(f"swap stage {stage_index}")
            swap_diffs.append(diff_image)
            swap_summary.append(
                {
                    "stage_index": int(stage_index),
                    "anchor_code": int(codes[stage_index]),
                    "donor_code": int(donor_codes[stage_index]),
                    "reconstruction_mad": float(abs_diff.mean()),
                    "reconstruction_l2": float(np.linalg.norm(abs_diff)),
                }
            )

        _save_image_grid(
            [full_reconstruction] + swap_images,
            ["anchor_full_recon"] + swap_titles,
            output_dir / "stage_swaps.png",
            cols=4,
        )
        _save_image_grid(
            swap_diffs,
            [f"diff stage {stage}" for stage in swap_stages],
            output_dir / "stage_swap_diffs.png",
            cols=4,
            cmap="bwr",
            signed=True,
        )

        summary = {
            "manifest_path": str(manifest_path),
            "model_path": str(model_path),
            "sample_index": int(sample_index),
            "sample_id": int(sample["sample_id"]),
            "donor_index": int(donor_index),
            "donor_sample_id": int(donor["sample_id"]),
            "num_stages": num_stages,
            "codebook_size": int(rvq_codebooks.shape[1]),
            "pca_dim": int(pca_components.shape[0]),
            "cumulative_stage_counts": [int(value) for value in cumulative_stage_counts],
            "swap_stages": [int(value) for value in swap_stages],
            "residual_norms": residual_norms,
            "swap_summary": swap_summary,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    finally:
        dataset.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RVQ representation capacity without involving the optical model.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/celeba_rvq/celeba_train_rvq_lm.json"),
        help="Path to the RVQ latent manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Anchor sample index in the RVQ dataset.",
    )
    parser.add_argument(
        "--donor-index",
        type=int,
        default=1,
        help="Donor sample index used for single-stage swap analysis.",
    )
    parser.add_argument(
        "--cumulative-stages",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated 1-indexed cumulative stage counts to visualize.",
    )
    parser.add_argument(
        "--swap-stages",
        type=str,
        default="0,1,2,3,7,15,31",
        help="Comma-separated 0-indexed stage ids to swap from the donor sample.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = _load_manifest(args.manifest.resolve())
    num_stages = int(payload["latent_spec"]["num_stages"])
    cumulative_stage_counts = _parse_positive_counts(args.cumulative_stages, max_value=num_stages)
    if num_stages not in cumulative_stage_counts:
        cumulative_stage_counts.append(num_stages)
        cumulative_stage_counts = sorted(set(cumulative_stage_counts))
    swap_stages = _parse_index_list(args.swap_stages, max_value=num_stages, one_indexed=False)
    summary = analyze_rvq_capacity(
        manifest_path=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        sample_index=int(args.sample_index),
        donor_index=int(args.donor_index),
        cumulative_stage_counts=cumulative_stage_counts,
        swap_stages=swap_stages,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
