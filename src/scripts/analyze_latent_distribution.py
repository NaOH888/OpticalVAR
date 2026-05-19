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
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from optical.data import NpzImageDataset


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_optional_label_name(manifest: dict[str, Any], label_index: int | None) -> str | None:
    if label_index is None:
        return None
    label_names = manifest.get("label_names")
    if isinstance(label_names, list) and 0 <= label_index < len(label_names):
        return str(label_names[label_index])
    return f"label_{label_index}"


def _load_arrays(
    *,
    manifest_path: Path,
    max_items: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    dataset = NpzImageDataset.from_manifest(manifest_path, max_items=max_items, channel_mode="keep")
    latents: list[np.ndarray] = []
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            if "latent" not in sample:
                raise KeyError("manifest dataset does not contain 'latent'")
            latents.append(sample["latent"].cpu().numpy().astype(np.float32, copy=False))
            if "image" in sample:
                images.append(sample["image"].cpu().numpy().astype(np.float32, copy=False))
            if "label" in sample:
                labels.append(sample["label"].cpu().numpy())
    finally:
        dataset.close()
    latent_array = np.stack(latents, axis=0)
    image_array = np.stack(images, axis=0) if images else None
    if not labels:
        label_array = None
    else:
        first = labels[0]
        if np.ndim(first) == 0:
            label_array = np.asarray(labels, dtype=np.int64)
        else:
            label_array = np.stack(labels, axis=0)
    return latent_array, image_array, label_array


def _pick_hist_values(latents: np.ndarray, *, max_values: int, rng: np.random.Generator) -> np.ndarray:
    flat = latents.reshape(-1)
    if flat.size <= max_values:
        return flat
    indices = rng.choice(flat.size, size=max_values, replace=False)
    return flat[indices]


def _compute_pairwise_distance_stats(flat_latents: np.ndarray, *, sample_count: int, rng: np.random.Generator) -> dict[str, float]:
    num_items = int(flat_latents.shape[0])
    if num_items < 2:
        return {
            "pairwise_l2_mean": 0.0,
            "pairwise_l2_std": 0.0,
            "pairwise_cosine_mean": 0.0,
            "pairwise_cosine_std": 0.0,
        }
    count = min(sample_count, num_items)
    if count < num_items:
        indices = rng.choice(num_items, size=count, replace=False)
        subset = flat_latents[indices]
    else:
        subset = flat_latents
    l2 = pairwise_distances(subset, metric="euclidean")
    cosine = pairwise_distances(subset, metric="cosine")
    mask = np.triu(np.ones_like(l2, dtype=bool), k=1)
    l2_values = l2[mask]
    cosine_values = cosine[mask]
    return {
        "pairwise_l2_mean": float(l2_values.mean()),
        "pairwise_l2_std": float(l2_values.std()),
        "pairwise_cosine_mean": float(cosine_values.mean()),
        "pairwise_cosine_std": float(cosine_values.std()),
    }


def _save_pca_variance_plot(output_path: Path, explained_ratio: np.ndarray) -> None:
    cumulative = np.cumsum(explained_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, explained_ratio.size + 1), explained_ratio, marker="o", linewidth=1.5)
    axes[0].set_title("PCA Explained Variance")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance Ratio")
    axes[0].grid(alpha=0.3)
    axes[1].plot(np.arange(1, cumulative.size + 1), cumulative, marker="o", linewidth=1.5)
    axes[1].set_title("PCA Cumulative Variance")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Cumulative Ratio")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resolve_scatter_colors(
    *,
    embedding_2d: np.ndarray,
    labels: np.ndarray | None,
    label_index: int | None,
) -> tuple[np.ndarray, str]:
    if labels is None or label_index is None:
        return embedding_2d[:, 0], "PC1"
    if labels.ndim == 1:
        return labels.astype(np.float32, copy=False), "label"
    if not (0 <= label_index < labels.shape[1]):
        return embedding_2d[:, 0], "PC1"
    return labels[:, label_index].astype(np.float32, copy=False), f"label[{label_index}]"


def _save_scatter_plot(
    output_path: Path,
    *,
    embedding_2d: np.ndarray,
    title: str,
    labels: np.ndarray | None,
    label_index: int | None,
    label_name: str | None,
) -> None:
    colors, color_label = _resolve_scatter_colors(
        embedding_2d=embedding_2d,
        labels=labels,
        label_index=label_index,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        cmap="viridis",
        s=10,
        alpha=0.75,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(label_name if label_name is not None else color_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_channel_stats_plot(output_path: Path, latents: np.ndarray) -> tuple[list[float], list[float]]:
    channel_mean = latents.mean(axis=(0, 2, 3))
    channel_std = latents.std(axis=(0, 2, 3))
    channels = np.arange(channel_mean.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(channels, channel_mean)
    axes[0].set_title("Channel Means")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Mean")
    axes[0].grid(alpha=0.2, axis="y")
    axes[1].bar(channels, channel_std)
    axes[1].set_title("Channel Std")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Std")
    axes[1].grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return channel_mean.astype(float).tolist(), channel_std.astype(float).tolist()


def _save_value_histogram(output_path: Path, values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=120, color="#2f6db3", alpha=0.85)
    ax.set_title("Latent Value Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pca_extreme_gallery(
    output_path: Path,
    *,
    images: np.ndarray | None,
    embedding_2d: np.ndarray,
) -> list[dict[str, int | float]]:
    if images is None:
        return []
    if images.ndim != 4 or images.shape[1] != 1:
        return []
    pc1 = embedding_2d[:, 0]
    pc2 = embedding_2d[:, 1]
    indices = [
        int(np.argmin(pc1)),
        int(np.argmax(pc1)),
        int(np.argmin(pc2)),
        int(np.argmax(pc2)),
    ]
    titles = ["PC1 min", "PC1 max", "PC2 min", "PC2 max"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    entries: list[dict[str, int | float]] = []
    for ax, index, title in zip(axes, indices, titles):
        ax.imshow(images[index, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title}\nidx={index}")
        ax.axis("off")
        entries.append(
            {
                "title": title,
                "index": index,
                "pc1": float(embedding_2d[index, 0]),
                "pc2": float(embedding_2d[index, 1]),
            }
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return entries


def _save_umap_plot(
    output_path: Path,
    *,
    flat_latents: np.ndarray,
    labels: np.ndarray | None,
    label_index: int | None,
    label_name: str | None,
    random_state: int,
) -> str:
    try:
        import umap  # type: ignore
    except ModuleNotFoundError:
        return "umap_not_installed"
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(flat_latents)
    _save_scatter_plot(
        output_path,
        embedding_2d=embedding,
        title="UMAP 2D Scatter",
        labels=labels,
        label_index=label_index,
        label_name=label_name,
    )
    return "ok"


def analyze_latent_distribution(
    *,
    manifest_path: Path,
    output_dir: Path,
    max_items: int | None,
    pca_components: int,
    hist_max_values: int,
    pairwise_sample_count: int,
    label_index: int | None,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    latents, images, labels = _load_arrays(manifest_path=manifest_path, max_items=max_items)
    flat_latents = latents.reshape(latents.shape[0], -1)
    latent_dim = int(flat_latents.shape[1])
    pca_dim = min(int(pca_components), int(flat_latents.shape[0]), latent_dim)
    if pca_dim < 2:
        raise ValueError("need at least 2 samples/components to visualize PCA")
    pca = PCA(n_components=pca_dim, random_state=random_seed)
    pca_embedding = pca.fit_transform(flat_latents)
    label_name = _resolve_optional_label_name(manifest, label_index)

    _save_pca_variance_plot(output_dir / "pca_variance.png", pca.explained_variance_ratio_)
    _save_scatter_plot(
        output_dir / "pca_scatter.png",
        embedding_2d=pca_embedding[:, :2],
        title="PCA 2D Scatter",
        labels=labels,
        label_index=label_index,
        label_name=label_name,
    )
    channel_mean, channel_std = _save_channel_stats_plot(output_dir / "channel_stats.png", latents)
    hist_values = _pick_hist_values(latents, max_values=hist_max_values, rng=rng)
    _save_value_histogram(output_dir / "latent_histogram.png", hist_values)
    pca_extremes = _save_pca_extreme_gallery(
        output_dir / "pca_extremes.png",
        images=images,
        embedding_2d=pca_embedding[:, :2],
    )
    umap_status = _save_umap_plot(
        output_dir / "umap_scatter.png",
        flat_latents=flat_latents,
        labels=labels,
        label_index=label_index,
        label_name=label_name,
        random_state=random_seed,
    )
    pairwise_stats = _compute_pairwise_distance_stats(
        flat_latents,
        sample_count=pairwise_sample_count,
        rng=rng,
    )
    summary = {
        "manifest_path": str(manifest_path),
        "num_items": int(latents.shape[0]),
        "latent_shape_nchw": list(latents.shape),
        "latent_dim": latent_dim,
        "pca_components": pca_dim,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "pca_cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).astype(float).tolist(),
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "global_mean": float(latents.mean()),
        "global_std": float(latents.std()),
        "global_min": float(latents.min()),
        "global_max": float(latents.max()),
        "label_index": label_index,
        "label_name": label_name,
        "umap_status": umap_status,
        "pca_extremes": pca_extremes,
        **pairwise_stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize latent distribution for NPZ latent datasets.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest path for latent dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write figures and summary.")
    parser.add_argument("--max-items", type=int, default=4096, help="Maximum number of samples to load.")
    parser.add_argument("--pca-components", type=int, default=32, help="Number of PCA components to fit.")
    parser.add_argument("--hist-max-values", type=int, default=200000, help="Maximum latent values sampled for histogram.")
    parser.add_argument("--pairwise-sample-count", type=int, default=512, help="Sample count used for pairwise distance stats.")
    parser.add_argument("--label-index", type=int, default=None, help="Optional label dimension for coloring scatter plots.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for subsampling and PCA/UMAP.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_latent_distribution(
        manifest_path=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        max_items=args.max_items,
        pca_components=args.pca_components,
        hist_max_values=args.hist_max_values,
        pairwise_sample_count=args.pairwise_sample_count,
        label_index=args.label_index,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
