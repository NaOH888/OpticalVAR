from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzImageDataset(Dataset):
    """Load image/label pairs from an offline NPZ archive."""

    def __init__(
        self,
        npz_path: str | Path | list[str] | list[Path] | tuple[str | Path, ...],
        *,
        image_key: str | None = "images",
        label_key: str | None = "labels",
        latent_key: str | None = None,
        latent_source: str | None = None,
        latent_type: str | None = None,
        latent_spec: dict | None = None,
        sample_id_key: str | None = None,
        max_items: int | None = None,
        dtype: torch.dtype = torch.float32,
        channel_mode: str = "keep",
    ) -> None:
        raw_paths = npz_path if isinstance(npz_path, (list, tuple)) else [npz_path]
        self.npz_paths = [Path(path) for path in raw_paths]
        if not self.npz_paths:
            raise ValueError("npz_path must contain at least one NPZ path")
        for path in self.npz_paths:
            if not path.exists():
                raise FileNotFoundError(f"NPZ dataset not found: {path}")

        self.image_key = None if image_key is None else str(image_key)
        self.label_key = None if label_key is None else str(label_key)
        self.latent_key = None if latent_key is None else str(latent_key)
        self.latent_source = None if latent_source is None else str(latent_source)
        self.latent_type = None if latent_type is None else str(latent_type)
        self.latent_spec = {} if latent_spec is None else dict(latent_spec)
        self.sample_id_key = None if sample_id_key is None else str(sample_id_key)
        self.dtype = dtype
        self.channel_mode = str(channel_mode)
        if self.channel_mode not in {"keep", "first", "mean"}:
            raise ValueError(
                "channel_mode must be one of {'keep', 'first', 'mean'}, "
                f"got {self.channel_mode!r}"
            )
        if self.latent_source is not None and self.latent_source not in {"cvae", "rvq"}:
            raise ValueError(
                "latent_source must be one of {'cvae', 'rvq'} when provided, "
                f"got {self.latent_source!r}"
            )
        if self.latent_type is not None and self.latent_type not in {"continuous_map", "discrete_code"}:
            raise ValueError(
                "latent_type must be one of {'continuous_map', 'discrete_code'} when provided, "
                f"got {self.latent_type!r}"
            )

        self.archives: list[np.lib.npyio.NpzFile] | None = None
        self.images: list[np.ndarray] | None = None
        self.labels: list[np.ndarray] | None = None
        self.latents: list[np.ndarray] | None = None
        self.sample_ids: list[np.ndarray] | None = None
        self.shard_lengths: list[int] = []
        self.cumulative_lengths: list[int] = []

        self._open_archives()
        if self.images is None and self.labels is None and self.latents is None and self.sample_ids is None:
            raise RuntimeError("failed to initialize NPZ dataset archives")
        total_items = 0
        shard_sources = self.images or self.labels or self.latents or self.sample_ids
        if shard_sources is None:
            raise RuntimeError("failed to resolve shard lengths for NPZ dataset")
        for array in shard_sources:
            shard_length = int(array.shape[0])
            total_items += shard_length
            self.shard_lengths.append(shard_length)
            self.cumulative_lengths.append(total_items)

        if max_items is None:
            self.length = total_items
        else:
            self.length = min(int(max_items), total_items)

    def _open_archives(self) -> None:
        if self.archives is not None:
            return
        archives = [np.load(path, allow_pickle=False) for path in self.npz_paths]
        images: list[np.ndarray] | None = [] if self.image_key is not None else None
        labels: list[np.ndarray] | None = [] if self.label_key is not None else None
        latents: list[np.ndarray] | None = [] if self.latent_key is not None else None
        sample_ids: list[np.ndarray] | None = [] if self.sample_id_key is not None else None
        for archive, path in zip(archives, self.npz_paths):
            if images is not None:
                if self.image_key not in archive.files:
                    raise KeyError(
                        f"image_key={self.image_key!r} is not present in {path.name}: {sorted(archive.files)}"
                    )
                images.append(archive[self.image_key])
            if labels is not None:
                if self.label_key not in archive.files:
                    raise KeyError(
                        f"label_key={self.label_key!r} is not present in {path.name}: {sorted(archive.files)}"
                    )
                labels.append(archive[self.label_key])
            if latents is not None:
                if self.latent_key not in archive.files:
                    raise KeyError(
                        f"latent_key={self.latent_key!r} is not present in {path.name}: {sorted(archive.files)}"
                    )
                latents.append(archive[self.latent_key])
            if sample_ids is not None:
                if self.sample_id_key not in archive.files:
                    raise KeyError(
                        f"sample_id_key={self.sample_id_key!r} is not present in {path.name}: {sorted(archive.files)}"
                    )
                sample_ids.append(archive[self.sample_id_key])
        self.archives = archives
        self.images = images
        self.labels = labels
        self.latents = latents
        self.sample_ids = sample_ids

    def close(self) -> None:
        if self.archives is not None:
            for archive in self.archives:
                archive.close()
        self.archives = None
        self.images = None
        self.labels = None if self.label_key is None else []
        self.latents = None if self.latent_key is None else []
        self.sample_ids = None if self.sample_id_key is None else []

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        archives = state.get("archives")
        if archives is not None:
            for archive in archives:
                archive.close()
        state["archives"] = None
        state["images"] = None
        state["labels"] = None if self.label_key is None else []
        state["latents"] = None if self.latent_key is None else []
        state["sample_ids"] = None if self.sample_id_key is None else []
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        max_items: int | None = None,
        dtype: torch.dtype = torch.float32,
        channel_mode: str = "keep",
    ) -> "NpzImageDataset":
        manifest_file = Path(manifest_path)
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        if "npz_files" in payload:
            npz_path = [manifest_file.parent / str(filename) for filename in payload["npz_files"]]
        else:
            npz_path = manifest_file.parent / manifest_file.name.replace(".json", ".npz")
        if "image_key" in payload:
            image_key = None if payload["image_key"] is None else str(payload["image_key"])
        else:
            legacy_image_key = payload.get("config", {}).get("image_key", "images")
            image_key = None if legacy_image_key is None else str(legacy_image_key)
        label_key = payload.get("label_key", payload.get("config", {}).get("label_key", "labels"))
        latent_key = payload.get("latent_key", payload.get("config", {}).get("latent_key"))
        latent_source = payload.get("latent_source")
        latent_type = payload.get("latent_type")
        latent_spec = payload.get("latent_spec")
        sample_id_key = payload.get("sample_id_key", payload.get("config", {}).get("sample_id_key"))
        return cls(
            npz_path=npz_path,
            image_key=image_key,
            label_key=None if label_key is None else str(label_key),
            latent_key=None if latent_key is None else str(latent_key),
            latent_source=None if latent_source is None else str(latent_source),
            latent_type=None if latent_type is None else str(latent_type),
            latent_spec=None if latent_spec is None else dict(latent_spec),
            sample_id_key=None if sample_id_key is None else str(sample_id_key),
            max_items=max_items,
            dtype=dtype,
            channel_mode=channel_mode,
        )

    def __len__(self) -> int:
        return self.length

    def _resolve_location(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.length:
            raise IndexError(f"index out of range: {index}")
        shard_index = bisect.bisect_right(self.cumulative_lengths, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
        local_index = index - shard_start
        return shard_index, local_index

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open_archives()
        if self.images is None and self.labels is None and self.latents is None and self.sample_ids is None:
            raise RuntimeError("dataset archives are not available")
        shard_index, local_index = self._resolve_location(index)
        if self.sample_ids is not None and len(self.sample_ids) > 0:
            sample_id_value = self.sample_ids[shard_index][local_index]
        else:
            sample_id_value = index
        sample: dict[str, torch.Tensor] = {
            "sample_id": torch.as_tensor(sample_id_value, dtype=torch.long),
        }
        if self.images is not None and len(self.images) > 0:
            image = torch.as_tensor(self.images[shard_index][local_index], dtype=self.dtype)
            if image.dim() == 3 and self.channel_mode == "first":
                image = image[:1]
            elif image.dim() == 3 and self.channel_mode == "mean":
                image = image.mean(dim=0, keepdim=True)
            sample["image"] = image
        if self.labels is not None and len(self.labels) > 0:
            label_value = self.labels[shard_index][local_index]
            if np.ndim(label_value) == 0:
                sample["label"] = torch.as_tensor(label_value, dtype=torch.long)
            else:
                sample["label"] = torch.as_tensor(label_value, dtype=self.dtype)
        if self.latents is not None and len(self.latents) > 0:
            latent_value = self.latents[shard_index][local_index]
            if self.latent_type == "discrete_code":
                sample["latent"] = torch.as_tensor(latent_value, dtype=torch.long)
            else:
                sample["latent"] = torch.as_tensor(latent_value, dtype=self.dtype)
        return sample


class ReferencedImageLatentDataset(Dataset):
    """Combine an image manifest and a latent-only manifest aligned by sample_id and order."""

    def __init__(
        self,
        image_dataset: NpzImageDataset,
        latent_dataset: NpzImageDataset,
    ) -> None:
        self.image_dataset = image_dataset
        self.latent_dataset = latent_dataset
        if len(self.image_dataset) != len(self.latent_dataset):
            raise ValueError(
                f"image and latent dataset lengths must match, got {len(self.image_dataset)} vs {len(self.latent_dataset)}"
            )

    @classmethod
    def from_latent_manifest(
        cls,
        manifest_path: str | Path,
        *,
        max_items: int | None = None,
        dtype: torch.dtype = torch.float32,
        channel_mode: str = "keep",
    ) -> "ReferencedImageLatentDataset":
        manifest_file = Path(manifest_path)
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        image_manifest_path = payload.get("image_manifest_path")
        if image_manifest_path is None:
            raise KeyError("latent manifest must contain image_manifest_path")
        image_manifest = (manifest_file.parent / str(image_manifest_path)).resolve()
        image_dataset = NpzImageDataset.from_manifest(
            image_manifest,
            max_items=max_items,
            dtype=dtype,
            channel_mode=channel_mode,
        )
        latent_dataset = NpzImageDataset.from_manifest(
            manifest_file,
            max_items=max_items,
            dtype=dtype,
            channel_mode=channel_mode,
        )
        return cls(image_dataset=image_dataset, latent_dataset=latent_dataset)

    def close(self) -> None:
        self.image_dataset.close()
        self.latent_dataset.close()

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_sample = self.image_dataset[index]
        latent_sample = self.latent_dataset[index]
        image_sample_id = int(image_sample["sample_id"])
        latent_sample_id = int(latent_sample["sample_id"])
        if image_sample_id != latent_sample_id:
            raise ValueError(
                f"image and latent sample_id mismatch at index {index}: {image_sample_id} vs {latent_sample_id}"
            )
        output = {
            "image": image_sample["image"],
            "sample_id": image_sample["sample_id"],
            "latent": latent_sample["latent"],
        }
        if "label" in latent_sample:
            output["label"] = latent_sample["label"]
        elif "label" in image_sample:
            output["label"] = image_sample["label"]
        return output
