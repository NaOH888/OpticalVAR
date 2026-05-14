from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


SPLIT_TO_PARTITION = {
    "train": 0,
    "valid": 1,
    "test": 2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare grayscale 176x176 CelebA NPZ shards directly from zip.")
    parser.add_argument("--root", type=Path, default=Path("dataset/celeba"), help="Dataset output root.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "valid", "test", "all"))
    parser.add_argument("--crop-size", type=int, default=176, help="Center crop size. Default keeps cVAE-compatible 176.")
    parser.add_argument("--shard-size", type=int, default=4096, help="Samples per NPZ shard. Use a positive value.")
    parser.add_argument("--compress", action="store_true", help="Use np.savez_compressed for smaller but slower output.")
    return parser.parse_args(argv)


def _read_partition_filenames(partition_path: Path, *, split: str) -> list[str]:
    selected: list[str] = []
    target_partition = SPLIT_TO_PARTITION.get(split)
    with partition_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            filename, partition_text = line.split()
            if split == "all" or int(partition_text) == target_partition:
                selected.append(filename)
    return selected


def _read_attr_table(attr_path: Path, *, selected_filenames: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    with attr_path.open("r", encoding="utf-8") as handle:
        _ = handle.readline()
        attr_names = handle.readline().split()
        attr_map: dict[str, np.ndarray] = {}
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            filename = parts[0]
            if filename not in selected_filenames:
                continue
            attr_values = np.asarray(parts[1:], dtype=np.int16)
            # CelebA attributes are stored as -1 / 1; convert to 0 / 1.
            attr_map[filename] = ((attr_values + 1) // 2).astype(np.float32, copy=False)
    return attr_names, attr_map


def _center_crop_rgb(image: Image.Image, crop_size: int) -> np.ndarray:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if crop_size > width or crop_size > height:
        raise ValueError(
            f"crop_size {crop_size} exceeds image size {(width, height)}"
        )
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped = rgb.crop((left, top, left + crop_size, top + crop_size))
    rgb_array = np.asarray(cropped, dtype=np.float32) / 255.0
    gray = (
        0.299 * rgb_array[..., 0]
        + 0.587 * rgb_array[..., 1]
        + 0.114 * rgb_array[..., 2]
    )
    return gray[np.newaxis, ...].astype(np.float16, copy=False)


def _save_shard(
    *,
    shard_index: int,
    shard_images: list[np.ndarray],
    shard_labels: list[np.ndarray],
    shard_sample_ids: list[int],
    output_prefix: str,
    root: Path,
    compress: bool,
) -> dict[str, object]:
    shard_name = f"{output_prefix}_part{shard_index:04d}.npz"
    shard_path = root / shard_name
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        shard_path,
        images=np.stack(shard_images, axis=0).astype(np.float16, copy=False),
        labels=np.stack(shard_labels, axis=0).astype(np.float32, copy=False),
        sample_ids=np.asarray(shard_sample_ids, dtype=np.int64),
    )
    return {
        "filename": shard_name,
        "num_items": int(len(shard_images)),
        "sample_id_start": int(shard_sample_ids[0]),
        "sample_id_end": int(shard_sample_ids[-1]),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.shard_size) <= 0:
        raise ValueError(f"shard_size must be positive, got {args.shard_size!r}")

    repo_root = Path(__file__).resolve().parents[2]
    root = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()
    celeba_root = root / "raw" / "celeba"
    zip_path = celeba_root / "img_align_celeba.zip"
    attr_path = celeba_root / "list_attr_celeba.txt"
    partition_path = celeba_root / "list_eval_partition.txt"

    for required_path in (zip_path, attr_path, partition_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required CelebA file not found: {required_path}")

    filenames = _read_partition_filenames(partition_path, split=str(args.split))
    filename_set = set(filenames)
    attr_names, attr_map = _read_attr_table(attr_path, selected_filenames=filename_set)
    missing_attrs = [filename for filename in filenames if filename not in attr_map]
    if missing_attrs:
        raise ValueError(f"Missing attribute rows for {len(missing_attrs)} files, first={missing_attrs[0]!r}")

    root.mkdir(parents=True, exist_ok=True)
    output_prefix = f"celeba_{args.split}_gray_{int(args.crop_size)}"

    shard_images: list[np.ndarray] = []
    shard_labels: list[np.ndarray] = []
    shard_sample_ids: list[int] = []
    shard_records: list[dict[str, object]] = []
    shard_index = 0

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        for sample_id, filename in enumerate(filenames):
            member_name = f"img_align_celeba/{filename}"
            with archive.open(member_name, mode="r") as member_handle:
                with Image.open(member_handle) as image:
                    shard_images.append(_center_crop_rgb(image, int(args.crop_size)))
            shard_labels.append(attr_map[filename])
            shard_sample_ids.append(sample_id)

            if len(shard_images) >= int(args.shard_size):
                shard_records.append(
                    _save_shard(
                        shard_index=shard_index,
                        shard_images=shard_images,
                        shard_labels=shard_labels,
                        shard_sample_ids=shard_sample_ids,
                        output_prefix=output_prefix,
                        root=root,
                        compress=bool(args.compress),
                    )
                )
                shard_index += 1
                shard_images = []
                shard_labels = []
                shard_sample_ids = []
                print(f"[prepare_celeba] saved shard {shard_index}, processed {sample_id + 1}/{len(filenames)}")
            elif (sample_id + 1) % 5000 == 0:
                print(f"[prepare_celeba] processed {sample_id + 1}/{len(filenames)}")

    if shard_images:
        shard_records.append(
            _save_shard(
                shard_index=shard_index,
                shard_images=shard_images,
                shard_labels=shard_labels,
                shard_sample_ids=shard_sample_ids,
                output_prefix=output_prefix,
                root=root,
                compress=bool(args.compress),
            )
        )
        print(f"[prepare_celeba] saved shard {shard_index + 1}, processed {len(filenames)}/{len(filenames)}")

    manifest = {
        "dataset_name": "celeba",
        "split": args.split,
        "num_items": int(len(filenames)),
        "image_key": "images",
        "label_key": "labels",
        "sample_id_key": "sample_ids",
        "npz_files": [record["filename"] for record in shard_records],
        "shards": shard_records,
        "label_names": attr_names,
        "label_semantics": "celeba_40_attributes_binary_0_1",
        "image_shape_nchw": [int(len(filenames)), 1, int(args.crop_size), int(args.crop_size)],
        "image_dtype": "float16",
        "image_value_range": [0.0, 1.0],
        "crop_size": int(args.crop_size),
        "grayscale_formula": "0.299*R + 0.587*G + 0.114*B",
        "channel_mode": "keep",
        "raw_zip_path": str(zip_path),
        "attribute_path": str(attr_path),
        "partition_path": str(partition_path),
        "shard_size": int(args.shard_size),
        "compress": bool(args.compress),
    }
    manifest_path = root / f"{output_prefix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "num_items": int(len(filenames)),
                "num_shards": int(len(shard_records)),
                "first_shard": shard_records[0]["filename"] if shard_records else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
