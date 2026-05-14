from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _unique_sorted_float(values: Iterable[str]) -> list[float]:
    return sorted({float(value) for value in values})


def _relative_or_name(target: Path, base_dir: Path) -> str:
    try:
        return str(target.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return target.name


def _infer_rcwa_summary(csv_path: Path) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"RCWA csv is empty: {csv_path}")

    sample_row = rows[0]
    required_columns = ("T0_Ex", "wx_0", "wy_0", "laythick_0", "wavelength")
    missing_columns = [name for name in required_columns if name not in sample_row]
    if missing_columns:
        raise KeyError(f"RCWA csv is missing required columns: {missing_columns}")

    wavelengths_nm = _unique_sorted_float(row["wavelength"] for row in rows)
    height_nm = _unique_sorted_float(row["laythick_0"] for row in rows)
    wx_nm = _unique_sorted_float(row["wx_0"] for row in rows)
    wy_nm = _unique_sorted_float(row["wy_0"] for row in rows)

    return {
        "row_count": len(rows),
        "wavelengths_nm": wavelengths_nm,
        "height_nm": height_nm,
        "wx_nm": wx_nm,
        "wy_nm": wy_nm,
        "source_complex_col": "T0_Ex",
        "source_wx_col": "wx_0",
        "source_wy_col": "wy_0",
        "source_height_col": "laythick_0",
        "source_wavelength_col": "wavelength",
        "grid_shape_nh_nwx_nwy_nl": [
            len(height_nm),
            len(wx_nm),
            len(wy_nm),
            len(wavelengths_nm),
        ],
    }


def _validate_npz(npz_path: Path, summary: dict[str, object]) -> None:
    npz = np.load(npz_path, allow_pickle=True)
    required_keys = {
        "height_vec_m",
        "wx_vec_m",
        "wy_vec_m",
        "wavelength_vec_m",
        "phase_map_rad",
        "amp_map",
    }
    missing_keys = required_keys.difference(npz.files)
    if missing_keys:
        raise KeyError(f"LUT npz is missing required keys: {sorted(missing_keys)}")

    expected_shape = tuple(summary["grid_shape_nh_nwx_nwy_nl"])
    phase_shape = tuple(npz["phase_map_rad"].shape)
    amp_shape = tuple(npz["amp_map"].shape)
    if phase_shape != expected_shape or amp_shape != expected_shape:
        raise ValueError(
            "LUT tensor shape mismatch: "
            f"expected={expected_shape}, phase_map_rad={phase_shape}, amp_map={amp_shape}"
        )


def build_manifest_payload(
    *,
    dataset_name: str,
    csv_path: Path,
    npz_path: Path,
    output_path: Path,
) -> dict[str, object]:
    summary = _infer_rcwa_summary(csv_path)
    _validate_npz(npz_path, summary)

    output_dir = output_path.resolve().parent
    return {
        "name": dataset_name,
        "format": "npz",
        "data_file": _relative_or_name(npz_path, output_dir),
        "fields": {
            "height": "height_vec_m",
            "wx": "wx_vec_m",
            "wy": "wy_vec_m",
            "wavelength": "wavelength_vec_m",
            "phase": "phase_map_rad",
            "amp": "amp_map",
        },
        "constraints": {
            "grid_shape_nh_nwx_nwy_nl": summary["grid_shape_nh_nwx_nwy_nl"],
            "height_nm_list": summary["height_nm"],
            "wx_nm_list": summary["wx_nm"],
            "wy_nm_list": summary["wy_nm"],
            "wavelength_nm_list": summary["wavelengths_nm"],
        },
        "defaults": {
            "raw_csv": _relative_or_name(csv_path, output_dir),
            "source_complex_col": summary["source_complex_col"],
            "source_wx_col": summary["source_wx_col"],
            "source_wy_col": summary["source_wy_col"],
            "source_height_col": summary["source_height_col"],
            "source_wavelength_col": summary["source_wavelength_col"],
        },
        "units": {
            "height": "m",
            "wx": "m",
            "wy": "m",
            "wavelength": "m",
            "raw_csv_height": "nm",
            "raw_csv_wx": "nm",
            "raw_csv_wy": "nm",
            "raw_csv_wavelength": "nm",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LUT meta/manifest json from RCWA raw csv and LUT npz.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to RCWA raw result_table csv.")
    parser.add_argument("--npz", type=Path, required=True, help="Path to LUT npz derived from the RCWA csv.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest json path.")
    parser.add_argument("--name", type=str, default=None, help="Dataset name written into the manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    npz_path = args.npz.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = args.name or output_path.parent.name
    payload = build_manifest_payload(
        dataset_name=dataset_name,
        csv_path=csv_path,
        npz_path=npz_path,
        output_path=output_path,
    )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote RCWA meta to {output_path}")


if __name__ == "__main__":
    main()
