from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class SourceConfig:
    wavelengths_m: tuple[float, ...]
    light_mode: Literal["phase", "amplitude", "intensity"]
    amplitude: float


@dataclass(slots=True)
class DetectorConfig:
    width_num: int
    height_num: int
    detector_unit_len_m: float


@dataclass(slots=True)
class PropagationErrorConfig:
    delta_z_m: float
    shift_x_m: float
    shift_y_m: float


@dataclass(slots=True)
class PropagationConfig:
    canvas_h: int | None
    canvas_w: int | None
    canvas_factor: float
    refractive_index: float
    use_bandlimit_window: bool
    evanescent_mode: Literal["keep", "cut"]
    fft_norm: Literal["ortho", "backward", "forward"]


@dataclass(slots=True)
class FDTDSourceConfig:
    name: str = "src"
    injection_axis: str = "z"
    direction: str = "Forward"


@dataclass(slots=True)
class FDTDMonitorConfig:
    name: str
    monitor_type: str = "2D Z-normal"


@dataclass(slots=True)
class FDTDProbeConfig:
    enabled: bool = False
    name: str = "field_probe"
    monitor_type: str = "2D Z-normal"


@dataclass(slots=True)
class FDTDMetaConfig:
    min_feature_m: float = 0.0


@dataclass(slots=True)
class LUTManifest:
    name: str
    format: str
    data_file: Path
    fields: dict[str, str]
    constraints: dict[str, Any]
    defaults: dict[str, Any]
    units: dict[str, str]

    @classmethod
    def from_file(cls, path: str | Path) -> "LUTManifest":
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"LUT manifest not found: {manifest_path}")

        suffix = manifest_path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        elif suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError("PyYAML is required to load YAML LUT manifests") from exc
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported LUT manifest format: {manifest_path.suffix}")

        data_file = (manifest_path.parent / payload["data_file"]).resolve()
        return cls(
            name=str(payload["name"]),
            format=str(payload["format"]),
            data_file=data_file,
            fields=dict(payload.get("fields", {})),
            constraints=dict(payload.get("constraints", {})),
            defaults=dict(payload.get("defaults", {})),
            units=dict(payload.get("units", {})),
        )
