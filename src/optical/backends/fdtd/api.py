from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _candidate_api_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("LUMERICAL_API_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    program_roots = (
        Path(r"C:\Program Files"),
        Path(r"D:\Program Files"),
    )
    for version in ("v242", "v241", "v232", "v231", "v222", "v221"):
        for root in program_roots:
            candidates.extend(
                [
                    root / "Lumerical" / version / "api" / "python",
                    root / "Lumerical" / f"{version}.0" / "api" / "python",
                    root / "Ansys Inc" / version / "Lumerical" / "api" / "python",
                    root / "Ansys Inc" / f"{version}.0" / "Lumerical" / "api" / "python",
                ]
            )

    return candidates


def ensure_lumapi_on_sys_path(extra_paths: list[str | Path] | None = None) -> Path:
    search_paths: list[Path] = []
    if extra_paths is not None:
        search_paths.extend(Path(path) for path in extra_paths)
    search_paths.extend(_candidate_api_paths())

    for path in search_paths:
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        return path

    raise ModuleNotFoundError(
        "Could not locate the Lumerical Python API path. "
        "Set LUMERICAL_API_PATH or add the api/python directory to sys.path."
    )


def load_lumapi(extra_paths: list[str | Path] | None = None) -> Any:
    try:
        import lumapi  # type: ignore
    except ModuleNotFoundError:
        ensure_lumapi_on_sys_path(extra_paths=extra_paths)
        import lumapi  # type: ignore
    return lumapi


@dataclass(slots=True)
class FDTDSpec:
    hide: bool = True
    server_args: dict[str, Any] | None = None
    extra_api_paths: list[str | Path] | None = None


@dataclass(slots=True)
class FDTDRegionSpec:
    x_span: float
    y_span: float
    z_span: float
    center_z: float = 0.0
    dimension: str = "3D"
    simulation_time: float = 200e-15
    mesh_accuracy: int = 2
    x_min_bc: str = "PML"
    x_max_bc: str = "PML"
    y_min_bc: str = "PML"
    y_max_bc: str = "PML"
    z_min_bc: str = "PML"
    z_max_bc: str = "PML"
    # 用于粗扫阶段提前收敛，后续若发现误差大可再单点收紧。
    auto_shutoff_min: float | None = None


@dataclass(slots=True)
class FDTDMeshRegionSpec:
    name: str
    x: float
    y: float
    z: float
    x_span: float
    y_span: float
    z_span: float
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None


@dataclass(slots=True)
class FDTDLayerContext:
    # Python 抽象传播链里的层参考面坐标。
    z: float
    # 当前层在 FDTD 几何里的真实底面 z。
    z_bottom: float
    # 当前层前方抽象平面与本层抽象平面之间的空气间隔。
    air_gap_before: float


class LumericalFDTD:
    """
    Thin wrapper around `lumapi.FDTD`.

    This keeps the `lumapi` import logic in one place and gives the rest of the
    codebase a stable entry point for future FDTD helpers.
    """

    def __init__(self, spec: FDTDSpec | None = None):
        self.spec = spec or FDTDSpec()
        self.lumapi = load_lumapi(extra_paths=self.spec.extra_api_paths)
        kwargs = dict(self.spec.server_args or {})
        kwargs["hide"] = self.spec.hide
        self.session = self.lumapi.FDTD(**kwargs)

    @property
    def module_path(self) -> str:
        return getattr(self.lumapi, "__file__", "<unknown>")

    def close(self) -> None:
        self.session.close()

    def new_project(self) -> None:
        self.session.newproject()

    def save(self, path: str | Path) -> None:
        self.session.save(str(path))

    def run(self) -> None:
        self.session.run()

    def get_result(self, object_name: str, result_name: str) -> Any:
        return self.session.getresult(object_name, result_name)

    def eval(self, script: str) -> Any:
        return self.session.eval(script)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.session, item)

    def __enter__(self) -> "LumericalFDTD":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FDTDBuilder:
    """Small adapter over `lumapi.FDTD` used by optic layers."""

    def __init__(self, fdtd: LumericalFDTD):
        self.fdtd = fdtd

    def reset_session(self) -> None:
        self.fdtd.new_project()

    def add_fdtd_region(self, spec: FDTDRegionSpec) -> None:
        self.fdtd.addfdtd()
        self.fdtd.set("dimension", spec.dimension)
        self.fdtd.set("x span", spec.x_span)
        self.fdtd.set("y span", spec.y_span)
        self.fdtd.set("z span", spec.z_span)
        self.fdtd.set("z", spec.center_z)
        self.fdtd.set("simulation time", spec.simulation_time)
        self.fdtd.set("mesh accuracy", spec.mesh_accuracy)
        self.fdtd.set("x min bc", spec.x_min_bc)
        self.fdtd.set("x max bc", spec.x_max_bc)
        self.fdtd.set("y min bc", spec.y_min_bc)
        self.fdtd.set("y max bc", spec.y_max_bc)
        self.fdtd.set("z min bc", spec.z_min_bc)
        self.fdtd.set("z max bc", spec.z_max_bc)
        if spec.auto_shutoff_min is not None:
            self.fdtd.set("auto shutoff min", spec.auto_shutoff_min)

    def add_rect(
        self,
        name: str,
        material: str,
        x: float,
        y: float,
        z: float,
        x_span: float,
        y_span: float,
        z_span: float,
    ) -> None:
        self.fdtd.addrect()
        self.fdtd.set("name", name)
        self.fdtd.set("material", material)
        self.fdtd.set("x", x)
        self.fdtd.set("y", y)
        self.fdtd.set("z", z)
        self.fdtd.set("x span", x_span)
        self.fdtd.set("y span", y_span)
        self.fdtd.set("z span", z_span)

    def add_plane_source(
        self,
        name: str,
        injection_axis: str,
        direction: str,
        x_span: float,
        y_span: float,
        z: float,
        wavelength_start: float,
        wavelength_stop: float,
        theta_deg: float | None = None,
        phi_deg: float | None = None,
        polarization_angle_deg: float | None = None,
    ) -> None:
        self.fdtd.addplane()
        self.fdtd.set("name", name)
        self.fdtd.set("injection axis", injection_axis)
        self.fdtd.set("direction", direction)
        self.fdtd.set("x span", x_span)
        self.fdtd.set("y span", y_span)
        self.fdtd.set("z", z)
        self.fdtd.set("wavelength start", wavelength_start)
        self.fdtd.set("wavelength stop", wavelength_stop)
        if theta_deg is not None:
            self.fdtd.set("angle theta", theta_deg)
        if phi_deg is not None:
            self.fdtd.set("angle phi", phi_deg)
        if polarization_angle_deg is not None:
            self.fdtd.set("polarization angle", polarization_angle_deg)

    def add_power_monitor(
        self,
        name: str,
        monitor_type: str,
        x_span: float,
        y_span: float,
        z: float,
        override_global_monitor_settings: bool | None = None,
        use_source_limits: bool | None = None,
        frequency_points: int | None = None,
    ) -> None:
        self.fdtd.addpower()
        self.fdtd.set("name", name)
        self.fdtd.set("monitor type", monitor_type)
        self.fdtd.set("x span", x_span)
        self.fdtd.set("y span", y_span)
        self.fdtd.set("z", z)
        if override_global_monitor_settings is not None:
            self.fdtd.set("override global monitor settings", int(bool(override_global_monitor_settings)))
        if use_source_limits is not None:
            self.fdtd.set("use source limits", int(bool(use_source_limits)))
        if frequency_points is not None:
            self.fdtd.set("frequency points", int(frequency_points))

    def add_profile_monitor(
        self,
        name: str,
        monitor_type: str,
        x_span: float,
        y_span: float,
        z: float,
        override_global_monitor_settings: bool | None = None,
        use_source_limits: bool | None = None,
        frequency_points: int | None = None,
    ) -> None:
        self.fdtd.addprofile()
        self.fdtd.set("name", name)
        self.fdtd.set("monitor type", monitor_type)
        self.fdtd.set("x span", x_span)
        self.fdtd.set("y span", y_span)
        self.fdtd.set("z", z)
        if override_global_monitor_settings is not None:
            self.fdtd.set("override global monitor settings", int(bool(override_global_monitor_settings)))
        if use_source_limits is not None:
            self.fdtd.set("use source limits", int(bool(use_source_limits)))
        if frequency_points is not None:
            self.fdtd.set("frequency points", int(frequency_points))

    def add_mesh_region(self, spec: FDTDMeshRegionSpec) -> None:
        # 单周期验证时需要在结构附近单独收紧网格，避免完全依赖全局 mesh accuracy。
        self.fdtd.addmesh()
        self.fdtd.set("name", spec.name)
        self.fdtd.set("x", spec.x)
        self.fdtd.set("y", spec.y)
        self.fdtd.set("z", spec.z)
        self.fdtd.set("x span", spec.x_span)
        self.fdtd.set("y span", spec.y_span)
        self.fdtd.set("z span", spec.z_span)
        if spec.dx is not None:
            self.fdtd.set("override x mesh", 1)
            self.fdtd.set("dx", spec.dx)
        if spec.dy is not None:
            self.fdtd.set("override y mesh", 1)
            self.fdtd.set("dy", spec.dy)
        if spec.dz is not None:
            self.fdtd.set("override z mesh", 1)
            self.fdtd.set("dz", spec.dz)

    def add_structure_group(self, name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.fdtd.addstructuregroup()
        self.fdtd.set("name", name)
        self.fdtd.set("x", x)
        self.fdtd.set("y", y)
        self.fdtd.set("z", z)

    def eval_script(self, script: str) -> Any:
        return self.fdtd.eval(script)
