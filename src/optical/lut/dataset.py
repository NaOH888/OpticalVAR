from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from optical.core.config import LUTManifest


class PhaseAmpHeightInterp(nn.Module):
    """
    LUT-backed phase/amplitude query provider.

    The LUT is defined on four discrete axes:
    - height
    - wx
    - wy
    - wavelength

    Wavelength is treated as a discrete key. Interpolation only happens in
    `(height, wx, wy)` on a single wavelength slice, and it is performed in the
    complex plane instead of directly on wrapped phase.
    """

    DEFAULT_KEY_MAP = {
        "height": "height_vec_m",
        "wx": "wx_vec_m",
        "wy": "wy_vec_m",
        "wavelength": "wavelength_vec_m",
        "phase": "phase_map_rad",
        "amp": "amp_map",
    }

    def __init__(
        self,
        height_vec_m: torch.Tensor,
        wx_vec_m: torch.Tensor,
        wy_vec_m: torch.Tensor,
        wavelength_vec_m: torch.Tensor,
        phase_map_rad: torch.Tensor,
        amp_map: torch.Tensor,
        wavelength_tol: float = 1e-12,
    ) -> None:
        super().__init__()
        self.register_buffer("height_vec_m", height_vec_m.flatten().float())
        self.register_buffer("wx_vec_m", wx_vec_m.flatten().float())
        self.register_buffer("wy_vec_m", wy_vec_m.flatten().float())
        self.register_buffer("wavelength_vec_m", wavelength_vec_m.flatten().float())
        self.register_buffer("phase_map_rad", phase_map_rad.float())
        self.register_buffer("amp_map", amp_map.float())
        self.wavelength_tol = float(wavelength_tol)

        if self.phase_map_rad.shape != self.amp_map.shape:
            raise ValueError("phase_map and amp_map must have identical shapes")

        expected_shape = (
            self.height_vec_m.numel(),
            self.wx_vec_m.numel(),
            self.wy_vec_m.numel(),
            self.wavelength_vec_m.numel(),
        )
        if tuple(self.phase_map_rad.shape) != expected_shape:
            raise ValueError(f"LUT shape should be {expected_shape}, got {tuple(self.phase_map_rad.shape)}")

    @classmethod
    def from_external_file(cls, file_path: str | Path, key_map: dict | None = None) -> "PhaseAmpHeightInterp":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"LUT file not found: {path}")

        resolved_key_map = dict(cls.DEFAULT_KEY_MAP)
        if key_map:
            resolved_key_map.update(key_map)

        suffix = path.suffix.lower()
        if suffix in [".npz", ".npy"]:
            obj = np.load(path, allow_pickle=True)
            getter = lambda key: obj[key]
        elif suffix == ".mat":
            try:
                import scipy.io as sio
            except Exception as exc:
                raise ImportError("scipy is required to load .mat files") from exc
            obj = sio.loadmat(path)
            getter = lambda key: obj[key]
        else:
            raise ValueError(f"Unsupported LUT format: {suffix}")

        height = torch.tensor(np.asarray(getter(resolved_key_map["height"])).squeeze(), dtype=torch.float32)
        wx = torch.tensor(np.asarray(getter(resolved_key_map["wx"])).squeeze(), dtype=torch.float32)
        wy = torch.tensor(np.asarray(getter(resolved_key_map["wy"])).squeeze(), dtype=torch.float32)
        wavelength = torch.tensor(np.asarray(getter(resolved_key_map["wavelength"])).squeeze(), dtype=torch.float32)
        phase = torch.tensor(np.asarray(getter(resolved_key_map["phase"])), dtype=torch.float32)
        amp = torch.tensor(np.asarray(getter(resolved_key_map["amp"])), dtype=torch.float32)
        return cls(height, wx, wy, wavelength, phase, amp)

    @classmethod
    def from_manifest(cls, manifest: LUTManifest | str | Path) -> "PhaseAmpHeightInterp":
        resolved_manifest = manifest if isinstance(manifest, LUTManifest) else LUTManifest.from_file(manifest)
        return cls.from_external_file(
            resolved_manifest.data_file,
            key_map=resolved_manifest.fields,
        )

    def bounds(self) -> dict[str, float]:
        return {
            "h_min": float(self.height_vec_m.min().item()),
            "h_max": float(self.height_vec_m.max().item()),
            "wx_min": float(self.wx_vec_m.min().item()),
            "wx_max": float(self.wx_vec_m.max().item()),
            "wy_min": float(self.wy_vec_m.min().item()),
            "wy_max": float(self.wy_vec_m.max().item()),
            "l_min": float(self.wavelength_vec_m.min().item()),
            "l_max": float(self.wavelength_vec_m.max().item()),
        }

    def _coord_to_grid(self, value: torch.Tensor, axis_vec: torch.Tensor) -> torch.Tensor:
        axis_vec = axis_vec.to(device=value.device, dtype=value.dtype)
        if axis_vec.numel() == 1:
            return torch.zeros_like(value)

        value_clamped = value.clamp(axis_vec[0], axis_vec[-1])
        flat = value_clamped.reshape(-1)
        hi = torch.searchsorted(axis_vec, flat, right=False).clamp(1, axis_vec.numel() - 1)
        lo = hi - 1

        axis_lo = axis_vec[lo]
        axis_hi = axis_vec[hi]
        interp = (flat - axis_lo) / (axis_hi - axis_lo + 1e-12)
        frac_idx = (lo.to(value.dtype) + interp).reshape_as(value)
        return frac_idx / float(axis_vec.numel() - 1) * 2.0 - 1.0

    def _match_wavelength_index(self, wavelength_value: torch.Tensor) -> int:
        lut_wavelengths = self.wavelength_vec_m.to(device=wavelength_value.device, dtype=wavelength_value.dtype)
        diff = torch.abs(lut_wavelengths - wavelength_value)
        idx = int(torch.argmin(diff).item())
        min_diff = float(diff[idx].item())
        if min_diff > self.wavelength_tol:
            allowed = ", ".join(f"{float(x.item()):.9e}" for x in self.wavelength_vec_m)
            raise ValueError(
                f"wavelength={float(wavelength_value.item()):.9e} is not in discrete LUT wavelengths; "
                f"allowed values are [{allowed}] with tol={self.wavelength_tol:.3e}"
            )
        return idx

    def _query_single_wavelength(
        self,
        height_m: torch.Tensor,
        wx_m: torch.Tensor,
        wy_m: torch.Tensor,
        wavelength_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (height_m.shape == wx_m.shape == wy_m.shape):
            raise ValueError("height_m, wx_m and wy_m must share the same shape")
        if height_m.dim() != 3:
            raise ValueError("height_m, wx_m and wy_m must be [B,H,W]")

        batch = int(height_m.shape[0])
        device = height_m.device

        grid_x = self._coord_to_grid(wy_m, self.wy_vec_m)
        grid_y = self._coord_to_grid(wx_m, self.wx_vec_m)
        grid_z = self._coord_to_grid(height_m, self.height_vec_m)
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(1)

        phase_slice = self.phase_map_rad[:, :, :, wavelength_index].to(device)
        amp_slice = self.amp_map[:, :, :, wavelength_index].to(device)
        real_vol = (amp_slice * torch.cos(phase_slice)).unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1, -1)
        imag_vol = (amp_slice * torch.sin(phase_slice)).unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1, -1)

        real = F.grid_sample(real_vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
        imag = F.grid_sample(imag_vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
        real = real.squeeze(1).squeeze(1)
        imag = imag.squeeze(1).squeeze(1)

        phase = torch.atan2(imag, real)
        amp = torch.sqrt(real.square() + imag.square() + 1e-12)
        return phase, amp

    def query(
        self,
        height_m: torch.Tensor,
        wx_m: torch.Tensor,
        wy_m: torch.Tensor,
        wavelength_m: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (height_m.shape == wx_m.shape == wy_m.shape):
            raise ValueError("height_m, wx_m and wy_m must share the same shape")
        if height_m.dim() != 3:
            raise ValueError("height_m, wx_m and wy_m must be [B,H,W]")

        batch = int(height_m.shape[0])
        device = height_m.device
        dtype = height_m.dtype

        if not torch.is_tensor(wavelength_m):
            wavelength_m = torch.tensor([float(wavelength_m)], device=device, dtype=dtype)
        wavelength_m = wavelength_m.to(device=device, dtype=dtype)
        if wavelength_m.dim() == 0:
            wavelength_m = wavelength_m.unsqueeze(0)
        if wavelength_m.numel() == 1:
            wavelength_m = wavelength_m.repeat(batch)
        if wavelength_m.numel() != batch:
            raise ValueError("wavelength batch size must be 1 or B")

        phase_out = torch.empty_like(height_m)
        amp_out = torch.empty_like(height_m)
        for wavelength_value in torch.unique(wavelength_m):
            mask = torch.isclose(wavelength_m, wavelength_value, atol=self.wavelength_tol, rtol=0.0)
            wavelength_index = self._match_wavelength_index(wavelength_value)
            phase_part, amp_part = self._query_single_wavelength(height_m[mask], wx_m[mask], wy_m[mask], wavelength_index)
            phase_out[mask] = phase_part
            amp_out[mask] = amp_part
        return phase_out, amp_out

    def forward(
        self,
        height_m: torch.Tensor,
        wx_m: torch.Tensor,
        wy_m: torch.Tensor,
        wavelength_m: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(height_m, wx_m, wy_m, wavelength_m)


class PhaseAmpDiscreteProvider(PhaseAmpHeightInterp):
    """Semantic alias for a provider that only supports discrete wavelengths."""

    @classmethod
    def from_interp(
        cls,
        interp: PhaseAmpHeightInterp,
        wavelength_tol: float | None = None,
    ) -> "PhaseAmpDiscreteProvider":
        return cls(
            height_vec_m=interp.height_vec_m,
            wx_vec_m=interp.wx_vec_m,
            wy_vec_m=interp.wy_vec_m,
            wavelength_vec_m=interp.wavelength_vec_m,
            phase_map_rad=interp.phase_map_rad,
            amp_map=interp.amp_map,
            wavelength_tol=interp.wavelength_tol if wavelength_tol is None else wavelength_tol,
        )


class MetaSurfaceLUT(nn.Module):
    """
    Unified LUT holder for forward query and LUT inverse design.

    `raw_provider` is the ground-truth sampled LUT.
    `dense_provider` is an optional long-lived dense cache built from raw data by
    complex-plane interpolation on `(wx, wy)`.
    """

    def __init__(
        self,
        raw_provider: PhaseAmpDiscreteProvider,
        dense_provider: PhaseAmpDiscreteProvider | None = None,
    ) -> None:
        super().__init__()
        self.raw_provider = raw_provider
        self.dense_provider = dense_provider

    @property
    def height_vec_m(self) -> torch.Tensor:
        return self.raw_provider.height_vec_m

    @property
    def wx_vec_m(self) -> torch.Tensor:
        provider = self.dense_provider if self.dense_provider is not None else self.raw_provider
        return provider.wx_vec_m

    @property
    def wy_vec_m(self) -> torch.Tensor:
        provider = self.dense_provider if self.dense_provider is not None else self.raw_provider
        return provider.wy_vec_m

    @property
    def wavelength_vec_m(self) -> torch.Tensor:
        return self.raw_provider.wavelength_vec_m

    @classmethod
    def from_external_file(
        cls,
        file_path: str | Path,
        key_map: dict | None = None,
        dense_wx_points: int | None = None,
        dense_wy_points: int | None = None,
        wavelength_tol: float = 1e-12,
    ) -> "MetaSurfaceLUT":
        raw_interp = PhaseAmpHeightInterp.from_external_file(file_path, key_map=key_map)
        raw_provider = PhaseAmpDiscreteProvider.from_interp(raw_interp, wavelength_tol=wavelength_tol)
        lut = cls(raw_provider=raw_provider, dense_provider=None)
        if dense_wx_points is not None and dense_wy_points is not None:
            lut.build_dense(num_wx=dense_wx_points, num_wy=dense_wy_points)
        return lut

    @classmethod
    def from_manifest(
        cls,
        manifest: LUTManifest | str | Path,
        dense_wx_points: int | None = None,
        dense_wy_points: int | None = None,
        wavelength_tol: float = 1e-12,
    ) -> "MetaSurfaceLUT":
        resolved_manifest = manifest if isinstance(manifest, LUTManifest) else LUTManifest.from_file(manifest)
        return cls.from_external_file(
            resolved_manifest.data_file,
            key_map=resolved_manifest.fields,
            dense_wx_points=dense_wx_points,
            dense_wy_points=dense_wy_points,
            wavelength_tol=wavelength_tol,
        )

    def bounds(self) -> dict[str, float]:
        return self.raw_provider.bounds()

    def _select_provider(self, use_raw: bool) -> PhaseAmpDiscreteProvider:
        if use_raw:
            return self.raw_provider
        if self.dense_provider is None:
            raise RuntimeError("dense LUT has not been built; call build_dense(...) first or use use_raw=True")
        return self.dense_provider

    def build_dense(self, num_wx: int, num_wy: int) -> "MetaSurfaceLUT":
        if num_wx < 2 or num_wy < 2:
            raise ValueError("num_wx and num_wy must both be >= 2")

        raw = self.raw_provider
        bounds = raw.bounds()
        wx_dense = torch.linspace(bounds["wx_min"], bounds["wx_max"], steps=num_wx, dtype=torch.float32)
        wy_dense = torch.linspace(bounds["wy_min"], bounds["wy_max"], steps=num_wy, dtype=torch.float32)
        wx_mesh, wy_mesh = torch.meshgrid(wx_dense, wy_dense, indexing="ij")

        num_heights = raw.height_vec_m.numel()
        num_wavelengths = raw.wavelength_vec_m.numel()
        phase_dense = torch.empty((num_heights, num_wx, num_wy, num_wavelengths), dtype=torch.float32)
        amp_dense = torch.empty_like(phase_dense)

        for height_idx, height_value in enumerate(raw.height_vec_m.detach().cpu()):
            height_map = torch.full((1, num_wx, num_wy), float(height_value), dtype=torch.float32)
            wx_map = wx_mesh.unsqueeze(0)
            wy_map = wy_mesh.unsqueeze(0)
            for wavelength_idx, wavelength_value in enumerate(raw.wavelength_vec_m.detach().cpu()):
                phase, amp = raw.query(height_map, wx_map, wy_map, float(wavelength_value))
                phase_dense[height_idx, :, :, wavelength_idx] = phase[0]
                amp_dense[height_idx, :, :, wavelength_idx] = amp[0]

        self.dense_provider = PhaseAmpDiscreteProvider(
            height_vec_m=raw.height_vec_m.detach().cpu().clone(),
            wx_vec_m=wx_dense,
            wy_vec_m=wy_dense,
            wavelength_vec_m=raw.wavelength_vec_m.detach().cpu().clone(),
            phase_map_rad=phase_dense,
            amp_map=amp_dense,
            wavelength_tol=raw.wavelength_tol,
        )
        return self

    def query(
        self,
        height_m: torch.Tensor,
        wx_m: torch.Tensor,
        wy_m: torch.Tensor,
        wavelength_m: torch.Tensor | float,
        use_raw: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provider = self._select_provider(use_raw=use_raw)
        return provider.query(height_m=height_m, wx_m=wx_m, wy_m=wy_m, wavelength_m=wavelength_m)

    def forward(
        self,
        height_m: torch.Tensor,
        wx_m: torch.Tensor,
        wy_m: torch.Tensor,
        wavelength_m: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(height_m=height_m, wx_m=wx_m, wy_m=wy_m, wavelength_m=wavelength_m, use_raw=False)

    def _match_height_index(self, provider: PhaseAmpDiscreteProvider, height_value: float) -> int:
        height_tensor = torch.tensor(float(height_value), dtype=provider.height_vec_m.dtype, device=provider.height_vec_m.device)
        diff = torch.abs(provider.height_vec_m - height_tensor)
        return int(torch.argmin(diff).item())

    def _prepare_inverse_design_candidates(
        self,
        provider: PhaseAmpDiscreteProvider,
        wavelengths_m: Sequence[float],
        fixed_height_m: float | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        if len(wavelengths_m) == 0:
            raise ValueError("wavelengths_m must not be empty")

        height_value = float(self.height_vec_m[0].item()) if fixed_height_m is None else float(fixed_height_m)
        height_index = self._match_height_index(provider, height_value)

        phase_flat_list: list[torch.Tensor] = []
        amp_flat_list: list[torch.Tensor] = []
        for wavelength_m in wavelengths_m:
            wavelength_tensor = torch.tensor(
                float(wavelength_m),
                dtype=provider.wavelength_vec_m.dtype,
                device=provider.wavelength_vec_m.device,
            )
            wavelength_index = provider._match_wavelength_index(wavelength_tensor)
            phase_candidates = provider.phase_map_rad[height_index, :, :, wavelength_index].detach().cpu()
            amp_candidates = provider.amp_map[height_index, :, :, wavelength_index].detach().cpu()
            phase_flat_list.append(phase_candidates.reshape(-1))
            amp_flat_list.append(amp_candidates.reshape(-1))

        wx_candidates = provider.wx_vec_m.detach().cpu()
        wy_candidates = provider.wy_vec_m.detach().cpu()
        wx_grid, wy_grid = torch.meshgrid(wx_candidates, wy_candidates, indexing="ij")
        wx_flat = wx_grid.reshape(-1)
        wy_flat = wy_grid.reshape(-1)
        return phase_flat_list, amp_flat_list, wx_flat, wy_flat

    @staticmethod
    def _normalize_target_phase_list(
        target_phases: torch.Tensor | Sequence[torch.Tensor],
        expected_count: int,
    ) -> tuple[list[torch.Tensor], torch.Size]:
        if isinstance(target_phases, torch.Tensor):
            if target_phases.dim() != 3:
                raise ValueError("target_phases tensor must be [C,H,W]")
            if int(target_phases.shape[0]) != int(expected_count):
                raise ValueError(f"target_phases has {int(target_phases.shape[0])} channels, but expected {int(expected_count)}")
            target_phase_list = [target_phases[idx].detach().cpu() for idx in range(int(expected_count))]
        else:
            if len(target_phases) != int(expected_count):
                raise ValueError(f"target_phases has length {len(target_phases)}, but expected {int(expected_count)}")
            target_phase_list = []
            for idx, phase_map in enumerate(target_phases):
                if not isinstance(phase_map, torch.Tensor):
                    raise TypeError(f"target_phases[{idx}] must be a torch.Tensor")
                if phase_map.dim() != 2:
                    raise ValueError(f"target_phases[{idx}] must be 2D [H,W], got shape {tuple(phase_map.shape)}")
                target_phase_list.append(phase_map.detach().cpu())

        spatial_shape = target_phase_list[0].shape
        for idx, phase_map in enumerate(target_phase_list[1:], start=1):
            if phase_map.shape != spatial_shape:
                raise ValueError(
                    f"all target phase maps must share the same shape; "
                    f"target_phases[0]={tuple(spatial_shape)}, target_phases[{idx}]={tuple(phase_map.shape)}"
                )
        return target_phase_list, spatial_shape

    @staticmethod
    def _normalize_inverse_design_vector(
        values: float | Sequence[float] | torch.Tensor,
        expected_count: int,
        *,
        name: str,
    ) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            flat = values.detach().cpu().flatten().float()
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            flat = torch.tensor([float(x) for x in values], dtype=torch.float32)
        else:
            flat = torch.full((int(expected_count),), float(values), dtype=torch.float32)

        if flat.numel() != int(expected_count):
            raise ValueError(f"{name} must contain {int(expected_count)} values, got {int(flat.numel())}")
        return flat

    def inverse_design(
        self,
        target_phase: torch.Tensor,
        wavelength_m: float,
        fixed_height_m: float | None = None,
        use_raw: bool = False,
        method: str = "complex",
        amp_weight: float = 0.15,
        target_amp: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provider = self._select_provider(use_raw=use_raw)
        if target_phase.dim() != 2:
            raise ValueError("target_phase must be a 2D [H,W] tensor")

        phase_flat_list, amp_flat_list, wx_flat, wy_flat = self._prepare_inverse_design_candidates(
            provider=provider,
            wavelengths_m=[float(wavelength_m)],
            fixed_height_m=fixed_height_m,
        )
        phase_flat = phase_flat_list[0]
        amp_flat = amp_flat_list[0]
        target_cpu = target_phase.detach().cpu().reshape(-1, 1)

        if method == "complex":
            target_real = float(target_amp) * torch.cos(target_cpu)
            target_imag = float(target_amp) * torch.sin(target_cpu)
            candidate_real = amp_flat[None, :] * torch.cos(phase_flat[None, :])
            candidate_imag = amp_flat[None, :] * torch.sin(phase_flat[None, :])
            score = (candidate_real - target_real).square() + (candidate_imag - target_imag).square()
        elif method == "phase":
            phase_err = torch.atan2(
                torch.sin(target_cpu - phase_flat[None, :]),
                torch.cos(target_cpu - phase_flat[None, :]),
            ).abs()
            score = phase_err + float(amp_weight) * torch.abs(float(target_amp) - amp_flat[None, :])
        else:
            raise ValueError(f"Unsupported inverse design method: {method}")

        best_idx = torch.argmin(score, dim=1)
        wx_map = wx_flat[best_idx].reshape_as(target_phase.detach().cpu())
        wy_map = wy_flat[best_idx].reshape_as(target_phase.detach().cpu())
        return wx_map, wy_map

    def inverse_design_multiwavelength(
        self,
        target_phases: torch.Tensor | Sequence[torch.Tensor],
        wavelengths_m: Sequence[float],
        fixed_height_m: float | None = None,
        use_raw: bool = False,
        target_amp: float | Sequence[float] | torch.Tensor = 1.0,
        wavelength_weights: Sequence[float] | torch.Tensor | None = None,
        amp_weight: float = 0.05,
        amp_balance_weight: float = 0.05,
        amp_efficiency_weight: float = 0.02,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(wavelengths_m) == 0:
            raise ValueError("wavelengths_m must not be empty")

        provider = self._select_provider(use_raw=use_raw)
        num_wavelengths = len(wavelengths_m)
        target_phase_list, spatial_shape = self._normalize_target_phase_list(
            target_phases=target_phases,
            expected_count=num_wavelengths,
        )
        target_amp_vec = self._normalize_inverse_design_vector(
            target_amp,
            expected_count=num_wavelengths,
            name="target_amp",
        )

        if wavelength_weights is None:
            weight_vec = torch.ones((num_wavelengths,), dtype=torch.float32)
        else:
            weight_vec = self._normalize_inverse_design_vector(
                wavelength_weights,
                expected_count=num_wavelengths,
                name="wavelength_weights",
            )
        if torch.any(weight_vec < 0):
            raise ValueError("wavelength_weights must be non-negative")
        weight_sum = float(weight_vec.sum().item())
        if weight_sum <= 0.0:
            raise ValueError("wavelength_weights must contain at least one positive value")
        weight_vec = weight_vec / weight_sum

        phase_flat_list, amp_flat_list, wx_flat, wy_flat = self._prepare_inverse_design_candidates(
            provider=provider,
            wavelengths_m=[float(value) for value in wavelengths_m],
            fixed_height_m=fixed_height_m,
        )

        num_pixels = int(target_phase_list[0].numel())
        num_candidates = int(wx_flat.numel())
        score = torch.zeros((num_pixels, num_candidates), dtype=torch.float32)

        for wavelength_idx in range(num_wavelengths):
            target_phase_flat = target_phase_list[wavelength_idx].reshape(-1, 1)
            candidate_phase_flat = phase_flat_list[wavelength_idx][None, :]
            candidate_amp_flat = amp_flat_list[wavelength_idx][None, :]

            phase_err = torch.atan2(
                torch.sin(target_phase_flat - candidate_phase_flat),
                torch.cos(target_phase_flat - candidate_phase_flat),
            )
            score = score + weight_vec[wavelength_idx] * phase_err.square()

            if float(amp_weight) != 0.0:
                score = score + (
                    float(amp_weight)
                    * weight_vec[wavelength_idx]
                    * (candidate_amp_flat - float(target_amp_vec[wavelength_idx].item())).square()
                )

        candidate_amp_stack = torch.stack(amp_flat_list, dim=0)
        weighted_mean_amp = torch.sum(weight_vec[:, None] * candidate_amp_stack, dim=0)

        if float(amp_balance_weight) != 0.0 and num_wavelengths > 1:
            weighted_amp_var = torch.sum(
                weight_vec[:, None] * (candidate_amp_stack - weighted_mean_amp[None, :]).square(),
                dim=0,
            )
            score = score + float(amp_balance_weight) * weighted_amp_var[None, :]

        if float(amp_efficiency_weight) != 0.0:
            efficiency_penalty = (1.0 - weighted_mean_amp).square()
            score = score + float(amp_efficiency_weight) * efficiency_penalty[None, :]

        best_idx = torch.argmin(score, dim=1)
        wx_map = wx_flat[best_idx].reshape(spatial_shape)
        wy_map = wy_flat[best_idx].reshape(spatial_shape)
        return wx_map, wy_map


__all__ = [
    "PhaseAmpHeightInterp",
    "PhaseAmpDiscreteProvider",
    "MetaSurfaceLUT",
]
