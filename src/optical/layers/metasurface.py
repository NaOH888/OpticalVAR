from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from optical.core.base import OpticLayer
from optical.core.config import FDTDMetaConfig, LUTManifest

if TYPE_CHECKING:
    from optical.backends.fdtd.api import FDTDBuilder, FDTDLayerContext


DEFAULT_CELL_PITCH = 250e-9


class MetaEncoder(OpticLayer):
    """Map metasurface geometry `(wx, wy)` to complex modulation via LUT queries."""

    def __init__(
        self,
        provider,
        phys_shape: torch.Tensor,
        wavelengths_m: tuple[float, ...],
        *,
        phase_wrap: str = "neg_pi_to_pi",
        dx: float = 1.0,
        fixed_height: float | None = None,
        base_height: float = 50e-9,
        period_x: float = DEFAULT_CELL_PITCH,
        period_y: float = DEFAULT_CELL_PITCH,
        pattern_material: str = "Si (Silicon) - Palik",
        base_material: str = "SiO2 (Glass) - Palik",
        phys_bound_min_m: float | None = None,
        phys_bound_max_m: float | None = None,
        trainable_phys_shape: bool = False,
        fdtd_config: FDTDMetaConfig | None = None,
    ):
        super().__init__(dx=dx)
        if fixed_height is None:
            raise ValueError("MetaEncoder requires constructor-time fixed_height")

        self.provider = provider
        self.fdtd_config = fdtd_config or FDTDMetaConfig()
        self.wavelengths_m = tuple(float(x) for x in wavelengths_m)
        self.phys_bound_min_m = None if phys_bound_min_m is None else float(phys_bound_min_m)
        self.phys_bound_max_m = None if phys_bound_max_m is None else float(phys_bound_max_m)

        normalized_phys_shape = self._normalize_phys_shape(phys_shape).detach().clone()
        normalized_phys_shape = self._clamp_phys_shape_to_effective_bounds(normalized_phys_shape)
        self.trainable_phys_shape = bool(trainable_phys_shape)
        if self.trainable_phys_shape:
            self._raw_phys_shape = torch.nn.Parameter(self._encode_phys_shape_to_raw(normalized_phys_shape))
        else:
            self.register_buffer("_phys_shape_buffer", normalized_phys_shape, persistent=True)

        self.phase_wrap = str(phase_wrap)
        self.fixed_height = float(fixed_height)
        self.base_height = float(base_height)
        self.period_x = float(period_x)
        self.period_y = float(period_y)
        self.pattern_material = pattern_material
        self.base_material = base_material
        self.grid_h = int(normalized_phys_shape.shape[-2])
        self.grid_w = int(normalized_phys_shape.shape[-1])
        self.width = float(self.grid_w) * self.period_x
        self.height = float(self.grid_h) * self.period_y
        self.sx = max(1, int(round(self.width / self.dx)))
        self.sy = max(1, int(round(self.height / self.dx)))

    @classmethod
    def from_manifest(
        cls,
        *,
        phys_shape: torch.Tensor,
        manifest_path: str | Path,
        wavelengths_m: tuple[float, ...],
        fixed_height: float,
        dx: float = 1.0,
        base_height: float = 50e-9,
        period_x: float = DEFAULT_CELL_PITCH,
        period_y: float = DEFAULT_CELL_PITCH,
        phase_wrap: str = "neg_pi_to_pi",
        pattern_material: str = "Si (Silicon) - Palik",
        base_material: str = "SiO2 (Glass) - Palik",
        phys_bound_min_m: float | None = None,
        phys_bound_max_m: float | None = None,
        trainable_phys_shape: bool = False,
        fdtd_config: FDTDMetaConfig | None = None,
    ) -> "MetaEncoder":
        from optical.lut.dataset import PhaseAmpHeightInterp

        manifest = LUTManifest.from_file(manifest_path)
        provider = PhaseAmpHeightInterp.from_manifest(manifest)
        return cls(
            provider=provider,
            phys_shape=phys_shape,
            wavelengths_m=wavelengths_m,
            phase_wrap=phase_wrap,
            dx=dx,
            fixed_height=fixed_height,
            base_height=base_height,
            period_x=period_x,
            period_y=period_y,
            pattern_material=pattern_material,
            base_material=base_material,
            phys_bound_min_m=phys_bound_min_m,
            phys_bound_max_m=phys_bound_max_m,
            trainable_phys_shape=trainable_phys_shape,
            fdtd_config=fdtd_config,
        )

    @property
    def raw_phys_shape(self) -> torch.nn.Parameter | None:
        if not self.trainable_phys_shape:
            return None
        return self._raw_phys_shape

    @property
    def phys_shape(self) -> torch.Tensor:
        return self._canonical_phys_shape()

    def _canonical_phys_shape(self) -> torch.Tensor:
        if self.trainable_phys_shape:
            return self._decode_raw_phys_shape(self._raw_phys_shape)
        return self._phys_shape_buffer

    @staticmethod
    def _normalize_phys_shape(phys_shape: torch.Tensor) -> torch.Tensor:
        if phys_shape.dim() == 3:
            phys_shape = phys_shape.unsqueeze(0)
        if phys_shape.dim() != 4 or phys_shape.shape[1] != 2:
            raise ValueError("phys_shape must be [B,2,H,W] or [2,H,W]")
        return phys_shape

    def _resolve_effective_phys_bounds(self) -> tuple[float, float, float, float]:
        shared_lower = 0.0 if self.phys_bound_min_m is None else float(self.phys_bound_min_m)
        shared_upper = float("inf") if self.phys_bound_max_m is None else float(self.phys_bound_max_m)
        wx_lower = shared_lower
        wx_upper = shared_upper
        wy_lower = shared_lower
        wy_upper = shared_upper

        if hasattr(self.provider, "bounds"):
            bounds = self.provider.bounds()
            wx_lower = max(wx_lower, float(bounds["wx_min"]))
            wx_upper = min(wx_upper, float(bounds["wx_max"]))
            wy_lower = max(wy_lower, float(bounds["wy_min"]))
            wy_upper = min(wy_upper, float(bounds["wy_max"]))

        if wx_upper <= wx_lower or wy_upper <= wy_lower:
            raise ValueError(
                "Effective physical bounds are invalid after intersecting configured bounds with provider bounds: "
                f"wx=({wx_lower}, {wx_upper}), wy=({wy_lower}, {wy_upper})"
            )
        return wx_lower, wx_upper, wy_lower, wy_upper

    def _bounds_tensor_like(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wx_lower, wx_upper, wy_lower, wy_upper = self._resolve_effective_phys_bounds()
        lower = torch.tensor([wx_lower, wy_lower], device=reference.device, dtype=reference.dtype).view(1, 2, 1, 1)
        upper = torch.tensor([wx_upper, wy_upper], device=reference.device, dtype=reference.dtype).view(1, 2, 1, 1)
        return lower, upper

    def _decode_raw_phys_shape(self, raw_phys_shape: torch.Tensor) -> torch.Tensor:
        lower, upper = self._bounds_tensor_like(raw_phys_shape)
        return lower + (upper - lower) * torch.sigmoid(raw_phys_shape)

    def _encode_phys_shape_to_raw(self, phys_shape: torch.Tensor) -> torch.Tensor:
        lower, upper = self._bounds_tensor_like(phys_shape)
        normalized = (phys_shape - lower) / (upper - lower)
        normalized = normalized.clamp(1.0e-6, 1.0 - 1.0e-6)
        return torch.log(normalized) - torch.log1p(-normalized)

    def _clamp_phys_shape_to_effective_bounds(self, phys_shape: torch.Tensor) -> torch.Tensor:
        lower, upper = self._bounds_tensor_like(phys_shape)
        return phys_shape.clamp(min=lower, max=upper)

    def _resolve_meta_phys_shape(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_phys_shape = self._canonical_phys_shape().to(device=device, dtype=dtype)
        if base_phys_shape.shape[0] == 1 and int(batch_size) > 1:
            base_phys_shape = base_phys_shape.expand(int(batch_size), -1, -1, -1)
        if base_phys_shape.shape[0] != int(batch_size):
            raise ValueError(f"phys_shape batch={base_phys_shape.shape[0]} does not match input batch={batch_size}")
        return self._clamp_phys_shape_to_effective_bounds(base_phys_shape)

    @staticmethod
    def _resize_complex_grid(
        modulation: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        src_h = int(modulation.shape[-2])
        src_w = int(modulation.shape[-1])
        if (src_h, src_w) == (int(target_h), int(target_w)):
            return modulation

        real = modulation.real
        imag = modulation.imag

        pooled_h = min(src_h, int(target_h))
        pooled_w = min(src_w, int(target_w))
        if pooled_h != src_h or pooled_w != src_w:
            real = F.adaptive_avg_pool2d(real, output_size=(pooled_h, pooled_w))
            imag = F.adaptive_avg_pool2d(imag, output_size=(pooled_h, pooled_w))

        if (pooled_h, pooled_w) != (int(target_h), int(target_w)):
            real = F.interpolate(real, size=(int(target_h), int(target_w)), mode="nearest")
            imag = F.interpolate(imag, size=(int(target_h), int(target_w)), mode="nearest")

        return torch.complex(real, imag)

    def _map_modulation_to_field_grid(
        self,
        modulation_meta: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if modulation_meta.dim() != 4:
            raise ValueError(f"modulation_meta must be [B,C,H,W], got {tuple(modulation_meta.shape)}")
        return self._resize_complex_grid(modulation_meta, target_h=int(target_h), target_w=int(target_w))

    @staticmethod
    def _fdtd_name_token(z_bottom: float) -> str:
        z_nm = int(round(float(z_bottom) * 1e9))
        if z_nm < 0:
            return f"neg_{abs(z_nm)}nm"
        return f"{z_nm}nm"

    def _wrap_phase(self, phase: torch.Tensor) -> torch.Tensor:
        if self.phase_wrap == "none":
            return phase
        return torch.atan2(torch.sin(phase), torch.cos(phase))

    def get_modulation(self, phys_shape: torch.Tensor, wavelengths: list[float]) -> list[torch.Tensor]:
        if phys_shape.dim() != 4 or phys_shape.shape[1] != 2:
            raise ValueError("phys_shape must be [B,2,H,W], where channels are wx and wy")
        if len(wavelengths) == 0:
            raise ValueError("wavelengths cannot be empty")
        if not hasattr(self.provider, "query"):
            raise ValueError("provider must implement query(height_m, wx_m, wy_m, wavelength_m)")

        batch_size, _, grid_h, grid_w = phys_shape.shape
        device = phys_shape.device
        wx_m = phys_shape[:, 0, :, :]
        wy_m = phys_shape[:, 1, :, :]
        height_m = torch.full((batch_size, grid_h, grid_w), self.fixed_height, dtype=phys_shape.dtype, device=device)

        modulations: list[torch.Tensor] = []
        for wavelength in wavelengths:
            phase, amp = self.provider.query(height_m, wx_m, wy_m, float(wavelength))
            phase = self._wrap_phase(phase)
            amp = amp.clamp(0.0, 1.0)
            modulations.append(torch.complex(amp * torch.cos(phase), amp * torch.sin(phase)))
        return modulations

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs):
        original_dim = input_field.dim()
        if original_dim == 2:
            input_field = input_field.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            input_field = input_field.unsqueeze(0)

        batch_size, channel_count, target_h, target_w = input_field.shape
        if len(self.wavelengths_m) != channel_count:
            raise ValueError(
                f"wavelength count {len(self.wavelengths_m)} does not match channel count {channel_count}"
            )

        phys_shape_meta = self._resolve_meta_phys_shape(
            batch_size=batch_size,
            device=input_field.device,
            dtype=input_field.real.dtype,
        )
        modulation_meta = torch.stack(self.get_modulation(phys_shape_meta, list(self.wavelengths_m)), dim=1)
        modulation = self._map_modulation_to_field_grid(
            modulation_meta,
            target_h=target_h,
            target_w=target_w,
        )
        output = input_field * modulation

        if original_dim == 3:
            return output.squeeze(0)
        if original_dim == 2:
            return output.squeeze(0).squeeze(0)
        return output

    def build_fdtd(self, builder: "FDTDBuilder", layer_ctx: "FDTDLayerContext") -> float:
        phys_shape = self._clamp_phys_shape_to_effective_bounds(
            self._canonical_phys_shape().detach().cpu().to(dtype=torch.float32)
        )
        wx_map = phys_shape[0, 0]
        wy_map = phys_shape[0, 1]
        grid_h, grid_w = wx_map.shape

        pattern_height = self.fixed_height
        total_x = float(grid_w) * self.period_x
        total_y = float(grid_h) * self.period_y
        pattern_bottom_z = float(layer_ctx.z_bottom)
        base_bottom_z = pattern_bottom_z + pattern_height
        pattern_center_z = pattern_bottom_z + 0.5 * pattern_height
        base_center_z = base_bottom_z + 0.5 * self.base_height

        min_feature = float(self.fdtd_config.min_feature_m)
        fdtd_name_token = self._fdtd_name_token(pattern_bottom_z)
        group_name = f"meta_pattern_group_{fdtd_name_token}"
        builder.add_structure_group(name=group_name, x=0.0, y=0.0, z=0.0)

        script_lines = [f'select("{group_name}");']
        for row in range(grid_h):
            y = (float(row) - (grid_h - 1) / 2.0) * self.period_y
            for col in range(grid_w):
                x = (float(col) - (grid_w - 1) / 2.0) * self.period_x
                wx = float(wx_map[row, col].item())
                wy = float(wy_map[row, col].item())
                if wx <= min_feature or wy <= min_feature:
                    continue

                script_lines.extend(
                    [
                        "addrect;",
                        f'set("name","meta_pattern_{fdtd_name_token}_{row}_{col}");',
                        f'set("material","{self.pattern_material}");',
                        f'set("x",{x:.12e});',
                        f'set("y",{y:.12e});',
                        f'set("z",{pattern_center_z:.12e});',
                        f'set("x span",{wx:.12e});',
                        f'set("y span",{wy:.12e});',
                        f'set("z span",{pattern_height:.12e});',
                        f'addtogroup("{group_name}");',
                    ]
                )

        builder.add_rect(
            name=f"meta_base_{fdtd_name_token}",
            material=self.base_material,
            x=0.0,
            y=0.0,
            z=base_center_z,
            x_span=total_x,
            y_span=total_y,
            z_span=self.base_height,
        )

        builder.eval_script("\n".join(script_lines))
        return base_bottom_z + self.base_height

    def fdtd_physical_thickness(self, layer_ctx: "FDTDLayerContext") -> float:
        return self.fixed_height + self.base_height


__all__ = ["DEFAULT_CELL_PITCH", "MetaEncoder"]
