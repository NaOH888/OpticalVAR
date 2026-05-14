from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from optical.core.base import OpticLayer, SourceLayer
from optical.core.config import PropagationConfig, PropagationErrorConfig
from optical.layers.detector import DetectorLayer


def _center_pad_to(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    squeezed = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeezed = True
    if x.dim() != 4:
        raise ValueError(f"_center_pad_to expects 3D or 4D tensor, got {x.shape}")

    _, _, h, w = x.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    if torch.is_complex(x):
        real = F.pad(x.real, padding, value=0)
        imag = F.pad(x.imag, padding, value=0)
        x = torch.complex(real, imag)
    else:
        x = F.pad(x, padding, value=0)

    if squeezed:
        x = x.squeeze(0)
    return x


def _center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    squeezed = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeezed = True
    if x.dim() != 4:
        raise ValueError(f"_center_crop expects 3D or 4D tensor, got {x.shape}")

    _, _, h, w = x.shape
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    end_h = start_h + target_h
    end_w = start_w + target_w

    if torch.is_complex(x):
        real = x.real[:, :, start_h:end_h, start_w:end_w]
        imag = x.imag[:, :, start_h:end_h, start_w:end_w]
        x = torch.complex(real, imag)
    else:
        x = x[:, :, start_h:end_h, start_w:end_w]

    if squeezed:
        x = x.squeeze(0)
    return x


def _center_replace(canvas: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    squeezed = False
    if canvas.dim() == 3:
        canvas = canvas.unsqueeze(0)
        squeezed = True
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)
    if canvas.dim() != 4 or patch.dim() != 4:
        raise ValueError(f"_center_replace expects 3D or 4D tensors, got {canvas.shape} and {patch.shape}")
    if int(canvas.shape[0]) != int(patch.shape[0]) or int(canvas.shape[1]) != int(patch.shape[1]):
        raise ValueError(
            "canvas and patch batch/channel dimensions must match: "
            f"canvas={tuple(canvas.shape[:2])}, patch={tuple(patch.shape[:2])}"
        )

    _, _, canvas_h, canvas_w = canvas.shape
    _, _, patch_h, patch_w = patch.shape
    if patch_h > canvas_h or patch_w > canvas_w:
        raise ValueError(
            "patch spatial size must not exceed canvas size: "
            f"patch={(patch_h, patch_w)}, canvas={(canvas_h, canvas_w)}"
        )

    start_h = (canvas_h - patch_h) // 2
    start_w = (canvas_w - patch_w) // 2
    result = canvas.clone()
    result[:, :, start_h : start_h + patch_h, start_w : start_w + patch_w] = patch
    if squeezed:
        result = result.squeeze(0)
    return result


def _compute_kz(fx: torch.Tensor, fy: torch.Tensor, wavelength: float, n: float = 1.0) -> torch.Tensor:
    k0 = 2.0 * torch.pi / float(wavelength)
    k = n * k0
    t_s = (k**2) - (2.0 * torch.pi * fx) ** 2 - (2.0 * torch.pi * fy) ** 2
    t_s_complex = t_s.to(torch.complex64 if t_s.dtype == torch.float32 else torch.complex128)
    return torch.sqrt(t_s_complex)


def _build_q_window(
    fx: torch.Tensor,
    fy: torch.Tensor,
    wavelength: float,
    dz: float,
    span_x: float,
    span_y: float,
) -> torch.Tensor:
    dz_abs = abs(float(dz))
    if dz_abs == 0.0:
        return torch.ones_like(fx, dtype=torch.bool)
    if span_x <= 0.0 or span_y <= 0.0:
        raise ValueError("span_x and span_y must be positive when building Q window")

    wavelength = float(wavelength)
    fx_limit = 1.0 / (wavelength * math.sqrt(1.0 + (2.0 * dz_abs / span_x) ** 2))
    fy_limit = 1.0 / (wavelength * math.sqrt(1.0 + (2.0 * dz_abs / span_y) ** 2))
    return (fx.abs() <= fx_limit) & (fy.abs() <= fy_limit)


def _fft2_centered(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    return torch.fft.fftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm=norm)


def _ifft2_centered(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(x, dim=(-2, -1), norm=norm), dim=(-2, -1))


class PropagateContext:
    """Fixed-canvas ASM propagation context with fixed x/y/z assembly error."""

    def __init__(
        self,
        *,
        propagation_config: PropagationConfig,
        error_config: PropagationErrorConfig,
        error_factor: float,
    ):
        self.layers: list[tuple[float, OpticLayer]] = []
        self._canvas_shape: tuple[int, int] | None = None
        self._canvas_dirty = True
        self.propagation_config = propagation_config
        self.error_config = error_config
        self.error_factor = float(error_factor)
        self._spectral_cache: dict[tuple, dict[str, torch.Tensor | float]] = {}
        self._transfer_cache: dict[tuple, dict[str, torch.Tensor | list[float | None]]] = {}

    def add_layer(self, layer: OpticLayer, z: float) -> None:
        self.layers.append((float(z), layer))
        self.layers.sort(key=lambda x: x[0])
        self._canvas_dirty = True
        self._transfer_cache.clear()

    @staticmethod
    def _resolve_source_wavelengths(source_layer: SourceLayer, channel_count: int) -> list[float]:
        if hasattr(source_layer, "config") and hasattr(source_layer.config, "wavelengths_m"):
            wavelengths = [float(x) for x in source_layer.config.wavelengths_m]
        elif hasattr(source_layer, "source_config") and hasattr(source_layer.source_config, "wavelengths_m"):
            wavelengths = [float(x) for x in source_layer.source_config.wavelengths_m]
        elif hasattr(source_layer, "wavelengths_m"):
            wavelengths = [float(x) for x in source_layer.wavelengths_m]
        else:
            raise ValueError(f"Could not resolve wavelengths from source layer {source_layer.__class__.__name__}")

        if len(wavelengths) != int(channel_count):
            raise ValueError(
                f"wavelength count {len(wavelengths)} does not match source output channels {channel_count}"
            )
        return wavelengths

    def _resolve_canvas_shape(self) -> tuple[int, int]:
        if self._canvas_shape is not None and not self._canvas_dirty:
            return self._canvas_shape

        explicit_h = self.propagation_config.canvas_h
        explicit_w = self.propagation_config.canvas_w
        if explicit_h is not None or explicit_w is not None:
            if explicit_h is None or explicit_w is None:
                raise ValueError("canvas_h and canvas_w must be provided together")
            canvas_h = int(explicit_h)
            canvas_w = int(explicit_w)
        else:
            max_h = max((int(getattr(layer, "sy", 0)) for _, layer in self.layers), default=0)
            max_w = max((int(getattr(layer, "sx", 0)) for _, layer in self.layers), default=0)
            if max_h <= 0 or max_w <= 0:
                raise ValueError("Could not infer active layer size for canvas construction")
            canvas_factor = float(self.propagation_config.canvas_factor)
            if canvas_factor < 1.0:
                raise ValueError("canvas_factor must be >= 1.0")
            canvas_h = int(math.ceil(max_h * canvas_factor))
            canvas_w = int(math.ceil(max_w * canvas_factor))

        self._canvas_shape = (max(canvas_h, 1), max(canvas_w, 1))
        self._canvas_dirty = False
        return self._canvas_shape

    @staticmethod
    def _spectral_cache_key(
        prefix: str,
        canvas_h: int,
        canvas_w: int,
        dx: float,
        device: torch.device,
        dtype: torch.dtype,
        wl_eff_list: tuple[float, ...] | None = None,
    ) -> tuple:
        key = (prefix, int(canvas_h), int(canvas_w), float(dx), str(device), str(dtype))
        if wl_eff_list is not None:
            key = key + tuple(float(wl) for wl in wl_eff_list)
        return key

    def _get_frequency_grid_cache(
        self,
        canvas_h: int,
        canvas_w: int,
        dx: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | float]:
        cache_key = self._spectral_cache_key(
            prefix="freq_grid",
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            dx=dx,
            device=device,
            dtype=dtype,
        )
        cached = self._spectral_cache.get(cache_key)
        if cached is not None:
            return cached

        fx_1d = torch.fft.fftfreq(canvas_w, dx, device=device, dtype=dtype)
        fy_1d = torch.fft.fftfreq(canvas_h, dx, device=device, dtype=dtype)
        fx, fy = torch.meshgrid(fx_1d, fy_1d, indexing="xy")
        cached = {
            "fx": fx,
            "fy": fy,
            "span_x": float(canvas_w) * float(dx),
            "span_y": float(canvas_h) * float(dx),
        }
        self._spectral_cache[cache_key] = cached
        return cached

    def _get_channel_spectral_cache(
        self,
        canvas_h: int,
        canvas_w: int,
        dx: float,
        device: torch.device,
        dtype: torch.dtype,
        wl_eff_list: list[float],
    ) -> dict[str, torch.Tensor | float]:
        wl_eff_tuple = tuple(float(wl) for wl in wl_eff_list)
        cache_key = self._spectral_cache_key(
            prefix="channel_spectral",
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            dx=dx,
            device=device,
            dtype=dtype,
            wl_eff_list=wl_eff_tuple,
        )
        cached = self._spectral_cache.get(cache_key)
        if cached is not None:
            return cached

        base_cache = self._get_frequency_grid_cache(
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            dx=dx,
            device=device,
            dtype=dtype,
        )
        fx = base_cache["fx"]
        fy = base_cache["fy"]
        kz_per_channel = []
        pass_mask_per_channel = []
        for wl_eff in wl_eff_tuple:
            kz = _compute_kz(fx, fy, float(wl_eff), 1.0)
            k = 2.0 * torch.pi / float(wl_eff)
            pass_mask = ((2.0 * torch.pi * fx) ** 2 + (2.0 * torch.pi * fy) ** 2) <= (k**2)
            kz_per_channel.append(kz)
            pass_mask_per_channel.append(pass_mask)

        cached = {
            "fx": fx,
            "fy": fy,
            "span_x": base_cache["span_x"],
            "span_y": base_cache["span_y"],
            "kz": torch.stack(kz_per_channel, dim=0),
            "pass_mask": torch.stack(pass_mask_per_channel, dim=0),
        }
        self._spectral_cache[cache_key] = cached
        return cached

    def _get_transfer_function_cache(
        self,
        *,
        canvas_h: int,
        canvas_w: int,
        dx: float,
        device: torch.device,
        dtype: torch.dtype,
        wl_eff_list: list[float],
        actual_dz: float,
    ) -> dict[str, torch.Tensor | list[float | None]]:
        wl_eff_tuple = tuple(float(wl) for wl in wl_eff_list)
        cache_key = self._spectral_cache_key(
            prefix="transfer",
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            dx=dx,
            device=device,
            dtype=dtype,
            wl_eff_list=wl_eff_tuple,
        ) + (
            f"{float(actual_dz):.12e}",
            str(self.propagation_config.evanescent_mode),
            bool(self.propagation_config.use_bandlimit_window),
        )
        cached = self._transfer_cache.get(cache_key)
        if cached is not None:
            return cached

        spectral_cache = self._get_channel_spectral_cache(
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            dx=dx,
            device=device,
            dtype=dtype,
            wl_eff_list=list(wl_eff_tuple),
        )
        fx = spectral_cache["fx"]
        fy = spectral_cache["fy"]
        span_x = float(spectral_cache["span_x"])
        span_y = float(spectral_cache["span_y"])
        kz_per_channel = spectral_cache["kz"]
        pass_mask_per_channel = spectral_cache["pass_mask"]

        transfer_per_channel = []
        q_keep_ratios: list[float | None] = []
        for i, wl_eff in enumerate(wl_eff_tuple):
            kz = kz_per_channel[i]
            h_phase = torch.exp(1j * kz * float(actual_dz))
            q_keep_ratio = None
            if self.propagation_config.use_bandlimit_window:
                q_window = _build_q_window(fx, fy, float(wl_eff), float(actual_dz), span_x, span_y)
                q_keep_ratio = float(q_window.to(torch.float32).mean().detach().cpu())
                h_phase = h_phase * q_window.to(h_phase.dtype)
            if self.propagation_config.evanescent_mode == "cut":
                h_phase = h_phase * pass_mask_per_channel[i].to(h_phase.dtype)
            transfer_per_channel.append(h_phase)
            q_keep_ratios.append(q_keep_ratio)

        cached = {
            "transfer": torch.stack(transfer_per_channel, dim=0),
            "q_keep_ratios": q_keep_ratios,
        }
        self._transfer_cache[cache_key] = cached
        return cached

    def _propagate_canvas_by_distance(
        self,
        field_canvas: torch.Tensor,
        *,
        actual_dz: float,
        dx: float,
        wavelengths_m: list[float],
        refractive_index: float,
        canvas_h: int,
        canvas_w: int,
    ) -> torch.Tensor:
        if float(actual_dz) == 0.0:
            return field_canvas

        wl_eff_list = [float(wl) / float(refractive_index) for wl in wavelengths_m]
        transfer_cache = self._get_transfer_function_cache(
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            dx=dx,
            device=field_canvas.device,
            dtype=field_canvas.real.dtype,
            wl_eff_list=wl_eff_list,
            actual_dz=float(actual_dz),
        )
        transfer = transfer_cache["transfer"]
        field_f = _fft2_centered(field_canvas, norm=self.propagation_config.fft_norm)
        return _ifft2_centered(
            field_f * transfer.unsqueeze(0),
            norm=self.propagation_config.fft_norm,
        )

    @staticmethod
    def _apply_complex_rigid_transform(
        field: torch.Tensor,
        shift_x_m: float,
        shift_y_m: float,
        dx: float,
    ) -> torch.Tensor:
        if abs(float(shift_x_m)) <= 0.0 and abs(float(shift_y_m)) <= 0.0:
            return field

        b, _, h, w = field.shape
        dtype = field.real.dtype
        xs = (torch.arange(w, device=field.device, dtype=dtype) - (w - 1) / 2.0) * float(dx)
        ys = (torch.arange(h, device=field.device, dtype=dtype) - (h - 1) / 2.0) * float(dx)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        src_x = grid_x - float(shift_x_m)
        src_y = grid_y - float(shift_y_m)

        if w > 1:
            src_x_norm = 2.0 * src_x / (float(w - 1) * float(dx))
        else:
            src_x_norm = torch.zeros_like(src_x)
        if h > 1:
            src_y_norm = 2.0 * src_y / (float(h - 1) * float(dx))
        else:
            src_y_norm = torch.zeros_like(src_y)

        grid = torch.stack((src_x_norm, src_y_norm), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        transformed_real = F.grid_sample(
            field.real,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        transformed_imag = F.grid_sample(
            field.imag,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return torch.complex(transformed_real, transformed_imag)

    def _resolve_detector_record(self) -> tuple[float, DetectorLayer]:
        for z, layer in reversed(self.layers):
            if isinstance(layer, DetectorLayer):
                return float(z), layer
        raise ValueError("No DetectorLayer found in propagation chain")

    def _measure_prefix_readout_at_detector_plane(
        self,
        *,
        field_canvas: torch.Tensor,
        current_z: float,
        current_dx: float,
        detector_z: float,
        detector_layer: DetectorLayer,
        wavelengths_m: list[float],
        refractive_index: float,
        canvas_h: int,
        canvas_w: int,
        error_scale: float,
    ) -> torch.Tensor:
        actual_dz = float(detector_z - current_z) + float(self.error_config.delta_z_m) * float(error_scale)
        detector_canvas = self._propagate_canvas_by_distance(
            field_canvas,
            actual_dz=actual_dz,
            dx=current_dx,
            wavelengths_m=wavelengths_m,
            refractive_index=refractive_index,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )
        detector_canvas = self._apply_complex_rigid_transform(
            detector_canvas,
            shift_x_m=float(self.error_config.shift_x_m) * float(error_scale),
            shift_y_m=float(self.error_config.shift_y_m) * float(error_scale),
            dx=current_dx,
        )
        detector_active = _center_crop(detector_canvas, detector_layer.sy, detector_layer.sx)
        _, detector_intensity = detector_layer.measure(detector_active)
        return detector_intensity

    def propagate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if len(self.layers) == 0:
            raise ValueError("PropagateContext has no layers")

        first_source_idx = None
        for idx, (_, layer) in enumerate(self.layers):
            if isinstance(layer, SourceLayer):
                first_source_idx = idx
                break
        if first_source_idx is None:
            raise ValueError("No SourceLayer found in propagation chain")

        z_source, source_layer = self.layers[first_source_idx]
        field_active = source_layer.forward(input_tensor)
        if field_active.dim() == 3:
            field_active = field_active.unsqueeze(0)
        if not torch.is_complex(field_active):
            raise ValueError("SourceLayer output must be complex field")

        canvas_h, canvas_w = self._resolve_canvas_shape()
        field_canvas = _center_pad_to(field_active, canvas_h, canvas_w)

        prev_z = z_source
        _, channel_count, prev_h, prev_w = field_active.shape
        prev_dx = source_layer.dx
        wavelengths_m = self._resolve_source_wavelengths(source_layer, channel_count)
        n_medium = float(self.propagation_config.refractive_index)
        error_scale = float(self.error_factor)

        for z, layer in self.layers[first_source_idx + 1 :]:
            nominal_dz = float(z - prev_z)
            actual_dz = nominal_dz + float(self.error_config.delta_z_m) * error_scale
            target_w = int(getattr(layer, "sx", prev_w))
            target_h = int(getattr(layer, "sy", prev_h))
            dx = prev_dx

            field_canvas = self._propagate_canvas_by_distance(
                field_canvas,
                actual_dz=actual_dz,
                dx=dx,
                wavelengths_m=wavelengths_m,
                refractive_index=n_medium,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
            )

            field_canvas = self._apply_complex_rigid_transform(
                field_canvas,
                shift_x_m=float(self.error_config.shift_x_m) * error_scale,
                shift_y_m=float(self.error_config.shift_y_m) * error_scale,
                dx=dx,
            )

            field_in_active = _center_crop(field_canvas, target_h, target_w)
            field_active = layer.forward(field_in_active)
            if field_active.dim() == 3:
                field_active = field_active.unsqueeze(0)
            if not torch.is_complex(field_active):
                raise ValueError(f"{layer.__class__.__name__} output must be complex field")
            if not isinstance(layer, DetectorLayer):
                field_canvas = _center_replace(field_canvas, field_active)

            prev_z = float(z)
            _, _, prev_h, prev_w = field_active.shape
            prev_dx = getattr(layer, "dx", prev_dx)

        return field_active

    def propagate_with_prefix_readouts(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(self.layers) == 0:
            raise ValueError("PropagateContext has no layers")

        detector_z, detector_layer = self._resolve_detector_record()

        first_source_idx = None
        for idx, (_, layer) in enumerate(self.layers):
            if isinstance(layer, SourceLayer):
                first_source_idx = idx
                break
        if first_source_idx is None:
            raise ValueError("No SourceLayer found in propagation chain")

        z_source, source_layer = self.layers[first_source_idx]
        field_active = source_layer.forward(input_tensor)
        if field_active.dim() == 3:
            field_active = field_active.unsqueeze(0)
        if not torch.is_complex(field_active):
            raise ValueError("SourceLayer output must be complex field")

        canvas_h, canvas_w = self._resolve_canvas_shape()
        field_canvas = _center_pad_to(field_active, canvas_h, canvas_w)

        prev_z = z_source
        _, channel_count, prev_h, prev_w = field_active.shape
        prev_dx = source_layer.dx
        wavelengths_m = self._resolve_source_wavelengths(source_layer, channel_count)
        n_medium = float(self.propagation_config.refractive_index)
        error_scale = float(self.error_factor)

        outputs: dict[str, torch.Tensor] = {}
        prefix_index = 0

        for z, layer in self.layers[first_source_idx + 1 :]:
            nominal_dz = float(z - prev_z)
            actual_dz = nominal_dz + float(self.error_config.delta_z_m) * error_scale
            target_w = int(getattr(layer, "sx", prev_w))
            target_h = int(getattr(layer, "sy", prev_h))
            dx = prev_dx

            field_canvas = self._propagate_canvas_by_distance(
                field_canvas,
                actual_dz=actual_dz,
                dx=dx,
                wavelengths_m=wavelengths_m,
                refractive_index=n_medium,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
            )
            field_canvas = self._apply_complex_rigid_transform(
                field_canvas,
                shift_x_m=float(self.error_config.shift_x_m) * error_scale,
                shift_y_m=float(self.error_config.shift_y_m) * error_scale,
                dx=dx,
            )

            field_in_active = _center_crop(field_canvas, target_h, target_w)

            if isinstance(layer, DetectorLayer):
                detector_layer.record_measurement(field_in_active)
                outputs["final_detector"] = detector_layer.I
                outputs["final_detector_field"] = detector_layer.E
                prev_z = float(z)
                break

            field_active = layer.forward(field_in_active)
            if field_active.dim() == 3:
                field_active = field_active.unsqueeze(0)
            if not torch.is_complex(field_active):
                raise ValueError(f"{layer.__class__.__name__} output must be complex field")

            field_canvas = _center_replace(field_canvas, field_active)
            prefix_index += 1
            current_dx = getattr(layer, "dx", prev_dx)
            outputs[f"prefix_readout_{prefix_index}"] = self._measure_prefix_readout_at_detector_plane(
                field_canvas=field_canvas,
                current_z=float(z),
                current_dx=current_dx,
                detector_z=detector_z,
                detector_layer=detector_layer,
                wavelengths_m=wavelengths_m,
                refractive_index=n_medium,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                error_scale=error_scale,
            )

            prev_z = float(z)
            _, _, prev_h, prev_w = field_active.shape
            prev_dx = current_dx

        if "final_detector" not in outputs:
            raise RuntimeError("Propagation finished without reaching DetectorLayer")

        return outputs


__all__ = ["PropagateContext"]
