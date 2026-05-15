from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from optical.data import MultiScaleFrequencyTargetTransform


class OpticalMultiscaleLoss(nn.Module):
    """
    Compute final-intensity, cumulative-scale, and band-residual losses.

    Expected keys:
    - model output: `final_detector`, `prefix_readout_1 ... prefix_readout_n`
    - target dict: `target_final`, `target_scale_1 ... target_scale_n`,
      `target_band_1 ... target_band_n`
    """

    def __init__(
        self,
        *,
        num_levels: int,
        final_weight: float = 1.0,
        scale_weight: float = 1.0,
        band_weight: float = 0.0,
        tv_weight: float = 0.0,
        background_weight: float = 0.0,
        background_threshold: float = 0.05,
        loss_type: Literal["mse", "l1"] = "mse",
        band_mode: Literal["prefix_difference", "frequency_transform"] = "prefix_difference",
        level_weights: Sequence[float] | None = None,
        band_transform: MultiScaleFrequencyTargetTransform | None = None,
    ) -> None:
        super().__init__()
        if int(num_levels) <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels!r}")
        if loss_type not in {"mse", "l1"}:
            raise ValueError(f"loss_type must be 'mse' or 'l1', got {loss_type!r}")
        if band_mode not in {"prefix_difference", "frequency_transform"}:
            raise ValueError(
                "band_mode must be 'prefix_difference' or 'frequency_transform', "
                f"got {band_mode!r}"
            )

        self.num_levels = int(num_levels)
        self.final_weight = float(final_weight)
        self.scale_weight = float(scale_weight)
        self.band_weight = float(band_weight)
        self.tv_weight = float(tv_weight)
        self.background_weight = float(background_weight)
        self.background_threshold = float(background_threshold)
        self.loss_type = loss_type
        self.band_mode = band_mode
        self.band_transform = band_transform

        if level_weights is None:
            self.level_weights = tuple(1.0 for _ in range(self.num_levels))
        else:
            self.level_weights = self._validate_level_weights(level_weights)

        if (
            self.band_weight != 0.0
            and self.band_mode == "frequency_transform"
            and self.band_transform is None
        ):
            self.band_transform = MultiScaleFrequencyTargetTransform(num_levels=self.num_levels)

    def _validate_level_weights(self, level_weights: Sequence[float]) -> tuple[float, ...]:
        weights = tuple(float(value) for value in level_weights)
        if len(weights) != self.num_levels:
            raise ValueError(
                f"level_weights length must equal num_levels={self.num_levels}, "
                f"got {len(weights)}"
            )
        return weights

    def set_level_weights(self, level_weights: Sequence[float]) -> None:
        self.level_weights = self._validate_level_weights(level_weights)

    def _base_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match, "
                f"got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if self.loss_type == "l1":
            return torch.mean(torch.abs(prediction - target))
        diff = prediction - target
        return torch.mean(diff.square())

    def _scale_aligned_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match, "
                f"got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        batch_size = int(prediction.shape[0])
        prediction_flat = prediction.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        numerator = torch.sum(prediction_flat * target_flat, dim=1, keepdim=True)
        denominator = torch.sum(prediction_flat.square(), dim=1, keepdim=True).clamp_min(1.0e-8)
        scale = numerator / denominator
        aligned_prediction = (prediction_flat * scale).reshape_as(prediction)
        return self._base_loss(aligned_prediction, target)

    @staticmethod
    def _correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match, "
                f"got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        batch_size = int(prediction.shape[0])
        prediction_flat = prediction.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        prediction_centered = prediction_flat - prediction_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        prediction_norm = torch.linalg.norm(prediction_centered, dim=1)
        target_norm = torch.linalg.norm(target_centered, dim=1)
        numerator = torch.sum(prediction_centered * target_centered, dim=1)

        both_small = (prediction_norm <= 1.0e-8) & (target_norm <= 1.0e-8)
        correlation = numerator / (prediction_norm * target_norm).clamp_min(1.0e-8)
        correlation = torch.where(both_small, torch.ones_like(correlation), correlation)
        return torch.mean(1.0 - correlation)

    @staticmethod
    def _readout_image(readout: torch.Tensor) -> torch.Tensor:
        if readout.dim() != 4:
            raise ValueError(f"readout must be [B,C,H,W], got {tuple(readout.shape)}")
        if int(readout.shape[1]) != 1:
            raise ValueError(
                "current multiscale loss expects single-channel detector intensities, "
                f"got channels={int(readout.shape[1])}"
            )
        return readout[:, 0]

    @staticmethod
    def _target_image(target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 4 and int(target.shape[1]) == 1:
            return target[:, 0]
        if target.dim() == 3:
            return target
        raise ValueError(
            "target must be [B,1,H,W] or [B,H,W] / [1,H,W], "
            f"got {tuple(target.shape)}"
        )

    def _extract_predicted_band(self, readout_image: torch.Tensor, level_idx: int) -> torch.Tensor:
        if self.band_mode == "prefix_difference":
            if level_idx == 1:
                return readout_image

            previous_key = f"prefix_readout_{level_idx - 1}"
            raise RuntimeError(
                f"prefix-difference band extraction for level {level_idx} must be handled in forward; "
                f"missing previous readout key {previous_key!r}"
            )

        if self.band_transform is None:
            raise RuntimeError("band_transform is not initialized")

        predicted_bands: list[torch.Tensor] = []
        for sample_image in readout_image:
            transformed = self.band_transform(sample_image)
            band_tensor = transformed[f"target_band_{level_idx}"]
            predicted_bands.append(band_tensor.squeeze(0))
        return torch.stack(predicted_bands, dim=0)

    @staticmethod
    def _tv_loss(image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3:
            raise ValueError(f"tv image must be [B,H,W], got {tuple(image.shape)}")
        vertical = torch.abs(image[:, 1:, :] - image[:, :-1, :]).mean()
        horizontal = torch.abs(image[:, :, 1:] - image[:, :, :-1]).mean()
        return vertical + horizontal

    def _background_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "background prediction and target shapes must match, "
                f"got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        mask = (target < self.background_threshold).to(dtype=prediction.dtype)
        denom = mask.sum().clamp_min(1.0)
        return ((prediction.square()) * mask).sum() / denom

    def forward(
        self,
        model_output: dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
        target_output: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | object],
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if "final_detector" not in model_output:
            raise KeyError("model_output must contain 'final_detector'")
        if "target_final" not in target_output:
            raise KeyError("target_output must contain 'target_final'")

        final_prediction = self._readout_image(model_output["final_detector"])  # type: ignore[arg-type]
        final_target = self._target_image(target_output["target_final"])  # type: ignore[arg-type]
        final_loss = self._scale_aligned_loss(final_prediction, final_target)

        scale_losses: list[torch.Tensor] = []
        band_losses: list[torch.Tensor] = []
        weighted_scale_loss = final_loss.new_zeros(())
        weighted_band_loss = final_loss.new_zeros(())
        previous_readout_image = torch.zeros_like(final_prediction)

        for level_idx in range(1, self.num_levels + 1):
            readout_key = f"prefix_readout_{level_idx}"
            scale_key = f"target_scale_{level_idx}"
            band_key = f"target_band_{level_idx}"
            if readout_key not in model_output:
                raise KeyError(f"model_output must contain {readout_key!r}")
            if scale_key not in target_output:
                raise KeyError(f"target_output must contain {scale_key!r}")
            if band_key not in target_output:
                raise KeyError(f"target_output must contain {band_key!r}")

            readout_image = self._readout_image(model_output[readout_key])  # type: ignore[arg-type]
            scale_target = self._target_image(target_output[scale_key])  # type: ignore[arg-type]
            level_weight = float(self.level_weights[level_idx - 1])
            scale_loss = self._scale_aligned_loss(readout_image, scale_target)
            scale_losses.append(scale_loss)
            weighted_scale_loss = weighted_scale_loss + (level_weight * scale_loss)

            if self.band_weight != 0.0:
                if self.band_mode == "prefix_difference":
                    predicted_band = readout_image - previous_readout_image
                else:
                    predicted_band = self._extract_predicted_band(readout_image, level_idx)
                band_target = self._target_image(target_output[band_key])  # type: ignore[arg-type]
                band_loss = self._correlation_loss(predicted_band, band_target)
                band_losses.append(band_loss)
                weighted_band_loss = weighted_band_loss + (level_weight * band_loss)

            previous_readout_image = readout_image

        total = (
            self.final_weight * final_loss
            + self.scale_weight * weighted_scale_loss
            + self.band_weight * weighted_band_loss
        )
        tv_loss = self._tv_loss(final_prediction) if self.tv_weight != 0.0 else final_loss.new_zeros(())
        background_loss = (
            self._background_loss(final_prediction, final_target)
            if self.background_weight != 0.0
            else final_loss.new_zeros(())
        )
        total = total + self.tv_weight * tv_loss
        total = total + self.background_weight * background_loss
        scale_loss_mean = torch.stack(scale_losses).mean() if scale_losses else final_loss.new_zeros(())
        if band_losses:
            band_loss_mean = torch.stack(band_losses).mean()
        else:
            band_loss_mean = final_loss.new_zeros(())

        return {
            "total_loss": total,
            "final_loss": final_loss,
            "scale_loss": scale_loss_mean,
            "band_loss": band_loss_mean,
            "tv_loss": tv_loss,
            "background_loss": background_loss,
            "scale_losses": tuple(scale_losses),
            "band_losses": tuple(band_losses),
        }
