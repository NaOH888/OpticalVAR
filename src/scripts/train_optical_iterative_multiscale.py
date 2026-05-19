from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from conditioning import ConditionEmbeddingLayer
from optical.core import DetectorConfig, PropagationConfig, PropagationErrorConfig, SourceConfig
from optical.data import FrequencyPathDataset, MultiScaleFrequencyTargetTransform, NpzImageDataset
from optical.layers import DetectorLayer, DiffractiveAmplitudeLayer, DiffractivePhaseLayer, SLMDeviceLayer
from optical.models import IterativeMultiscaleEncoder, IterativeMultiscaleOpticalModel, IterativeOpticalDecoder
from optical.models.multiscale import OpticalPrefixReadoutDecoder
from scripts.train_optical_multiscale import (
    _build_initial_amplitude_map,
    _build_initial_phase_map,
    _expand_between_layer_distances,
    _load_config,
    _move_batch_to_device,
    _resolve_multiscale_cutoffs,
    _resolve_path,
    _resolve_frozen_phase_layers,
    _resolve_phase_layer_geometry,
    _resolve_surface_modulation_mode,
    _seed_everything,
)
from vae import build_perceptual_loss


class IterativeStepLoss(nn.Module):
    def __init__(
        self,
        *,
        num_steps: int,
        loss_type: str = "l1",
        step_weights: list[float] | tuple[float, ...] | None = None,
        perceptual_weight: float = 0.0,
        perceptual_loss_fn: nn.Module | None = None,
        latent_diversity_weight: float = 0.0,
        latent_diversity_margin: float = 0.05,
        state_normalization: str = "mean_power",
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.loss_type = str(loss_type)
        if step_weights is None:
            self.step_weights = tuple(1.0 for _ in range(self.num_steps))
        else:
            self.step_weights = tuple(float(value) for value in step_weights)
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual_loss_fn = perceptual_loss_fn
        self.latent_diversity_weight = float(latent_diversity_weight)
        self.latent_diversity_margin = float(latent_diversity_margin)
        self.state_normalization = str(state_normalization)
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.loss_type not in {"l1", "mse"}:
            raise ValueError(f"loss_type must be 'l1' or 'mse', got {self.loss_type!r}")
        if len(self.step_weights) != self.num_steps:
            raise ValueError(
                f"step_weights length must equal num_steps={self.num_steps}, got {len(self.step_weights)}"
            )
        if any(value <= 0.0 for value in self.step_weights):
            raise ValueError("step_weights must all be positive")
        if self.perceptual_weight != 0.0 and self.perceptual_loss_fn is None:
            raise ValueError("perceptual_loss_fn must be provided when perceptual_weight is non-zero")
        if self.state_normalization != "mean_power":
            raise ValueError(
                "state_normalization must be 'mean_power' for the first implementation, "
                f"got {self.state_normalization!r}"
            )

    @staticmethod
    def _normalize_image(image: torch.Tensor) -> torch.Tensor:
        mean_power = image.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        return image / mean_power

    def _base_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction and target must match, got {tuple(prediction.shape)} vs {tuple(target.shape)}")
        if self.loss_type == "l1":
            return torch.mean(torch.abs(prediction - target))
        return torch.mean((prediction - target).square())

    def _scale_aligned_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction and target must match, got {tuple(prediction.shape)} vs {tuple(target.shape)}")
        batch_size = int(prediction.shape[0])
        prediction_flat = prediction.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        numerator = torch.sum(prediction_flat * target_flat, dim=1, keepdim=True)
        denominator = torch.sum(prediction_flat.square(), dim=1, keepdim=True).clamp_min(1.0e-8)
        scale = numerator / denominator
        aligned_prediction = (prediction_flat * scale).reshape_as(prediction)
        return self._base_loss(aligned_prediction, target)

    def _latent_diversity_loss(
        self,
        *,
        prediction: torch.Tensor,
        latent_input: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.dim() != 4 or int(prediction.shape[1]) != 1:
            raise ValueError(f"prediction must be [B,1,H,W], got {tuple(prediction.shape)}")
        if latent_input.shape[0] != prediction.shape[0]:
            raise ValueError(
                "latent_input batch size must match prediction batch size, "
                f"got {tuple(latent_input.shape)} vs {tuple(prediction.shape)}"
            )
        batch_size = int(prediction.shape[0])
        if batch_size < 2:
            return prediction.new_zeros(())

        prediction_flat = prediction.reshape(batch_size, -1)
        pred_pairwise = torch.mean(
            torch.abs(prediction_flat.unsqueeze(1) - prediction_flat.unsqueeze(0)),
            dim=-1,
        )

        if latent_input.is_floating_point():
            latent_flat = latent_input.reshape(batch_size, -1)
            latent_norm = torch.linalg.norm(latent_flat, dim=1, keepdim=True).clamp_min(1.0e-8)
            latent_unit = latent_flat / latent_norm
            cosine_similarity = torch.matmul(latent_unit, latent_unit.T).clamp(-1.0, 1.0)
            latent_pairwise = 0.5 * (1.0 - cosine_similarity)
        else:
            latent_flat = latent_input.reshape(batch_size, -1)
            latent_pairwise = torch.mean(
                (latent_flat.unsqueeze(1) != latent_flat.unsqueeze(0)).to(dtype=prediction.dtype),
                dim=-1,
            )

        target_margin = self.latent_diversity_margin * latent_pairwise.to(dtype=prediction.dtype)
        pairwise_penalty = torch.relu(target_margin - pred_pairwise)
        upper_mask = torch.triu(
            torch.ones((batch_size, batch_size), device=prediction.device, dtype=torch.bool),
            diagonal=1,
        )
        if not torch.any(upper_mask):
            return prediction.new_zeros(())
        return torch.mean(pairwise_penalty[upper_mask])

    def forward(
        self,
        *,
        predictions: tuple[torch.Tensor, ...],
        batch: dict[str, Any],
        latent_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if len(predictions) != self.num_steps:
            raise ValueError(f"expected {self.num_steps} predictions, got {len(predictions)}")
        step_losses: list[torch.Tensor] = []
        step_perceptual_losses: list[torch.Tensor] = []
        for index, prediction in enumerate(predictions, start=1):
            target_key = f"target_scale_{index}"
            if target_key not in batch:
                raise KeyError(f"batch is missing {target_key!r}")
            target = batch[target_key]
            if target.dim() == 3:
                target = target.unsqueeze(1)
            target = target.to(dtype=torch.float32)
            base_loss = self._scale_aligned_loss(prediction, target)
            apply_perceptual = index == self.num_steps
            perceptual_loss = (
                self.perceptual_loss_fn(prediction, target)
                if apply_perceptual and self.perceptual_weight != 0.0 and self.perceptual_loss_fn is not None
                else base_loss.new_zeros(())
            )
            step_losses.append(base_loss)
            step_perceptual_losses.append(perceptual_loss)
        if not step_losses:
            raise RuntimeError("step loss list must not be empty")
        weight_sum = float(sum(self.step_weights))
        weighted_scale_total = sum(
            weight * loss for weight, loss in zip(self.step_weights, step_losses)
        )
        scale_loss = weighted_scale_total / weight_sum
        latent_diversity_loss = scale_loss.new_zeros(())
        if self.latent_diversity_weight != 0.0:
            if latent_input is None:
                raise ValueError("latent_input must be provided when latent_diversity_weight is non-zero")
            latent_diversity_loss = self._latent_diversity_loss(
                prediction=predictions[0],
                latent_input=latent_input,
            )
        final_loss = step_losses[-1]
        final_perceptual = step_perceptual_losses[-1] if step_perceptual_losses else final_loss.new_zeros(())
        return {
            "total_loss": (
                scale_loss
                + self.perceptual_weight * final_perceptual
                + self.latent_diversity_weight * latent_diversity_loss
            ),
            "final_loss": final_loss,
            "scale_loss": scale_loss,
            "band_loss": scale_loss.new_zeros(()),
            "tv_loss": scale_loss.new_zeros(()),
            "background_loss": scale_loss.new_zeros(()),
            "perceptual_loss": final_perceptual,
            "latent_diversity_loss": latent_diversity_loss,
            "scale_losses": tuple(step_losses),
        }


def _build_dataset_and_loader(
    config: dict[str, Any],
    *,
    config_dir: Path,
    repo_root: Path,
) -> tuple[FrequencyPathDataset, DataLoader]:
    dataset_cfg = dict(config["dataset"])
    multiscale_cfg = dict(config["multiscale"])
    manifest_path = _resolve_path(dataset_cfg["manifest_path"], config_dir=config_dir, repo_root=repo_root)
    base_dataset = NpzImageDataset.from_manifest(
        manifest_path,
        max_items=dataset_cfg.get("max_items"),
        channel_mode=str(dataset_cfg.get("channel_mode", "keep")),
    )
    target_transform = MultiScaleFrequencyTargetTransform(
        num_levels=int(multiscale_cfg["num_levels"]),
        max_freq_fraction=float(multiscale_cfg.get("max_freq_fraction", 1.0)),
        transition_width=float(multiscale_cfg.get("transition_width", 0.05)),
        cutoff_mode=str(multiscale_cfg.get("cutoff_mode", "linear")),
        cutoffs=_resolve_multiscale_cutoffs(multiscale_cfg, config_dir=config_dir, repo_root=repo_root),
    )
    dataset = FrequencyPathDataset(
        base_dataset,
        target_transform,
        image_key="image",
        label_key="label",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_cfg.get("batch_size", 8)),
        shuffle=bool(dataset_cfg.get("shuffle", True)),
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        drop_last=bool(dataset_cfg.get("drop_last", False)),
    )
    return dataset, loader


def _build_condition_layer(config: dict[str, Any]) -> tuple[ConditionEmbeddingLayer | None, int | None]:
    encoder_cfg = dict(config["encoder"])
    condition_mode = encoder_cfg.get("condition_mode")
    class_conditional = bool(encoder_cfg.get("class_conditional", False))
    if condition_mode is None and not class_conditional:
        return None, None
    condition_embed_dim = int(encoder_cfg.get("condition_embed_dim", 128))
    resolved_mode = "class_index" if condition_mode is None else str(condition_mode)
    layer = ConditionEmbeddingLayer(
        mode=resolved_mode,
        output_dim=condition_embed_dim,
        num_classes=int(encoder_cfg.get("num_classes", 0)) if resolved_mode == "class_index" else None,
        input_dim=encoder_cfg.get("condition_input_dim") if resolved_mode == "attribute_vector" else None,
        embed_dim=int(encoder_cfg.get("class_embed_dim", 128)),
        hidden_dim=encoder_cfg.get("condition_hidden_dim"),
    )
    return layer, condition_embed_dim


def _build_optical_decoder(config: dict[str, Any]) -> OpticalPrefixReadoutDecoder:
    optical_cfg = dict(config["optical"])
    optical_num_layers = int(optical_cfg.get("num_layers", config["multiscale"]["num_levels"]))
    if optical_num_layers <= 0:
        raise ValueError("optical.num_layers must be positive")
    source_cfg = SourceConfig(
        wavelengths_m=tuple(float(value) for value in optical_cfg["source"]["wavelengths_m"]),
        light_mode=str(optical_cfg["source"]["light_mode"]),
        amplitude=float(optical_cfg["source"]["amplitude"]),
    )
    slm_cfg = dict(optical_cfg["slm"])
    slm = SLMDeviceLayer(
        pixel_pitch_x_m=float(slm_cfg["pixel_pitch_x_m"]),
        pixel_pitch_y_m=float(slm_cfg["pixel_pitch_y_m"]),
        pixel_count_x=int(slm_cfg["pixel_count_x"]),
        pixel_count_y=int(slm_cfg["pixel_count_y"]),
        dx=float(slm_cfg["dx_m"]),
        fill_factor=float(slm_cfg.get("fill_factor", 1.0)),
        phase_alpha=float(slm_cfg.get("phase_alpha", 2.0)),
        phase_bit_depth=slm_cfg.get("phase_bit_depth"),
        source_config=source_cfg,
    )
    phase_cfg = dict(optical_cfg["phase_layer"])
    modulation_mode = _resolve_surface_modulation_mode(phase_cfg)
    frozen_layers = _resolve_frozen_phase_layers(phase_cfg=phase_cfg, num_levels=optical_num_layers)

    optical_layers: list[nn.Module] = []
    for layer_index in range(optical_num_layers):
        width_m, height_m, grid_h, grid_w = _resolve_phase_layer_geometry(phase_cfg=phase_cfg, slm=slm)
        if modulation_mode == "phase":
            layer = DiffractivePhaseLayer(
                width_m=width_m,
                height_m=height_m,
                dx_m=slm.dx,
                channels=len(source_cfg.wavelengths_m),
                wavelengths_m=source_cfg.wavelengths_m,
                alpha_pi=float(phase_cfg.get("alpha_pi", 2.0)),
                share_across_channels=bool(phase_cfg.get("share_across_channels", True)),
                reference_wavelength_m=phase_cfg.get("reference_wavelength_m"),
                phase_grid_height=grid_h,
                phase_grid_width=grid_w,
                initial_phase_map_rad=_build_initial_phase_map(
                    phase_cfg=phase_cfg,
                    source_cfg=source_cfg,
                    phase_grid_height=grid_h,
                    phase_grid_width=grid_w,
                ),
            )
        else:
            layer = DiffractiveAmplitudeLayer(
                width_m=width_m,
                height_m=height_m,
                dx_m=slm.dx,
                channels=len(source_cfg.wavelengths_m),
                share_across_channels=bool(phase_cfg.get("share_across_channels", True)),
                amplitude_grid_height=grid_h,
                amplitude_grid_width=grid_w,
                initial_amplitude_map=_build_initial_amplitude_map(
                    phase_cfg=phase_cfg,
                    source_cfg=source_cfg,
                    amplitude_grid_height=grid_h,
                    amplitude_grid_width=grid_w,
                ),
            )
        if frozen_layers[layer_index]:
            layer.requires_grad_(False)
        optical_layers.append(layer)

    detector_cfg = DetectorConfig(
        width_num=int(optical_cfg["detector"]["width_num"]),
        height_num=int(optical_cfg["detector"]["height_num"]),
        detector_unit_len_m=float(optical_cfg["detector"]["detector_unit_len_m"]),
    )
    detector = DetectorLayer(config=detector_cfg, dx_m=slm.dx)
    propagation_cfg = PropagationConfig(
        canvas_h=optical_cfg["propagation"].get("canvas_h"),
        canvas_w=optical_cfg["propagation"].get("canvas_w"),
        canvas_factor=float(optical_cfg["propagation"].get("canvas_factor", 1.0)),
        refractive_index=float(optical_cfg["propagation"].get("refractive_index", 1.0)),
        use_bandlimit_window=bool(optical_cfg["propagation"].get("use_bandlimit_window", False)),
        evanescent_mode=str(optical_cfg["propagation"].get("evanescent_mode", "keep")),
        fft_norm=str(optical_cfg["propagation"].get("fft_norm", "ortho")),
    )
    error_cfg = PropagationErrorConfig(
        delta_z_m=float(optical_cfg["error"].get("delta_z_m", 0.0)),
        shift_x_m=float(optical_cfg["error"].get("shift_x_m", 0.0)),
        shift_y_m=float(optical_cfg["error"].get("shift_y_m", 0.0)),
    )
    return OpticalPrefixReadoutDecoder(
        slm_layer=slm,
        optical_layers=tuple(optical_layers),
        detector_layer=detector,
        distance_slm_to_first_layer_m=float(optical_cfg["distances_m"]["slm_to_first_layer_m"]),
        distance_between_layers_m=_expand_between_layer_distances(
            optical_cfg["distances_m"]["between_layers_m"],
            num_levels=optical_num_layers,
        ),
        distance_last_layer_to_detector_m=float(optical_cfg["distances_m"]["last_layer_to_detector_m"]),
        propagation_config=propagation_cfg,
        error_config=error_cfg,
        default_error_factor=float(optical_cfg["error"].get("error_factor", 1.0)),
    )


def _build_model(config: dict[str, Any], *, sample_item: dict[str, Any]) -> IterativeMultiscaleOpticalModel:
    if "latent" not in sample_item:
        raise KeyError("iterative training requires dataset samples to contain 'latent'")
    latent = sample_item["latent"]
    if not isinstance(latent, torch.Tensor):
        latent = torch.as_tensor(latent)
    if latent.dim() != 3:
        raise ValueError(f"latent must be [C,H,W], got {tuple(latent.shape)}")

    iterative_cfg = dict(config["iterative"])
    encoder_cfg = dict(config["encoder"])
    condition_layer, condition_embed_dim = _build_condition_layer(config)
    encoder = IterativeMultiscaleEncoder(
        latent_channels=int(latent.shape[0]),
        latent_height=int(latent.shape[1]),
        latent_width=int(latent.shape[2]),
        output_height=int(encoder_cfg.get("output_height", sample_item["target_final"].shape[-2])),
        output_width=int(encoder_cfg.get("output_width", sample_item["target_final"].shape[-1])),
        num_steps=int(iterative_cfg["num_steps"]),
        step_embedding_dim=int(iterative_cfg["step_embedding_dim"]),
        condition_layer=condition_layer,
        condition_embed_dim=condition_embed_dim,
        latent_stage_channels=encoder_cfg.get("latent_channels"),
        prev_image_channels=encoder_cfg.get("prev_image_channels"),
        use_prev_image=bool(iterative_cfg.get("use_prev_image", True)),
        fusion_hidden_dim=int(encoder_cfg.get("fusion_hidden_dim", 128)),
        dropout_prob=float(encoder_cfg.get("dropout_prob", 0.0)),
        weight_init=str(encoder_cfg.get("weight_init", "kaiming_uniform")),
        output_weight_init=str(encoder_cfg.get("output_weight_init", "xavier_uniform")),
        upsample_mode=str(encoder_cfg.get("upsample_mode", "bilinear")),
    )
    decoder = IterativeOpticalDecoder(optical_decoder=_build_optical_decoder(config))
    return IterativeMultiscaleOpticalModel(
        encoder=encoder,
        decoder=decoder,
        state_normalization=str(iterative_cfg.get("state_normalization", "mean_power")),
    )


def _build_condition_batch(batch: dict[str, Any], *, config: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    encoder_cfg = dict(config["encoder"])
    condition_mode = encoder_cfg.get("condition_mode")
    if condition_mode is None and not bool(encoder_cfg.get("class_conditional", False)):
        return None
    if "label" not in batch:
        raise KeyError("dataset batch must contain 'label' when conditioning is enabled")
    if condition_mode == "attribute_vector":
        return batch["label"].to(device=device, dtype=torch.float32)
    return batch["label"].to(device=device, dtype=torch.long).reshape(-1)


def _build_loss(config: dict[str, Any], *, device: torch.device, num_steps: int) -> IterativeStepLoss:
    loss_cfg = dict(config["loss"])
    iterative_cfg = dict(config["iterative"])
    perceptual_loss_fn = build_perceptual_loss(
        {
            "perceptual_weight": float(loss_cfg.get("perceptual_weight", 0.0)),
            "perceptual_weights": loss_cfg.get("perceptual_weights", "imagenet"),
            "perceptual_feature_layers": loss_cfg.get("perceptual_feature_layers", [3, 8, 15, 22]),
        }
    )
    if perceptual_loss_fn is not None:
        perceptual_loss_fn = perceptual_loss_fn.to(device)
    return IterativeStepLoss(
        num_steps=num_steps,
        loss_type=str(loss_cfg.get("loss_type", "l1")),
        step_weights=loss_cfg.get("step_weights"),
        perceptual_weight=float(loss_cfg.get("perceptual_weight", 0.0)),
        perceptual_loss_fn=perceptual_loss_fn,
        latent_diversity_weight=float(loss_cfg.get("latent_diversity_weight", 0.0)),
        latent_diversity_margin=float(loss_cfg.get("latent_diversity_margin", 0.05)),
        state_normalization=str(iterative_cfg.get("state_normalization", "mean_power")),
    )


def _build_optimizer(model: IterativeMultiscaleOpticalModel, *, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(train_cfg.get("lr", 1.0e-3))
    encoder_lr = float(train_cfg.get("encoder_lr", lr))
    decoder_lr = float(train_cfg.get("decoder_lr", lr))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    param_groups = [
        {
            "params": [param for param in model.encoder.parameters() if param.requires_grad],
            "lr": encoder_lr,
            "group_name": "encoder",
        },
        {
            "params": [param for param in model.decoder.parameters() if param.requires_grad],
            "lr": decoder_lr,
            "group_name": "decoder",
        },
    ]
    if not any(group["params"] for group in param_groups):
        raise ValueError("No trainable parameters found after applying freeze settings")
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.Adam(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )


def _resolve_resume_path(
    train_cfg: dict[str, Any],
    *,
    cli_resume: str | None,
    config_dir: Path,
    repo_root: Path,
) -> Path | None:
    raw_value = cli_resume if cli_resume is not None else train_cfg.get("resume_from")
    if raw_value is None:
        return None
    return _resolve_path(str(raw_value), config_dir=config_dir, repo_root=repo_root)


def _load_resume_checkpoint(
    *,
    model: IterativeMultiscaleOpticalModel,
    optimizer: torch.optim.Optimizer,
    resume_path: Path,
    device: torch.device,
    resume_optimizer: bool,
    resume_strict: bool,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=resume_strict)
    if resume_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    metrics = dict(checkpoint.get("metrics", {}))
    start_epoch = int(checkpoint.get("epoch", metrics.get("epoch", 0)))
    return start_epoch, metrics, checkpoint


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train(
    config: dict[str, Any],
    *,
    config_path: Path,
    resume_path_override: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = config_path.resolve().parent
    runtime_cfg = dict(config.get("runtime", {}))
    train_cfg = dict(config["training"])
    iterative_cfg = dict(config["iterative"])
    multiscale_cfg = dict(config["multiscale"])

    num_steps = int(iterative_cfg["num_steps"])
    if num_steps != int(multiscale_cfg["num_levels"]):
        raise ValueError("iterative.num_steps must equal multiscale.num_levels in the first implementation")

    _seed_everything(int(runtime_cfg.get("seed", 42)))
    device = torch.device(str(runtime_cfg.get("device", "cpu")))
    resume_path = resume_path_override or _resolve_resume_path(
        train_cfg,
        cli_resume=None,
        config_dir=config_dir,
        repo_root=repo_root,
    )
    dataset, loader = _build_dataset_and_loader(config, config_dir=config_dir, repo_root=repo_root)
    try:
        sample_item = dataset[0]
        model = _build_model(config, sample_item=sample_item).to(device)
        criterion = _build_loss(config, device=device, num_steps=num_steps)
        optimizer = _build_optimizer(model, train_cfg=train_cfg)

        output_dir = _resolve_path(str(train_cfg["output_dir"]), config_dir=config_dir, repo_root=repo_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "history.jsonl"
        (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        max_epochs = int(train_cfg.get("epochs", 1))
        log_interval = int(train_cfg.get("log_interval", 10))
        max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
        grad_clip_norm = train_cfg.get("grad_clip_norm")
        latest_metrics: dict[str, Any] = {}
        start_epoch = 0
        resumed_from: str | None = None
        previous_metrics: dict[str, Any] | None = None
        optimizer_group_lrs = {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }

        if resume_path is not None:
            resume_optimizer = bool(train_cfg.get("resume_optimizer", True))
            resume_strict = bool(train_cfg.get("resume_strict", True))
            start_epoch, previous_metrics, _ = _load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                resume_path=resume_path,
                device=device,
                resume_optimizer=resume_optimizer,
                resume_strict=resume_strict,
            )
            resumed_from = str(resume_path)
            print(
                f"[resume] path={resume_path} start_epoch={start_epoch} "
                f"resume_optimizer={resume_optimizer} resume_strict={resume_strict}"
            )
            if previous_metrics:
                print(f"[resume] previous_metrics={json.dumps(previous_metrics, ensure_ascii=False)}")
        if start_epoch >= max_epochs:
            raise ValueError(
                f"training.epochs={max_epochs} must be greater than resumed epoch {start_epoch}"
            )

        for epoch_idx in range(start_epoch, max_epochs):
            model.train()
            running_total = 0.0
            running_final = 0.0
            running_scale = 0.0
            running_band = 0.0
            running_tv = 0.0
            running_background = 0.0
            running_latent_div = 0.0
            running_perceptual = 0.0
            running_scale_steps = [0.0 for _ in range(num_steps)]
            step_count = 0

            for batch_idx, batch in enumerate(loader, start=1):
                batch = _move_batch_to_device(batch, device)
                latent = batch["latent"].to(device=device, dtype=torch.float32)
                condition = _build_condition_batch(batch, config=config, device=device)
                trajectory = model(
                    latent=latent,
                    condition=condition if config["encoder"].get("condition_mode") == "attribute_vector" else None,
                    class_labels=condition if config["encoder"].get("condition_mode") != "attribute_vector" else None,
                    num_steps=num_steps,
                    detach_prev_state=bool(iterative_cfg.get("detach_prev_state", False)),
                )
                loss_output = criterion(predictions=trajectory["predictions"], batch=batch, latent_input=latent)
                total_loss = loss_output["total_loss"]

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

                running_total += float(total_loss.detach().cpu())
                running_final += float(loss_output["final_loss"].detach().cpu())
                running_scale += float(loss_output["scale_loss"].detach().cpu())
                running_band += float(loss_output["band_loss"].detach().cpu())
                running_tv += float(loss_output["tv_loss"].detach().cpu())
                running_background += float(loss_output["background_loss"].detach().cpu())
                running_perceptual += float(loss_output["perceptual_loss"].detach().cpu())
                latent_div_value = float(loss_output["latent_diversity_loss"].detach().cpu())
                running_latent_div += latent_div_value
                step_losses = loss_output["scale_losses"]
                for step_index, step_loss in enumerate(step_losses):
                    running_scale_steps[step_index] += float(step_loss.detach().cpu())
                step_count += 1

                if batch_idx % log_interval == 0:
                    avg_scale_steps = [value / step_count for value in running_scale_steps]
                    print(
                        f"[epoch {epoch_idx + 1}/{max_epochs}] step={batch_idx} "
                        f"total={running_total / step_count:.6f} "
                        f"final={running_final / step_count:.6f} "
                        f"scale={running_scale / step_count:.6f} "
                        f"band={running_band / step_count:.6f} "
                        f"tv={running_tv / step_count:.6f} "
                        f"bg={running_background / step_count:.6f} "
                        f"perc={running_perceptual / step_count:.6f} "
                        f"latent_div={running_latent_div / step_count:.6f} "
                        f"scale_losses={avg_scale_steps} "
                    )

                if max_steps_per_epoch is not None and batch_idx >= int(max_steps_per_epoch):
                    break

            if step_count == 0:
                raise RuntimeError("training dataloader produced zero steps")

            latest_metrics = {
                "epoch": epoch_idx + 1,
                "total_loss": running_total / step_count,
                "final_loss": running_final / step_count,
                "scale_loss": running_scale / step_count,
                "scale_losses": [value / step_count for value in running_scale_steps],
                "band_loss": running_band / step_count,
                "tv_loss": running_tv / step_count,
                "background_loss": running_background / step_count,
                "perceptual_loss": running_perceptual / step_count,
                "latent_diversity_loss": running_latent_div / step_count,
                "optimizer_lrs": optimizer_group_lrs,
            }
            print(
                f"[epoch {epoch_idx + 1}/{max_epochs}] total={latest_metrics['total_loss']:.6f} "
                f"final={latest_metrics['final_loss']:.6f} "
                f"scale={latest_metrics['scale_loss']:.6f} "
                f"scale_losses={latest_metrics['scale_losses']} "
                f"band={latest_metrics['band_loss']:.6f} "
                f"tv={latest_metrics['tv_loss']:.6f} "
                f"bg={latest_metrics['background_loss']:.6f} "
                f"perc={latest_metrics['perceptual_loss']:.6f} "
                f"latent_div={latest_metrics['latent_diversity_loss']:.6f} "
                f"lrs={latest_metrics['optimizer_lrs']}"
            )
            _append_jsonl(history_path, latest_metrics)
            torch.save(
                {
                    "epoch": epoch_idx + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": latest_metrics,
                    "config": config,
                },
                output_dir / "latest.pt",
            )

        return {
            "output_dir": str(output_dir),
            "latest_checkpoint": str(output_dir / "latest.pt"),
            "history_path": str(history_path),
            "metrics": latest_metrics,
            "resumed_from": resumed_from,
            "previous_metrics": previous_metrics,
        }
    finally:
        dataset.base_dataset.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the iterative multiscale optical model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config file.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    result = train(
        _load_config(config_path),
        config_path=config_path,
        resume_path_override=args.resume_from.resolve() if args.resume_from is not None else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
