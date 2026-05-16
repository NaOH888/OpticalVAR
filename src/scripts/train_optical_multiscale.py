from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from conditioning import ConditionEmbeddingLayer, ConditionalLatentFusion, ContinuousMapLatentProjector, DiscreteCodeLatentProjector, LatentEmbeddingLayer
from optical.core import DetectorConfig, PropagationConfig, PropagationErrorConfig, SourceConfig
from optical.data import FrequencyPathDataset, MultiScaleFrequencyTargetTransform, NpzImageDataset, ReferencedImageLatentDataset
from optical.layers import DetectorLayer, DiffractivePhaseLayer, SLMDeviceLayer
from optical.losses import OpticalMultiscaleLoss
from optical.models import LatentPhaseMapEncoder, OpticalMultiscaleModel, OpticalPrefixReadoutDecoder, PhaseMapEncoder
from vae import build_perceptual_loss


def _load_config(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyYAML is required to load YAML configs") from exc
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"config root must be a mapping, got {type(payload).__name__}")
        return payload
    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def _resolve_path(path_value: str, *, config_dir: Path, repo_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return (repo_root / candidate).resolve()


def _resolve_multiscale_cutoffs(
    multiscale_cfg: dict[str, Any],
    *,
    config_dir: Path,
    repo_root: Path,
) -> tuple[float, ...] | None:
    if "cutoffs" in multiscale_cfg and multiscale_cfg["cutoffs"] is not None:
        return tuple(float(value) for value in multiscale_cfg["cutoffs"])
    cutoffs_path_value = multiscale_cfg.get("cutoffs_path")
    if cutoffs_path_value is None:
        return None
    cutoffs_path = _resolve_path(str(cutoffs_path_value), config_dir=config_dir, repo_root=repo_root)
    payload = json.loads(cutoffs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"multiscale cutoff file root must be a mapping, got {type(payload).__name__}")
    if "cutoffs" not in payload:
        raise KeyError(f"multiscale cutoff file {cutoffs_path} must contain 'cutoffs'")
    return tuple(float(value) for value in payload["cutoffs"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset_and_loader(
    config: dict[str, Any],
    *,
    config_dir: Path,
    repo_root: Path,
) -> tuple[FrequencyPathDataset, DataLoader, MultiScaleFrequencyTargetTransform]:
    dataset_cfg = dict(config["dataset"])
    multiscale_cfg = dict(config["multiscale"])
    dataset_path = _resolve_path(dataset_cfg["manifest_path"], config_dir=config_dir, repo_root=repo_root)
    manifest_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if "image_manifest_path" in manifest_payload:
        base_dataset = ReferencedImageLatentDataset.from_latent_manifest(
            dataset_path,
            max_items=dataset_cfg.get("max_items"),
            channel_mode=str(dataset_cfg.get("channel_mode", "keep")),
        )
    else:
        base_dataset = NpzImageDataset.from_manifest(
            dataset_path,
            max_items=dataset_cfg.get("max_items"),
            channel_mode=str(dataset_cfg.get("channel_mode", "keep")),
        )
    target_transform = MultiScaleFrequencyTargetTransform(
        num_levels=int(multiscale_cfg["num_levels"]),
        max_freq_fraction=float(multiscale_cfg.get("max_freq_fraction", 1.0)),
        transition_width=float(multiscale_cfg.get("transition_width", 0.05)),
        cutoffs=_resolve_multiscale_cutoffs(
            multiscale_cfg,
            config_dir=config_dir,
            repo_root=repo_root,
        ),
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
    return dataset, loader, target_transform


def _expand_between_layer_distances(raw_value: Any, *, num_levels: int) -> tuple[float, ...]:
    if num_levels <= 1:
        return ()
    if isinstance(raw_value, (int, float)):
        return tuple(float(raw_value) for _ in range(num_levels - 1))
    values = tuple(float(item) for item in raw_value)
    if len(values) != num_levels - 1:
        raise ValueError(
            f"distance_between_layers_m must have length {num_levels - 1}, got {len(values)}"
        )
    return values


def _build_initial_phase_map(
    *,
    phase_cfg: dict[str, Any],
    source_cfg: SourceConfig,
    phase_grid_height: int,
    phase_grid_width: int,
) -> torch.Tensor:
    init_mode = str(phase_cfg.get("init_mode", "constant"))
    share_across_channels = bool(phase_cfg.get("share_across_channels", True))
    channels = 1 if share_across_channels else len(source_cfg.wavelengths_m)
    alpha_pi = float(phase_cfg.get("alpha_pi", 2.0))
    phase_period_rad = alpha_pi * float(torch.pi)
    shape = (channels, phase_grid_height, phase_grid_width) if channels > 1 else (phase_grid_height, phase_grid_width)

    if init_mode == "constant":
        initial_phase_value_rad = float(phase_cfg.get("initial_phase_value_rad", 0.0))
        return torch.full(shape, fill_value=initial_phase_value_rad, dtype=torch.float32)
    if init_mode == "uniform":
        init_min_rad = float(phase_cfg.get("init_min_rad", 0.0))
        init_max_rad = float(phase_cfg.get("init_max_rad", phase_period_rad))
        if init_max_rad <= init_min_rad:
            raise ValueError(
                f"phase init range must satisfy init_max_rad > init_min_rad, got {(init_min_rad, init_max_rad)}"
            )
        return torch.empty(shape, dtype=torch.float32).uniform_(init_min_rad, init_max_rad)
    raise ValueError(f"Unsupported phase init_mode: {init_mode!r}")


def _resolve_phase_layer_geometry(
    *,
    phase_cfg: dict[str, Any],
    slm: SLMDeviceLayer,
) -> tuple[float, float, int, int]:
    phase_grid_height = slm.sy if phase_cfg.get("phase_grid_height") is None else int(phase_cfg["phase_grid_height"])
    phase_grid_width = slm.sx if phase_cfg.get("phase_grid_width") is None else int(phase_cfg["phase_grid_width"])
    if phase_grid_height <= 0 or phase_grid_width <= 0:
        raise ValueError(
            "phase_grid_height and phase_grid_width must be positive, "
            f"got {(phase_grid_height, phase_grid_width)}"
        )

    phase_pitch_x_m = float(phase_cfg.get("phase_pitch_x_m", slm.pixel_pitch_x_m))
    phase_pitch_y_m = float(phase_cfg.get("phase_pitch_y_m", slm.pixel_pitch_y_m))
    if phase_pitch_x_m <= 0.0 or phase_pitch_y_m <= 0.0:
        raise ValueError(
            "phase_pitch_x_m and phase_pitch_y_m must be positive, "
            f"got {(phase_pitch_x_m, phase_pitch_y_m)}"
        )

    width_m = phase_pitch_x_m * float(phase_grid_width)
    height_m = phase_pitch_y_m * float(phase_grid_height)
    return width_m, height_m, phase_grid_height, phase_grid_width


def _build_model(
    config: dict[str, Any],
    *,
    sample_item: dict[str, Any],
) -> OpticalMultiscaleModel:
    multiscale_cfg = dict(config["multiscale"])
    optical_cfg = dict(config["optical"])
    encoder_cfg = dict(config["encoder"])
    num_levels = int(multiscale_cfg["num_levels"])

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
    optical_layers = []
    for _ in range(num_levels):
        width_m, height_m, phase_grid_height, phase_grid_width = _resolve_phase_layer_geometry(
            phase_cfg=phase_cfg,
            slm=slm,
        )
        optical_layers.append(
            DiffractivePhaseLayer(
                width_m=width_m,
                height_m=height_m,
                dx_m=slm.dx,
                channels=len(source_cfg.wavelengths_m),
                wavelengths_m=source_cfg.wavelengths_m,
                alpha_pi=float(phase_cfg.get("alpha_pi", 2.0)),
                share_across_channels=bool(phase_cfg.get("share_across_channels", True)),
                reference_wavelength_m=phase_cfg.get("reference_wavelength_m"),
                phase_grid_height=phase_grid_height,
                phase_grid_width=phase_grid_width,
                initial_phase_map_rad=_build_initial_phase_map(
                    phase_cfg=phase_cfg,
                    source_cfg=source_cfg,
                    phase_grid_height=phase_grid_height,
                    phase_grid_width=phase_grid_width,
                ),
            )
        )

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

    decoder = OpticalPrefixReadoutDecoder(
        slm_layer=slm,
        optical_layers=tuple(optical_layers),
        detector_layer=detector,
        distance_slm_to_first_layer_m=float(optical_cfg["distances_m"]["slm_to_first_layer_m"]),
        distance_between_layers_m=_expand_between_layer_distances(
            optical_cfg["distances_m"]["between_layers_m"],
            num_levels=num_levels,
        ),
        distance_last_layer_to_detector_m=float(optical_cfg["distances_m"]["last_layer_to_detector_m"]),
        propagation_config=propagation_cfg,
        error_config=error_cfg,
        default_error_factor=float(optical_cfg["error"].get("error_factor", 1.0)),
    )

    latent_embed_dim = int(encoder_cfg.get("latent_embed_dim", encoder_cfg.get("hidden_dim", 512)))
    condition_embed_dim = int(encoder_cfg.get("condition_embed_dim", encoder_cfg.get("class_embed_dim", 128)))
    fused_dim = int(encoder_cfg.get("fused_dim", encoder_cfg.get("hidden_dim", 512)))
    fusion_mode = str(encoder_cfg.get("fusion_mode", "concat"))
    fusion_hidden_dim = encoder_cfg.get("fusion_hidden_dim")
    condition_mode = encoder_cfg.get("condition_mode")
    class_conditional = bool(encoder_cfg.get("class_conditional", False))

    if "latent" in sample_item:
        sample_latent = sample_item["latent"]
        if not isinstance(sample_latent, torch.Tensor):
            sample_latent = torch.as_tensor(sample_latent)
        if sample_latent.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            discrete_shape = sample_latent.reshape(-1).shape[0]
            codebook_size = encoder_cfg.get("rvq_codebook_size")
            if codebook_size is None:
                raise ValueError("encoder.rvq_codebook_size must be provided for discrete latent inputs")
            latent_projector = DiscreteCodeLatentProjector(
                num_codebooks=discrete_shape,
                codebook_size=int(codebook_size),
                code_embed_dim=int(encoder_cfg.get("rvq_code_embed_dim", latent_embed_dim)),
                output_dim=latent_embed_dim,
                hidden_dim=encoder_cfg.get("latent_hidden_dim"),
                fuse_codebooks=str(encoder_cfg.get("rvq_fuse_codebooks", "sum")),
            )
        else:
            latent_shape = tuple(int(v) for v in sample_latent.shape)
            latent_projector = ContinuousMapLatentProjector(
                input_dim=int(torch.as_tensor(latent_shape).prod().item()),
                output_dim=latent_embed_dim,
                hidden_dim=encoder_cfg.get("latent_hidden_dim"),
            )
    else:
        latent_projector = ContinuousMapLatentProjector(
            input_dim=int(encoder_cfg.get("noise_channels", 1))
            * int(encoder_cfg.get("input_height", sample_item["target_final"].shape[-2]))
            * int(encoder_cfg.get("input_width", sample_item["target_final"].shape[-1])),
            output_dim=latent_embed_dim,
            hidden_dim=encoder_cfg.get("latent_hidden_dim"),
        )

    condition_layer = None
    fusion_layer = None
    if condition_mode is not None or class_conditional:
        resolved_condition_mode = "class_index" if condition_mode is None else str(condition_mode)
        condition_layer = ConditionEmbeddingLayer(
            mode=resolved_condition_mode,
            output_dim=condition_embed_dim,
            num_classes=int(encoder_cfg.get("num_classes", 0)) if resolved_condition_mode == "class_index" else None,
            input_dim=encoder_cfg.get("condition_input_dim") if resolved_condition_mode == "attribute_vector" else None,
            embed_dim=int(encoder_cfg.get("class_embed_dim", 128)),
            hidden_dim=encoder_cfg.get("condition_hidden_dim"),
        )
        fusion_layer = ConditionalLatentFusion(
            latent_dim=latent_embed_dim,
            condition_dim=condition_embed_dim,
            output_dim=fused_dim,
            mode=fusion_mode,
            hidden_dim=fusion_hidden_dim,
        )
        phase_input_dim = fused_dim
    else:
        phase_input_dim = latent_embed_dim

    encoder = LatentPhaseMapEncoder(
        latent_layer=LatentEmbeddingLayer(projector=latent_projector),
        condition_layer=condition_layer,
        fusion_layer=fusion_layer,
        phase_map_encoder=PhaseMapEncoder(
            input_dim=phase_input_dim,
            output_height=int(encoder_cfg.get("output_height", slm.pixel_count_y)),
            output_width=int(encoder_cfg.get("output_width", slm.pixel_count_x)),
            hidden_dim=int(encoder_cfg.get("hidden_dim", 512)),
            phase_alpha_pi=float(encoder_cfg.get("phase_alpha_pi", 2.0)),
            weight_init=str(encoder_cfg.get("weight_init", "kaiming_uniform")),
            output_weight_init=str(encoder_cfg.get("output_weight_init", "xavier_uniform")),
        ),
    )
    return OpticalMultiscaleModel(
        encoder=encoder,
        optical_decoder=decoder,
        upsample_mode=str(encoder_cfg.get("upsample_mode", "nearest")),
    )


def _build_loss(
    config: dict[str, Any],
    *,
    multiscale_transform: MultiScaleFrequencyTargetTransform,
    device: torch.device,
) -> OpticalMultiscaleLoss:
    loss_cfg = dict(config["loss"])
    num_levels = int(config["multiscale"]["num_levels"])
    band_mode = str(loss_cfg.get("band_mode", "prefix_difference"))
    band_transform = multiscale_transform if band_mode == "frequency_transform" else None
    perceptual_loss_fn = build_perceptual_loss(
        {
            "perceptual_weight": float(loss_cfg.get("perceptual_weight", 0.0)),
            "perceptual_weights": loss_cfg.get("perceptual_weights", "imagenet"),
            "perceptual_feature_layers": loss_cfg.get("perceptual_feature_layers", [3, 8, 15, 22]),
        }
    )
    if perceptual_loss_fn is not None:
        perceptual_loss_fn = perceptual_loss_fn.to(device)
    return OpticalMultiscaleLoss(
        num_levels=num_levels,
        final_weight=float(loss_cfg.get("final_weight", 1.0)),
        scale_weight=float(loss_cfg.get("scale_weight", 1.0)),
        band_weight=float(loss_cfg.get("band_weight", 0.0)),
        tv_weight=float(loss_cfg.get("tv_weight", 0.0)),
        background_weight=float(loss_cfg.get("background_weight", 0.0)),
        background_threshold=float(loss_cfg.get("background_threshold", 0.05)),
        perceptual_weight=float(loss_cfg.get("perceptual_weight", 0.0)),
        perceptual_loss_fn=perceptual_loss_fn,
        loss_type=str(loss_cfg.get("loss_type", "mse")),
        band_mode=band_mode,
        level_weights=loss_cfg.get("level_weights"),
        band_transform=band_transform,
    )


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        elif isinstance(value, tuple):
            output[key] = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in value)
        else:
            output[key] = value
    return output


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_level_weight_schedule(
    config: dict[str, Any],
    *,
    epoch_idx: int,
) -> tuple[float, ...] | None:
    train_cfg = dict(config["training"])
    schedule_cfg = train_cfg.get("level_weight_schedule")
    if schedule_cfg is None:
        return None
    if not isinstance(schedule_cfg, dict):
        raise TypeError("training.level_weight_schedule must be a mapping when provided")

    mode = str(schedule_cfg.get("mode", "swing"))
    num_levels = int(config["multiscale"]["num_levels"])
    if mode != "swing":
        raise ValueError(f"Unsupported level_weight_schedule mode: {mode!r}")

    base_weights = tuple(float(value) for value in schedule_cfg.get("base_weights", config["loss"].get("level_weights", [1.0] * num_levels)))
    if len(base_weights) != num_levels:
        raise ValueError(
            f"training.level_weight_schedule.base_weights must have length {num_levels}, got {len(base_weights)}"
        )

    positions_raw = schedule_cfg.get("positions")
    if positions_raw is None:
        center = (num_levels - 1) / 2.0
        positions = tuple(center - float(index) for index in range(num_levels))
    else:
        positions = tuple(float(value) for value in positions_raw)
        if len(positions) != num_levels:
            raise ValueError(
                f"training.level_weight_schedule.positions must have length {num_levels}, got {len(positions)}"
            )

    amplitude = float(schedule_cfg.get("amplitude", 0.0))
    period_epochs = int(schedule_cfg.get("period_epochs", max(int(train_cfg.get("epochs", 1)) - 1, 1)))
    if period_epochs <= 0:
        raise ValueError("training.level_weight_schedule.period_epochs must be positive")

    phase = 2.0 * math.pi * float(epoch_idx) / float(period_epochs)
    offset = amplitude * math.cos(phase)
    weights = [base_weight + offset * position for base_weight, position in zip(base_weights, positions)]

    min_weights_raw = schedule_cfg.get("min_weights")
    if min_weights_raw is not None:
        min_weights = tuple(float(value) for value in min_weights_raw)
        if len(min_weights) != num_levels:
            raise ValueError(
                f"training.level_weight_schedule.min_weights must have length {num_levels}, got {len(min_weights)}"
            )
        weights = [max(weight, lower) for weight, lower in zip(weights, min_weights)]

    max_weights_raw = schedule_cfg.get("max_weights")
    if max_weights_raw is not None:
        max_weights = tuple(float(value) for value in max_weights_raw)
        if len(max_weights) != num_levels:
            raise ValueError(
                f"training.level_weight_schedule.max_weights must have length {num_levels}, got {len(max_weights)}"
            )
        weights = [min(weight, upper) for weight, upper in zip(weights, max_weights)]

    return tuple(float(value) for value in weights)


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
    model: OpticalMultiscaleModel,
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


def _resolve_sample_ids(batch: dict[str, Any], *, batch_size: int) -> torch.Tensor:
    if "sample_id" not in batch:
        raise KeyError("dataset batch must contain 'sample_id' for fixed latent generation")
    sample_ids = batch["sample_id"]
    if not isinstance(sample_ids, torch.Tensor):
        sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
    sample_ids = sample_ids.to(dtype=torch.long).reshape(-1)
    if int(sample_ids.shape[0]) != batch_size:
        raise ValueError(
            f"sample_id batch size must match target batch size {batch_size}, got {int(sample_ids.shape[0])}"
        )
    return sample_ids


def _build_fixed_latent_batch(
    sample_ids: torch.Tensor,
    *,
    latent_seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    encoder_cfg = dict(config["encoder"])
    latent_shape = (
        int(encoder_cfg.get("noise_channels", 1)),
        int(encoder_cfg.get("input_height")),
        int(encoder_cfg.get("input_width")),
    )
    latents: list[torch.Tensor] = []
    for sample_id in sample_ids.tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(latent_seed) + int(sample_id))
        latents.append(torch.randn(latent_shape, generator=generator, dtype=torch.float32))
    return torch.stack(latents, dim=0).to(device=device)


def _resolve_condition_batch(
    batch: dict[str, Any],
    *,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    encoder_cfg = dict(config["encoder"])
    condition_mode = encoder_cfg.get("condition_mode")
    if condition_mode is None and not bool(encoder_cfg.get("class_conditional", False)):
        return None
    if "label" not in batch:
        raise KeyError("dataset batch must contain 'label' when conditioning is enabled")
    if condition_mode == "attribute_vector":
        return batch["label"].to(device=device, dtype=torch.float32)
    return batch["label"].to(device=device, dtype=torch.long).reshape(-1)


def _build_model_inputs(
    batch: dict[str, Any],
    *,
    model: OpticalMultiscaleModel,
    config: dict[str, Any],
    device: torch.device,
    latent_seed_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if "latent" in batch:
        latent = batch["latent"].to(device=device)
        if latent.is_floating_point():
            latent = latent.to(dtype=torch.float32)
        return latent, _resolve_condition_batch(batch, config=config, device=device)

    batch_size = int(batch["target_final"].shape[0])
    sample_ids = _resolve_sample_ids(batch, batch_size=batch_size)
    encoder_cfg = dict(config["encoder"])
    latent_seed = int(
        encoder_cfg.get(
            "latent_seed",
            0 if latent_seed_override is None else int(latent_seed_override),
        )
    )
    if latent_seed_override is not None:
        latent_seed = int(latent_seed_override)
    noise = _build_fixed_latent_batch(
        sample_ids,
        latent_seed=latent_seed,
        config=config,
        device=device,
    )
    return noise, _resolve_condition_batch(batch, config=config, device=device)


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

    seed = int(runtime_cfg.get("seed", 42))
    _seed_everything(seed)

    device = torch.device(str(runtime_cfg.get("device", "cpu")))
    dataset, loader, multiscale_transform = _build_dataset_and_loader(
        config,
        config_dir=config_dir,
        repo_root=repo_root,
    )
    sample_item = dataset[0]
    model = _build_model(config, sample_item=sample_item).to(device)
    criterion = _build_loss(config, multiscale_transform=multiscale_transform, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    output_dir = _resolve_path(str(train_cfg["output_dir"]), config_dir=config_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    config_copy_path = output_dir / "resolved_config.json"
    config_copy_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    max_epochs = int(train_cfg.get("epochs", 1))
    max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
    log_interval = int(train_cfg.get("log_interval", 10))
    latest_metrics: dict[str, Any] = {}
    start_epoch = 0
    resumed_from: str | None = None
    previous_metrics: dict[str, Any] | None = None

    resume_path = resume_path_override or _resolve_resume_path(
        train_cfg,
        cli_resume=None,
        config_dir=config_dir,
        repo_root=repo_root,
    )
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
        scheduled_level_weights = _resolve_level_weight_schedule(config, epoch_idx=epoch_idx)
        if scheduled_level_weights is not None:
            criterion.set_level_weights(scheduled_level_weights)
        model.train()
        running_total = 0.0
        running_final = 0.0
        running_scale = 0.0
        running_band = 0.0
        running_tv = 0.0
        running_background = 0.0
        running_perceptual = 0.0
        step_count = 0

        for step_idx, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, device)
            model_input, class_labels = _build_model_inputs(
                batch,
                model=model,
                config=config,
                device=device,
            )
            model_output = model(model_input, class_labels=class_labels)
            loss_output = criterion(model_output, batch)
            total_loss = loss_output["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            running_total += float(total_loss.detach().cpu())
            running_final += float(loss_output["final_loss"].detach().cpu())
            running_scale += float(loss_output["scale_loss"].detach().cpu())
            running_band += float(loss_output["band_loss"].detach().cpu())
            running_tv += float(loss_output["tv_loss"].detach().cpu())
            running_background += float(loss_output["background_loss"].detach().cpu())
            running_perceptual += float(loss_output["perceptual_loss"].detach().cpu())
            step_count += 1

            if step_idx % log_interval == 0:
                print(
                    f"[epoch {epoch_idx + 1}/{max_epochs}] "
                    f"step={step_idx} total={running_total / step_count:.6f} "
                    f"final={running_final / step_count:.6f} "
                    f"scale={running_scale / step_count:.6f} "
                    f"band={running_band / step_count:.6f} "
                    f"tv={running_tv / step_count:.6f} "
                    f"bg={running_background / step_count:.6f} "
                    f"perc={running_perceptual / step_count:.6f}"
                )

            if max_steps_per_epoch is not None and step_idx >= int(max_steps_per_epoch):
                break

        if step_count == 0:
            raise RuntimeError("Training dataloader produced zero steps")

        latest_metrics = {
            "epoch": epoch_idx + 1,
            "total_loss": running_total / step_count,
            "final_loss": running_final / step_count,
            "scale_loss": running_scale / step_count,
            "band_loss": running_band / step_count,
            "tv_loss": running_tv / step_count,
            "background_loss": running_background / step_count,
            "perceptual_loss": running_perceptual / step_count,
            "level_weights": list(criterion.level_weights),
        }
        print(
            f"[epoch {epoch_idx + 1}/{max_epochs}] "
            f"total={latest_metrics['total_loss']:.6f} "
            f"final={latest_metrics['final_loss']:.6f} "
            f"scale={latest_metrics['scale_loss']:.6f} "
            f"band={latest_metrics['band_loss']:.6f} "
            f"tv={latest_metrics['tv_loss']:.6f} "
            f"bg={latest_metrics['background_loss']:.6f} "
            f"perc={latest_metrics['perceptual_loss']:.6f} "
            f"level_weights={latest_metrics['level_weights']}"
        )
        _append_jsonl(history_path, latest_metrics)

        checkpoint = {
            "epoch": epoch_idx + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": latest_metrics,
            "config": config,
            "resumed_from": resumed_from,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if bool(train_cfg.get("save_every_epoch", False)):
            torch.save(checkpoint, output_dir / f"epoch_{epoch_idx + 1:03d}.pt")

    return {
        "output_dir": str(output_dir),
        "latest_checkpoint": str(output_dir / "latest.pt"),
        "history_path": str(history_path),
        "metrics": latest_metrics,
        "resumed_from": resumed_from,
        "start_epoch": start_epoch,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multiscale optical model from a config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config file.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    result = train(
        config,
        config_path=config_path,
        resume_path_override=None if args.resume is None else args.resume.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
