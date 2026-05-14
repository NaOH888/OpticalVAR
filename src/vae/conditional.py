from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from conditioning import ConditionEmbeddingLayer
from pythae.models import VAE, VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder


class ConditionalEncoder(BaseEncoder):
    def __init__(
        self,
        *,
        input_channels: int,
        image_size: int,
        latent_dim: int,
        condition_mode: str,
        num_classes: int | None,
        condition_input_dim: int | None,
        condition_embed_dim: int,
        condition_hidden_dim: int | None,
        condition_channels: int,
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.condition_embedding = ConditionEmbeddingLayer(
            mode=str(condition_mode),
            output_dim=int(condition_channels),
            num_classes=None if num_classes is None else int(num_classes),
            input_dim=None if condition_input_dim is None else int(condition_input_dim),
            embed_dim=int(condition_embed_dim),
            hidden_dim=condition_hidden_dim,
        )
        channels = tuple(int(v) for v in hidden_channels)
        if len(channels) == 0:
            raise ValueError("hidden_channels must contain at least one value")
        downsample_factor = 2 ** len(channels)
        if self.image_size % downsample_factor != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by 2**len(hidden_channels)={downsample_factor}"
            )
        layers: list[nn.Module] = []
        in_channels = int(input_channels) + int(condition_channels)
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.SiLU())
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        feature_size = self.image_size // downsample_factor
        flattened_dim = channels[-1] * feature_size * feature_size
        self.to_mu = nn.Linear(flattened_dim, int(latent_dim))
        self.to_log_var = nn.Linear(flattened_dim, int(latent_dim))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> ModelOutput:
        condition = self.condition_embedding(labels.to(device=x.device)).to(dtype=x.dtype)
        condition = condition.view(x.shape[0], -1, 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        hidden = self.features(torch.cat((x, condition), dim=1)).reshape(x.shape[0], -1)
        return ModelOutput(
            embedding=self.to_mu(hidden),
            log_covariance=self.to_log_var(hidden),
        )


class ConditionalDecoder(BaseDecoder):
    def __init__(
        self,
        *,
        output_channels: int,
        image_size: int,
        latent_dim: int,
        condition_mode: str,
        num_classes: int | None,
        condition_input_dim: int | None,
        condition_embed_dim: int,
        condition_hidden_dim: int | None,
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.condition_embedding = ConditionEmbeddingLayer(
            mode=str(condition_mode),
            output_dim=int(condition_embed_dim),
            num_classes=None if num_classes is None else int(num_classes),
            input_dim=None if condition_input_dim is None else int(condition_input_dim),
            embed_dim=int(condition_embed_dim),
            hidden_dim=condition_hidden_dim,
        )
        channels = tuple(int(v) for v in hidden_channels)
        if len(channels) == 0:
            raise ValueError("hidden_channels must contain at least one value")
        upsample_factor = 2 ** len(channels)
        if self.image_size % upsample_factor != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by 2**len(hidden_channels)={upsample_factor}"
            )
        feature_size = self.image_size // upsample_factor
        self.fc = nn.Linear(int(latent_dim) + int(condition_embed_dim), channels[0] * feature_size * feature_size)
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.SiLU())
        layers.append(nn.ConvTranspose2d(channels[-1], int(output_channels), kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)
        self.feature_channels = channels[0]
        self.feature_size = feature_size

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> ModelOutput:
        condition = self.condition_embedding(labels.to(device=z.device)).to(dtype=z.dtype)
        hidden = self.fc(torch.cat((z, condition), dim=1))
        hidden = hidden.view(z.shape[0], self.feature_channels, self.feature_size, self.feature_size)
        return ModelOutput(reconstruction=self.decoder(hidden))


class PerceptualLoss(nn.Module):
    def __init__(self, *, feature_layers: tuple[int, ...], weights: str | None = "imagenet") -> None:
        super().__init__()
        resolved_weights: VGG16_Weights | None
        if weights is None or str(weights).lower() == "none":
            resolved_weights = None
        elif str(weights).lower() == "imagenet":
            resolved_weights = VGG16_Weights.IMAGENET1K_V1
        else:
            resolved_weights = None

        try:
            features = vgg16(weights=resolved_weights).features
        except Exception as exc:
            raise RuntimeError(
                "failed to initialize perceptual backbone; use training.perceptual_weights='none' "
                "or provide cached torchvision weights"
            ) from exc

        if weights not in {None, "none", "imagenet"} and str(weights).lower() != "imagenet":
            state_dict = torch.load(str(weights), map_location="cpu")
            features.load_state_dict(state_dict, strict=False)

        self.features = features.eval()
        self.feature_layers = tuple(sorted({int(v) for v in feature_layers}))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[1] == 1:
            prediction = prediction.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        prediction = (prediction - self.mean.to(device=prediction.device, dtype=prediction.dtype)) / self.std.to(
            device=prediction.device, dtype=prediction.dtype
        )
        target = (target - self.mean.to(device=target.device, dtype=target.dtype)) / self.std.to(
            device=target.device, dtype=target.dtype
        )

        loss = prediction.new_zeros(())
        prediction_features = prediction
        target_features = target
        for layer_idx, layer in enumerate(self.features):
            prediction_features = layer(prediction_features)
            target_features = layer(target_features)
            if layer_idx in self.feature_layers:
                loss = loss + F.l1_loss(prediction_features, target_features, reduction="mean")
        return loss


class ConditionalVAE(VAE):
    def __init__(
        self,
        model_config: VAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        *,
        reconstruction_mode: str = "bce",
    ) -> None:
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)
        self.reconstruction_mode = str(reconstruction_mode)

    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.reconstruction_mode == "mse":
            recon_loss = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        elif self.reconstruction_mode == "l1":
            recon_loss = F.l1_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        elif self.reconstruction_mode == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        else:
            raise ValueError(f"unsupported reconstruction_loss: {self.reconstruction_mode}")

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return (recon_loss + kld).mean(dim=0), recon_loss.mean(dim=0), kld.mean(dim=0)

    def forward(self, inputs, **kwargs) -> ModelOutput:
        x = inputs["data"]
        labels = inputs["labels"].to(device=x.device)
        encoder_output = self.encoder(x, labels)
        mu = encoder_output.embedding
        log_var = encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z, labels)["reconstruction"]
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)
        return ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
            mu=mu,
            log_var=log_var,
        )

    def encode(self, x: torch.Tensor, labels: torch.Tensor, *, sample_posterior: bool) -> ModelOutput:
        encoder_output = self.encoder(x, labels)
        mu = encoder_output.embedding
        log_var = encoder_output.log_covariance
        if sample_posterior:
            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)
        else:
            z = mu
        return ModelOutput(z=z, mu=mu, log_var=log_var)

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, labels)["reconstruction"]


def build_cvae(model_cfg: dict) -> ConditionalVAE:
    latent_height = int(model_cfg["latent_height"])
    latent_width = int(model_cfg["latent_width"])
    latent_dim = latent_height * latent_width
    condition_mode = str(model_cfg.get("condition_mode", "class_index"))
    num_classes = model_cfg.get("num_classes")
    condition_input_dim = model_cfg.get("condition_input_dim")
    condition_embed_dim = int(model_cfg.get("condition_embed_dim", model_cfg.get("class_embed_dim", 32)))
    condition_hidden_dim = model_cfg.get("condition_hidden_dim")
    encoder_hidden = tuple(int(v) for v in model_cfg.get("encoder_hidden_channels", [32, 64, 128]))
    decoder_hidden = tuple(int(v) for v in model_cfg.get("decoder_hidden_channels", [128, 64, 32]))
    encoder = ConditionalEncoder(
        input_channels=int(model_cfg.get("input_channels", 1)),
        image_size=int(model_cfg.get("image_size", 32)),
        latent_dim=latent_dim,
        condition_mode=condition_mode,
        num_classes=None if num_classes is None else int(num_classes),
        condition_input_dim=None if condition_input_dim is None else int(condition_input_dim),
        condition_embed_dim=condition_embed_dim,
        condition_hidden_dim=None if condition_hidden_dim is None else int(condition_hidden_dim),
        condition_channels=int(model_cfg.get("condition_channels", 4)),
        hidden_channels=encoder_hidden,
    )
    decoder = ConditionalDecoder(
        output_channels=int(model_cfg.get("input_channels", 1)),
        image_size=int(model_cfg.get("image_size", 32)),
        latent_dim=latent_dim,
        condition_mode=condition_mode,
        num_classes=None if num_classes is None else int(num_classes),
        condition_input_dim=None if condition_input_dim is None else int(condition_input_dim),
        condition_embed_dim=condition_embed_dim,
        condition_hidden_dim=None if condition_hidden_dim is None else int(condition_hidden_dim),
        hidden_channels=decoder_hidden,
    )
    reconstruction_mode = str(model_cfg.get("reconstruction_loss", "bce"))
    model_config = VAEConfig(
        input_dim=(
            int(model_cfg.get("input_channels", 1)),
            int(model_cfg.get("image_size", 32)),
            int(model_cfg.get("image_size", 32)),
        ),
        latent_dim=latent_dim,
        reconstruction_loss="mse" if reconstruction_mode == "l1" else reconstruction_mode,
    )
    return ConditionalVAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder,
        reconstruction_mode=reconstruction_mode,
    )


def build_perceptual_loss(train_cfg: dict) -> PerceptualLoss | None:
    perceptual_weight = float(train_cfg.get("perceptual_weight", 0.0))
    if perceptual_weight <= 0.0:
        return None
    feature_layers = tuple(int(v) for v in train_cfg.get("perceptual_feature_layers", [3, 8, 15, 22]))
    return PerceptualLoss(
        feature_layers=feature_layers,
        weights=train_cfg.get("perceptual_weights", "imagenet"),
    )
