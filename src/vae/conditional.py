from __future__ import annotations

import torch
import torch.nn as nn

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
        hidden_channels: tuple[int, int, int],
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
        c1, c2, c3 = (int(v) for v in hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(int(input_channels) + int(condition_channels), c1, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        feature_size = self.image_size // 8
        flattened_dim = c3 * feature_size * feature_size
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
        hidden_channels: tuple[int, int, int],
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
        c1, c2, c3 = (int(v) for v in hidden_channels)
        feature_size = self.image_size // 8
        self.fc = nn.Linear(int(latent_dim) + int(condition_embed_dim), c1 * feature_size * feature_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(c3, int(output_channels), kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.feature_channels = c1
        self.feature_size = feature_size

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> ModelOutput:
        condition = self.condition_embedding(labels.to(device=z.device)).to(dtype=z.dtype)
        hidden = self.fc(torch.cat((z, condition), dim=1))
        hidden = hidden.view(z.shape[0], self.feature_channels, self.feature_size, self.feature_size)
        return ModelOutput(reconstruction=self.decoder(hidden))


class ConditionalVAE(VAE):
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
    model_config = VAEConfig(
        input_dim=(
            int(model_cfg.get("input_channels", 1)),
            int(model_cfg.get("image_size", 32)),
            int(model_cfg.get("image_size", 32)),
        ),
        latent_dim=latent_dim,
        reconstruction_loss=str(model_cfg.get("reconstruction_loss", "bce")),
    )
    return ConditionalVAE(model_config=model_config, encoder=encoder, decoder=decoder)
