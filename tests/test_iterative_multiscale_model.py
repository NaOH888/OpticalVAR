from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conditioning import ConditionEmbeddingLayer
from optical.core import DetectorConfig, PropagationConfig, PropagationErrorConfig, SourceConfig
from optical.layers import DetectorLayer, DiffractivePhaseLayer, SLMDeviceLayer
from optical.models import IterativeMultiscaleEncoder, IterativeMultiscaleOpticalModel, IterativeOpticalDecoder
from optical.models.multiscale import OpticalPrefixReadoutDecoder


def _default_propagation_config() -> PropagationConfig:
    return PropagationConfig(
        canvas_h=None,
        canvas_w=None,
        canvas_factor=1.0,
        refractive_index=1.0,
        use_bandlimit_window=False,
        evanescent_mode="keep",
        fft_norm="ortho",
    )


class IterativeMultiscaleModelTests(unittest.TestCase):
    def _build_decoder(self) -> IterativeOpticalDecoder:
        source_config = SourceConfig(
            wavelengths_m=(532e-9,),
            light_mode="phase",
            amplitude=1.0,
        )
        slm = SLMDeviceLayer(
            pixel_pitch_x_m=1e-6,
            pixel_pitch_y_m=1e-6,
            pixel_count_x=8,
            pixel_count_y=8,
            dx=1e-6,
            fill_factor=1.0,
            phase_alpha=2.0,
            phase_bit_depth=None,
            source_config=source_config,
        )
        phase_1 = DiffractivePhaseLayer(
            width_m=8e-6,
            height_m=8e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=source_config.wavelengths_m,
            initial_phase_map_rad=torch.zeros((8, 8), dtype=torch.float32),
        )
        phase_2 = DiffractivePhaseLayer(
            width_m=8e-6,
            height_m=8e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=source_config.wavelengths_m,
            initial_phase_map_rad=torch.full((8, 8), 0.5, dtype=torch.float32),
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=8,
                height_num=8,
                detector_unit_len_m=1e-6,
            ),
            dx_m=1e-6,
        )
        optical_decoder = OpticalPrefixReadoutDecoder(
            slm_layer=slm,
            optical_layers=(phase_1, phase_2),
            detector_layer=detector,
            distance_slm_to_first_layer_m=2e-6,
            distance_between_layers_m=(2e-6,),
            distance_last_layer_to_detector_m=2e-6,
            propagation_config=_default_propagation_config(),
            error_config=PropagationErrorConfig(
                delta_z_m=0.0,
                shift_x_m=0.0,
                shift_y_m=0.0,
            ),
        )
        return IterativeOpticalDecoder(optical_decoder=optical_decoder)

    def test_iterative_encoder_supports_attribute_conditions(self) -> None:
        encoder = IterativeMultiscaleEncoder(
            latent_channels=4,
            latent_height=2,
            latent_width=2,
            output_height=8,
            output_width=8,
            num_steps=5,
            step_embedding_dim=8,
            condition_layer=ConditionEmbeddingLayer(
                mode="attribute_vector",
                input_dim=6,
                output_dim=12,
                hidden_dim=16,
            ),
            condition_embed_dim=12,
            latent_stage_channels=(24, 16, 12),
            prev_image_channels=(12, 8),
            fusion_hidden_dim=16,
        )
        prev_image = torch.rand((2, 1, 8, 8), dtype=torch.float32)
        latent = torch.rand((2, 4, 2, 2), dtype=torch.float32)
        cond_a = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=torch.float32)
        cond_b = torch.zeros((2, 6), dtype=torch.float32)

        output_a = encoder(prev_image=prev_image, latent=latent, timesteps=0, condition=cond_a)
        output_b = encoder(prev_image=prev_image, latent=latent, timesteps=0, condition=cond_b)

        self.assertEqual(tuple(output_a.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(output_a).all().item())
        self.assertFalse(torch.allclose(output_a, output_b))

    def test_iterative_model_unrolls_multiple_steps(self) -> None:
        encoder = IterativeMultiscaleEncoder(
            latent_channels=4,
            latent_height=2,
            latent_width=2,
            output_height=8,
            output_width=8,
            num_steps=4,
            step_embedding_dim=8,
            latent_stage_channels=(24, 16, 12),
            prev_image_channels=(12, 8),
            fusion_hidden_dim=16,
        )
        model = IterativeMultiscaleOpticalModel(
            encoder=encoder,
            decoder=self._build_decoder(),
            state_normalization="mean_power",
        )
        latent = torch.rand((2, 4, 2, 2), dtype=torch.float32)

        trajectory = model(latent=latent, num_steps=4, detach_prev_state=False)

        self.assertEqual(len(trajectory["predictions"]), 4)
        self.assertEqual(len(trajectory["states"]), 4)
        self.assertEqual(len(trajectory["control_maps"]), 4)
        self.assertEqual(tuple(trajectory["predictions"][0].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(trajectory["states"][0].shape), (2, 1, 8, 8))
        self.assertGreater(float(trajectory["states"][0].mean().item()), 0.0)

    def test_iterative_encoder_can_disable_prev_image_branch(self) -> None:
        encoder = IterativeMultiscaleEncoder(
            latent_channels=4,
            latent_height=2,
            latent_width=2,
            output_height=8,
            output_width=8,
            num_steps=4,
            step_embedding_dim=8,
            latent_stage_channels=(24, 16, 12),
            prev_image_channels=(12, 8),
            use_prev_image=False,
            fusion_hidden_dim=16,
        )
        latent = torch.rand((2, 4, 2, 2), dtype=torch.float32)
        prev_a = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        prev_b = torch.ones((2, 1, 8, 8), dtype=torch.float32)

        output_a = encoder(prev_image=prev_a, latent=latent, timesteps=0)
        output_b = encoder(prev_image=prev_b, latent=latent, timesteps=0)

        self.assertTrue(torch.allclose(output_a, output_b))


if __name__ == "__main__":
    unittest.main()
