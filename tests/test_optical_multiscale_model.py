from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optical.core import DetectorConfig, PropagationConfig, PropagationErrorConfig, SourceConfig
from optical.layers import DetectorLayer, DiffractivePhaseLayer, SLMDeviceLayer
from optical.models import (
    ConditionEmbeddingLayer,
    ConditionalPhaseSLMEncoder,
    DigitalPrefixReadoutDecoder,
    LatentPhaseMapEncoder,
    OpticalMultiscaleModel,
    OpticalPrefixReadoutDecoder,
    PhaseMapEncoder,
)
from conditioning import ConditionalLatentFusion, ContinuousMapLatentProjector, DiscreteCodeLatentProjector, LatentEmbeddingLayer


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


class OpticalMultiscaleModelTests(unittest.TestCase):
    def test_condition_embedding_layer_supports_attribute_vectors(self) -> None:
        layer = ConditionEmbeddingLayer(
            mode="attribute_vector",
            input_dim=5,
            output_dim=3,
            hidden_dim=7,
        )
        output = layer(torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_encoder_supports_explicit_weight_initialization(self) -> None:
        encoder = ConditionalPhaseSLMEncoder(
            input_channels=1,
            input_height=4,
            input_width=4,
            output_height=4,
            output_width=4,
            hidden_dim=16,
            class_conditional=True,
            num_classes=10,
            class_embed_dim=8,
            class_condition_channels=2,
            weight_init="xavier_uniform",
            output_weight_init="xavier_uniform",
            embedding_init_std=0.01,
        )

        self.assertEqual(tuple(encoder.fc1.weight.shape), (16, 48))
        self.assertTrue(torch.isfinite(encoder.fc1.weight).all().item())
        self.assertTrue(torch.isfinite(encoder.fc3.weight).all().item())
        self.assertIsNotNone(encoder.condition_embedding)
        embedding = getattr(encoder.condition_embedding, "embedding", None)
        self.assertIsNotNone(embedding)
        self.assertLess(float(embedding.weight.std()), 0.05)

    def test_conditional_phase_slm_encoder_supports_time_and_class_embeddings(self) -> None:
        encoder = ConditionalPhaseSLMEncoder(
            input_channels=1,
            input_height=2,
            input_width=2,
            output_height=3,
            output_width=3,
            hidden_dim=32,
            phase_alpha_pi=2.0,
            time_conditional=True,
            time_embedding_type="positional",
            time_embedding_dim=16,
            class_conditional=True,
            num_classes=10,
            class_embed_dim=8,
        )
        sample = torch.tensor(
            [
                [[[0.0, 0.5], [1.0, 0.25]]],
                [[[0.1, 0.2], [0.3, 0.4]]],
            ],
            dtype=torch.float32,
        )
        timesteps = torch.tensor([1, 5], dtype=torch.long)
        class_labels = torch.tensor([2, 7], dtype=torch.long)

        phase_map = encoder(sample, timesteps=timesteps, class_labels=class_labels)
        phase_map_other = encoder(sample, timesteps=timesteps, class_labels=torch.tensor([1, 1], dtype=torch.long))

        self.assertEqual(tuple(phase_map.shape), (2, 1, 3, 3))
        self.assertTrue(torch.all(phase_map >= 0.0).item())
        self.assertTrue(torch.all(phase_map < encoder.phase_period_rad).item())
        self.assertFalse(torch.allclose(phase_map, phase_map_other))

    def test_conditional_phase_slm_encoder_supports_attribute_conditions(self) -> None:
        encoder = ConditionalPhaseSLMEncoder(
            input_channels=1,
            input_height=4,
            input_width=4,
            output_height=4,
            output_width=4,
            hidden_dim=32,
            condition_mode="attribute_vector",
            condition_input_dim=5,
            class_condition_channels=3,
        )
        sample = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        cond_a = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32)
        cond_b = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
        phase_a = encoder(sample, condition=cond_a)
        phase_b = encoder(sample, condition=cond_b)
        self.assertEqual(tuple(phase_a.shape), (2, 1, 4, 4))
        self.assertFalse(torch.allclose(phase_a, phase_b))

    def test_phase_map_encoder_projects_vector_to_phase_map(self) -> None:
        encoder = PhaseMapEncoder(
            input_dim=12,
            output_height=4,
            output_width=5,
            hidden_dim=16,
            phase_alpha_pi=2.0,
        )
        output = encoder(torch.rand((3, 12), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 1, 4, 5))
        self.assertTrue(torch.all(output >= 0.0).item())
        self.assertTrue(torch.all(output < encoder.phase_period_rad).item())

    def test_latent_phase_map_encoder_supports_discrete_codes_and_condition(self) -> None:
        encoder = LatentPhaseMapEncoder(
            latent_layer=LatentEmbeddingLayer(
                projector=DiscreteCodeLatentProjector(
                    num_codebooks=4,
                    codebook_size=16,
                    code_embed_dim=8,
                    output_dim=10,
                    hidden_dim=12,
                )
            ),
            condition_layer=ConditionEmbeddingLayer(
                mode="attribute_vector",
                input_dim=5,
                output_dim=10,
                hidden_dim=12,
            ),
            fusion_layer=ConditionalLatentFusion(
                latent_dim=10,
                condition_dim=10,
                output_dim=14,
                mode="concat",
                hidden_dim=16,
            ),
            phase_map_encoder=PhaseMapEncoder(
                input_dim=14,
                output_height=4,
                output_width=4,
                hidden_dim=16,
                phase_alpha_pi=2.0,
            ),
        )
        output = encoder(
            torch.tensor([[1, 2, 3, 4], [0, 5, 7, 9]], dtype=torch.long),
            condition=torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32),
        )
        self.assertEqual(tuple(output.shape), (2, 1, 4, 4))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_digital_prefix_readout_decoder_returns_multiscale_images(self) -> None:
        decoder = DigitalPrefixReadoutDecoder(
            input_height=4,
            input_width=4,
            output_height=6,
            output_width=6,
            num_levels=3,
            hidden_channels=(8, 16),
            output_activation="sigmoid",
        )

        outputs = decoder(torch.zeros((2, 1, 4, 4), dtype=torch.float32))

        self.assertEqual(tuple(outputs["prefix_readout_1"].shape), (2, 1, 6, 6))
        self.assertEqual(tuple(outputs["prefix_readout_2"].shape), (2, 1, 6, 6))
        self.assertEqual(tuple(outputs["prefix_readout_3"].shape), (2, 1, 6, 6))
        self.assertEqual(tuple(outputs["final_detector"].shape), (2, 1, 6, 6))
        self.assertEqual(len(outputs["prefix_readouts"]), 3)
        self.assertTrue(torch.allclose(outputs["prefix_readout_3"], outputs["final_detector"]))

    def test_prefix_readout_decoder_returns_detector_plane_prefixes(self) -> None:
        source_config = SourceConfig(
            wavelengths_m=(532e-9,),
            light_mode="phase",
            amplitude=1.0,
        )
        slm = SLMDeviceLayer(
            pixel_pitch_x_m=1e-6,
            pixel_pitch_y_m=1e-6,
            pixel_count_x=4,
            pixel_count_y=4,
            dx=1e-6,
            fill_factor=1.0,
            phase_alpha=2.0,
            phase_bit_depth=None,
            source_config=source_config,
        )
        phase_1 = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=source_config.wavelengths_m,
            initial_phase_map_rad=torch.zeros((4, 4), dtype=torch.float32),
        )
        phase_2 = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=source_config.wavelengths_m,
            initial_phase_map_rad=torch.full((4, 4), math.pi / 3, dtype=torch.float32),
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=2,
                height_num=2,
                detector_unit_len_m=2e-6,
            ),
            dx_m=1e-6,
        )
        decoder = OpticalPrefixReadoutDecoder(
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
            default_error_factor=1.0,
        )

        outputs = decoder(torch.zeros((1, 1, 4, 4), dtype=torch.float32), error_factor=2.5)

        self.assertEqual(decoder._propagation_context.error_factor, 1.0)
        self.assertEqual(tuple(outputs["prefix_readout_1"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(outputs["prefix_readout_2"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(outputs["final_detector"].shape), (1, 1, 2, 2))
        self.assertEqual(len(outputs["prefix_readouts"]), 2)
        self.assertTrue(torch.allclose(outputs["prefix_readout_2"], outputs["final_detector"], atol=1e-5))

    def test_optical_multiscale_model_runs_end_to_end(self) -> None:
        source_config = SourceConfig(
            wavelengths_m=(532e-9,),
            light_mode="phase",
            amplitude=1.0,
        )
        slm = SLMDeviceLayer(
            pixel_pitch_x_m=1e-6,
            pixel_pitch_y_m=1e-6,
            pixel_count_x=4,
            pixel_count_y=4,
            dx=1e-6,
            fill_factor=1.0,
            phase_alpha=2.0,
            phase_bit_depth=None,
            source_config=source_config,
        )
        phase_1 = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=source_config.wavelengths_m,
            initial_phase_map_rad=torch.zeros((4, 4), dtype=torch.float32),
        )
        phase_2 = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=source_config.wavelengths_m,
            initial_phase_map_rad=torch.full((4, 4), math.pi / 6, dtype=torch.float32),
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=2,
                height_num=2,
                detector_unit_len_m=2e-6,
            ),
            dx_m=1e-6,
        )
        decoder = OpticalPrefixReadoutDecoder(
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
        encoder = ConditionalPhaseSLMEncoder(
            input_channels=1,
            input_height=2,
            input_width=2,
            output_height=2,
            output_width=2,
            hidden_dim=16,
            phase_alpha_pi=2.0,
        )
        model = OpticalMultiscaleModel(
            encoder=encoder,
            optical_decoder=decoder,
            upsample_mode="nearest",
        )
        sample = torch.tensor([[[[0.0, 0.5], [0.25, 1.0]]]], dtype=torch.float32)

        outputs = model(sample)

        self.assertEqual(tuple(outputs["encoder_output"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(outputs["slm_input"].shape), (1, 1, 4, 4))
        self.assertEqual(len(outputs["prefix_readouts"]), 2)
        self.assertIn("prefix_readout_1", outputs)
        self.assertIn("prefix_readout_2", outputs)
        self.assertIn("final_detector", outputs)
        self.assertTrue(torch.allclose(outputs["prefix_readout_2"], outputs["final_detector"], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
