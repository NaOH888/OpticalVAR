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

from conditioning import ConditionEmbeddingLayer
from optical.core import DetectorConfig, PropagationConfig, PropagationErrorConfig, SourceConfig
from optical.layers import DetectorLayer, DiffractivePhaseLayer, SLMDeviceLayer
from optical.models import (
    OpticalMultiscaleModel,
    OpticalPrefixReadoutDecoder,
    SpatialPhaseMapEncoder,
)


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
    def test_spatial_phase_map_encoder_supports_attribute_conditions(self) -> None:
        encoder = SpatialPhaseMapEncoder(
            input_channels=4,
            input_height=2,
            input_width=2,
            output_height=8,
            output_width=8,
            hidden_dim=48,
            hidden_channels=(32, 24, 16),
            phase_alpha_pi=2.0,
            condition_layer=ConditionEmbeddingLayer(
                mode="attribute_vector",
                input_dim=5,
                output_dim=12,
                hidden_dim=16,
            ),
            condition_dim=12,
            upsample_mode="bilinear",
        )
        sample = torch.rand((2, 4, 2, 2), dtype=torch.float32)
        cond_a = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32)
        cond_b = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)

        phase_a = encoder(sample, condition=cond_a)
        phase_b = encoder(sample, condition=cond_b)

        self.assertEqual(tuple(phase_a.shape), (2, 1, 8, 8))
        self.assertTrue(torch.all(phase_a >= 0.0).item())
        self.assertTrue(torch.all(phase_a < encoder.phase_period_rad).item())
        self.assertFalse(torch.allclose(phase_a, phase_b))

    def test_spatial_phase_map_encoder_supports_class_labels(self) -> None:
        encoder = SpatialPhaseMapEncoder(
            input_channels=1,
            input_height=4,
            input_width=4,
            output_height=4,
            output_width=4,
            hidden_dim=32,
            condition_layer=ConditionEmbeddingLayer(
                mode="class_index",
                num_classes=10,
                output_dim=8,
                embed_dim=6,
            ),
            condition_dim=8,
        )
        sample = torch.rand((2, 1, 4, 4), dtype=torch.float32)
        labels = torch.tensor([2, 7], dtype=torch.long)

        output = encoder(sample, class_labels=labels)

        self.assertEqual(tuple(output.shape), (2, 1, 4, 4))
        self.assertTrue(torch.isfinite(output).all().item())

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

    def test_optical_multiscale_model_runs_end_to_end_with_spatial_encoder(self) -> None:
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
        encoder = SpatialPhaseMapEncoder(
            input_channels=4,
            input_height=2,
            input_width=2,
            output_height=2,
            output_width=2,
            hidden_dim=24,
        )
        model = OpticalMultiscaleModel(
            encoder=encoder,
            optical_decoder=decoder,
            upsample_mode="nearest",
        )
        sample = torch.rand((1, 4, 2, 2), dtype=torch.float32)

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
