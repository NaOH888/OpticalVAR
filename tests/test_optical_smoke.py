from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optical.backends.fdtd.api import FDTDLayerContext
from optical.core import (
    DetectorConfig,
    FDTDMetaConfig,
    FDTDMonitorConfig,
    FDTDProbeConfig,
    FDTDSourceConfig,
    PropagateContext,
    PropagationConfig,
    PropagationErrorConfig,
    SourceConfig,
)
from optical.layers import DetectorLayer, DiffractivePhaseLayer, FieldProbeLayer, LightSourceLayer, MetaEncoder


class _DummyLUTProvider:
    def bounds(self) -> dict[str, float]:
        return {
            "wx_min": 0.2e-6,
            "wx_max": 1.2e-6,
            "wy_min": 0.2e-6,
            "wy_max": 1.2e-6,
        }

    def query(
        self,
        height_m: torch.Tensor,
        wx_m: torch.Tensor,
        wy_m: torch.Tensor,
        wavelength_m: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del height_m, wavelength_m
        phase = (wx_m - wy_m) * 1.0e6
        amp = torch.ones_like(phase)
        return phase, amp


class _DummyBuilder:
    def __init__(self) -> None:
        self.plane_sources: list[dict] = []
        self.profile_monitors: list[dict] = []
        self.rects: list[dict] = []
        self.groups: list[dict] = []
        self.scripts: list[str] = []

    def add_plane_source(self, **kwargs) -> None:
        self.plane_sources.append(kwargs)

    def add_profile_monitor(self, **kwargs) -> None:
        self.profile_monitors.append(kwargs)

    def add_rect(self, **kwargs) -> None:
        self.rects.append(kwargs)

    def add_structure_group(self, **kwargs) -> None:
        self.groups.append(kwargs)

    def eval_script(self, script: str) -> None:
        self.scripts.append(script)


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


class OpticalSmokeTests(unittest.TestCase):
    def test_canvas_shape_uses_largest_physical_layer_size(self) -> None:
        source = LightSourceLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            config=SourceConfig(
                wavelengths_m=(532e-9,),
                light_mode="phase",
                amplitude=1.0,
            ),
        )
        phase = DiffractivePhaseLayer(
            width_m=6e-6,
            height_m=8e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=(532e-9,),
            initial_phase_map_rad=torch.zeros((2, 3), dtype=torch.float32),
            phase_grid_height=2,
            phase_grid_width=3,
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=2,
                height_num=2,
                detector_unit_len_m=1e-6,
            ),
            dx_m=1e-6,
        )

        propagation = PropagateContext(
            propagation_config=_default_propagation_config(),
            error_config=PropagationErrorConfig(
                delta_z_m=0.0,
                shift_x_m=0.0,
                shift_y_m=0.0,
            ),
            error_factor=1.0,
        )
        propagation.add_layer(source, z=0.0)
        propagation.add_layer(phase, z=2e-6)
        propagation.add_layer(detector, z=4e-6)

        self.assertEqual(propagation._resolve_canvas_shape(), (8, 6))

    def test_prefix_readouts_are_measured_on_detector_plane(self) -> None:
        source = LightSourceLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            config=SourceConfig(
                wavelengths_m=(532e-9,),
                light_mode="phase",
                amplitude=1.0,
            ),
        )
        phase_1 = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=(532e-9,),
            initial_phase_map_rad=torch.zeros((4, 4), dtype=torch.float32),
        )
        phase_2 = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=(532e-9,),
            initial_phase_map_rad=torch.full((4, 4), torch.pi / 4, dtype=torch.float32),
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=2,
                height_num=2,
                detector_unit_len_m=2e-6,
            ),
            dx_m=1e-6,
        )

        propagation = PropagateContext(
            propagation_config=_default_propagation_config(),
            error_config=PropagationErrorConfig(
                delta_z_m=0.0,
                shift_x_m=0.0,
                shift_y_m=0.0,
            ),
            error_factor=1.0,
        )
        propagation.add_layer(source, z=0.0)
        propagation.add_layer(phase_1, z=2e-6)
        propagation.add_layer(phase_2, z=4e-6)
        propagation.add_layer(detector, z=6e-6)

        outputs = propagation.propagate_with_prefix_readouts(
            torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        )

        self.assertIn("prefix_readout_1", outputs)
        self.assertIn("prefix_readout_2", outputs)
        self.assertIn("final_detector", outputs)
        self.assertEqual(tuple(outputs["prefix_readout_1"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(outputs["prefix_readout_2"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(outputs["final_detector"].shape), (1, 1, 2, 2))
        self.assertTrue(torch.allclose(outputs["prefix_readout_2"], outputs["final_detector"], atol=1e-5))

    def test_source_phase_detector_pipeline(self) -> None:
        source = LightSourceLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            config=SourceConfig(
                wavelengths_m=(532e-9,),
                light_mode="phase",
                amplitude=1.0,
            ),
        )
        phase = DiffractivePhaseLayer(
            width_m=4e-6,
            height_m=4e-6,
            dx_m=1e-6,
            channels=1,
            wavelengths_m=(532e-9,),
            initial_phase_map_rad=torch.full((4, 4), torch.pi / 2, dtype=torch.float32),
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=2,
                height_num=2,
                detector_unit_len_m=2e-6,
            ),
            dx_m=1e-6,
        )

        propagation = PropagateContext(
            propagation_config=_default_propagation_config(),
            error_config=PropagationErrorConfig(
                delta_z_m=0.0,
                shift_x_m=0.0,
                shift_y_m=0.0,
            ),
            error_factor=1.0,
        )
        propagation.add_layer(source, z=0.0)
        propagation.add_layer(phase, z=3e-6)
        propagation.add_layer(detector, z=6e-6)

        output = propagation.propagate(torch.zeros((1, 1, 4, 4), dtype=torch.float32))
        self.assertTrue(torch.is_complex(output))
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4))
        self.assertIsNotNone(detector.E)
        self.assertIsNotNone(detector.I)
        self.assertEqual(tuple(detector.E.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(detector.I.shape), (1, 1, 2, 2))

    def test_source_metasurface_detector_pipeline(self) -> None:
        source = LightSourceLayer(
            width_m=2e-6,
            height_m=2e-6,
            dx_m=1e-6,
            config=SourceConfig(
                wavelengths_m=(532e-9,),
                light_mode="amplitude",
                amplitude=1.0,
            ),
        )
        metasurface = MetaEncoder(
            provider=_DummyLUTProvider(),
            phys_shape=torch.tensor(
                [
                    [[0.6e-6, 0.8e-6], [0.7e-6, 0.9e-6]],
                    [[0.6e-6, 0.7e-6], [0.8e-6, 0.9e-6]],
                ],
                dtype=torch.float32,
            ),
            wavelengths_m=(532e-9,),
            dx=1e-6,
            fixed_height=600e-9,
            period_x=1e-6,
            period_y=1e-6,
        )
        detector = DetectorLayer(
            config=DetectorConfig(
                width_num=2,
                height_num=2,
                detector_unit_len_m=1e-6,
            ),
            dx_m=1e-6,
        )

        propagation = PropagateContext(
            propagation_config=_default_propagation_config(),
            error_config=PropagationErrorConfig(
                delta_z_m=0.0,
                shift_x_m=0.0,
                shift_y_m=0.0,
            ),
            error_factor=1.0,
        )
        propagation.add_layer(source, z=0.0)
        propagation.add_layer(metasurface, z=2e-6)
        propagation.add_layer(detector, z=4e-6)

        output = propagation.propagate(torch.ones((1, 1, 2, 2), dtype=torch.float32))
        self.assertTrue(torch.is_complex(output))
        self.assertIsNotNone(detector.I)
        self.assertEqual(tuple(detector.I.shape), (1, 1, 2, 2))
        self.assertTrue(torch.isfinite(detector.I).all().item())

    def test_fixed_xyz_error_changes_output(self) -> None:
        image = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        image[0, 0, 1, 2] = 1.0

        def build_context(error_config: PropagationErrorConfig) -> tuple[PropagateContext, FieldProbeLayer]:
            source = LightSourceLayer(
                width_m=4e-6,
                height_m=4e-6,
                dx_m=1e-6,
                config=SourceConfig(
                    wavelengths_m=(532e-9,),
                    light_mode="amplitude",
                    amplitude=1.0,
                ),
            )
            probe = FieldProbeLayer(dx=1e-6, name="probe")
            context = PropagateContext(
                propagation_config=_default_propagation_config(),
                error_config=error_config,
                error_factor=1.0,
            )
            context.add_layer(source, z=0.0)
            context.add_layer(probe, z=6e-6)
            return context, probe

        baseline_context, baseline_probe = build_context(
            PropagationErrorConfig(delta_z_m=0.0, shift_x_m=0.0, shift_y_m=0.0)
        )
        shift_context, shift_probe = build_context(
            PropagationErrorConfig(delta_z_m=0.0, shift_x_m=1e-6, shift_y_m=0.0)
        )
        dz_context, dz_probe = build_context(
            PropagationErrorConfig(delta_z_m=1e-6, shift_x_m=0.0, shift_y_m=0.0)
        )

        baseline_context.propagate(image)
        shift_context.propagate(image)
        dz_context.propagate(image)

        self.assertIsNotNone(baseline_probe.E)
        self.assertIsNotNone(shift_probe.E)
        self.assertIsNotNone(dz_probe.E)
        self.assertFalse(torch.allclose(baseline_probe.E, shift_probe.E))
        self.assertFalse(torch.allclose(baseline_probe.E, dz_probe.E))

    def test_fdtd_component_configs(self) -> None:
        builder = _DummyBuilder()
        layer_ctx = FDTDLayerContext(z=0.0, z_bottom=1e-6, air_gap_before=0.0)

        source = LightSourceLayer(
            width_m=2e-6,
            height_m=2e-6,
            dx_m=1e-6,
            config=SourceConfig(
                wavelengths_m=(532e-9, 633e-9),
                light_mode="amplitude",
                amplitude=1.0,
            ),
            fdtd_config=FDTDSourceConfig(
                name="src_custom",
                injection_axis="x",
                direction="Backward",
            ),
        )
        detector = DetectorLayer(
            config=DetectorConfig(width_num=2, height_num=2, detector_unit_len_m=1e-6),
            dx_m=1e-6,
            fdtd_config=FDTDMonitorConfig(
                name="det_custom",
                monitor_type="2D X-normal",
            ),
        )
        probe = FieldProbeLayer(
            dx=1e-6,
            name="probe_custom",
            fdtd_config=FDTDProbeConfig(
                enabled=True,
                name="probe_monitor",
                monitor_type="2D Y-normal",
            ),
        )
        probe.width = 2e-6
        probe.height = 2e-6
        metasurface = MetaEncoder(
            provider=_DummyLUTProvider(),
            phys_shape=torch.tensor(
                [
                    [[0.4e-6, 0.8e-6], [0.9e-6, 0.5e-6]],
                    [[0.4e-6, 0.8e-6], [0.9e-6, 0.5e-6]],
                ],
                dtype=torch.float32,
            ),
            wavelengths_m=(532e-9,),
            dx=1e-6,
            fixed_height=600e-9,
            base_height=100e-9,
            period_x=1e-6,
            period_y=1e-6,
            fdtd_config=FDTDMetaConfig(min_feature_m=0.6e-6),
        )

        source.build_fdtd(builder, layer_ctx)
        detector.build_fdtd(builder, layer_ctx)
        probe.build_fdtd(builder, layer_ctx)
        metasurface.build_fdtd(builder, layer_ctx)

        self.assertEqual(builder.plane_sources[0]["name"], "src_custom")
        self.assertEqual(builder.plane_sources[0]["injection_axis"], "x")
        self.assertEqual(builder.plane_sources[0]["direction"], "Backward")
        self.assertEqual(builder.profile_monitors[0]["name"], "det_custom")
        self.assertEqual(builder.profile_monitors[1]["name"], "probe_monitor")
        self.assertEqual(builder.groups[0]["name"], "meta_pattern_group_1000nm")
        self.assertEqual(builder.rects[0]["name"], "meta_base_1000nm")
        self.assertEqual(builder.scripts[0].count("addrect;"), 2)


if __name__ == "__main__":
    unittest.main()
