"""Station-feature selection, units, and optional-field isolation."""

import itertools
import math
import unittest

import torch

from emulator.data import station_features_from_json


class StationMetadataTests(unittest.TestCase):
    def test_independent_feature_switches_preserve_geography_and_units(self):
        station = dict(lat=30, lon=-90, elevation_m=20, bathymetry_m=50)
        trig = [0.5, math.sqrt(3) / 2, -1.0, 0.0]
        for elevation, bathymetry in itertools.product((False, True), repeat=2):
            with self.subTest(elevation=elevation, bathymetry=bathymetry):
                features = station_features_from_json(
                    station, use_site_elevation=elevation, use_bathymetry=bathymetry,
                )
                expected = [1 / 3, -0.5] + ([2.0] if elevation else []) + trig + ([5.0] if bathymetry else [])
                torch.testing.assert_close(features, torch.tensor(expected), rtol=1e-6, atol=1e-7)
        expected_default = torch.tensor([1 / 3, -0.5, 2.0, *trig])
        torch.testing.assert_close(station_features_from_json(station), expected_default, rtol=1e-6, atol=1e-7)

    def test_disabled_features_and_node_coordinates_do_not_affect_inputs(self):
        station = dict(lat=30, lon=-90, elevation_m=20, bathymetry_m=50)
        changed = dict(station, bathymetry_m=float("nan"), bathymetry_node=dict(latitude=0, longitude=0))
        torch.testing.assert_close(station_features_from_json(station), station_features_from_json(changed), rtol=0, atol=0)
        changed = dict(station, elevation_m=float("nan"))
        options = dict(use_site_elevation=False, use_bathymetry=True)
        torch.testing.assert_close(station_features_from_json(station, **options), station_features_from_json(changed, **options), rtol=0, atol=0)

    def test_enabled_bathymetry_requires_a_finite_value(self):
        for value in (None, "invalid", float("nan"), float("inf")):
            station = dict(lat=30, lon=-90)
            if value is not None:
                station["bathymetry_m"] = value
            self.assertTrue(torch.isfinite(station_features_from_json(station)).all())
            with self.assertRaisesRegex(ValueError, "requires finite bathymetry_m"):
                station_features_from_json(station, use_bathymetry=True)


if __name__ == "__main__":
    unittest.main()
