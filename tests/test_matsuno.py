import inspect
import unittest

import numpy as np

from xr_ccew import matsuno


class MatsunoBackgroundWindTests(unittest.TestCase):
    """Regression tests for constant-background-wind Matsuno curves."""

    latitude = 12.0
    equivalent_depth_m = 25.0
    background_u = 12.5

    def _doppler_shift(self, zonal_wavenumber):
        _, perimeter = matsuno.beta_parameters(self.latitude)
        return (
            self.background_u
            * np.asarray(zonal_wavenumber, dtype=float)
            * matsuno.SECONDS_PER_DAY
            / perimeter
        )

    def _frequency_cases(self):
        return (
            ("kelvin", matsuno.kelvin_frequency, np.array([1.0, 4.0]), {}),
            ("mrg", matsuno.mrg_frequency, np.array([-1.0, -4.0]), {}),
            ("eig0", matsuno.eig0_frequency, np.array([1.0, 4.0]), {}),
            ("er", matsuno.er_frequency, np.array([-1.0, -4.0]), {"n": 1}),
            ("eig", matsuno.eig_frequency, np.array([1.0, 4.0]), {"n": 1}),
            ("wig", matsuno.wig_frequency, np.array([-1.0, -4.0]), {"n": 1}),
        )

    def test_frequency_functions_accept_keyword_only_background_wind(self):
        functions = (
            matsuno.kelvin_frequency,
            matsuno.mrg_frequency,
            matsuno.eig0_frequency,
            matsuno.er_frequency,
            matsuno.eig_frequency,
            matsuno.wig_frequency,
            matsuno.profile_curve_frequency,
        )

        for function in functions:
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters
                for name in ("background_u", "background_v"):
                    parameter = parameters[name]
                    self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertEqual(parameter.default, 0.0)

    def test_frequency_functions_apply_signed_zonal_doppler_shift(self):
        for name, function, zonal_wavenumber, kwargs in self._frequency_cases():
            with self.subTest(curve=name):
                resting = function(
                    zonal_wavenumber,
                    self.equivalent_depth_m,
                    latitude=self.latitude,
                    **kwargs,
                )
                explicit_resting = function(
                    zonal_wavenumber,
                    self.equivalent_depth_m,
                    latitude=self.latitude,
                    background_u=0.0,
                    background_v=0.0,
                    **kwargs,
                )
                shifted = function(
                    zonal_wavenumber,
                    self.equivalent_depth_m,
                    latitude=self.latitude,
                    background_u=self.background_u,
                    **kwargs,
                )

                np.testing.assert_allclose(explicit_resting, resting, equal_nan=True)
                np.testing.assert_allclose(
                    shifted,
                    resting + self._doppler_shift(zonal_wavenumber),
                    equal_nan=True,
                )

    def test_profile_curve_frequency_forwards_signed_zonal_doppler_shift(self):
        for name, _, zonal_wavenumber, kwargs in self._frequency_cases():
            with self.subTest(curve=name):
                resting = matsuno.profile_curve_frequency(
                    name,
                    zonal_wavenumber,
                    self.equivalent_depth_m,
                    meridional_mode_number=kwargs.get("n"),
                    latitude=self.latitude,
                )
                shifted = matsuno.profile_curve_frequency(
                    name,
                    zonal_wavenumber,
                    self.equivalent_depth_m,
                    meridional_mode_number=kwargs.get("n"),
                    latitude=self.latitude,
                    background_u=self.background_u,
                )

                np.testing.assert_allclose(
                    shifted,
                    resting + self._doppler_shift(zonal_wavenumber),
                    equal_nan=True,
                )

    def test_frequency_functions_reject_nonzero_meridional_background_wind(self):
        for name, function, zonal_wavenumber, kwargs in self._frequency_cases():
            with self.subTest(curve=name):
                with self.assertRaises(NotImplementedError):
                    function(
                        zonal_wavenumber,
                        self.equivalent_depth_m,
                        latitude=self.latitude,
                        background_v=1.0,
                        **kwargs,
                    )

        with self.assertRaises(NotImplementedError):
            matsuno.profile_curve_frequency(
                "kelvin",
                np.array([1.0]),
                self.equivalent_depth_m,
                background_v=1.0,
            )


if __name__ == "__main__":
    unittest.main()
