import unittest

import numpy as np
import xarray as xr

import xr_ccew as tw


class SpectralFilteringTests(unittest.TestCase):
    def test_space_time_power_peak_uses_eastward_positive_convention(self):
        data = tw.synthetic_wave(period_days=8, zonal_wavenumber=5, meridional_mode="flat")

        power = tw.power_spectrum(data).isel(lat=0)
        positive = power.where(power.frequency > 0, drop=True)
        peak = positive.where(positive == positive.max(), drop=True)

        self.assertAlmostEqual(float(peak.frequency.values[0]), 1.0 / 8.0)
        self.assertAlmostEqual(float(peak.zonal_wavenumber.values[0]), 5.0)

    def test_kelvin_filter_keeps_matching_eastward_wave(self):
        eastward = tw.synthetic_wave(
            period_days=8,
            zonal_wavenumber=5,
            amplitude=1.0,
            propagation="eastward",
            meridional_mode="symmetric",
        )
        westward = tw.synthetic_wave(
            period_days=8,
            zonal_wavenumber=5,
            amplitude=1.0,
            propagation="westward",
            meridional_mode="symmetric",
        )
        mixed = eastward + westward

        filtered = tw.filter_field(mixed, "Kelvin")
        error = float(np.abs(filtered - eastward).max())

        self.assertLess(error, 1e-10)

    def test_symmetry_decomposition_handles_off_equatorial_grid(self):
        lat = np.array([-18.0, -13.0, -8.0, -3.0, 2.0, 7.0, 12.0, 17.0])
        time = np.arange(4)
        lon = np.arange(0.0, 360.0, 90.0)
        values = np.broadcast_to(lat[None, :, None], (time.size, lat.size, lon.size))
        data = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
        )

        antisymmetric = tw.symmetric_antisymmetric_component(
            data,
            "antisymmetric",
            extrapolate=True,
        )
        symmetric = tw.symmetric_antisymmetric_component(
            data,
            "symmetric",
            extrapolate=True,
        )

        self.assertLess(float(np.abs(antisymmetric - data.sortby("lat")).max()), 1e-12)
        self.assertLess(float(np.abs(symmetric).max()), 1e-12)


class IdentityPreservationTests(unittest.TestCase):
    """Preprocessing must not let coordinate attributes reach the variable."""

    def _field(self):
        time = xr.date_range("2000-01-01", periods=256, freq="D")
        lat = np.arange(-10.0, 10.1, 2.5)
        lon = np.arange(0.0, 360.0, 10.0)
        rng = np.random.default_rng(0)
        da = xr.DataArray(
            rng.standard_normal((time.size, lat.size, lon.size)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="olr",
            attrs={"units": "W m-2", "long_name": "outgoing longwave radiation"},
        )
        # A CF-compliant time axis, as read off a real NetCDF file. `bounds`
        # and the conflicting `long_name` are what used to corrupt the result.
        da.time.attrs.update(
            {"standard_name": "time", "long_name": "Time", "bounds": "time_bnds", "axis": "T"}
        )
        return da

    def _assert_identity_preserved(self, out, source):
        self.assertEqual(out.name, source.name)
        self.assertEqual(out.attrs, source.attrs)

    def test_remove_mean_and_linear_trend_preserves_identity(self):
        da = self._field()
        self._assert_identity_preserved(tw.remove_mean_and_linear_trend(da), da)

    def test_remove_harmonics_preserves_identity(self):
        da = self._field()
        out = tw.remove_harmonics_of_seasonal_cycle(da, num_harmonics=3)
        self._assert_identity_preserved(out, da)
        # The specific regression: the design matrix is built from `time.dt.*`
        # and used to carry the time axis's attributes onto the anomaly.
        for leaked in ("standard_name", "bounds", "axis"):
            self.assertNotIn(leaked, out.attrs)
        self.assertEqual(out.attrs["long_name"], "outgoing longwave radiation")

    def test_apply_window_preserves_identity(self):
        da = self._field()
        out = tw.apply_window(da, dim="time", window="tukey", pct=0.1)
        self._assert_identity_preserved(out, da)

    def test_preprocessing_chain_preserves_identity(self):
        da = self._field()
        out = tw.remove_mean_and_linear_trend(da)
        out = tw.remove_harmonics_of_seasonal_cycle(out, num_harmonics=3)
        out = tw.filter_field(out, "Kelvin")
        self.assertEqual(out.name, da.name)
        for leaked in ("standard_name", "bounds", "axis"):
            self.assertNotIn(leaked, out.attrs)
        self.assertEqual(out.attrs["units"], "W m-2")

    def test_preprocessing_leaves_source_untouched(self):
        da = self._field()
        before = dict(da.attrs)
        tw.remove_harmonics_of_seasonal_cycle(da, num_harmonics=3)
        self.assertEqual(da.attrs, before)


if __name__ == "__main__":
    unittest.main()

