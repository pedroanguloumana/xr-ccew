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

    def test_td_type_filter_keeps_westward_wave_inside_the_box(self):
        westward = tw.synthetic_wave(
            period_days=4,
            zonal_wavenumber=8,
            amplitude=1.0,
            propagation="westward",
            meridional_mode="symmetric",
        )
        eastward = tw.synthetic_wave(
            period_days=4,
            zonal_wavenumber=8,
            amplitude=1.0,
            propagation="eastward",
            meridional_mode="symmetric",
        )
        mixed = westward + eastward

        filtered = tw.filter_field(mixed, "td")
        error = float(np.abs(filtered - westward).max())
        self.assertLess(error, 1e-10)

    def test_td_type_mask_is_the_kiladis_2006_parallelogram(self):
        frequency = np.fft.fftshift(np.fft.fftfreq(96, d=1.0))
        wavenumber = np.arange(-72, 72, dtype=float)
        spectrum = xr.DataArray(
            np.zeros((frequency.size, wavenumber.size)),
            dims=("frequency", "zonal_wavenumber"),
            coords={"frequency": frequency, "zonal_wavenumber": wavenumber},
        )
        mask = tw.make_filter_mask(spectrum, "TD-type", include_conjugates=False)

        # Both sloped edges drop 1/84 cpd per unit wavenumber from k=-20.
        f, k = xr.broadcast(spectrum.frequency, spectrum.zonal_wavenumber)
        drop = (k + 20.0) / 84.0
        expected = (
            (k >= -20) & (k <= -6) & (f >= 0.3 - drop - 1e-9) & (f <= 0.5 - drop + 1e-9)
        )
        self.assertTrue(bool(expected.any()))
        self.assertTrue(bool((mask.transpose(*expected.dims) == expected).all()))

        column = lambda kk: mask.sel(zonal_wavenumber=kk)  # noqa: E731
        kept = lambda kk: column(kk).frequency.where(column(kk), drop=True).values  # noqa: E731
        np.testing.assert_allclose(kept(-20).min(), 29 / 96)
        np.testing.assert_allclose(kept(-20).max(), 47 / 96)
        np.testing.assert_allclose(kept(-6).min(), 13 / 96)
        np.testing.assert_allclose(kept(-6).max(), 32 / 96)  # 1/3 lies on the grid: kept
        self.assertEqual(kept(-5).size, 0)
        self.assertEqual(kept(-21).size, 0)

        # At k=-13 the upper edge sits exactly on the 40/96 grid line.
        self.assertTrue(bool(column(-13).sel(frequency=40 / 96, method="nearest")))
        self.assertFalse(bool(column(-13).sel(frequency=41 / 96, method="nearest")))
        # Inside the bounding rectangle but above the sloped edge.
        self.assertFalse(bool(column(-10).sel(frequency=0.45, method="nearest")))

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


class NonStandardCalendarTests(unittest.TestCase):
    """CMIP output uses noleap and 360_day calendars, which xarray decodes to
    cftime objects rather than datetime64."""

    CALENDARS = ("proleptic_gregorian", "noleap", "360_day")

    def _field(self, calendar, periods=800):
        time = xr.date_range(
            "2000-01-01",
            periods=periods,
            freq="D",
            calendar=calendar,
            use_cftime=(calendar != "proleptic_gregorian"),
        )
        lat = np.arange(-10.0, 10.1, 5.0)
        lon = np.arange(0.0, 360.0, 20.0)
        rng = np.random.default_rng(0)
        return xr.DataArray(
            rng.standard_normal((periods, lat.size, lon.size)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="pr",
        )

    def test_pipeline_runs_on_every_calendar(self):
        for calendar in self.CALENDARS:
            with self.subTest(calendar=calendar):
                da = self._field(calendar)
                out = tw.remove_mean_and_linear_trend(da)
                out = tw.remove_harmonics_of_seasonal_cycle(out, num_harmonics=3)
                segments = tw.segment_data(out, segment_days=96, overlap_days=30)
                self.assertGreater(len(segments), 0)
                tapered = tw.apply_window(segments[0], dim="time", window="tukey", pct=0.1)
                power = tw.power_spectrum(tapered)
                self.assertTrue(np.isfinite(power).all())
                filtered = tw.filter_field(out, "Kelvin")
                self.assertEqual(filtered.shape, da.shape)
                tw.add_time_days(segments[0])

    def test_segmentation_matches_across_calendars(self):
        counts = {
            calendar: len(
                tw.segment_data(
                    self._field(calendar), segment_days=96, overlap_days=30
                )
            )
            for calendar in self.CALENDARS
        }
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_annual_cycle_removed_on_360_day_calendar(self):
        """The regression that a hard-coded 365-day year would fail.

        A 360_day model year scored against a 365-day period drifts five days
        per year, so the fitted harmonic slips out of phase and a large part
        of the seasonal cycle survives.
        """
        periods = 360 * 10
        time = xr.date_range(
            "2000-01-01", periods=periods, freq="D", calendar="360_day", use_cftime=True
        )
        day_of_year = np.asarray(time.dayofyear if hasattr(time, "dayofyear")
                                 else [t.dayofyr for t in time], dtype=float)
        amplitude = 5.0
        seasonal = amplitude * np.sin(2 * np.pi * (day_of_year - 1) / 360.0)
        da = xr.DataArray(seasonal, dims=("time",), coords={"time": time}, name="pr")
        da = da.expand_dims(lat=[0.0]).transpose("time", "lat")

        residual = tw.remove_harmonics_of_seasonal_cycle(da, num_harmonics=1)
        self.assertLess(float(np.abs(residual).max()), 0.01 * amplitude)

    def test_year_length_follows_the_calendar(self):
        expected = {"proleptic_gregorian": 365.2425, "noleap": 365.0, "360_day": 360.0}
        for calendar, days in expected.items():
            with self.subTest(calendar=calendar):
                da = self._field(calendar, periods=64)
                self.assertAlmostEqual(tw.spectral._year_length(da.time), days)

    def test_numeric_time_still_rejected(self):
        da = xr.DataArray(
            np.zeros((10, 1)),
            dims=("time", "lat"),
            coords={"time": np.arange(10.0), "lat": [0.0]},
        )
        with self.assertRaises(TypeError):
            tw.remove_harmonics_of_seasonal_cycle(da, num_harmonics=1)


if __name__ == "__main__":
    unittest.main()

