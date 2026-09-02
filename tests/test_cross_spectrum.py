import unittest

import numpy as np
import xarray as xr

import xr_ccew as tw


def _pair(phase_b_rad=0.0, n_time=256, amplitude_b=0.5):
    a = tw.synthetic_wave(period_days=8, zonal_wavenumber=5, n_time=n_time, meridional_mode="flat")
    b = tw.synthetic_wave(period_days=8, zonal_wavenumber=5, n_time=n_time, meridional_mode="flat",
                          amplitude=amplitude_b, phase0_rad=phase_b_rad)
    return a.isel(lat=[0]), b.isel(lat=[0])


class CrossSpectrumTests(unittest.TestCase):
    def test_cross_of_a_field_with_itself_is_its_power(self):
        a, _ = _pair()
        np.testing.assert_allclose(tw.cross_spectrum(a, a).real.values, tw.power_spectrum(a).values, rtol=1e-12, atol=1e-9)
        self.assertTrue(np.allclose(tw.cross_spectrum(a, a).imag.values, 0.0, atol=1e-9))

    def test_in_phase_pair_is_real_and_positive_at_the_wave_bin(self):
        a, b = _pair(phase_b_rad=0.0)
        cross = tw.cross_spectrum(a, b).isel(lat=0)
        peak = cross.sel(frequency=1 / 8, zonal_wavenumber=5, method="nearest")
        self.assertGreater(float(peak.real), 0.0)
        self.assertLess(abs(float(peak.imag)), 1e-6 * abs(float(peak.real)))

    def test_quarter_period_pair_is_purely_imaginary(self):
        # cos(k lon - omega t + pi/2) = a(t - T/4): b is a delayed by a quarter period (b lags a)
        a, b = _pair(phase_b_rad=np.pi / 2)
        cross = tw.cross_spectrum(a, b).isel(lat=0)
        peak = cross.sel(frequency=1 / 8, zonal_wavenumber=5, method="nearest")
        self.assertLess(abs(float(peak.real)), 1e-6 * abs(float(peak.imag)))
        self.assertGreater(float(peak.imag), 0.0)   # documented sign: lagging partner gives a positive imaginary part
        # and the leading partner the opposite sign
        _, b_lead = _pair(phase_b_rad=-np.pi / 2)
        lead = tw.cross_spectrum(a, b_lead).isel(lat=0).sel(frequency=1 / 8, zonal_wavenumber=5, method="nearest")
        self.assertLess(float(lead.imag), 0.0)

    def test_anti_phase_pair_is_real_and_negative(self):
        a, b = _pair(phase_b_rad=np.pi)
        peak = tw.cross_spectrum(a, b).isel(lat=0).sel(frequency=1 / 8, zonal_wavenumber=5, method="nearest")
        self.assertLess(float(peak.real), 0.0)


class SegmentAveragedSpectrumTests(unittest.TestCase):
    def _fields(self):
        time = xr.date_range("2000-01-01", periods=700, freq="D")
        lat = np.arange(-10.0, 10.1, 5.0)
        lon = np.arange(0.0, 360.0, 10.0)
        rng = np.random.default_rng(0)
        base = rng.standard_normal((time.size, lat.size, lon.size))
        a = xr.DataArray(base, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}, name="a", attrs={"units": "m s-1"})
        b = xr.DataArray(0.6 * base + 0.8 * rng.standard_normal(base.shape), dims=a.dims, coords=a.coords, name="b", attrs={"units": "g kg-1"})
        return a, b

    def test_power_and_cross_share_one_pipeline(self):
        a, _ = self._fields()
        power = tw.segment_averaged_spectrum(a, segment_days=128, overlap_days=64)
        cross = tw.segment_averaged_spectrum(a, a, segment_days=128, overlap_days=64)
        np.testing.assert_allclose(cross.real.values, power.values, rtol=1e-12, atol=1e-12)
        self.assertEqual(power.attrs["n_segments"], cross.attrs["n_segments"])
        self.assertEqual(power.attrs["samples_per_segment"], 128)

    def test_parseval_sum_equals_segment_covariance(self):
        """The bins sum to the taper-corrected covariance of the tapered segment anomalies, exactly."""
        a, b = self._fields()
        cross = tw.segment_averaged_spectrum(a, b, segment_days=128, overlap_days=64, num_harmonics=0)
        # rebuild the segment covariance the same way the function does
        fa = tw.remove_mean_and_linear_trend(a, dim="time")
        fb = tw.remove_mean_and_linear_trend(b, dim="time")
        segs_a = [s.isel(time=slice(0, 128)) for s in tw.segment_data(fa, segment_days=128, overlap_days=64) if s.sizes["time"] >= 128]
        segs_b = [s.isel(time=slice(0, 128)) for s in tw.segment_data(fb, segment_days=128, overlap_days=64) if s.sizes["time"] >= 128]
        ones = xr.DataArray(np.ones(128), dims=("time",), coords={"time": segs_a[0]["time"]})
        w = tw.apply_window(ones, dim="time", window="tukey", pct=0.10)
        mean_w2 = float((w**2).mean())
        covs = []
        for sa, sb in zip(segs_a, segs_b):
            sa = tw.apply_window(tw.remove_mean_and_linear_trend(sa, dim="time"), dim="time", pct=0.10)
            sb = tw.apply_window(tw.remove_mean_and_linear_trend(sb, dim="time"), dim="time", pct=0.10)
            covs.append(float((sa * sb).mean(("time", "lon")).mean("lat")) / mean_w2)
        expected = float(np.mean(covs))
        total = float(cross.real.sum())
        self.assertAlmostEqual(total / expected, 1.0, places=9)

    def test_single_segment_two_dim_input_without_lat(self):
        a, b = self._fields()
        cross = tw.segment_averaged_spectrum(a.isel(lat=0), b.isel(lat=0), segment_days=128, overlap_days=64)
        self.assertEqual(cross.dims, ("frequency", "zonal_wavenumber"))
        self.assertTrue(np.iscomplexobj(cross.values))


if __name__ == "__main__":
    unittest.main()
