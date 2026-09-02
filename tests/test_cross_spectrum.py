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
        self.assertLess(float(peak.imag), 0.0)   # standard convention: a lagging partner gives a negative imaginary part
        # and the leading partner the opposite sign
        _, b_lead = _pair(phase_b_rad=-np.pi / 2)
        lead = tw.cross_spectrum(a, b_lead).isel(lat=0).sel(frequency=1 / 8, zonal_wavenumber=5, method="nearest")
        self.assertGreater(float(lead.imag), 0.0)

    def test_negative_frequency_half_plane_is_the_complex_conjugate(self):
        a, b = _pair(phase_b_rad=0.7)
        cross = tw.cross_spectrum(a, b).isel(lat=0)
        pos = cross.sel(frequency=1 / 8, zonal_wavenumber=5, method="nearest")
        neg = cross.sel(frequency=-1 / 8, zonal_wavenumber=-5, method="nearest")
        np.testing.assert_allclose(complex(neg.values), np.conj(complex(pos.values)), rtol=1e-10, atol=1e-12)


EARTH_RADIUS_M = 6.371e6


def _damped_advection(ubar_ms, tau_days, zonal_wavenumber, period_days, *, gradient=1e-6, n_days=500, dt_days=0.05, spinup_days=100):
    """Integrate dq/dt + ubar dq/dx = -g v - q/tau with RK4 for v = cos(k x - omega t), sampled daily after spin-up.

    Returns the daily (time, lon) fields v and q and the intrinsic frequency omega - ubar k in rad/day.
    With zonal_wavenumber < 0 the prescribed wave moves westward (it appears at negative wavenumber
    and positive frequency in this library); with > 0 eastward.
    """
    lon = np.arange(0.0, 360.0, 2.5)
    x = np.deg2rad(lon) * EARTH_RADIUS_M
    omega = 2.0 * np.pi / (period_days * 86400.0)
    k_m = zonal_wavenumber / EARTH_RADIUS_M
    kx = np.fft.fftfreq(lon.size, d=x[1] - x[0]) * 2.0 * np.pi
    tau = tau_days * 86400.0
    h = dt_days * 86400.0
    n_steps = int(round((n_days + spinup_days) / dt_days))
    per_day = int(round(1.0 / dt_days))

    def rhs(q, t):
        v = np.cos(k_m * x - omega * t)
        dqdx = np.fft.ifft(1j * kx * np.fft.fft(q)).real
        return -ubar_ms * dqdx - gradient * v - q / tau

    q = np.zeros(lon.size)
    v_out, q_out = [], []
    for i in range(n_steps):
        t = i * h
        k1 = rhs(q, t)
        k2 = rhs(q + 0.5 * h * k1, t + 0.5 * h)
        k3 = rhs(q + 0.5 * h * k2, t + 0.5 * h)
        k4 = rhs(q + h * k3, t + h)
        q = q + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)      # q is now at time t + h
        if (i + 1) % per_day == 0 and i + 1 >= spinup_days * per_day:
            v_out.append(np.cos(k_m * x - omega * (t + h)))   # sample v at the same instant as q
            q_out.append(q.copy())
    time = xr.date_range("2000-01-01", periods=len(v_out), freq="D")
    coords = {"time": time, "lon": lon}
    v_da = xr.DataArray(np.array(v_out), dims=("time", "lon"), coords=coords, name="v")
    q_da = xr.DataArray(np.array(q_out), dims=("time", "lon"), coords=coords, name="q")
    return v_da, q_da, (omega - ubar_ms * k_m) * 86400.0


class DampedAdvectionPhaseTests(unittest.TestCase):
    """Regression test for the quadrature sign: q = -g v / (1/tau - i omega_intr) lags the anti-phase point by arctan(omega_intr tau)."""

    def _lag_from_antiphase_deg(self, v, q, zonal_wavenumber, period_days):
        cross = tw.cross_spectrum(v, q)
        peak = cross.sel(frequency=1.0 / period_days, zonal_wavenumber=zonal_wavenumber, method="nearest")
        self.assertLess(float(peak.real), 0.0)   # down-gradient covariance for g > 0
        return np.degrees(np.arctan2(float(peak.imag), -float(peak.real))), cross

    def test_westward_wave_lag_equals_arctan_omega_tau(self):
        v, q, omega_intr = _damped_advection(ubar_ms=-5.0, tau_days=5.0, zonal_wavenumber=-4, period_days=5.0)
        lag, _ = self._lag_from_antiphase_deg(v, q, -4, 5.0)
        self.assertAlmostEqual(lag, np.degrees(np.arctan(omega_intr * 5.0)), delta=1.5)

    def test_eastward_wave_lag_equals_arctan_omega_tau(self):
        v, q, omega_intr = _damped_advection(ubar_ms=-5.0, tau_days=2.0, zonal_wavenumber=4, period_days=4.0)
        lag, _ = self._lag_from_antiphase_deg(v, q, 4, 4.0)
        self.assertAlmostEqual(lag, np.degrees(np.arctan(omega_intr * 2.0)), delta=1.5)

    def test_reversed_intrinsic_propagation_reverses_the_lag(self):
        # a strong easterly carries the westward wave eastward relative to the flow: omega_intr < 0, q leads the anti-phase point
        v, q, omega_intr = _damped_advection(ubar_ms=-40.0, tau_days=5.0, zonal_wavenumber=-4, period_days=5.0)
        self.assertLess(omega_intr, 0.0)
        lag, _ = self._lag_from_antiphase_deg(v, q, -4, 5.0)
        self.assertLess(lag, 0.0)
        self.assertAlmostEqual(lag, np.degrees(np.arctan(omega_intr * 5.0)), delta=1.5)

    def test_both_half_planes_give_the_same_lag(self):
        v, q, omega_intr = _damped_advection(ubar_ms=-5.0, tau_days=5.0, zonal_wavenumber=-4, period_days=5.0)
        lag_pos, cross = self._lag_from_antiphase_deg(v, q, -4, 5.0)
        neg = cross.sel(frequency=-1.0 / 5.0, zonal_wavenumber=4, method="nearest")
        # the conjugate bin: applying the sign of the frequency recovers the same lag
        lag_neg = np.degrees(np.arctan2(np.sign(float(neg.frequency)) * float(neg.imag), -float(neg.real)))
        self.assertAlmostEqual(lag_pos, lag_neg, places=6)
        self.assertAlmostEqual(lag_pos, np.degrees(np.arctan(omega_intr * 5.0)), delta=1.5)

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
