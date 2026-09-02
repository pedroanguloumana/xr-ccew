import unittest

import numpy as np
import pandas as pd
import xarray as xr

import xr_ccew as tw


def _time(freq, periods=64):
    return xr.DataArray(pd.date_range("2000-01-01", periods=periods, freq=freq), dims=("time",), name="time")


class NyquistTests(unittest.TestCase):
    def test_nyquist_of_daily_and_twice_daily_records(self):
        self.assertAlmostEqual(tw.nyquist_frequency(_time("D")), 0.5)
        self.assertAlmostEqual(tw.nyquist_frequency(_time("12h")), 1.0)
        self.assertAlmostEqual(tw.nyquist_frequency(_time("6h")), 2.0)

    def test_accepts_numeric_days_and_datetime_index(self):
        self.assertAlmostEqual(tw.nyquist_frequency(np.arange(0.0, 10.0, 0.5)), 1.0)
        self.assertAlmostEqual(tw.nyquist_frequency(pd.date_range("2000-01-01", periods=8, freq="D")), 0.5)

    def test_irregular_time_is_rejected(self):
        irregular = xr.DataArray(
            pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"]), dims=("time",), name="time"
        )
        with self.assertRaises(ValueError):
            tw.nyquist_frequency(irregular)


class FrequencyCeilingTests(unittest.TestCase):
    def test_default_ceiling_on_daily_data_is_0p40(self):
        eig0 = tw.apply_frequency_ceiling("n=0 EIG", _time("D"))
        self.assertAlmostEqual(eig0.frequency_max, 0.40)
        self.assertAlmostEqual(eig0.frequency_ceiling, 0.40)
        self.assertEqual(eig0.name, "n=0 EIG")
        # the built-in profile itself is untouched
        self.assertAlmostEqual(tw.wave_profile("n=0 EIG").frequency_max, 0.55)

    def test_fraction_controls_the_ceiling(self):
        eig0 = tw.apply_frequency_ceiling("n=0 EIG", _time("D"), fraction=0.9)
        self.assertAlmostEqual(eig0.frequency_max, 0.45)
        self.assertAlmostEqual(eig0.frequency_ceiling, 0.45)

    def test_profile_below_ceiling_keeps_its_frequency_max(self):
        mrg = tw.apply_frequency_ceiling(tw.wave_profile("MRG"), _time("D"))
        self.assertAlmostEqual(mrg.frequency_max, 1.0 / 3.0)
        self.assertAlmostEqual(mrg.frequency_ceiling, 0.40)

    def test_twice_daily_data_leaves_eig0_untruncated(self):
        eig0 = tw.apply_frequency_ceiling("n=0 EIG", _time("12h"))
        self.assertAlmostEqual(eig0.frequency_max, 0.55)
        self.assertAlmostEqual(eig0.frequency_ceiling, 0.80)

    def test_band_entirely_above_ceiling_raises(self):
        fast = tw.wave_profile("KW").with_updates(frequency_min=0.45, frequency_max=0.6)
        with self.assertRaises(ValueError):
            tw.apply_frequency_ceiling(fast, _time("D"))

    def test_bad_fraction_raises(self):
        with self.assertRaises(ValueError):
            tw.apply_frequency_ceiling("MRG", _time("D"), fraction=0.0)
        with self.assertRaises(ValueError):
            tw.apply_frequency_ceiling("MRG", _time("D"), fraction=1.5)

    def test_recorded_ceiling_is_validated_on_later_updates(self):
        eig0 = tw.apply_frequency_ceiling("n=0 EIG", _time("D"))
        with self.assertRaises(ValueError):
            eig0.with_updates(frequency_max=0.5)

    def test_truncated_profile_masks_nothing_above_the_ceiling(self):
        data = tw.synthetic_wave(period_days=3, zonal_wavenumber=5, n_time=128, meridional_mode="flat")
        spectrum = tw.power_spectrum(data).isel(lat=0)
        eig0 = tw.apply_frequency_ceiling("n=0 EIG", data.time)
        mask = tw.make_filter_mask(spectrum, eig0, include_conjugates=False)
        above = mask.where(mask.frequency > eig0.frequency_ceiling + 1e-12, other=False)
        self.assertFalse(bool(above.any()))
        self.assertTrue(bool(mask.any()))

    def test_as_dict_reports_the_ceiling(self):
        eig0 = tw.apply_frequency_ceiling("n=0 EIG", _time("D"))
        params = eig0.as_dict()
        self.assertAlmostEqual(params["frequency_ceiling"], 0.40)
        self.assertAlmostEqual(params["frequency_max"], 0.40)
        self.assertEqual(params["name"], "n=0 EIG")


if __name__ == "__main__":
    unittest.main()
