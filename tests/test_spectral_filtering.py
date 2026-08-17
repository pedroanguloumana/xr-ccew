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


if __name__ == "__main__":
    unittest.main()

