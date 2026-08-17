"""Synthetic datasets for examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def synthetic_wave(
    *,
    period_days: float = 8.0,
    zonal_wavenumber: int = 5,
    amplitude: float = 1.0,
    phase0_rad: float = 0.0,
    propagation: str = "eastward",
    meridional_mode: str = "symmetric",
    n_time: int = 96,
    n_lon: int = 144,
    start: str = "2000-01-01",
    time_step_days: float = 1.0,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    lat0: float = 0.0,
    lat_sigma_deg: float | None = 10.0,
    name: str = "synthetic_wave",
) -> xr.DataArray:
    """Generate a gridded traveling wave with known frequency and wavenumber."""
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    if propagation not in {"eastward", "westward", "standing"}:
        raise ValueError("propagation must be 'eastward', 'westward', or 'standing'")
    if meridional_mode not in {"symmetric", "antisymmetric", "gaussian", "flat"}:
        raise ValueError("unsupported meridional_mode")

    if lat is None:
        lat = np.arange(-20.0, 20.1, 2.5)
    if lon is None:
        lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)

    time = pd.date_range(start=start, periods=n_time, freq=pd.to_timedelta(time_step_days, unit="D"))
    t_days = xr.DataArray(
        np.arange(n_time, dtype=float) * time_step_days,
        dims=("time",),
        coords={"time": time},
    )
    lon_da = xr.DataArray(np.asarray(lon, dtype=float), dims=("lon",), coords={"lon": lon})
    lat_da = xr.DataArray(np.asarray(lat, dtype=float), dims=("lat",), coords={"lat": lat})

    if lat_sigma_deg is None or meridional_mode == "flat":
        envelope = xr.ones_like(lat_da)
    else:
        y = (lat_da - lat0) / float(lat_sigma_deg)
        if meridional_mode in {"symmetric", "gaussian"}:
            envelope = np.exp(-0.5 * y**2)
        else:
            envelope = y * np.exp(-0.5 * y**2)

    omega = 2.0 * np.pi / period_days
    lon_phase = 2.0 * np.pi * int(zonal_wavenumber) * lon_da / 360.0
    time_phase = omega * t_days

    if propagation == "eastward":
        phase = lon_phase - time_phase + phase0_rad
        wave_xt = np.cos(phase)
    elif propagation == "westward":
        phase = -lon_phase - time_phase + phase0_rad
        wave_xt = np.cos(phase)
    else:
        wave_xt = np.cos(lon_phase + phase0_rad) * np.cos(time_phase)

    wave = amplitude * envelope * wave_xt
    wave = wave.transpose("time", "lat", "lon")
    wave.name = name
    wave.attrs.update(
        {
            "synthetic": True,
            "period_days": period_days,
            "zonal_wavenumber": int(zonal_wavenumber),
            "propagation": propagation,
            "meridional_mode": meridional_mode,
        }
    )
    return wave

