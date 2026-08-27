"""Core xarray preprocessing and spectral operations."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import numpy as np
import xarray as xr


def symmetric_antisymmetric_component(
    da: xr.DataArray,
    which: Literal["symmetric", "antisymmetric", "both"] = "both",
    *,
    lat_dim: str = "lat",
    extrapolate: bool = False,
) -> xr.DataArray:
    """Return the equatorially symmetric or antisymmetric component.

    Mirrored latitudes are interpolated, so the latitude grid does not need to
    contain exact +/- pairs or an exact equatorial point.
    """
    if lat_dim not in da.dims:
        raise ValueError(f"lat_dim={lat_dim!r} not in da.dims={da.dims}")
    if which not in {"symmetric", "antisymmetric", "both"}:
        raise ValueError("which must be 'symmetric', 'antisymmetric', or 'both'")

    sorted_da = da.sortby(lat_dim)
    source_lat = sorted_da[lat_dim].astype(float)
    target_lat = -source_lat
    output_dtype = np.result_type(sorted_da.dtype, float)

    mirrored = xr.apply_ufunc(
        _interp_1d,
        sorted_da,
        source_lat,
        target_lat,
        input_core_dims=[[lat_dim], [lat_dim], [lat_dim]],
        output_core_dims=[[lat_dim]],
        kwargs={"extrapolate": extrapolate},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[output_dtype],
        dask_gufunc_kwargs={"output_sizes": {lat_dim: sorted_da.sizes[lat_dim]}},
    )
    mirrored = mirrored.assign_coords({lat_dim: sorted_da[lat_dim]}).transpose(*sorted_da.dims)

    symmetric = 0.5 * (sorted_da + mirrored)
    antisymmetric = 0.5 * (sorted_da - mirrored)

    if which == "symmetric":
        return symmetric
    if which == "antisymmetric":
        return antisymmetric
    return symmetric + antisymmetric


def _restore_identity(out: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
    """Restore `source`'s name and attributes on a derived array.

    xarray merges operand attributes across a binary op and drops the keys
    that conflict. Several operations here subtract or multiply by an
    intermediate built from a coordinate of `source` -- the harmonic design
    matrix, the linear trend, the taper -- and those intermediates inherit
    that coordinate's attributes. Left alone, the result picks up the time
    axis's `standard_name`/`bounds`/`axis` (writing a `bounds` attribute
    that points at a variable no longer in the dataset, which is CF-invalid)
    and loses any of the variable's own attributes that collide, such as
    `long_name`. Each of these operations returns the same physical
    quantity as `source`, so its identity carries over unchanged.
    """
    out.name = source.name
    out.attrs = dict(source.attrs)
    return out


def remove_mean_and_linear_trend(da: xr.DataArray, *, dim: str = "time") -> xr.DataArray:
    """Remove the mean and least-squares linear trend along one dimension."""
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not in da.dims={da.dims}")

    x = xr.DataArray(np.arange(da.sizes[dim], dtype=float), dims=(dim,), coords={dim: da[dim]})
    x0 = x - x.mean(dim)
    y0 = da - da.mean(dim, skipna=True)
    slope = (y0 * x0).sum(dim, skipna=True) / (x0 * x0).sum(dim)
    intercept = da.mean(dim, skipna=True)
    trend = intercept + slope * x0
    return _restore_identity(da - trend, da)


def remove_harmonics_of_seasonal_cycle(
    da: xr.DataArray,
    *,
    num_harmonics: int = 3,
    time_dim: str = "time",
) -> xr.DataArray:
    """Remove harmonics of the annual cycle without removing the mean."""
    if time_dim not in da.dims:
        raise ValueError(f"time_dim={time_dim!r} not in da.dims={da.dims}")
    if num_harmonics < 1:
        return da

    t = da[time_dim]
    if not np.issubdtype(t.dtype, np.datetime64):
        raise TypeError("remove_harmonics_of_seasonal_cycle expects datetime64 time")

    frac_day = (
        t.dt.hour / 24.0
        + t.dt.minute / 1440.0
        + t.dt.second / 86400.0
        + getattr(t.dt, "microsecond", 0) / 86400.0 / 1e6
    )
    doy0 = (t.dt.dayofyear - 1).astype(float)
    theta = 2.0 * np.pi * (doy0 + frac_day) / 365.0

    cols = []
    for harmonic in range(1, num_harmonics + 1):
        cols.extend([np.sin(harmonic * theta), np.cos(harmonic * theta)])

    design = xr.concat(cols, dim="_harmonic").transpose(time_dim, "_harmonic").astype(float)
    design.attrs = {}

    def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return beta

    beta = xr.apply_ufunc(
        _lstsq,
        design,
        da,
        input_core_dims=[[time_dim, "_harmonic"], [time_dim]],
        output_core_dims=[["_harmonic"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"_harmonic": 2 * num_harmonics}},
    )
    fitted = xr.dot(design, beta, dim="_harmonic")
    return _restore_identity(da - fitted, da)


def apply_window(
    da: xr.DataArray,
    *,
    dim: str = "time",
    window: Literal["tukey", "hann", "hanning", "hamming", "blackman"] = "tukey",
    pct: float | None = 0.10,
    points_per_end: int | None = None,
) -> xr.DataArray:
    """Multiply data by a 1-D window along one dimension."""
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not in da.dims={da.dims}")

    weights = _window_values(
        da.sizes[dim],
        window=window,
        pct=pct,
        points_per_end=points_per_end,
    )
    w_da = xr.DataArray(weights, dims=(dim,), coords={dim: da[dim]}, name=f"{window}_window")
    return _restore_identity(da * w_da, da)


def segment_data(
    da: xr.DataArray,
    *,
    segment_days: int = 96,
    overlap_days: int = 30,
    time_dim: str = "time",
) -> list[xr.DataArray]:
    """Split a datetime-indexed array into overlapping full-length segments."""
    import pandas as pd

    if time_dim not in da.dims:
        raise ValueError(f"time_dim={time_dim!r} not in da.dims={da.dims}")
    if segment_days <= 0:
        raise ValueError("segment_days must be positive")
    if overlap_days >= segment_days:
        raise ValueError("overlap_days must be smaller than segment_days")

    earliest_day = pd.Timestamp(da[time_dim].min().item())
    latest_day = pd.Timestamp(da[time_dim].max().item())
    step = pd.Timedelta(days=segment_days - overlap_days)
    segment = pd.Timedelta(days=segment_days)
    total = latest_day - earliest_day
    n_segments = max(0, int((total - segment) / step) + 1)

    return [
        da.sel({time_dim: slice(earliest_day + i * step, earliest_day + i * step + segment)})
        for i in range(n_segments)
    ]


def add_time_days(da: xr.DataArray, *, time_dim: str = "time") -> xr.DataArray:
    """Swap a datetime time dimension for numeric days since the first sample."""
    if time_dim not in da.dims:
        raise ValueError(f"time_dim={time_dim!r} not in da.dims={da.dims}")
    t = da[time_dim]
    if not np.issubdtype(t.dtype, np.datetime64):
        raise TypeError("add_time_days expects a datetime64 time coordinate")

    time_days = (t - t.isel({time_dim: 0})) / np.timedelta64(1, "D")
    out = da.assign_coords(time_days=time_days).swap_dims({time_dim: "time_days"})
    out = out.drop_vars(time_dim)
    out["time_days"].attrs["units"] = "days since segment start"
    return out


def space_time_fft(
    da: xr.DataArray,
    *,
    time_dim: str = "time",
    lon_dim: str = "lon",
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
    shift: bool = True,
) -> xr.DataArray:
    """Return the 2-D FFT over time and longitude.

    Positive frequency and positive zonal wavenumber are coordinated so that
    an eastward wave written as cos(k * lon - omega * time) appears at
    positive frequency and positive zonal wavenumber.
    """
    _require_dims(da, (time_dim, lon_dim))
    dt_days = _coordinate_spacing_days(da[time_dim])
    dlon_degrees = _coordinate_spacing_float(da[lon_dim], name=lon_dim)
    if dlon_degrees <= 0:
        raise ValueError("longitude coordinate must be increasing")

    axes = (da.get_axis_num(time_dim), da.get_axis_num(lon_dim))
    fft_data = _fftn(da.data, axes=axes, shift=shift)

    frequency = -np.fft.fftfreq(da.sizes[time_dim], d=dt_days)
    zonal_wavenumber = np.fft.fftfreq(da.sizes[lon_dim], d=dlon_degrees) * 360.0
    if shift:
        frequency = np.fft.fftshift(frequency)
        zonal_wavenumber = np.fft.fftshift(zonal_wavenumber)

    dims = tuple(
        frequency_dim if dim == time_dim else wavenumber_dim if dim == lon_dim else dim
        for dim in da.dims
    )
    coords = _non_transformed_coords(da, transformed_dims={time_dim, lon_dim})
    coords[frequency_dim] = (frequency_dim, frequency)
    coords[wavenumber_dim] = (wavenumber_dim, zonal_wavenumber)

    out = xr.DataArray(
        fft_data,
        dims=dims,
        coords=coords,
        name=f"{da.name}_fft" if da.name else None,
        attrs=dict(da.attrs),
    )
    out[frequency_dim].attrs["units"] = "cycles day-1"
    out[wavenumber_dim].attrs["units"] = "cycles per 360 degrees longitude"
    out.attrs.update(
        {
            "fft_time_dim": time_dim,
            "fft_lon_dim": lon_dim,
            "fft_shifted": shift,
            "frequency_convention": "positive frequency and positive zonal wavenumber are eastward",
        }
    )
    return out


def space_time_ifft(
    ft: xr.DataArray,
    *,
    template: xr.DataArray | None = None,
    time_dim: str = "time",
    lon_dim: str = "lon",
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
    shift: bool = True,
    real: bool = True,
) -> xr.DataArray:
    """Invert a space-time FFT produced by :func:`space_time_fft`."""
    _require_dims(ft, (frequency_dim, wavenumber_dim))

    axes = (ft.get_axis_num(frequency_dim), ft.get_axis_num(wavenumber_dim))
    data = ft.data
    if shift:
        data = _ifftshift(data, axes=axes)
    inverse = _ifftn(data, axes=axes)
    if real:
        inverse = inverse.real

    dims = tuple(time_dim if dim == frequency_dim else lon_dim if dim == wavenumber_dim else dim for dim in ft.dims)
    coords: dict[Hashable, object] = {}
    if template is not None:
        for dim in dims:
            if dim in template.coords:
                coords[dim] = template.coords[dim]
    else:
        for old_dim, new_dim in zip(ft.dims, dims):
            if old_dim not in {frequency_dim, wavenumber_dim} and old_dim in ft.coords:
                coords[new_dim] = ft.coords[old_dim]

    return xr.DataArray(
        inverse,
        dims=dims,
        coords=coords,
        name=template.name if template is not None else ft.name,
        attrs=dict(template.attrs) if template is not None else dict(ft.attrs),
    )


def power_spectrum(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    lon_dim: str = "lon",
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
    scaling: Literal["none", "density", "spectrum"] = "none",
    shift: bool = True,
) -> xr.DataArray:
    """Compute space-time power from a field or from an existing FFT."""
    if frequency_dim in data.dims and wavenumber_dim in data.dims:
        ft = data
    else:
        ft = space_time_fft(
            data,
            time_dim=time_dim,
            lon_dim=lon_dim,
            frequency_dim=frequency_dim,
            wavenumber_dim=wavenumber_dim,
            shift=shift,
        )

    power = (ft * ft.conj()).real
    if scaling != "none":
        df = _coordinate_spacing_float(power[frequency_dim], name=frequency_dim, absolute=True)
        dk = _coordinate_spacing_float(power[wavenumber_dim], name=wavenumber_dim, absolute=True)
        cell = df * dk
        if scaling == "density":
            power = power * cell
        elif scaling == "spectrum":
            power = power * (cell**2)
        else:
            raise ValueError("scaling must be 'none', 'density', or 'spectrum'")

    power.name = f"{data.name}_power" if data.name else "power"
    power[frequency_dim].attrs["units"] = "cycles day-1"
    power[wavenumber_dim].attrs["units"] = "cycles per 360 degrees longitude"
    return power


def background_spectrum(
    power: xr.DataArray,
    *,
    wavenumber_dim: str = "zonal_wavenumber",
    frequency_dim: str = "frequency",
    passes_wavenumber: int = 10,
    passes_frequency: int = 10,
) -> xr.DataArray:
    """Compute a simple repeated 1-2-1 smoothed spectral background."""
    bg = _smooth_121(power, dim=wavenumber_dim, n_passes=passes_wavenumber)
    bg = _smooth_121(bg, dim=frequency_dim, n_passes=passes_frequency)
    bg.name = f"{power.name}_background" if power.name else "background"
    return bg


def _smooth_121(da: xr.DataArray, *, dim: str, n_passes: int) -> xr.DataArray:
    if n_passes < 0:
        raise ValueError("n_passes must be non-negative")
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not in da.dims={da.dims}")

    out = da
    weights = xr.DataArray([0.25, 0.5, 0.25], dims=("_smooth_window",))
    for _ in range(n_passes):
        padded = out.pad({dim: (1, 1)}, mode="edge")
        rolled = padded.rolling({dim: 3}, center=True).construct("_smooth_window")
        out = (rolled * weights).sum("_smooth_window").isel({dim: slice(1, -1)})
        out = out.assign_coords({dim: da[dim]})
    return out


def _interp_1d(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    extrapolate: bool,
) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    values = np.asarray(values)

    if source.size < 2:
        raise ValueError("at least two latitude points are required for interpolation")

    out = np.interp(target, source, values)
    outside = (target < source[0]) | (target > source[-1])

    if extrapolate:
        left = target < source[0]
        right = target > source[-1]
        left_slope = (values[1] - values[0]) / (source[1] - source[0])
        right_slope = (values[-1] - values[-2]) / (source[-1] - source[-2])
        out[left] = values[0] + left_slope * (target[left] - source[0])
        out[right] = values[-1] + right_slope * (target[right] - source[-1])
    else:
        out = out.astype(np.result_type(out.dtype, float), copy=False)
        out[outside] = np.nan

    return out


def _window_values(
    n: int,
    *,
    window: str,
    pct: float | None,
    points_per_end: int | None,
) -> np.ndarray:
    if n < 1:
        raise ValueError("window length must be positive")
    name = window.lower()
    x = np.arange(n, dtype=float)

    if name == "tukey":
        if points_per_end is not None:
            alpha = min(1.0, (2.0 * points_per_end) / float(n))
        else:
            if pct is None:
                raise ValueError("provide pct or points_per_end for a Tukey window")
            alpha = float(pct)
        if alpha <= 0:
            return np.ones(n)
        if alpha >= 1:
            return np.hanning(n)
        w = np.ones(n)
        first = x < alpha * (n - 1) / 2.0
        last = x >= (n - 1) * (1.0 - alpha / 2.0)
        w[first] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[first] / (alpha * (n - 1)) - 1.0)))
        w[last] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * x[last] / (alpha * (n - 1)) - 2.0 / alpha + 1.0))
        )
        return w
    if name in {"hann", "hanning"}:
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unsupported window {window!r}")


def _coordinate_spacing_days(coord: xr.DataArray) -> float:
    values = np.asarray(coord.values)
    if values.size < 2:
        raise ValueError(f"coordinate {coord.name!r} must contain at least two values")

    if np.issubdtype(values.dtype, np.datetime64):
        diffs = np.diff(values) / np.timedelta64(1, "D")
    elif np.issubdtype(values.dtype, np.timedelta64):
        diffs = np.diff(values) / np.timedelta64(1, "D")
    else:
        diffs = np.diff(values.astype(float))

    diffs = np.asarray(diffs, dtype=float)
    if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
        raise ValueError(f"coordinate {coord.name!r} must be regularly spaced")
    if diffs[0] <= 0:
        raise ValueError(f"coordinate {coord.name!r} must be increasing")
    return float(diffs[0])


def _coordinate_spacing_float(
    coord: xr.DataArray,
    *,
    name: str,
    absolute: bool = False,
) -> float:
    values = np.asarray(coord.values, dtype=float)
    if values.size < 2:
        raise ValueError(f"coordinate {name!r} must contain at least two values")
    diffs = np.diff(values)
    test_diffs = np.abs(diffs) if absolute else diffs
    if not np.allclose(test_diffs, test_diffs[0], rtol=1e-5, atol=1e-8):
        raise ValueError(f"coordinate {name!r} must be regularly spaced")
    out = abs(test_diffs[0]) if absolute else test_diffs[0]
    return float(out)


def _require_dims(da: xr.DataArray, dims: tuple[str, ...]) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"missing required dimensions {missing}; got dims={da.dims}")


def _non_transformed_coords(
    da: xr.DataArray,
    *,
    transformed_dims: set[str],
) -> dict[Hashable, object]:
    coords: dict[Hashable, object] = {}
    for name, coord in da.coords.items():
        if name in transformed_dims:
            continue
        if all(dim not in transformed_dims for dim in coord.dims):
            coords[name] = coord
    return coords


def _fftn(data: object, *, axes: tuple[int, int], shift: bool) -> object:
    if hasattr(data, "chunks"):
        try:
            import dask.array as dask_array
        except ImportError as exc:
            raise ImportError("Dask-backed FFT requires installing xr-ccew[dask]") from exc
        out = dask_array.fft.fftn(data, axes=axes)
        return dask_array.fft.fftshift(out, axes=axes) if shift else out

    out = np.fft.fftn(data, axes=axes)
    return np.fft.fftshift(out, axes=axes) if shift else out


def _ifftn(data: object, *, axes: tuple[int, int]) -> object:
    if hasattr(data, "chunks"):
        import dask.array as dask_array

        return dask_array.fft.ifftn(data, axes=axes)
    return np.fft.ifftn(data, axes=axes)


def _ifftshift(data: object, *, axes: tuple[int, int]) -> object:
    if hasattr(data, "chunks"):
        import dask.array as dask_array

        return dask_array.fft.ifftshift(data, axes=axes)
    return np.fft.ifftshift(data, axes=axes)
