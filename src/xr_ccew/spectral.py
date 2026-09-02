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


# Days in a calendar year, keyed by the CF calendar name. The annual-cycle
# harmonics are fitted against day-of-year, so the period has to be the year
# length of the data's own calendar: a 360_day model run scored against a
# 365-day year drifts a full five days per year and leaves a large residual
# seasonal cycle behind.
_CALENDAR_YEAR_LENGTH = {
    "360_day": 360.0,
    "365_day": 365.0,
    "noleap": 365.0,
    "366_day": 366.0,
    "all_leap": 366.0,
    "julian": 365.25,
    "standard": 365.2425,
    "gregorian": 365.2425,
    "proleptic_gregorian": 365.2425,
}

_DEFAULT_YEAR_LENGTH = 365.2425


def _year_length(coord: xr.DataArray) -> float:
    """Days per year for the calendar of a time coordinate."""
    calendar = getattr(coord.dt, "calendar", None) if hasattr(coord, "dt") else None
    return _CALENDAR_YEAR_LENGTH.get(str(calendar), _DEFAULT_YEAR_LENGTH)


def _is_datetime_like(values: np.ndarray) -> bool:
    """True for numpy datetimes and for cftime objects (which are dtype=object)."""
    return np.issubdtype(values.dtype, np.datetime64) or (
        values.dtype == object and values.size > 0 and hasattr(values.flat[0], "timetuple")
    )


def _days_since_start(values: np.ndarray) -> np.ndarray:
    """Days elapsed since the first sample.

    Handles numpy datetime64, cftime datetimes under any CF calendar (which
    xarray stores as an object array, and which subtract to
    ``datetime.timedelta``), and coordinates that are already numeric.
    """
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros(0, dtype=float)
    if np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(
        values.dtype, np.timedelta64
    ):
        return np.asarray((values - values[0]) / np.timedelta64(1, "D"), dtype=float)
    if values.dtype == object:
        first = values[0]
        return np.array(
            [(value - first).total_seconds() / 86400.0 for value in values], dtype=float
        )
    return values.astype(float) - float(values[0])


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
    if not _is_datetime_like(np.asarray(t.values)):
        raise TypeError(
            "remove_harmonics_of_seasonal_cycle expects a datetime64 or cftime "
            f"time coordinate; got dtype={t.dtype}"
        )
    year_length = _year_length(t)

    frac_day = (
        t.dt.hour / 24.0
        + t.dt.minute / 1440.0
        + t.dt.second / 86400.0
        + getattr(t.dt, "microsecond", 0) / 86400.0 / 1e6
    )
    doy0 = (t.dt.dayofyear - 1).astype(float)
    theta = 2.0 * np.pi * (doy0 + frac_day) / year_length

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
    """Split a datetime-indexed array into overlapping full-length segments.

    Segment boundaries are computed as day offsets from the first sample, so
    this works for any CF calendar, not only the real-world one.
    """
    if time_dim not in da.dims:
        raise ValueError(f"time_dim={time_dim!r} not in da.dims={da.dims}")
    if segment_days <= 0:
        raise ValueError("segment_days must be positive")
    if overlap_days >= segment_days:
        raise ValueError("overlap_days must be smaller than segment_days")

    days = _days_since_start(np.asarray(da[time_dim].values))
    if days.size == 0:
        return []

    step = float(segment_days - overlap_days)
    total = float(days[-1] - days[0])
    n_segments = max(0, int((total - segment_days) / step) + 1)

    segments = []
    for i in range(n_segments):
        start = i * step
        # Inclusive of both endpoints, matching the label-based slice this
        # replaces.
        selected = np.flatnonzero((days >= start) & (days <= start + segment_days))
        segments.append(da.isel({time_dim: selected}))
    return segments


def add_time_days(da: xr.DataArray, *, time_dim: str = "time") -> xr.DataArray:
    """Swap a datetime time dimension for numeric days since the first sample."""
    if time_dim not in da.dims:
        raise ValueError(f"time_dim={time_dim!r} not in da.dims={da.dims}")
    t = da[time_dim]
    values = np.asarray(t.values)
    if not _is_datetime_like(values):
        raise TypeError(
            f"add_time_days expects a datetime64 or cftime time coordinate; got dtype={t.dtype}"
        )

    time_days = xr.DataArray(
        _days_since_start(values), dims=(time_dim,), coords={time_dim: t}
    )
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


def cross_spectrum(
    data_a: xr.DataArray,
    data_b: xr.DataArray,
    *,
    time_dim: str = "time",
    lon_dim: str = "lon",
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
    scaling: Literal["none", "density", "spectrum"] = "none",
    shift: bool = True,
) -> xr.DataArray:
    """Complex space-time cross-spectrum S_AB = conj(A) * B of two fields or two FFTs.

    The real part is the cospectrum (in-phase covariance per bin) and the
    imaginary part the quadrature spectrum. ``cross_spectrum(x, x).real``
    equals ``power_spectrum(x)`` bin for bin. Summed over every bin (with the
    normalisation used by :func:`segment_averaged_spectrum`) the real part is
    the covariance of the two series, which is what makes a band partition of
    a covariance close exactly (Parseval).

    Phase convention (standard, the same as ``scipy.signal.csd``): the phase
    ``arg(S_AB)`` is the phase of B minus the phase of A in physical time, so
    on the positive-frequency half-plane a partner B that *leads* A has a
    positive imaginary part and a partner that *lags* A by a quarter period
    has a negative one; the negative-frequency half-plane holds the complex
    conjugate. Because :func:`space_time_fft` negates the frequency axis
    (positive frequency and wavenumber = eastward), each transform on that
    axis is the conjugate of the physical-time transform, so the standard
    cross-spectrum is obtained as ``A * conj(B)`` of the library transforms.
    Verified against an integrated damped-advection solution with known lag
    in ``tests/test_cross_spectrum.py``.

    History: before September 2026 this function returned ``conj(A) * B`` of
    the library transforms, i.e. the complex conjugate of the standard
    cross-spectrum (opposite quadrature sign); the real part was unaffected.
    """
    fts = []
    for data in (data_a, data_b):
        if frequency_dim in data.dims and wavenumber_dim in data.dims:
            fts.append(data)
        else:
            fts.append(
                space_time_fft(
                    data,
                    time_dim=time_dim,
                    lon_dim=lon_dim,
                    frequency_dim=frequency_dim,
                    wavenumber_dim=wavenumber_dim,
                    shift=shift,
                )
            )
    ft_a, ft_b = fts
    cross = ft_a * ft_b.conj()   # = conj(A) B in the physical time convention (see docstring)
    if scaling != "none":
        df = _coordinate_spacing_float(cross[frequency_dim], name=frequency_dim, absolute=True)
        dk = _coordinate_spacing_float(cross[wavenumber_dim], name=wavenumber_dim, absolute=True)
        cell = df * dk
        if scaling == "density":
            cross = cross * cell
        elif scaling == "spectrum":
            cross = cross * (cell**2)
        else:
            raise ValueError("scaling must be 'none', 'density', or 'spectrum'")
    name_a = data_a.name or "a"
    name_b = data_b.name or "b"
    cross.name = f"{name_a}_{name_b}_cross"
    cross.attrs = {
        "description": f"cross-spectrum S_ab = conj(A) B of {name_a} and {name_b} in the physical time convention; real = cospectrum, imag = quadrature spectrum",
        "phase_convention": "arg(S_ab) = phase(b) - phase(a): positive when b leads a on the positive-frequency half-plane (same as scipy.signal.csd)",
    }
    cross[frequency_dim].attrs["units"] = "cycles day-1"
    cross[wavenumber_dim].attrs["units"] = "cycles per 360 degrees longitude"
    return cross


def segment_averaged_spectrum(
    data_a: xr.DataArray,
    data_b: xr.DataArray | None = None,
    *,
    segment_days: int = 96,
    overlap_days: int = 30,
    component: Literal["symmetric", "antisymmetric"] | None = None,
    num_harmonics: int = 3,
    detrend_segments: bool = True,
    window: Literal["tukey", "hann", "hanning", "hamming", "blackman"] = "tukey",
    window_pct: float = 0.10,
    average_over_lat: bool = True,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Segment-averaged space-time power (one field) or cross-spectrum (two fields).

    One preprocessing chain serves both, so a band partition of a cospectrum
    closes against the power and covariance computed the same way:

    1. optional equatorial ``component`` (symmetric or antisymmetric) of each field;
    2. full-record mean, linear trend and the first ``num_harmonics`` annual
       harmonics removed (``num_harmonics=0`` skips the harmonics);
    3. overlapping segments of ``segment_days`` with ``overlap_days``; each
       segment is trimmed to exactly ``segment_days / dt`` samples so that
       every sampling interval shares one frequency grid with
       ``df = 1 / segment_days`` (``segment_data`` itself returns one sample
       more, both endpoints inclusive);
    4. per segment, mean and trend removed again if ``detrend_segments``, a
       taper of the given ``window``, the space-time FFT, and
       ``conj(A) * B`` (or ``|A|^2``);
    5. average over segments and, if present and ``average_over_lat``, over
       ``lat_dim``.

    Normalisation: each bin is divided by ``(N_t N_x)^2 mean(w^2)`` with w the
    taper, so the sum over all (frequency, wavenumber) bins equals the
    (latitude-mean) covariance of the tapered segment anomalies corrected for
    the taper's variance loss. For one field the result is real; for two it is
    complex, with the real part the cospectrum.

    The parameters, the number of segments and the normalisation are recorded
    in ``attrs``.
    """
    def _prepare(data: xr.DataArray) -> xr.DataArray:
        field = data
        if component is not None:
            field = symmetric_antisymmetric_component(field, component, lat_dim=lat_dim)
        field = remove_mean_and_linear_trend(field, dim=time_dim)
        if num_harmonics:
            field = remove_harmonics_of_seasonal_cycle(field, num_harmonics=num_harmonics, time_dim=time_dim)
        return field.load()

    field_a = _prepare(data_a)
    field_b = _prepare(data_b) if data_b is not None else None

    dt_days = _coordinate_spacing_days(field_a[time_dim])
    n_time = int(round(segment_days / dt_days))
    n_lon = field_a.sizes[lon_dim]

    def _segments(field: xr.DataArray) -> list[xr.DataArray]:
        return [
            seg.isel({time_dim: slice(0, n_time)})
            for seg in segment_data(field, segment_days=segment_days, overlap_days=overlap_days, time_dim=time_dim)
            if seg.sizes[time_dim] >= n_time
        ]

    segments_a = _segments(field_a)
    if not segments_a:
        raise ValueError("record shorter than one segment")
    segments_b = _segments(field_b) if field_b is not None else None
    if segments_b is not None and len(segments_b) != len(segments_a):
        raise ValueError("the two fields produced different segment counts; they must share a time coordinate")

    ones = xr.DataArray(np.ones(n_time), dims=(time_dim,), coords={time_dim: segments_a[0][time_dim]})
    taper = apply_window(ones, dim=time_dim, window=window, pct=window_pct)
    mean_w2 = float((taper**2).mean())
    norm = 1.0 / ((n_time * n_lon) ** 2 * mean_w2)

    def _tapered_fft(seg: xr.DataArray) -> xr.DataArray:
        if detrend_segments:
            seg = remove_mean_and_linear_trend(seg, dim=time_dim)
        seg = apply_window(seg, dim=time_dim, window=window, pct=window_pct)
        return space_time_fft(seg, time_dim=time_dim, lon_dim=lon_dim)

    total = None
    for i, seg_a in enumerate(segments_a):
        ft_a = _tapered_fft(seg_a)
        if segments_b is None:
            spec = power_spectrum(ft_a) * norm
        else:
            ft_b = _tapered_fft(segments_b[i])
            spec = cross_spectrum(ft_a, ft_b) * norm
        total = spec if total is None else total + spec
    mean_spec = total / len(segments_a)
    if average_over_lat and lat_dim in mean_spec.dims:
        mean_spec = mean_spec.mean(lat_dim)

    name_a = data_a.name or "a"
    if data_b is None:
        mean_spec.name = f"{name_a}_power"
        long_name = "segment-averaged space-time power"
        units = f"({data_a.attrs.get('units', '1')})^2 per bin"
    else:
        name_b = data_b.name or "b"
        mean_spec.name = f"{name_a}_{name_b}_cross"
        long_name = "segment-averaged space-time cross-spectrum (real: cospectrum, imag: quadrature)"
        units = f"({data_a.attrs.get('units', '1')})({data_b.attrs.get('units', '1')}) per bin"
    mean_spec.attrs = {
        "long_name": long_name,
        "units": units,
        "component": component or "full",
        "segment_days": segment_days,
        "overlap_days": overlap_days,
        "n_segments": len(segments_a),
        "samples_per_segment": n_time,
        "sampling_interval_days": dt_days,
        "nyquist_frequency_cpd": 0.5 / dt_days,
        "num_seasonal_harmonics": num_harmonics,
        "detrend_segments": str(detrend_segments),
        "window": f"{window} pct={window_pct}",
        "normalisation": "conj(A)B/(N_t N_x)^2/mean(w^2); sums to the (lat-mean) covariance of the tapered segment anomalies",
        "latitude_range": f"{float(data_a[lat_dim].min()):g} to {float(data_a[lat_dim].max()):g}" if lat_dim in data_a.dims else "n/a",
    }
    return mean_spec


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

    diffs = np.diff(_days_since_start(values))
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
