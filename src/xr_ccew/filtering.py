"""Profile-based filtering in space-time spectral coordinates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import xarray as xr

from .matsuno import profile_curve_frequency
from .profiles import WaveProfile, wave_profiles
from .spectral import space_time_fft, space_time_ifft, symmetric_antisymmetric_component


def make_filter_mask(
    spectrum: xr.DataArray,
    profiles: str | WaveProfile | Iterable[str | WaveProfile],
    *,
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
    include_conjugates: bool = True,
) -> xr.DataArray:
    """Create a boolean filter mask from one or more wave profiles."""
    resolved = _coerce_profiles(profiles)
    if not resolved:
        raise ValueError("at least one wave profile is required")
    if frequency_dim not in spectrum.dims or wavenumber_dim not in spectrum.dims:
        raise ValueError(
            f"spectrum must include dims {frequency_dim!r} and {wavenumber_dim!r}; "
            f"got {spectrum.dims}"
        )

    frequency = spectrum[frequency_dim]
    wavenumber = spectrum[wavenumber_dim]
    combined = xr.zeros_like(
        xr.broadcast(frequency, wavenumber)[0],
        dtype=bool,
    )

    for profile in resolved:
        combined = combined | _profile_mask(
            frequency,
            wavenumber,
            profile,
            frequency_dim=frequency_dim,
            wavenumber_dim=wavenumber_dim,
        )

    if include_conjugates:
        combined = add_conjugate_partner(
            combined,
            frequency_dim=frequency_dim,
            wavenumber_dim=wavenumber_dim,
        )

    combined.name = "filter_mask"
    combined.attrs["wave_profiles"] = ", ".join(profile.name for profile in resolved)
    return combined


def apply_filter_mask(
    spectrum: xr.DataArray,
    mask: xr.DataArray,
    *,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """Apply a boolean mask to a spectral field."""
    return spectrum.where(mask, other=fill_value)


def filter_field(
    data: xr.DataArray | xr.Dataset,
    profiles: str | WaveProfile | Iterable[str | WaveProfile],
    *,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
    apply_profile_symmetry: bool = True,
    shift: bool = True,
) -> xr.DataArray | xr.Dataset:
    """Filter a field by one or more wave profiles.

    When profile symmetry is applied, each profile filters the corresponding
    symmetric, antisymmetric, or full field component before the pieces are
    summed back together.
    """
    resolved = _coerce_profiles(profiles)
    if isinstance(data, xr.Dataset):
        return data.map(
            lambda da: filter_field(
                da,
                resolved,
                time_dim=time_dim,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
                frequency_dim=frequency_dim,
                wavenumber_dim=wavenumber_dim,
                apply_profile_symmetry=apply_profile_symmetry,
                shift=shift,
            )
            if time_dim in da.dims and lon_dim in da.dims
            else da
        )

    if not resolved:
        raise ValueError("at least one wave profile is required")

    groups: dict[str, list[WaveProfile]] = defaultdict(list)
    for profile in resolved:
        key = profile.symmetry if apply_profile_symmetry else "both"
        groups[key].append(profile)

    filtered_parts: list[xr.DataArray] = []
    for symmetry, group_profiles in groups.items():
        if symmetry == "both":
            component = data
        else:
            if lat_dim not in data.dims:
                raise ValueError(
                    f"profile symmetry requires lat_dim={lat_dim!r}; got dims={data.dims}"
                )
            component = symmetric_antisymmetric_component(data, symmetry, lat_dim=lat_dim)

        spectrum = space_time_fft(
            component,
            time_dim=time_dim,
            lon_dim=lon_dim,
            frequency_dim=frequency_dim,
            wavenumber_dim=wavenumber_dim,
            shift=shift,
        )
        mask = make_filter_mask(
            spectrum,
            group_profiles,
            frequency_dim=frequency_dim,
            wavenumber_dim=wavenumber_dim,
            include_conjugates=True,
        )
        filtered_spectrum = apply_filter_mask(spectrum, mask)
        filtered_parts.append(
            space_time_ifft(
                filtered_spectrum,
                template=component,
                time_dim=time_dim,
                lon_dim=lon_dim,
                frequency_dim=frequency_dim,
                wavenumber_dim=wavenumber_dim,
                shift=shift,
                real=True,
            )
        )

    out = filtered_parts[0]
    for part in filtered_parts[1:]:
        out = out + part
    out.name = data.name
    out.attrs.update(data.attrs)
    out.attrs["filtered_wave_profiles"] = ", ".join(profile.name for profile in resolved)
    return out


def add_conjugate_partner(
    mask: xr.DataArray,
    *,
    frequency_dim: str = "frequency",
    wavenumber_dim: str = "zonal_wavenumber",
) -> xr.DataArray:
    """Add the (-frequency, -wavenumber) partner needed for real-valued inverse FFTs."""
    if frequency_dim not in mask.dims or wavenumber_dim not in mask.dims:
        raise ValueError(
            f"mask dims are {mask.dims}, expected {frequency_dim!r} and {wavenumber_dim!r}"
        )

    frequency = np.asarray(mask[frequency_dim].values)
    wavenumber = np.asarray(mask[wavenumber_dim].values)
    frequency_partner = np.argmin(np.abs(frequency[None, :] + frequency[:, None]), axis=1)
    wavenumber_partner = np.argmin(np.abs(wavenumber[None, :] + wavenumber[:, None]), axis=1)

    base = np.asarray(mask.fillna(False).data, dtype=bool)
    f_axis = mask.dims.index(frequency_dim)
    k_axis = mask.dims.index(wavenumber_dim)

    moved = np.moveaxis(base, (f_axis, k_axis), (0, 1))
    partner = moved[np.ix_(frequency_partner, wavenumber_partner)]
    combined = moved | partner
    restored = np.moveaxis(combined, (0, 1), (f_axis, k_axis))

    return xr.DataArray(restored, coords=mask.coords, dims=mask.dims, name=mask.name, attrs=mask.attrs)


def _profile_mask(
    frequency: xr.DataArray,
    wavenumber: xr.DataArray,
    profile: WaveProfile,
    *,
    frequency_dim: str,
    wavenumber_dim: str,
) -> xr.DataArray:
    freq_grid, k_grid = xr.broadcast(frequency, wavenumber)
    rectangular = (
        (freq_grid >= profile.frequency_min)
        & (freq_grid <= profile.frequency_max)
        & (k_grid >= profile.k_min)
        & (k_grid <= profile.k_max)
    )

    curve_mask = _profile_curve_mask(
        frequency,
        wavenumber,
        profile,
        frequency_dim=frequency_dim,
        wavenumber_dim=wavenumber_dim,
    )
    return rectangular & curve_mask


def _profile_curve_mask(
    frequency: xr.DataArray,
    wavenumber: xr.DataArray,
    profile: WaveProfile,
    *,
    frequency_dim: str,
    wavenumber_dim: str,
) -> xr.DataArray:
    if profile.curve_name is None:
        return xr.ones_like(xr.broadcast(frequency, wavenumber)[0], dtype=bool)
    if not (
        np.isfinite(profile.equivalent_depth_min)
        and np.isfinite(profile.equivalent_depth_max)
    ):
        return xr.ones_like(xr.broadcast(frequency, wavenumber)[0], dtype=bool)

    k_values = np.asarray(wavenumber.values, dtype=float)
    lower = profile_curve_frequency(
        profile.curve_name,
        k_values,
        profile.equivalent_depth_min,
        meridional_mode_number=profile.meridional_mode_number,
    )
    upper = profile_curve_frequency(
        profile.curve_name,
        k_values,
        profile.equivalent_depth_max,
        meridional_mode_number=profile.meridional_mode_number,
    )
    finite = np.isfinite(lower) & np.isfinite(upper)

    in_k_bounds = (k_values >= profile.k_min) & (k_values <= profile.k_max)
    if not np.any(finite & in_k_bounds):
        return xr.ones_like(xr.broadcast(frequency, wavenumber)[0], dtype=bool)

    f_min = xr.DataArray(
        np.minimum(lower, upper),
        dims=(wavenumber_dim,),
        coords={wavenumber_dim: wavenumber},
    )
    f_max = xr.DataArray(
        np.maximum(lower, upper),
        dims=(wavenumber_dim,),
        coords={wavenumber_dim: wavenumber},
    )
    finite_da = xr.DataArray(
        finite,
        dims=(wavenumber_dim,),
        coords={wavenumber_dim: wavenumber},
    )
    freq_grid, f_min_grid = xr.broadcast(frequency, f_min)
    _, f_max_grid = xr.broadcast(frequency, f_max)
    _, finite_grid = xr.broadcast(frequency, finite_da)
    return finite_grid & (freq_grid >= f_min_grid) & (freq_grid <= f_max_grid)


def _coerce_profiles(
    profiles: str | WaveProfile | Iterable[str | WaveProfile],
) -> tuple[WaveProfile, ...]:
    if isinstance(profiles, (str, WaveProfile)):
        return wave_profiles(profiles)
    return wave_profiles(*profiles)

