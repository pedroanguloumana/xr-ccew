"""Sampling-rate helpers: Nyquist frequency and profile frequency ceilings.

A wave profile is defined in continuous (wavenumber, frequency) space, but a
sampled record only resolves frequencies below its Nyquist frequency, and
power close to Nyquist is unreliable. :func:`apply_frequency_ceiling`
truncates a profile at a configurable fraction of the Nyquist frequency of the
record it will be applied to, so the truncation is derived from the data's own
time coordinate rather than hard-coded per call site. The n=0 EIG profile is
the usual case: its high-wavenumber tail steepens towards two-day periods,
which daily data cannot resolve, while its low-wavenumber portion is fine.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .profiles import WaveProfile, wave_profile
from .spectral import _coordinate_spacing_days

TimeLike = xr.DataArray | np.ndarray | Sequence


def sampling_interval_days(time: TimeLike) -> float:
    """Sampling interval in days of a regularly spaced time coordinate.

    Datetime coordinates are interpreted in days; numeric coordinates are
    assumed to be in days already. Irregular spacing raises ``ValueError``.
    """
    coord = time if isinstance(time, xr.DataArray) else xr.DataArray(
        np.asarray(time), dims=("time",), name="time"
    )
    return _coordinate_spacing_days(coord)


def nyquist_frequency(time: TimeLike) -> float:
    """Nyquist frequency, in cycles per day, of a regularly spaced time coordinate."""
    return 0.5 / sampling_interval_days(time)


def apply_frequency_ceiling(
    profile: str | WaveProfile,
    time: TimeLike,
    *,
    fraction: float = 0.8,
) -> WaveProfile:
    """Return ``profile`` with ``frequency_max`` capped at ``fraction`` x Nyquist.

    The ceiling is ``fraction * nyquist_frequency(time)``. For daily data the
    default ``fraction=0.8`` gives 0.40 cycles per day (a 2.5-day period);
    ``fraction=0.9`` gives 0.45. A profile whose ``frequency_max`` is already
    below the ceiling is returned unchanged apart from ``frequency_ceiling``
    being recorded. A profile that lies entirely above the ceiling cannot be
    resolved by the record and raises ``ValueError`` rather than silently
    returning an empty band.

    The ceiling is a statement about frequency, not wavenumber: for a
    dispersive profile the wavenumber at which the band crosses the ceiling
    moves with equivalent depth, so callers should report the retained
    wavenumber range from the resulting filter mask rather than express the
    truncation as a wavenumber cutoff.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    resolved = wave_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(resolved, WaveProfile):
        raise TypeError(f"expected a profile name or WaveProfile, got {type(profile)!r}")

    ceiling = float(fraction * nyquist_frequency(time))
    if resolved.frequency_min >= ceiling:
        raise ValueError(
            f"{resolved.name}: frequency_min ({resolved.frequency_min:g} cpd) is at or above "
            f"the ceiling ({ceiling:g} cpd = {fraction:g} x Nyquist); the band is not "
            "resolvable at this sampling interval"
        )
    return resolved.with_updates(
        frequency_max=min(resolved.frequency_max, ceiling),
        frequency_ceiling=ceiling,
    )
