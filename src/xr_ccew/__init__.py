"""Xarray tools for convectively coupled equatorial wave analysis."""

from .filtering import add_conjugate_partner, apply_filter_mask, filter_field, make_filter_mask
from .profiles import WaveProfile, list_wave_profiles, wave_profile, wave_profiles
from .nyquist import apply_frequency_ceiling, nyquist_frequency, sampling_interval_days
from .spectral import (
    add_time_days,
    apply_window,
    background_spectrum,
    cross_spectrum,
    power_spectrum,
    segment_averaged_spectrum,
    remove_harmonics_of_seasonal_cycle,
    remove_mean_and_linear_trend,
    segment_data,
    space_time_fft,
    space_time_ifft,
    symmetric_antisymmetric_component,
)
from .synthetic import synthetic_wave

__all__ = [
    "WaveProfile",
    "add_conjugate_partner",
    "add_time_days",
    "apply_frequency_ceiling",
    "apply_filter_mask",
    "apply_window",
    "background_spectrum",
    "cross_spectrum",
    "filter_field",
    "list_wave_profiles",
    "make_filter_mask",
    "nyquist_frequency",
    "power_spectrum",
    "remove_harmonics_of_seasonal_cycle",
    "remove_mean_and_linear_trend",
    "sampling_interval_days",
    "segment_averaged_spectrum",
    "segment_data",
    "space_time_fft",
    "space_time_ifft",
    "symmetric_antisymmetric_component",
    "synthetic_wave",
    "wave_profile",
    "wave_profiles",
]

