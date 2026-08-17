"""Matsuno dispersion utilities used by the built-in wave profiles."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6.371008e6
GRAVITY = 9.80665
EARTH_ANGULAR_SPEED = 7.292e-05
SECONDS_PER_DAY = 24.0 * 60.0 * 60.0


def beta_parameters(latitude: float = 0.0) -> tuple[float, float]:
    """Return beta-plane parameter and latitude-circle perimeter."""
    latitude_rad = np.deg2rad(abs(latitude))
    beta = 2.0 * EARTH_ANGULAR_SPEED * np.cos(latitude_rad) / EARTH_RADIUS_M
    perimeter = 2.0 * np.pi * EARTH_RADIUS_M * np.cos(latitude_rad)
    return beta, perimeter


def zonal_wavenumber_to_rad_per_meter(
    zonal_wavenumber: np.ndarray | float,
    *,
    latitude: float = 0.0,
) -> np.ndarray:
    """Convert global zonal wavenumber to angular wavenumber in rad m-1."""
    _, perimeter = beta_parameters(latitude)
    return 2.0 * np.pi * np.asarray(zonal_wavenumber, dtype=float) / perimeter


def angular_frequency_to_cycles_per_day(angular_frequency: np.ndarray | float) -> np.ndarray:
    """Convert angular frequency in rad s-1 to cycles day-1."""
    return np.asarray(angular_frequency, dtype=float) * SECONDS_PER_DAY / (2.0 * np.pi)


def kelvin_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    latitude: float = 0.0,
) -> np.ndarray:
    """Kelvin-wave frequency in cycles day-1."""
    k = np.asarray(zonal_wavenumber, dtype=float)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    omega = np.sqrt(GRAVITY * equivalent_depth_m) * k_rad
    out = angular_frequency_to_cycles_per_day(omega)
    return np.where(k > 0, out, np.nan)


def mrg_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    latitude: float = 0.0,
) -> np.ndarray:
    """Mixed Rossby-gravity-wave frequency in cycles day-1."""
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega = sqrt_gh * k_rad * (
            0.5 - 0.5 * np.sqrt(1.0 + (4.0 * beta / (k_rad * k_rad * sqrt_gh)))
        )
    out = angular_frequency_to_cycles_per_day(omega)
    return np.where(k < 0, out, np.nan)


def eig0_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    latitude: float = 0.0,
) -> np.ndarray:
    """n=0 eastward inertial-gravity frequency in cycles day-1."""
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega = sqrt_gh * k_rad * (
            0.5 + 0.5 * np.sqrt(1.0 + (4.0 * beta / (k_rad * k_rad * sqrt_gh)))
        )
    out = angular_frequency_to_cycles_per_day(omega)
    return np.where(k > 0, out, np.nan)


def er_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    n: int,
    latitude: float = 0.0,
) -> np.ndarray:
    """Equatorial Rossby frequency in cycles day-1."""
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        guess = -beta * k_rad / (k_rad * k_rad + (2.0 * n + 1.0) * beta / sqrt_gh)

    return _frequency_from_dispersion_roots(
        k_rad,
        equivalent_depth_m,
        n=n,
        beta=beta,
        guess=np.abs(guess),
        valid=k < 0,
    )


def eig_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    n: int,
    latitude: float = 0.0,
) -> np.ndarray:
    """Eastward inertial-gravity frequency in cycles day-1."""
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)
    guess = np.sqrt((2.0 * n + 1.0) * beta * sqrt_gh + (k_rad**2) * GRAVITY * equivalent_depth_m)

    return _frequency_from_dispersion_roots(
        k_rad,
        equivalent_depth_m,
        n=n,
        beta=beta,
        guess=guess,
        valid=k > 0,
    )


def wig_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    n: int,
    latitude: float = 0.0,
) -> np.ndarray:
    """Westward inertial-gravity frequency in cycles day-1."""
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)
    guess = np.sqrt((2.0 * n + 1.0) * beta * sqrt_gh + (k_rad**2) * GRAVITY * equivalent_depth_m)

    return _frequency_from_dispersion_roots(
        k_rad,
        equivalent_depth_m,
        n=n,
        beta=beta,
        guess=guess,
        valid=k < 0,
    )


def profile_curve_frequency(
    curve_name: str,
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    meridional_mode_number: int | None = None,
    latitude: float = 0.0,
) -> np.ndarray:
    """Evaluate one of the named profile curves in cycles day-1."""
    name = curve_name.lower()
    if name == "kelvin":
        return kelvin_frequency(zonal_wavenumber, equivalent_depth_m, latitude=latitude)
    if name == "mrg":
        return mrg_frequency(zonal_wavenumber, equivalent_depth_m, latitude=latitude)
    if name == "eig0":
        return eig0_frequency(zonal_wavenumber, equivalent_depth_m, latitude=latitude)

    if meridional_mode_number is None:
        raise ValueError(f"curve {curve_name!r} requires a meridional mode number")

    if name == "er":
        return er_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            n=meridional_mode_number,
            latitude=latitude,
        )
    if name == "eig":
        return eig_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            n=meridional_mode_number,
            latitude=latitude,
        )
    if name == "wig":
        return wig_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            n=meridional_mode_number,
            latitude=latitude,
        )

    raise ValueError(f"unknown curve name {curve_name!r}")


def _frequency_from_dispersion_roots(
    k_rad: np.ndarray,
    equivalent_depth_m: float,
    *,
    n: int,
    beta: float,
    guess: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    out = np.full(np.shape(k_rad), np.nan, dtype=float)
    flat_k = np.ravel(k_rad)
    flat_guess = np.ravel(np.asarray(guess, dtype=float))
    flat_valid = np.ravel(np.asarray(valid, dtype=bool))
    flat_out = np.ravel(out)

    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)
    gh = GRAVITY * equivalent_depth_m

    for i, (ki, target, is_valid) in enumerate(zip(flat_k, flat_guess, flat_valid)):
        if not is_valid or not np.isfinite(ki) or ki == 0.0:
            continue

        coeffs = [
            1.0,
            0.0,
            -gh * (ki * ki + beta * (2.0 * n + 1.0) / sqrt_gh),
            -ki * beta * gh,
        ]
        roots = np.roots(coeffs)
        real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-10)].real
        positive_roots = real_roots[real_roots > 0.0]
        if positive_roots.size == 0:
            continue
        flat_out[i] = positive_roots[np.argmin(np.abs(positive_roots - target))]

    return angular_frequency_to_cycles_per_day(out)

