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


def zonal_wind_frequency_shift(
    zonal_wavenumber: np.ndarray | float,
    background_u: float,
    *,
    latitude: float = 0.0,
) -> np.ndarray:
    """Return the frequency shift from a constant zonal wind in cycles day-1.

    Positive ``background_u`` is eastward.  The returned shift is signed with
    zonal wavenumber, so an eastward background flow raises eastward branches
    and lowers westward branches.
    """
    u = _coerce_constant_wind(background_u, name="background_u")
    _, perimeter = beta_parameters(latitude)
    return np.asarray(zonal_wavenumber, dtype=float) * u * SECONDS_PER_DAY / perimeter


def _validate_background_wind(
    background_u: float,
    background_v: float,
) -> tuple[float, float]:
    """Validate the constant-flow approximation supported by this module."""
    u = _coerce_constant_wind(background_u, name="background_u")
    v = _coerce_constant_wind(background_v, name="background_v")
    if v != 0.0:
        raise NotImplementedError(
            "background_v must be 0.0 m s-1: a nonzero constant meridional flow "
            "is not a stationary Matsuno beta-plane basic state and cannot be "
            "represented by this frequency-zonal-wavenumber formulation."
        )
    return u, v


def _coerce_constant_wind(value: float, *, name: str) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite scalar in m s-1") from exc
    if array.ndim != 0:
        raise TypeError(f"{name} must be a finite scalar in m s-1")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _apply_constant_background_wind(
    intrinsic_frequency: np.ndarray,
    zonal_wavenumber: np.ndarray | float,
    *,
    background_u: float,
    background_v: float,
    latitude: float,
) -> np.ndarray:
    """Convert resting-state Matsuno frequencies to observed frequencies."""
    u, _ = _validate_background_wind(background_u, background_v)
    if u == 0.0:
        return intrinsic_frequency
    return intrinsic_frequency + zonal_wind_frequency_shift(
        zonal_wavenumber,
        u,
        latitude=latitude,
    )


def kelvin_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """Kelvin-wave frequency in cycles day-1.

    ``background_u`` applies the constant-zonal-advection Doppler
    approximation. ``background_v`` must be zero; a meridional background
    flow is not representable by the one-dimensional Matsuno curves.
    """
    k = np.asarray(zonal_wavenumber, dtype=float)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    omega = np.sqrt(GRAVITY * equivalent_depth_m) * k_rad
    out = angular_frequency_to_cycles_per_day(omega)
    intrinsic = np.where(k > 0, out, np.nan)
    return _apply_constant_background_wind(
        intrinsic,
        k,
        background_u=background_u,
        background_v=background_v,
        latitude=latitude,
    )


def mrg_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """Mixed Rossby-gravity-wave frequency in cycles day-1.

    ``background_u`` applies the constant-zonal-advection Doppler
    approximation. ``background_v`` must be zero.
    """
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega = sqrt_gh * k_rad * (
            0.5 - 0.5 * np.sqrt(1.0 + (4.0 * beta / (k_rad * k_rad * sqrt_gh)))
        )
    out = angular_frequency_to_cycles_per_day(omega)
    intrinsic = np.where(k < 0, out, np.nan)
    return _apply_constant_background_wind(
        intrinsic,
        k,
        background_u=background_u,
        background_v=background_v,
        latitude=latitude,
    )


def eig0_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """n=0 eastward inertial-gravity frequency in cycles day-1.

    ``background_u`` applies the constant-zonal-advection Doppler
    approximation. ``background_v`` must be zero.
    """
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega = sqrt_gh * k_rad * (
            0.5 + 0.5 * np.sqrt(1.0 + (4.0 * beta / (k_rad * k_rad * sqrt_gh)))
        )
    out = angular_frequency_to_cycles_per_day(omega)
    intrinsic = np.where(k > 0, out, np.nan)
    return _apply_constant_background_wind(
        intrinsic,
        k,
        background_u=background_u,
        background_v=background_v,
        latitude=latitude,
    )


def er_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    n: int,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """Equatorial Rossby frequency in cycles day-1.

    ``background_u`` applies the constant-zonal-advection Doppler
    approximation. ``background_v`` must be zero.
    """
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        guess = -beta * k_rad / (k_rad * k_rad + (2.0 * n + 1.0) * beta / sqrt_gh)

    intrinsic = _frequency_from_dispersion_roots(
        k_rad,
        equivalent_depth_m,
        n=n,
        beta=beta,
        guess=np.abs(guess),
        valid=k < 0,
    )
    return _apply_constant_background_wind(
        intrinsic,
        k,
        background_u=background_u,
        background_v=background_v,
        latitude=latitude,
    )


def eig_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    n: int,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """Eastward inertial-gravity frequency in cycles day-1.

    ``background_u`` applies the constant-zonal-advection Doppler
    approximation. ``background_v`` must be zero.
    """
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)
    guess = np.sqrt((2.0 * n + 1.0) * beta * sqrt_gh + (k_rad**2) * GRAVITY * equivalent_depth_m)

    intrinsic = _frequency_from_dispersion_roots(
        k_rad,
        equivalent_depth_m,
        n=n,
        beta=beta,
        guess=guess,
        valid=k > 0,
    )
    return _apply_constant_background_wind(
        intrinsic,
        k,
        background_u=background_u,
        background_v=background_v,
        latitude=latitude,
    )


def wig_frequency(
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    n: int,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """Westward inertial-gravity frequency in cycles day-1.

    ``background_u`` applies the constant-zonal-advection Doppler
    approximation. ``background_v`` must be zero.
    """
    k = np.asarray(zonal_wavenumber, dtype=float)
    beta, _ = beta_parameters(latitude)
    k_rad = zonal_wavenumber_to_rad_per_meter(k, latitude=latitude)
    sqrt_gh = np.sqrt(GRAVITY * equivalent_depth_m)
    guess = np.sqrt((2.0 * n + 1.0) * beta * sqrt_gh + (k_rad**2) * GRAVITY * equivalent_depth_m)

    intrinsic = _frequency_from_dispersion_roots(
        k_rad,
        equivalent_depth_m,
        n=n,
        beta=beta,
        guess=guess,
        valid=k < 0,
    )
    return _apply_constant_background_wind(
        intrinsic,
        k,
        background_u=background_u,
        background_v=background_v,
        latitude=latitude,
    )


def profile_curve_frequency(
    curve_name: str,
    zonal_wavenumber: np.ndarray | float,
    equivalent_depth_m: float,
    *,
    meridional_mode_number: int | None = None,
    latitude: float = 0.0,
    background_u: float = 0.0,
    background_v: float = 0.0,
) -> np.ndarray:
    """Evaluate one of the named profile curves in observed cycles day-1.

    With the default resting basic state (``background_u=background_v=0``),
    this returns the standard Matsuno curves. A nonzero ``background_u``
    applies ``f_observed = f_Matsuno + U * s * 86400 / perimeter``, where
    ``s`` is signed global zonal wavenumber and ``perimeter`` is in metres.
    A nonzero ``background_v`` is unsupported because the current
    frequency-zonal-wavenumber representation has no meridional wavenumber
    or meridional eigenproblem.
    """
    name = curve_name.lower()
    if name == "kelvin":
        return kelvin_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            latitude=latitude,
            background_u=background_u,
            background_v=background_v,
        )
    if name == "mrg":
        return mrg_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            latitude=latitude,
            background_u=background_u,
            background_v=background_v,
        )
    if name == "eig0":
        return eig0_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            latitude=latitude,
            background_u=background_u,
            background_v=background_v,
        )

    if meridional_mode_number is None:
        raise ValueError(f"curve {curve_name!r} requires a meridional mode number")

    if name == "er":
        return er_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            n=meridional_mode_number,
            latitude=latitude,
            background_u=background_u,
            background_v=background_v,
        )
    if name == "eig":
        return eig_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            n=meridional_mode_number,
            latitude=latitude,
            background_u=background_u,
            background_v=background_v,
        )
    if name == "wig":
        return wig_frequency(
            zonal_wavenumber,
            equivalent_depth_m,
            n=meridional_mode_number,
            latitude=latitude,
            background_u=background_u,
            background_v=background_v,
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
