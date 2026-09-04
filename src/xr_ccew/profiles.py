"""Wave profile definitions and lookup helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class WaveProfile:
    """Editable definition of a wave family in frequency-wavenumber space."""

    name: str
    k_min: float
    k_max: float
    frequency_min: float
    frequency_max: float
    equivalent_depth_min: float
    equivalent_depth_max: float
    symmetry: str
    direction: str
    curve_name: str | None = None
    meridional_mode_number: int | None = None
    # Optional convex polygon of (zonal_wavenumber, frequency) vertices that
    # further restricts the band, for regions that are neither rectangles nor
    # dispersion-curve bands (e.g. the sloped TD-type box of Kiladis et al.
    # 2006). The filter mask is the intersection of this polygon with the
    # k/frequency bounds above, so apply_frequency_ceiling still truncates it.
    wavenumber_frequency_polygon: tuple[tuple[float, float], ...] | None = None
    aliases: tuple[str, ...] = ()
    description: str = ""
    references: tuple[str, ...] = ()
    # Set by :func:`xr_ccew.apply_frequency_ceiling`: the Nyquist-derived
    # ceiling (cycles day-1) that frequency_max was truncated to. Recorded on
    # the profile so provenance metadata can state which ceiling was applied.
    frequency_ceiling: float | None = None

    def __post_init__(self) -> None:
        if self.k_min > self.k_max:
            raise ValueError(f"{self.name}: k_min must be <= k_max")
        if self.frequency_min > self.frequency_max:
            raise ValueError(f"{self.name}: frequency_min must be <= frequency_max")
        if self.equivalent_depth_min > self.equivalent_depth_max:
            raise ValueError(
                f"{self.name}: equivalent_depth_min must be <= equivalent_depth_max"
            )
        if self.frequency_ceiling is not None and self.frequency_max > self.frequency_ceiling:
            raise ValueError(
                f"{self.name}: frequency_max ({self.frequency_max}) exceeds the recorded "
                f"frequency_ceiling ({self.frequency_ceiling})"
            )
        if self.symmetry not in {"symmetric", "antisymmetric", "both"}:
            raise ValueError(
                f"{self.name}: symmetry must be 'symmetric', 'antisymmetric', or 'both'"
            )
        if self.direction not in {"eastward", "westward", "both"}:
            raise ValueError(
                f"{self.name}: direction must be 'eastward', 'westward', or 'both'"
            )
        polygon = _validate_polygon(self.name, self.wavenumber_frequency_polygon)
        object.__setattr__(self, "wavenumber_frequency_polygon", polygon)

    @property
    def frequency_bounds(self) -> tuple[float, float]:
        return self.frequency_min, self.frequency_max

    @property
    def wavenumber_bounds(self) -> tuple[float, float]:
        return self.k_min, self.k_max

    @property
    def equivalent_depth_bounds(self) -> tuple[float, float]:
        return self.equivalent_depth_min, self.equivalent_depth_max

    def with_updates(self, **updates: object) -> "WaveProfile":
        """Return a modified copy of the profile."""
        return replace(self, **updates)

    def as_dict(self) -> dict[str, object]:
        """Every parameter as a plain dict, for provenance metadata."""
        return asdict(self)


def _validate_polygon(
    name: str, polygon: object
) -> tuple[tuple[float, float], ...] | None:
    """Coerce ``polygon`` to float vertex pairs and require a convex shape."""
    if polygon is None:
        return None
    try:
        vertices = tuple((float(k), float(f)) for k, f in polygon)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: wavenumber_frequency_polygon must be a sequence of "
            "(wavenumber, frequency) pairs"
        ) from exc
    if len(vertices) < 3:
        raise ValueError(f"{name}: wavenumber_frequency_polygon needs at least 3 vertices")
    if not all(np.isfinite(value) for vertex in vertices for value in vertex):
        raise ValueError(f"{name}: wavenumber_frequency_polygon vertices must be finite")

    # Convex iff every consecutive edge pair turns the same way (collinear
    # triples are allowed). This also rejects bow-tie self-intersections.
    turns: set[bool] = set()
    n = len(vertices)
    for i in range(n):
        (x0, y0), (x1, y1), (x2, y2) = vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cross) > 1e-12:
            turns.add(cross > 0)
    if len(turns) != 1:
        raise ValueError(
            f"{name}: wavenumber_frequency_polygon must be a convex, non-degenerate polygon"
        )
    return vertices


REFERENCE_WK99 = (
    "Wheeler, M. C., and Kiladis, G. N. (1999), Convectively coupled "
    "equatorial waves: Analysis of clouds and temperature in the "
    "wavenumber-frequency domain."
)

REFERENCE_KILADIS06 = (
    "Kiladis, G. N., Thorncroft, C. D., and Hall, N. M. J. (2006), "
    "Three-dimensional structure and dynamics of African easterly waves. "
    "Part I: Observations, J. Atmos. Sci., 63, 2212-2230."
)

REFERENCE_KILADIS09 = (
    "Kiladis, G. N., Wheeler, M. C., Haertel, P. T., Straub, K. H., and "
    "Roundy, P. E. (2009), Convectively coupled equatorial waves, "
    "Rev. Geophys., 47, RG2003."
)


KELVIN = WaveProfile(
    name="KW",
    aliases=("Kelvin", "Kelvin wave"),
    k_min=1,
    k_max=14,
    frequency_min=1.0 / 30.0,
    frequency_max=1.0 / 2.5,
    equivalent_depth_min=8,
    equivalent_depth_max=90,
    symmetry="symmetric",
    direction="eastward",
    curve_name="kelvin",
    description="Convectively coupled Kelvin wave.",
    references=(REFERENCE_WK99,),
)

EQUATORIAL_ROSSBY_N0 = WaveProfile(
    name="n=0 ER",
    aliases=("ER0", "n0 ER", "equatorial rossby n0"),
    k_min=2,
    k_max=14,
    frequency_min=0.0,
    frequency_max=0.55,
    equivalent_depth_min=8,
    equivalent_depth_max=90,
    symmetry="antisymmetric",
    direction="eastward",
    curve_name="er",
    meridional_mode_number=0,
    description="n=0 equatorial Rossby profile from the reference project.",
    references=(REFERENCE_WK99,),
)

EQUATORIAL_ROSSBY_N1 = WaveProfile(
    name="n=1 ER",
    aliases=("ER", "ER1", "Rossby", "equatorial rossby", "n1 ER"),
    k_min=-10,
    k_max=-1,
    frequency_min=1.0 / 30.0,
    frequency_max=1.0,
    equivalent_depth_min=8,
    equivalent_depth_max=90,
    symmetry="symmetric",
    direction="westward",
    curve_name="er",
    meridional_mode_number=1,
    description="n=1 equatorial Rossby wave.",
    references=(REFERENCE_WK99,),
)

MIXED_ROSSBY_GRAVITY = WaveProfile(
    name="MRG",
    aliases=("mixed rossby gravity", "mixed rossby-gravity"),
    k_min=-10,
    k_max=-1,
    frequency_min=0.1,
    frequency_max=1.0 / 3.0,
    equivalent_depth_min=8,
    equivalent_depth_max=90,
    symmetry="antisymmetric",
    direction="westward",
    curve_name="mrg",
    description="Mixed Rossby-gravity wave.",
    references=(REFERENCE_WK99,),
)

MJO = WaveProfile(
    name="MJO",
    aliases=("madden julian oscillation", "madden-julian oscillation"),
    k_min=1,
    k_max=5,
    frequency_min=1.0 / 96.0,
    frequency_max=1.0 / 30.0,
    equivalent_depth_min=-np.inf,
    equivalent_depth_max=np.inf,
    symmetry="both",
    direction="eastward",
    curve_name=None,
    description="Madden-Julian oscillation band.",
    references=(REFERENCE_WK99,),
)

EASTWARD_INERTIAL_GRAVITY_N0 = WaveProfile(
    name="n=0 EIG",
    aliases=("EIG0", "n0 EIG", "eastward inertial gravity n0"),
    k_min=1,
    k_max=14,
    frequency_min=1.0 / 4.0,
    frequency_max=0.55,
    equivalent_depth_min=12,
    equivalent_depth_max=50,
    symmetry="antisymmetric",
    direction="eastward",
    curve_name="eig0",
    meridional_mode_number=0,
    description="n=0 eastward inertial-gravity wave.",
    references=(REFERENCE_WK99,),
)

WESTWARD_INERTIAL_GRAVITY_N2 = WaveProfile(
    name="n=2 WIG",
    aliases=("WIG2", "n2 WIG", "westward inertial gravity n2"),
    k_min=-14,
    k_max=-1,
    frequency_min=0.0,
    frequency_max=1.0,
    equivalent_depth_min=12,
    equivalent_depth_max=90,
    symmetry="antisymmetric",
    direction="westward",
    curve_name="wig",
    meridional_mode_number=2,
    description="n=2 westward inertial-gravity wave.",
    references=(REFERENCE_WK99,),
)

WESTWARD_INERTIAL_GRAVITY_N1 = WaveProfile(
    name="n=1 WIG",
    aliases=("WIG", "WIG1", "n1 WIG", "westward inertial gravity", "westward ig"),
    k_min=-14,
    k_max=-1,
    frequency_min=0.0,
    frequency_max=1.0,
    equivalent_depth_min=12,
    equivalent_depth_max=90,
    symmetry="symmetric",
    direction="westward",
    curve_name="wig",
    meridional_mode_number=1,
    description="n=1 westward inertial-gravity wave.",
    references=(REFERENCE_WK99,),
)

TD_TYPE = WaveProfile(
    name="TD-type",
    aliases=(
        "TD",
        "TD type",
        "TD wave",
        "TD-type wave",
        "tropical depression",
        "tropical depression type",
        "tropical depression-type",
        "easterly wave",
        "easterly waves",
    ),
    k_min=-20,
    k_max=-6,
    frequency_min=2.0 / 15.0,
    frequency_max=0.5,
    equivalent_depth_min=-np.inf,
    equivalent_depth_max=np.inf,
    symmetry="both",
    direction="westward",
    curve_name=None,
    # Heavy box of Kiladis et al. (2006), Fig. 1: a parallelogram spanning
    # westward wavenumbers 6-20 whose sloped edges drop 1/84 cpd per
    # wavenumber, so the band covers 2-3.3-day periods at k=-20 and
    # 3-7.5-day periods at k=-6.
    wavenumber_frequency_polygon=(
        (-20, 0.3),
        (-6, 2.0 / 15.0),
        (-6, 1.0 / 3.0),
        (-20, 0.5),
    ),
    description=(
        "Tropical depression-type (easterly wave) band of Kiladis et al. "
        "(2006): a parallelogram in wavenumber-frequency space over westward "
        "wavenumbers 6-20, from 0.3-0.5 cpd at k=-20 to 0.133-0.333 cpd at "
        "k=-6, with no dispersion-curve bounds."
    ),
    references=(REFERENCE_KILADIS06, REFERENCE_KILADIS09),
)

BUILTIN_PROFILES: tuple[WaveProfile, ...] = (
    KELVIN,
    EQUATORIAL_ROSSBY_N0,
    EQUATORIAL_ROSSBY_N1,
    MIXED_ROSSBY_GRAVITY,
    MJO,
    EASTWARD_INERTIAL_GRAVITY_N0,
    WESTWARD_INERTIAL_GRAVITY_N2,
    WESTWARD_INERTIAL_GRAVITY_N1,
    TD_TYPE,
)

PROFILE_GROUPS: dict[str, tuple[WaveProfile, ...]] = {
    "all": BUILTIN_PROFILES,
    "rossby": (EQUATORIAL_ROSSBY_N0, EQUATORIAL_ROSSBY_N1),
    "equatorialrossby": (EQUATORIAL_ROSSBY_N0, EQUATORIAL_ROSSBY_N1),
    "er": (EQUATORIAL_ROSSBY_N0, EQUATORIAL_ROSSBY_N1),
    "inertialgravity": (
        EASTWARD_INERTIAL_GRAVITY_N0,
        WESTWARD_INERTIAL_GRAVITY_N1,
        WESTWARD_INERTIAL_GRAVITY_N2,
    ),
    "ig": (
        EASTWARD_INERTIAL_GRAVITY_N0,
        WESTWARD_INERTIAL_GRAVITY_N1,
        WESTWARD_INERTIAL_GRAVITY_N2,
    ),
}


def list_wave_profiles(*, include_groups: bool = False) -> tuple[str, ...]:
    """Return available built-in profile names."""
    names = tuple(profile.name for profile in BUILTIN_PROFILES)
    if include_groups:
        return names + tuple(sorted(PROFILE_GROUPS))
    return names


def wave_profile(name: str) -> WaveProfile:
    """Return one built-in profile by name or alias."""
    key = _normalize_name(name)
    index = _profile_index()
    try:
        return index[key]
    except KeyError as exc:
        available = ", ".join(list_wave_profiles(include_groups=True))
        raise KeyError(f"unknown wave profile {name!r}; available profiles/groups: {available}") from exc


def wave_profiles(*items: str | WaveProfile | Iterable[str | WaveProfile]) -> tuple[WaveProfile, ...]:
    """Return one or more profiles, expanding known group names."""
    if len(items) == 1 and _is_profile_iterable(items[0]):
        items = tuple(items[0])  # type: ignore[assignment]

    out: list[WaveProfile] = []
    for item in items:
        if isinstance(item, WaveProfile):
            out.append(item)
            continue
        if not isinstance(item, str):
            raise TypeError(f"expected profile names or WaveProfile objects, got {type(item)!r}")

        group = PROFILE_GROUPS.get(_normalize_name(item))
        if group is not None:
            out.extend(group)
        else:
            out.append(wave_profile(item))

    return tuple(out)


def _profile_index() -> dict[str, WaveProfile]:
    index: dict[str, WaveProfile] = {}
    for profile in BUILTIN_PROFILES:
        for name in (profile.name, *profile.aliases):
            index[_normalize_name(name)] = profile
    return index


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_profile_iterable(item: object) -> bool:
    return isinstance(item, Iterable) and not isinstance(item, (str, bytes, WaveProfile))

