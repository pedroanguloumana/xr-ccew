"""Wave profile definitions and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    aliases: tuple[str, ...] = ()
    description: str = ""
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.k_min > self.k_max:
            raise ValueError(f"{self.name}: k_min must be <= k_max")
        if self.frequency_min > self.frequency_max:
            raise ValueError(f"{self.name}: frequency_min must be <= frequency_max")
        if self.equivalent_depth_min > self.equivalent_depth_max:
            raise ValueError(
                f"{self.name}: equivalent_depth_min must be <= equivalent_depth_max"
            )
        if self.symmetry not in {"symmetric", "antisymmetric", "both"}:
            raise ValueError(
                f"{self.name}: symmetry must be 'symmetric', 'antisymmetric', or 'both'"
            )
        if self.direction not in {"eastward", "westward", "both"}:
            raise ValueError(
                f"{self.name}: direction must be 'eastward', 'westward', or 'both'"
            )

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


REFERENCE_WK99 = (
    "Wheeler, M. C., and Kiladis, G. N. (1999), Convectively coupled "
    "equatorial waves: Analysis of clouds and temperature in the "
    "wavenumber-frequency domain."
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

BUILTIN_PROFILES: tuple[WaveProfile, ...] = (
    KELVIN,
    EQUATORIAL_ROSSBY_N0,
    EQUATORIAL_ROSSBY_N1,
    MIXED_ROSSBY_GRAVITY,
    MJO,
    EASTWARD_INERTIAL_GRAVITY_N0,
    WESTWARD_INERTIAL_GRAVITY_N2,
    WESTWARD_INERTIAL_GRAVITY_N1,
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

