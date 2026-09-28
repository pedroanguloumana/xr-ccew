# xr-ccew

`xr-ccew` is a small Python library for convectively coupled equatorial wave
analysis on gridded `xarray` data.

The package is currently in its first implementation milestone. The core goal
is to make wave definitions inspectable and editable, then use those profiles
to build composable power-spectrum and filtering workflows.

## Install for Development

Use the project conda environment:

```bash
conda env create -f environment.yml
conda activate xr_ccew
```

If the environment already exists, update it from the same file:

```bash
conda env update -n xr_ccew -f environment.yml --prune
```

If the repository directory is renamed or moved, refresh the editable install
from the new repository root:

```bash
conda run -n xr_ccew python -m pip install -e .
```

## Basic Usage

```python
import xr_ccew as tw

data = tw.synthetic_wave(period_days=8, zonal_wavenumber=5)
waves = tw.wave_profiles("Kelvin")

filtered = tw.filter_field(data, waves)
power = tw.power_spectrum(data)
```

## Constant Background Wind

By default, filters use Matsuno dispersion curves for a resting basic state.
You can apply a constant eastward-positive zonal background wind in metres per
second:

```python
filtered = tw.filter_field(data, "Kelvin", background_u=10.0)
```

This uses the constant-zonal-advection Doppler approximation. For signed global
zonal wavenumber `s`, the observed-frequency shift is
`background_u * s * 86400 / perimeter` cycles day-1. Profile frequency bounds,
including sloped polygon bounds, and equivalent-depth curves are treated as
intrinsic/rest-state bounds, then shifted into the observed frame together. The
default `background_u=0.0` therefore reproduces the resting Matsuno behavior.

`background_v=0.0` is accepted for an explicit resting meridional wind. A
nonzero constant meridional wind is rejected: it is not a stationary beta-plane
Matsuno basic state and cannot be represented by this library's
frequency/zonal-wavenumber filters without a separate meridional eigenproblem.
The zonal option is consequently a Doppler approximation, not a full balanced
mean-flow Matsuno solver.

## Built-in Wave Profiles

The initial built-in profiles are ported from the reference project:

- Kelvin wave (`KW`)
- n=0 equatorial Rossby (`n=0 ER`)
- n=1 equatorial Rossby (`n=1 ER`)
- Mixed Rossby-gravity (`MRG`)
- Madden-Julian oscillation (`MJO`)
- n=0 eastward inertial gravity (`n=0 EIG`)
- n=1 westward inertial gravity (`n=1 WIG`)
- n=2 westward inertial gravity (`n=2 WIG`)
- Tropical depression-type / easterly wave band (`TD-type`): the Kiladis et
  al. (2006) parallelogram over westward wavenumbers 6-20, spanning
  2-3.3-day periods at k=-20 and 3-7.5-day periods at k=-6, with no
  dispersion curves

Besides rectangular and dispersion-curve bounds, a profile can carry a convex
`wavenumber_frequency_polygon` of (wavenumber, frequency) vertices; the filter
mask is the intersection of the polygon with the profile's k/frequency bounds.

Profiles can be inspected and copied with modifications:

```python
kelvin = tw.wave_profile("Kelvin")
slow_kelvin = kelvin.with_updates(frequency_max=0.25)
```

## Frequency Ceiling

A sampled record only resolves frequencies below its Nyquist frequency, and
the high-wavenumber tail of a dispersive profile such as `n=0 EIG` runs into
it. Derive the truncation from the data's own time coordinate instead of
hard-coding it:

```python
eig0 = tw.apply_frequency_ceiling("n=0 EIG", data.time)            # 0.8 x Nyquist
eig0_sensitivity = tw.apply_frequency_ceiling("n=0 EIG", data.time, fraction=0.9)
eig0.frequency_ceiling                                              # recorded for provenance
tw.nyquist_frequency(data.time)
```

For daily data the default gives a 0.40 cycles-per-day ceiling (2.5 days). The
ceiling is recorded on the returned profile, and `profile.as_dict()` gives
every parameter for metadata.

## Cross-Spectra and Segment Averaging

`cross_spectrum(a, b)` returns the complex space-time cross-spectrum
`conj(FFT[a]) * FFT[b]` (real part: cospectrum; imaginary part: quadrature
spectrum), and `segment_averaged_spectrum(a, b=None, ...)` runs one
preprocessing chain (component, detrending, seasonal harmonics, overlapping
segments, taper, FFT) for either a power spectrum or a cross-spectrum, with a
normalisation whose sum over all bins is the covariance of the tapered segment
anomalies. That is what lets a band partition of a covariance close exactly
(Parseval):

```python
cross = tw.segment_averaged_spectrum(v, q, segment_days=128, overlap_days=64)
cospectrum = cross.real
mask = tw.make_filter_mask(cospectrum, "MRG", include_conjugates=True)
band_covariance = float(cospectrum.where(mask, 0.0).sum())
```

## Data Conventions

Core functions expect a latitude/longitude/time grid. The default dimension
names are `time`, `lat`, and `lon`, but most functions accept dimension-name
arguments.

- Time must be regularly spaced. Datetime coordinates are interpreted in days.
- Longitude must be regularly spaced and increasing.
- Positive frequency and positive zonal wavenumber represent eastward-propagating
  waves in the profile masks.
- Symmetric and antisymmetric decomposition is interpolation-based, so exact
  latitude pairs around the equator are not required.

Plotting is intentionally outside the core package for now.

## Gallery

Reference-project figure recipes live in `gallery/`. They are notebooks that
show the API as the figure is made. The first gallery notebook regenerates the
NOAA OLR symmetric and antisymmetric spectra PDFs, and the second displays the
primary regions of every built-in filter, including a Doppler-shifted MRG:

```bash
conda activate xr_ccew
jupyter lab gallery/01_noaa_olr_spectra.ipynb
jupyter lab gallery/02_wave_filter_regions.ipynb
```

## Test

The tests are written with the standard library test runner:

```bash
conda activate xr_ccew
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```
