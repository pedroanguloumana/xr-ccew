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
NOAA OLR symmetric and antisymmetric spectra PDFs:

```bash
conda activate xr_ccew
jupyter lab gallery/01_noaa_olr_spectra.ipynb
```

## Test

The tests are written with the standard library test runner:

```bash
conda activate xr_ccew
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```
