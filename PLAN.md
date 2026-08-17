# Suggested Plan of Attack

Source note: `DEV_PLAN.md` was not present in the workspace; this plan is based on the available `DEV_PLAN_v0.md`.
USER: Good, this is what I meant.
## Project Goal

Build a reusable, pip-installable Python library for analyzing tropical wave signals in gridded `xarray` datasets. The library should support both Wheeler-Kiladis-style power spectra and filtered fields in latitude/longitude/time space.

The central design idea should be that wave definitions live in editable, extensible wave profiles rather than being hard-coded into individual routines.

## Guiding Principles

- Keep public APIs centered on `xarray.Dataset` and `xarray.DataArray`.
- Make wave profiles explicit, inspectable, serializable, and easy to extend.
- Expose the intermediate scientific operations that users may want to compose themselves.
- Prefer small, tested numerical primitives over one large pipeline function.
- Preserve metadata, coordinates, dimensions, and attributes wherever practical.
- Avoid destructive repository operations unless explicitly approved.

## Phase 1: Establish the Package Skeleton

1. Choose and create the package name.
   - Candidate: `gridded_tropical_waves`
   - Note: the example import in `DEV_PLAN_v0.md` says `gridded_tropica_waves`, which looks like a typo worth confirming.
   USER: Yes, that is a typo. Project name should actually be `xr_ccew`. 
2. Add standard packaging files.
   - `pyproject.toml`
   - `README.md`
   - `LICENSE`, if desired
   - `src/gridded_tropical_waves/`
   - `tests/`
   USER: Again,  project name should actually be `xr_ccew`. 

3. Define core dependencies.
   - Required: `numpy`, `xarray`, `scipy`
   - Likely useful: `dask`, `netCDF4`, `matplotlib`
   - Optional/dev: `pytest`, `ruff`, `mypy` or `pyright`

4. Add basic tooling.
   - Unit test command
   - Lint/format command
   - Minimal CI later, once the package shape stabilizes

## Phase 2: Define Data and API Conventions

1. Document expected input conventions.
   - Time dimension name and calendar handling
   - Longitude convention: `0..360` vs `-180..180`
   - Latitude ordering
   - Required sampling regularity
   - Handling of missing data

2. Add lightweight validation helpers.
   - Confirm required dimensions exist
   - Confirm time spacing is regular
   - Normalize dimension names where possible
   - Raise clear errors when assumptions fail

3. Decide whether functions accept both `Dataset` and `DataArray`.
   - Recommended: support both, but implement core operations on `DataArray` first.

## Phase 3: Build Wave Profiles

1. Create a `WaveProfile` model.
   - Name
   - Equivalent depth or dispersion parameters, where relevant
   - Wavenumber bounds
   - Frequency/period bounds
   - Symmetry mode, if needed
   - Filter mask configuration
   - Human-readable description and references

2. Provide built-in profiles.
   - Kelvin
   - Equatorial Rossby
   - Mixed Rossby-gravity
   - Inertio-gravity modes, if needed
   - MJO or tropical disturbance bands, if desired

3. Add profile composition helpers.
   - Example: `wave_profiles("Kelvin", "Rossby")`
   - Allow user-modified copies
   - Allow custom profile registration or direct profile objects

## Phase 4: Implement Spectral Primitives

1. Implement preprocessing routines.
   - Remove mean
   - Detrend
   - Taper/window
   - Optional latitude weighting
   - Optional symmetric/antisymmetric decomposition

2. Implement transform routines.
   - Time FFT
   - Longitude FFT
   - Frequency and zonal wavenumber coordinate construction
   - Consistent metadata on spectral outputs

3. Implement power spectrum routines.
   - Raw power
   - Background spectrum smoothing
   - Ratio-to-background fields
   - Wheeler-Kiladis plotting-ready output

4. Keep these operations public and composable.
   - Example names: `preprocess`, `space_time_fft`, `power_spectrum`, `background_spectrum`

## Phase 5: Implement Filtering

1. Build mask generation from wave profiles.
   - Convert profile bounds into frequency/wavenumber masks
   - Support combining multiple profiles
   - Preserve enough metadata to explain what was filtered

2. Apply masks in spectral space.
   - Forward FFT
   - Apply mask
   - Inverse FFT
   - Restore original coordinates and metadata

3. Expose high-level and low-level filtering APIs.
   - Low-level: build/apply masks
   - High-level: filter a field by selected wave profiles

4. Add examples matching the target usage.
   - `waves = tw.wave_profiles("Kelvin", "Rossby")`
   - `filtered = tw.filter_field(data, waves)`

## Phase 6: Testing Strategy

1. Start with synthetic datasets.
   - Known sinusoidal signals in time/longitude
   - Known frequency/wavenumber peaks
   - Known signals that should pass or fail a filter

2. Test metadata behavior.
   - Coordinates preserved
   - Dimension order sensible
   - Attributes handled predictably

3. Test wave profile behavior.
   - Built-in profile lookup
   - Custom profile creation
   - Multiple-profile composition

4. Add regression tests once prior repository behavior is available.
   - Reproduce a small known Wheeler-Kiladis calculation
   - Compare filtered output against trusted prior outputs

## Phase 7: Documentation and Examples

1. Write a README with the intended workflow.
   - Installation
   - Basic wave profile usage
   - Filtering example
   - Power spectrum example

2. Add short example scripts or notebooks.
   - Synthetic data demo
   - NOAA-like gridded data demo
   - Wheeler-Kiladis diagram demo

3. Document extension points.
   - Creating a custom wave profile
   - Adjusting filter bounds
   - Composing a processing pipeline

## Phase 8: Prior Repository Review

The development note mentions an earlier repository used to recreate Wheeler and Kiladis 1999 figures. Once that code is available, review it specifically for:

- Existing numerical algorithms worth preserving
- API pain points to avoid
- Assumptions that were project-specific and should become configurable
- Test fixtures or known outputs that can seed regression tests
- Plotting conventions for Wheeler-Kiladis diagrams

## Proposed Initial Milestone

The first useful milestone should be a minimal package that can:

1. Install locally with `pip install -e .`.
2. Define and inspect built-in wave profiles.
3. Generate a synthetic gridded `xarray.DataArray`.
4. Compute a space-time power spectrum.
5. Apply a simple profile-based frequency/wavenumber filter.
6. Pass focused tests for synthetic known-frequency signals.

That milestone creates a working spine for the library before adding more wave families, plotting polish, background spectra, or compatibility layers.

## Open Questions

1. What should the final package/import name be?
USER: `xr-ccew`
2. Where is the prior repository, and should this project copy code from it or use it only as reference?
USER: It is /Users/pedro/Projects/Libraries/Gridded_Tropical_Wave_Filtering/OLD_REFERENCE_PROJECT
3. Which wave types should be built in for version 0.1?
USER: The waves that exist in /Users/pedro/Projects/Libraries/Gridded_Tropical_Wave_Filtering/OLD_REFERENCE_PROJECT
4. Should plotting be part of the core package or an optional extra?
USER: Optional extra. Focus is on the data crunching; making figures pretty is a user problem.
5. Should filtering support Dask-backed arrays from the beginning?
USER: Sure, why not?
6. What input data conventions should be mandatory versus auto-normalized?
USER: Lat/lon grid. Also, not an answer to your question, but make sure you handle the case where there's an off-equatorial grid for the symmetric/antisymmetric decomposition. 
