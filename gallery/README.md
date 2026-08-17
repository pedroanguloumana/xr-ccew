# Gallery

This folder contains notebook-based figure galleries based on the old reference
project. The notebooks are intended to show the `xr_ccew` API as each figure is
made, not just produce files in batch.

The first gallery entry recreates the two committed reference spectra:

- `NOAA_OLR_symmetric_spectra.pdf`
- `NOAA_OLR_antisymmetric_spectra.pdf`

Open the notebook from the repository root:

```bash
conda activate xr_ccew
jupyter lab gallery/01_noaa_olr_spectra.ipynb
```

By default, the notebook reads:

```text
OLD_REFERENCE_PROJECT/data/olr.2xdaily.1979-2022.nc
```

and writes regenerated figures to:

```text
gallery/figures/
```

For a faster smoke-test run outside Jupyter, set environment variables before
executing the notebook:

```bash
XR_CCEW_GALLERY_MAX_SEGMENTS=1 \
XR_CCEW_GALLERY_OUTPUT_DIR=/tmp/xr_ccew_gallery_smoke \
jupyter nbconvert --to notebook --execute gallery/01_noaa_olr_spectra.ipynb
```
