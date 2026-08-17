# Aim of this project
This project is to create a reusable, pip-installable library (hosted on github) that can be used by me to analyze the tropical wave signals in gridded (xarray) datasets. 

# Prior code:
A copy of a prior repository, where a version of what I want to build here was built just for this project of re-creating figures from Wheeler Kiladis 1999. Please look at this repo for an understanding of what I want:

# What I want: target usage
I want to be able to do something like
```python
import gridded_tropica_waves as tw

data = load_noaa_data()

waves = tw.wave_profiles('Kelvin', 'Rossby')
def processing_pipeline(ds, waves_to_process):
    ds_foo = tw.foo(ds)
    ds_foo_bar = tw.bar(ds_foo, waves_to_process)
    ...
    return ds.foo_bar_...

filtered = processing_pipeline(data, waves)
```

The key design points:
- wave information is stored as a profile so it can be extended/modified easily
- the aim will be both power spectra (wheeler-kiladis diagrams) and using the filtered fields (in lat/lon/time). So the functions that generate that need to be public, not private. 

## How to work with me
- DO NOT TAKE DESTRUCTIVE ACTIONS WITHOUT MY APPROVAL
- Ask questions if you are unsure. 





