# pyCDFTOOLS

My adaptations [CDFTOOLS](https://github.com/meom-group/CDFTOOLS) for Python analysing NEMO data, but using the [xgcm](https://github.com/xgcm) formalism via [xnemogcm](https://github.com/rcaneill/xnemogcm), with some attempt to mirror the functionality and coding structure of CDFTOOLS; this really is just a wrapper on top of xnemogcm. The main reason for doing this is because I don't like doing analysis through Fortran... but of course the extra functionality with DASK (chunking and parallelism) is very desirable for larger dataset analysis.

This is split out from whatever is in my other [NEMO repository](https://github.com/julianmak/NEMO-related/tree/master), which is a Fortran-like way of writing things. Those won't be updated anymore.

## Installation / Setup (12/06/2025)

1. Create an environment called `pycdftools` from the `environment.yml` file.
    ```
    conda env create -f environment.yml
    ```
    It takes a while (~20-30 mins, depends on platform) to finish the build and installation.

2. Activate the environment.

    ```
    conda activate pycdftools
    ```

3. Setup the rest using `pip`. (Note: don't use the shebang line (i.e. `#!/usr/bin/env python3`) in `pyCDFTOOLS/__init__.py`, otherwise the conda-packaged python will not be used and the `pyCDFTOOLS` will not installed correctly)

    ```
    pip install .
    ```

## To do:

- [ ] testing of DASK chunking and parallel computation
- [ ] examples with DASK functionality
- [ ] upload of sample notebooks
- [ ] upload of brainfart notebooks detailing what things to do (and not to do) for a FAQ later
- [ ] data pre-processing for use with xnemogcm
- [ ] upload of sample data (notably outputs from CDFTOOLS) and developing testing workflow
- [ ] update references to NEMO-related readthedocs
- [ ] set up some containers of sorts?
- [ ] testing probably in UNAGI, Andrew Styles' Weddell gyre model, ORCA2 and 1, maybe GYRE_PISCES 
- [x] claim the repository name 8-)

## Status

* (21 Nov 2023) Probably don't use this for anything serious, no sanity check and code testing has been done whatsoever. Plan is to first do a few subroutines, decide on subroutine formatting and coding standards, do a test workflow, before doing more subroutines.
