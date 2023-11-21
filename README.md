# pyCDFTOOLS

My adaptations [CDFTOOLS](https://github.com/meom-group/CDFTOOLS) for Python use but using the [xgcm](https://github.com/xgcm) formalism via [xnemogcm](https://github.com/rcaneill/xnemogcm), with some attempt to mirror the functionality and coding structure of CDFTOOLS; this is really more of a wrapper on top of xnemogcm. The main reason for doing this is because I don't like doing analysis through Fortran...

This is split out from whatever is in my other [NEMO repository](https://github.com/julianmak/NEMO-related/tree/master), which is a Fortran-like way of writing things.

## To do:

- [ ] testing of DASK chunking and parallel computation
- [ ] examples with DASK functionality
- [ ] upload of sample notebooks
- [ ] upload of brainfart notebooks detailing what things to do (and not t do) for a FAQ later
- [ ] data pre-processing for use with xnemogcm
- [ ] upload of sample data (notably outputs from CDFTOOLS) and developing testing workflow
- [ ] update references to NEMO-related readthedocs
- [ ] set up some containers of sorts?
- [ ] testing probably in UNAGI, Andrew Styles' Weddell gyre model, ORCA2 and 1, maybe GYRE_PISCES 
- [x] claim the repository name

## Status

* (21 Nov 2023) Probably don't use this for anything serious, no sanity check and code testing has been done whatsoever. Plan is to first do a few subroutines, decide on subroutine formatting and coding standards, do a test workflow, before doing more subroutines.
