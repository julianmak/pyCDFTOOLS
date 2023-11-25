# JM: 25 Nov 2023

"""
Module to house the I/O subroutines for processed data.
"""

def save_netcdf(ds, varname=None, filename=None):
    """
    Promotes the processed DataArray to Dataset and provides naming.
    
    Filename can be a path (with the .nc suffix), otherwise outputs as 
    ds.attrs["standard_name"].nc
    """
    
    if varname is None:
        varname = ds.attrs["standard_name"]
    if filename is None:
        filename = ds.attrs["standard_name"] + ".nc"

    # output
    ds.to_dataset(name=varname).to_netcdf(filename)
    
    print(f"variable name {varname}, output to {filename}")
