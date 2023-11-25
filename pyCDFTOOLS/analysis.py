# JM: 25 Nov 2023

"""
Module to house the subroutines that focuses on analysing the data. Naming to mirror CDFTOOLS (e.g. cdfmoc computes the moc at fixed height etc.)
"""

# TODO: only going to write a few subroutines, in order to see how imbuement and coding structure is going to work

#-------------------------------------------------------------------------------
    
def cdfcurl(grid, ds, un, vn, jk=0, **bd):
    """Computes the 2d curl at a fixed horizontal level (e.g. vertical component 
    of relative vorticity) in a way consistent with the NEMO finite volume 
    discretisation. 
    
    Expects a 2d field input. Computes this along ALL time by default; this is 
    not a necessarily a problem until evaluation.
    
    *** NOTE ***
    By default attributes assume the inputs are *velocities*, so the output is 
    a relative vorticity. Modify this as appropriate if need be.
    """

    # this is put onto the F-points
    socurl  = (  grid.diff(vn * ds.e2v, "X", **bd) 
               - grid.diff(un * ds.e1u, "Y", **bd) 
              ) / (ds.e1f * ds.e2f) * ds.fmask[jk, :, :]
               
    # imbue some useful attributes
    socurl["gdept_1d"] = ds.gdept_1d[jk]
    socurl.attrs["standard_name"] = "socurl"
    socurl.attrs["long_name"]     = "Relative_vorticity (curl)"
    socurl.attrs["units"]         = "s-1"
        
    return socurl
    
#-------------------------------------------------------------------------------
