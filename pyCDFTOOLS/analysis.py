# JM: 25 Nov 2023

"""
Module to house the subroutines that focuses on analysing the data. Naming to 
mirror CDFTOOLS (e.g. cdfmoc computes the moc at fixed height etc.)
"""

# TODO: only going to write a few subroutines, in order to see how imbuement and 
#       coding structure is going to work

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

    # 1) calculation, puts the output onto the F-points
    socurl  = (  grid.diff(vn * ds.e2v, "X", **bd) 
               - grid.diff(un * ds.e1u, "Y", **bd) 
              ) / (ds.e1f * ds.e2f) * ds.fmask[jk, :, :]
               
    # imbue some useful variables/attributes
    socurl["glamf"] = ds.glamf
    socurl["gphif"] = ds.gphif
    socurl["gdept_1d"] = ds.gdept_1d[jk]
    socurl.attrs["standard_name"] = "socurl"
    socurl.attrs["long_name"]     = "Relative_vorticity (curl)"
    socurl.attrs["units"]         = "s-1"
        
    return socurl
    
#-------------------------------------------------------------------------------

def cdfmoc(grid, ds, vn, **bd):
    """Computes the meridional overturning circulation in depth co-ordinates,
       in a way consistent with CDFTOOLS/cdfmoc
    
    Expects a 3d field input. Computes this along ALL time by default; this is 
    not a necessarily a problem until evaluation.
    
    *** NOTE ***
    By default attributes assume the inputs are *velocities*, and the output is
    a transport. Modify this as appropriate if need be.
    """
    
    # input is (z, y, x), grid_V, units of m s-1

    # 1) zonal integral and multiply by e3v; (z, y), grid_V and units of m3 s-1
    moc = (vn * ds.e1v * ds.e3v_0).sum(dim="x_c")

    # 2) cumulative sum in k from TOP; sum in Z puts it onto z_f, units of Sv
    #    then removes the total integral (the last entry)
    moc = grid.cumsum(moc, axis="Z", **bd) / 1e6
    
    moc -= moc.isel({"z_f" : -1})
    
    # imbue some useful variables/attributes
    moc["gphiv"] = ds.gphiv[:, 1]  # placeholder only
    moc["gdepw_1d"] = ds.gdepw_1d
    moc.attrs["standard_name"] = "moc"
    moc.attrs["long_name"]     = "Meridional Overturning Circulation (avg at fixed depth)"
    moc.attrs["units"]         = "Sv"

    return moc
