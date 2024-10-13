# JM: 25 Nov 2023

"""
Module to house the subroutines that focuses on analysing the data. Naming to 
mirror CDFTOOLS (e.g. cdfmoc computes the moc at fixed height etc.)
"""

import xnemogcm as xn
import xgcm as xgcm

# TODO: only going to write a few subroutines, in order to see how imbuement and 
#       coding structure is going to work

# TODO: add check for variable having right dimensions?

#-------------------------------------------------------------------------------
    
def cdfcurl(grid, ds, un, vn, jk=0, **bd):
    """Computes the 2d curl at a fixed horizontal level (e.g. vertical component 
    of relative vorticity) in a way consistent with the NEMO finite volume 
    discretisation. 
    
    Expects a 2d field input on U and V-grid. Computes this along ALL time by 
    default; this is not a necessarily a problem until evaluation.
    
    *** NOTE ***
    By default attributes assume the inputs are *velocities*, so the output is 
    a relative vorticity. Modify this as appropriate if need be.
    """
    
    # 1) check if W or T variable first
    var_T, var_W = False, False
    if "z_c" in list(un.coords):
        var_T = True
        z_ind = un["z_c"].values.astype('int')
    elif "z_f" in list(un.coords):
        var_W = True
        z_ind = (un["z_f"].values + 1).astype('int')  # because shifted down
    else:
        print("=== WARNING: no z_c information found, assume it is a T-grid variable ===")
        var_T = True
        z_ind = 0

    # 1) calculation, puts the output onto the F-points
    if var_T:
        socurl  = (  grid.diff(vn * ds.e2v, "X", **bd) 
                   - grid.diff(un * ds.e1u, "Y", **bd) 
                  ) / (ds.e1f * ds.e2f) * ds.fmask.isel(z_c=z_ind)

    elif var_W:
        socurl  = (  grid.diff(vn * ds.e2v, "X", **bd) 
                   - grid.diff(un * ds.e1u, "Y", **bd) 
                  ) / (ds.e1f * ds.e2f) * ds.fmask.isel(z_f=z_ind)
               
    # end) imbue some useful variables/attributes
    socurl.attrs["standard_name"] = "socurl"
    socurl.attrs["long_name"]     = "Relative_vorticity (curl)"
    socurl.attrs["units"]         = "s-1"
        
    return socurl
    
#-------------------------------------------------------------------------------

def cdfmoc(grid, ds, vn, **bd):
    """Computes the meridional overturning circulation in depth co-ordinates,
       in a way consistent with CDFTOOLS/cdfmoc
    
    Expects a 3d field input on V-grid. Computes this along ALL time by default; 
    this is not a necessarily a problem until evaluation.
    
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
    
    # end) imbue some useful variables/attributes
    moc["gphiv"] = ds.gphiv[:, 1]  # placeholder only
    moc["gdepw_1d"] = ds.gdepw_1d
    moc.attrs["standard_name"] = "moc"
    moc.attrs["long_name"]     = "Meridional Overturning Circulation (avg at fixed depth)"
    moc.attrs["units"]         = "Sv"

    return moc
    
#-------------------------------------------------------------------------------

def cdfzonalmean(grid, ds, da, **bd):
    """Computes the zonal mean (really the mean in the i-dimension). Assumes at
    least 2d with x and y dimension.
    
    Do not include the mask, in order for inheriting attributes. Computes along 
    ALL other dimensions by default; this is not a necessarily a problem until 
    evaluation.
    
    *** NOTE ***
    Inhereits the attributes of the input field. Modify this as appropriate if 
    need be.
    """
    
    # 1) check if W or T variable first and grab some indices
    var_T, var_W = False, False
    if "z_c" in list(da.coords):
        var_T = True
        z_ind = da["z_c"].values.astype('int')
    elif "z_f" in list(da.coords):
        var_W = True
        z_ind = (da["z_f"].values + 1).astype('int')  # because shifted down
    else:
        print("=== WARNING: no z_c information found, assume it is a T-grid variable ===")
        var_T = True
        z_ind = 0
    
    # 2) do integral of variable and metric along i-direction, then divide each 
    #    other (so i-mean)
    # W-variables treated as if it were T-variables
    
    # UGLY: consider doing the imbuement of the variable grid location at the 
    #       point when the files are loaded so it can be read off...
    
    # T-variable
    if all(x in list(da.coords) for x in ["x_c", "y_c"]):
        x_ind = da["x_c"].values.astype('int')
        y_ind = da["y_c"].values.astype('int')
        e1 = ds.e1t.isel(y_c=y_ind, x_c=x_ind)
        if var_T:
            mask = ds.tmask.isel(z_c=z_ind, y_c=y_ind, x_c=x_ind)
        elif var_W:
            mask = ds.tmask.isel(z_f=z_ind, y_c=y_ind, x_c=x_ind)
        zonalmean = (da * mask).sum(dim="x_c") / (e1 * mask).sum(dim="x_c")
        zonalmean["gphit"] = ds["gphit"].isel(y_c=y_ind, x_c=1)  # placeholder only
 
    # U-variable
    elif all(x in list(da.coords) for x in ["x_f", "y_c"]):
        x_ind = da["x_f"].values.astype('int')
        y_ind = da["y_c"].values.astype('int')
        e1 = ds.e1u.isel(y_c=y_ind, x_f=x_ind)
        if var_T:
            mask = ds.umask.isel(z_c=z_ind, y_c=y_ind, x_f=x_ind)
        elif var_W:
            mask = ds.umask.isel(z_f=z_ind, y_c=y_ind, x_f=x_ind)
        zonalmean = (da * mask).sum(dim="x_f") / (e1 * mask).sum(dim="x_f")
        zonalmean["gphiu"] = ds["gphiu"].isel(y_c=y_ind, x_f=1)  # placeholder only

    # V-variable
    elif all(x in list(da.coords) for x in ["x_c", "y_f"]):
        x_ind = da["x_c"].values.astype('int')
        y_ind = da["y_f"].values.astype('int')
        e1 = ds.e1v.isel(y_f=y_ind, x_c=x_ind)
        if var_T:
            mask = ds.vmask.isel(z_c=z_ind, y_f=y_ind, x_c=x_ind)
        elif var_W:
            mask = ds.vmask.isel(z_f=z_ind, y_f=y_ind, x_c=x_ind)
        zonalmean = (da * mask).sum(dim="x_c") / (e1 * mask).sum(dim="x_c")
        zonalmean["gphiv"] = ds["gphiv"].isel(y_f=y_ind, x_c=1)  # placeholder only

    # F-variable
    elif all(x in list(da.coords) for x in ["x_f", "y_f"]):  
        x_ind = da["x_f"].values.astype('int')
        y_ind = da["y_f"].values.astype('int')
        e1 = ds.e1f.isel(y_f=y_ind, x_f=x_ind)
        if var_T:
            mask = ds.fmask.isel(z_c=z_ind, y_f=y_ind, x_f=x_ind)
        elif var_W:
            mask = ds.fmask.isel(z_f=z_ind, y_f=y_ind, x_f=x_ind) 
        zonalmean = (da * mask).sum(dim="x_f") / (e1 * mask).sum(dim="x_f")
        zonalmean["gphif"] = ds["gphif"].isel(y_f=y_ind, x_f=1)  # placeholder only
        
    else:
        print("=== WARNING: no valid combo of (x,y) dimension grabbed, CHECK ===")
        print(f"  the list of coords grabbed as list(ds.coords) = {list(da.coords)}")
        return np.nan
        
    # end) imbue some useful variables/attributes
    zonalmean.attrs = da.attrs
        
    return zonalmean
    
#-------------------------------------------------------------------------------

def cdfz2sig(ds, da, sigma, sigma_coord, method="linear", **bd):
    """Performs a vertical co-ordinate transformation; sigma is just a 
    placeholder, can be density, temperature, other depth, etc. By default does
    a linear interpolation.
    
    Do not include the mask, in order for inheriting attributes. Computes along 
    ALL other dimensions by default; this is not a necessarily a problem until 
    evaluation.
    
    *** NOTE ***
    Inhereits the attributes of the input field. Modify this as appropriate if 
    need be.
    """
    
    # 0) define a temporary coords and grid for exclusive use with 
    #    xgcm.transform (needs "periodic=False" and an "outer" definition for
    #    the conservative transform option)
    coords = {"X": {"right" : "x_f", "center":"x_c"},   # xU > xT
              "Y": {"right" : "y_f", "center":"y_c"},   # yV > yT
              "Z": {"center": "z_c", "outer" :"z_f"},
              "T": {"center": "t"},
             }
    grid = xgcm.Grid(ds, coords=coords, metrics=xn.get_metrics(ds), periodic=False)
    
    # 1) check if W or T variable first and grab some indices
    var_T, var_W = False, False
    if "z_c" in list(da.coords):
        var_T = True
        z_ind = da["z_c"].values.astype('int')
    elif "z_f" in list(da.coords):
        var_W = True
        z_ind = (da["z_f"].values + 1).astype('int')  # because shifted down
    else:
        print("=== WARNING: no z information found! ===")
        print(f"  the list of coords grabbed as list(ds.coords) = {list(da.coords)}")
        return np.nan
        
    # T-variable
    if all(x in list(da.coords) for x in ["x_c", "y_c"]):
        x_ind = da["x_c"].values.astype('int')
        y_ind = da["y_c"].values.astype('int')
        if var_T:
            mask = ds.tmask.isel(z_c=z_ind, y_c=y_ind, x_c=x_ind)
        elif var_W:
            mask = ds.tmask.isel(z_f=z_ind, y_c=y_ind, x_c=x_ind)
 
    # U-variable
    elif all(x in list(da.coords) for x in ["x_f", "y_c"]):
        x_ind = da["x_f"].values.astype('int')
        y_ind = da["y_c"].values.astype('int')
        if var_T:
            mask = ds.umask.isel(z_c=z_ind, y_c=y_ind, x_f=x_ind)
        elif var_W:
            mask = ds.umask.isel(z_f=z_ind,y_c=y_ind, x_f=x_ind)

    # V-variable
    elif all(x in list(da.coords) for x in ["x_c", "y_f"]):
        x_ind = da["x_c"].values.astype('int')
        y_ind = da["y_f"].values.astype('int')
        if var_T:
            mask = ds.vmask.isel(z_c=z_ind, y_f=y_ind, x_c=x_ind)
        elif var_W:
            mask = ds.vmask.isel(z_f=z_ind, y_f=y_ind, x_c=x_ind)

    # F-variable
    elif all(x in list(da.coords) for x in ["x_f", "y_f"]):  
        x_ind = da["x_f"].values.astype('int')
        y_ind = da["y_f"].values.astype('int')
        if var_T:
            mask = ds.fmask.isel(z_c=z_ind, y_f=y_ind, x_f=x_ind)
        elif var_W:
            mask = ds.fmask.isel(z_f=z_ind, y_f=y_ind, x_f=x_ind)
        
    else:
        print("=== WARNING: no valid combo of (x,y) dimension grabbed, CHECK ===")
        print(f"  the list of coords grabbed as list(ds.coords) = {list(da.coords)}")
        return np.nan
        
    # 2) do transformation
    if method=="conservative":  # extra steps to trigger fewer warnings
        sigma_var = grid.interp_like(sigma, da)
        sigma = grid.interp(sigma_var, "Z", boundary="extend")

    da_in_sigma = grid.transform(da * mask, "Z", sigma_coord, 
                                 target_data=sigma, 
                                 method=method)
                                 
    # one liner for conservative transformation
    # (a ton of warnings are triggered if done this way)
    # da_in_sigma = grid.transform(da * mask, 'Z', sigma_coord, 
    #                              target_data=sigma,
    #                              method=method)
        
    # end) imbue some useful variables/attributes
    da_in_sigma.attrs = da.attrs
    da_in_sigma = da_in_sigma.rename({sigma.name : "sigma"})
    if "t" in da_in_sigma.dims:
        da_in_sigma = da_in_sigma.transpose("t", "sigma", ...)
    else:
        da_in_sigma = da_in_sigma.transpose("sigma", ...)

    return da_in_sigma
    
    
    
    
    
    
    
    
    
