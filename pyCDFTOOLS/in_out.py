# JM: 25 Nov 2023

"""
Module to house the I/O subroutines for processed data.
"""

import glob
import numpy as np

#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------

def read_transport(data_dir, filename, key, match=None):
    """
    Given a data_dir, process all files with the "*_transport*" generated from
    SECTIONS_DIADCT, and spit out some of its contents
    
    Give it a unique case sensitive key (look into the file itself to check)
    
    Inputs:
    data_dir = string for data directory
    filename = filename to match (put in wildcards, e.g. "volume_transport*" if need be)
    key      = string for data to grab
    
    Optional input:
    match    = string (a number) to grab the specified entry under the same key
                      if None, then grab the total (e.g. 26N 1st section)
    
    Returns:
        t, data (as datetime["D"] and float64 arrays)
    """

    file_list = []
    for file in glob.glob(data_dir + f"{filename}"):
        file_list.append(file) 
      
    if not file_list:
        print("no files grabbed, are you in the right directory?")
        print("no files grabbed, are you in the right directory?")
        print("no files grabbed, are you in the right directory?")
    
    # do a sort according to some tag (usually timestamp, otherwise run_number)
    file_list.sort()

    data_dump = []
    for filename in file_list:
        with open(filename) as f:
            for line in f:
                if key in line: # this needs to match case for case
                    data_dump.append(line)

    data_redump = []
    if match is None:
        for line in range(len(data_dump)):
            if "total" in data_dump[line]:
                data_redump.append(data_dump[line])
    else:
        for line in range(len(data_dump)):
            if data_dump[line].split()[5] == match:
                data_redump.append(data_dump[line])

    # define some lists to dump entries, then turn them into arrays of np.datatime64 and a floats
    t = []
    data = []
    for i in range(len(data_redump)):
        yyyymmdd = data_redump[i].split()[0]
        t.append(np.datetime64(f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}", "D"))
        data.append(float(data_redump[i].split()[11]))

    return np.asarray(t), np.asarray(data)
