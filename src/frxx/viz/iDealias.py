from .iPyart import iPyart
from ..proc import splashDealias
import netCDF4
import numpy as np
from pathlib import Path

dealiasCorrectionVar = {
    "name": "dealias_correction",
    "fields": {
        "type": "i2",
        "fill_value": 0,
        "dims": ("time", "range"),
        "long_name": "velocity_dealias_correction_offset",
        "units": "2x nyqist velocity m/s",
        "grid_mapping": "grid_mapping",
        "coordinates": "time range",
        "data": "tbd"
    },
}

velCopyVar = {
    "name": "VEL_OLD",
    "fields": {
        "type": "i2",
        "fill_value": -32768,
        "dims": ("time", "range"),
        "long_name": "doppler_velocity",
        "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
        "units": "m/s",
        "scale_factor": 0.01,
        "add_offset": 0.0,
        "grid_mapping": "grid_mapping",
        "coordinates": "time range",
        "data": "tbd"
    }
}

def _addVar(filePath, var):
    coreVarFields = ["type", "fill_value", "dims", "data"]

    ncfile = netCDF4.Dataset(filePath, 'r+', format='NETCDF4')
    varname = var["name"]
    varDict = var["fields"]
    ncvar = ncfile.createVariable(varname, varDict["type"], varDict["dims"], 
                            fill_value=varDict["fill_value"])
    for key, val in varDict.items():
        if not key in coreVarFields:
            setattr(ncvar, key, val)
    ncvar[:] = varDict["data"]

    ncfile.close()
    

def _getVelAndShape(filePath):
    ncfile = netCDF4.Dataset(filePath, 'r', format='NETCDF4')
    shape = ncfile["VEL"].shape
    data = np.ma.array(ncfile["VEL"][:])
    ncfile.close()
    return data, shape

def _setVel(filePath, velField):
    ncfile = netCDF4.Dataset(filePath, 'r+', format='NETCDF4')
    ncfile["VEL"][:] = velField
    ncfile.close()

def _isEdited(filePath):
    ncfile = netCDF4.Dataset(filePath, 'r', format='NETCDF4')
    edited = 'VEL_OLD' in ncfile.variables
    ncfile.close()
    return edited

def undoAllEdits(filePath):
    if not _isEdited(filePath):
        return False
    filePath = Path(filePath)
    ncfile = netCDF4.Dataset(filePath, 'r+', format='NETCDF4')
    oldvel = np.array(ncfile['VEL_OLD'][:])
    ncfile['VEL'][:] = oldvel
    
    newfile = netCDF4.Dataset(filePath.parent/"tmp.nc", 'w', format='NETCDF4')
    newfile.setncatts({attr: ncfile.getncattr(attr) for attr in ncfile.ncattrs()})
    for name, dimension in ncfile.dimensions.items():
        newfile.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None)
        )

    for name, variable in ncfile.variables.items():
            if not (name in ['VEL_OLD', 'dealias_correction']):
                var_out = newfile.createVariable(name, variable.datatype, variable.dimensions)
                var_out.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
                var_out[:] = variable[:]

    ncfile.close()
    newfile.close()

    filePath.unlink()
    (filePath.parent/"tmp.nc").rename(filePath)

    return True
    

def iDealias(cfradDir, fileNum, gateMask):
    #If we are not done editing, we need to undo everything in the edit history
    editsConfirmed = False
    #treat this like a stack, push edits, pop to undo
    editHistory = []
    
    #initialize iPyart so we have a way to get filename
    display = iPyart(cfradDir, fileNum)
    display.freezeTime()
    display.freezeField()


    #get filepath
    filePath = display.getFilePath()
    vel, shape = _getVelAndShape(filePath)
    
    #if backups and history doesnt exist lets create it
    if not _isEdited(filePath):
        closeEarlyIsDeleteAndCopy = True
        velCopyVar["fields"]["data"] = vel
        dealiasCorrectionVar["fields"]["data"] = np.zeros(shape)

        _addVar(filePath, velCopyVar)
        _addVar(filePath, dealiasCorrectionVar)
    else:
        #will have to just undo one by one
        closeEarlyIsDeleteAndCopy = False

    #then take gate filter and apply it new vel. Make sure you figure out how to get NaN over to i4 in netcdf. has to do with fill value i think.
    vel.mask = gateMask
    _setVel(filePath, vel)
