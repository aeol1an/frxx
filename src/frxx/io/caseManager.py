from pathlib import Path

import numpy as np
import xarray as xr

from ..core import IQ, moments, spectra

supportedInputs = ["rk"]

def frxxDataFromFile(filename: str | Path):
    ds = xr.open_dataset(filename)
    if "frxx_data_type" not in ds.attrs:
        raise AttributeError("Netcdf file not created by frxx.")
    
    type = ds.attrs["frxx_data_type"]
    if type == "IQ":
        return IQ(ds)
    elif type == "moments":
        return moments(ds)
    elif type == "spectra":
        return spectra(ds)