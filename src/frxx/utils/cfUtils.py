import xarray as xr
import numpy as np

from datetime import datetime

import warnings


def _alignTime(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
	selfStart = datetime.fromisoformat(ds1.attrs["start_datetime"])
	otherStart = datetime.fromisoformat(ds2.attrs["start_datetime"])
	
	offset_seconds = (otherStart - selfStart).total_seconds()
	
	ds2 = ds2.assign_coords(time=ds2["time"] + offset_seconds)
	ds2["time"].attrs["units"] = ds1["time"].attrs["units"]

	return ds2
def _alignSweep(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
	lastSweep = ds1["sweep"].values[-1]
	ds2 = ds2.assign_coords(sweep=ds2["sweep"] + lastSweep+1)
	
	lastSweepNumber = ds1["sweep_number"].values[-1]
	ds2["sweep_number"].data = ds2["sweep_number"] + lastSweepNumber+1
	
	timeLength = len(ds1["time"])
	ds2["sweep_start_ray_index"].data = ds2["sweep_start_ray_index"].values + timeLength
	ds2["sweep_end_ray_index"].data = ds2["sweep_end_ray_index"].values + timeLength

	return ds2

def cfConcat(ds1: xr.Dataset, ds2: xr.Dataset, newSweep: bool = True) -> xr.Dataset:
	if ds1["volume_number"].data.item() != ds2["volume_number"].data.item():
		warning = ("volume_number in both files should be equal before concatenation "
						"to prevent second volume number from being overwritten")
		warnings.warn(warning)

		ds1Cpy = ds1.copy(deep=False)
		ds2Cpy = ds2.copy(deep=False)

		ds2Cpy = _alignTime(ds1Cpy, ds2Cpy)
		ds2Cpy = _alignSweep(ds1Cpy, ds2Cpy)

		ds1Cpy.attrs["end_datetime"] = ds2Cpy.attrs["end_datetime"]
		ds1Cpy.attrs["time_coverage_end"] = ds2Cpy.attrs["time_coverage_end"]
		ds1Cpy["time_coverage_end"].values = ds2Cpy["time_coverage_end"].values

		if set(ds1Cpy.variables) != set(ds2Cpy.variables):
			raise ValueError(f"Variable mismatch: {set(ds1Cpy.variables)} vs {set(ds2Cpy.variables)}")

		merged = xr.concat(
			[ds1Cpy, ds2Cpy], 
			dim = "time",
			data_vars="minimal",
			coords='minimal',
			compat='override'
		)
		newSweepVarLen = len(merged["sweep"])
		merged = merged.assign_coords(sweep=np.arange(newSweepVarLen))
		for var in merged.variables:
			if "sweep" in merged[var].dims:
				if var == "sweep":
					#dont need to modify the coordinate itself
					continue
				merged[var].loc[{"sweep":ds2Cpy["sweep"].values}] = ds2Cpy[var].values
		
		if not newSweep:
			mergedSweep = len(ds1Cpy["sweep"])-1
			endSweep = newSweepVarLen-1

			merged["sweep_start_ray_index"].values[mergedSweep+1] = merged["sweep_start_ray_index"].values[mergedSweep]
			merged["sweep_start_ray_index"].values[mergedSweep:endSweep] = merged["sweep_start_ray_index"].values[mergedSweep+1:endSweep+1]
			merged["sweep_start_ray_index"].values[endSweep] = -1
			merged["sweep_end_ray_index"].values[mergedSweep:endSweep] = merged["sweep_end_ray_index"].values[mergedSweep+1:endSweep+1]
			merged["sweep_end_ray_index"].values[endSweep] = -1
			merged["sweep_numer"].values[mergedSweep:endSweep] -= 1
			merged["sweep_numer"].values[endSweep] = -1

			merged = merged.isel(sweep=slice(0, endSweep))
		
		if np.any(np.diff(merged["time"])<0):
			raise ValueError("Time variable is non-decreasing! "
							 "Concat is only for time-adjacent files.")

		if type(merged) != xr.Dataset:
			raise TypeError("Something went wrong. Somehow we have a dataarray.")

		return merged	