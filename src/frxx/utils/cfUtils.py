import xarray as xr
import numpy as np

from datetime import datetime, timedelta

from typing import Tuple
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

def cfConcat(ds1: xr.Dataset, ds2: xr.Dataset, newSweep: bool = False) -> xr.Dataset:
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
	
def cfBreakAt(ds: xr.Dataset, index: int, newVol: bool = False) -> Tuple[xr.Dataset | None, xr.Dataset | None]:
	n_times = len(ds["time"])

	if index < 0 or index > n_times:
		raise IndexError(
			f"index {index} is out of bounds for time dimension of length {n_times}"
		)

	if index == 0:
		warnings.warn("index is 0; first dataset will be None")
		ds2 = ds.copy(deep=False)
		if newVol:
			ds2["volume_number"].data = ds2["volume_number"].data + 1
		return None, ds2

	if index == n_times:
		warnings.warn(
			f"index equals time dimension length ({n_times}); second dataset will be None"
		)
		return ds.copy(deep=False), None

	starts = ds["sweep_start_ray_index"].values.astype(int)
	ends = ds["sweep_end_ray_index"].values.astype(int)

	clean = index in starts

	if clean:
		split_sweep_idx = int(np.where(starts == index)[0][0])

		ds1 = ds.isel(
			time=slice(0, index),
			sweep=slice(0, split_sweep_idx)
		).copy(deep=False)

		ds2 = ds.isel(
			time=slice(index, None),
			sweep=slice(split_sweep_idx, None)
		).copy(deep=False)

		ds2["sweep_start_ray_index"].data = ds2["sweep_start_ray_index"].values - index
		ds2["sweep_end_ray_index"].data = ds2["sweep_end_ray_index"].values - index

	else:
		matches = np.where((starts <= index) & (ends >= index))[0]
		if len(matches) == 0:
			raise ValueError(
				f"index {index} does not fall within any sweep. "
				"There may be a gap in sweep ray indices."
			)
		containing = int(matches[0])

		ds1 = ds.isel(
			time=slice(0, index),
			sweep=slice(0, containing + 1)
		).copy(deep=False)

		ds2 = ds.isel(
			time=slice(index, None),
			sweep=slice(containing, None)
		).copy(deep=False)

		ds1_ends = ds1["sweep_end_ray_index"].values.copy()
		ds1_ends[-1] = index - 1
		ds1["sweep_end_ray_index"].data = ds1_ends

		ds2["sweep_start_ray_index"].data = ds2["sweep_start_ray_index"].values - index
		ds2["sweep_end_ray_index"].data = ds2["sweep_end_ray_index"].values - index

		ds2_starts = ds2["sweep_start_ray_index"].values.copy()
		ds2_starts[0] = 0
		ds2["sweep_start_ray_index"].data = ds2_starts

		ds2["sweep_number"].data = ds2["sweep_number"].values + 1

	ds1 = ds1.assign_coords(sweep=np.arange(len(ds1["sweep"])))
	ds2 = ds2.assign_coords(sweep=np.arange(len(ds2["sweep"])))

	if newVol:
		ds2["volume_number"].data = ds2["volume_number"].data + 1

		original_start = datetime.fromisoformat(ds.attrs["start_datetime"])
		ds2_time_offset = float(ds["time"].values[index])
		ds2_start_dt = original_start + timedelta(seconds=ds2_time_offset)
		ds1_end_dt = original_start + timedelta(seconds=float(ds["time"].values[index - 1]))

		time_attrs = ds2["time"].attrs.copy()
		ds2["time"].data = ds2["time"].values - ds2_time_offset
		ds2["time"].attrs = time_attrs

		ds2["time"].attrs["units"] = f"seconds since {ds2_start_dt.isoformat().replace('+00:00', 'Z')}"

		ds1_end_str = ds1_end_dt.isoformat()
		ds1.attrs["end_datetime"] = ds1_end_str
		ds1.attrs["time_coverage_end"] = ds1_end_str.replace('+00:00', 'Z')
		if "time_coverage_end" in ds1:
			ds1_end_str = ds1_end_str.replace('+00:00', 'Z') +\
				(len(ds1.ds["string_length_32"]) - len(ds1_end_str.replace('+00:00', 'Z')))*' '
			ds1["time_coverage_end"].data = np.array([c for c in ds1_end_str], dtype="|S1")

		ds2_start_str = ds2_start_dt.isoformat()
		ds2.attrs["start_datetime"] = ds2_start_str
		ds2.attrs["time_coverage_start"] = ds2_start_str.replace('+00:00', 'Z')
		if "time_coverage_start" in ds2:
			ds2_start_str = ds2_start_str.replace('+00:00', 'Z') +\
				(len(ds2.ds["string_length_32"]) - len(ds2_start_str.replace('+00:00', 'Z')))*' '
			ds2["time_coverage_start"].data = np.array([c for c in ds2_start_str], dtype="|S1")

	return ds1, ds2