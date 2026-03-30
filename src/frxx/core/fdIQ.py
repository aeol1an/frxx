import numpy as np
import xarray as xr

from pathlib import Path
import json

from numpy.typing import NDArray
from typing import List, Tuple, Union, cast
from collections.abc import Sequence
import warnings

from datetime import datetime

from ..utils import pathUtils, sourceFile
from .frxxData import _FILL_VALUES, frxxData

class IQ(frxxData):
	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()
		
		self.nonDataVars += [
			"iqdim",
			"pol",
			"noise",
			"Zcal",
			"Dcal",
			"Pcal",
		]

		self.requiredBools["iqdim"] = False
		self.requiredBools["pol"] = False
		self.requiredBools["noise"] = False
		self.requiredBools["calibration"] = False
		self.requiredBools["source_file"] = False

		if ds is None:
			#create new
			self.ds.attrs["frxx_data_type"] = "IQ"

			#add iq dimension
			self.ds = self.ds.assign_coords(
				iqdim=np.arange(2)
			)
			self.ds["iqdim"].attrs = {
				"long_name": "In-Phase/Quadrature",
				"indices": "index 0 real component, index 1 imaginary component"
			}
			self.ds["iqdim"].encoding = {
				"dtype": "int32",
				"_FillValue": _FILL_VALUES["int32"]
			}
			self.requiredBools["iqdim"] = True

		else:
			self.ds = ds.copy(deep=False)
			self.checkRequiredFields()
			valid = self.validateSelf()
			if not valid:
				raise RuntimeError("Invalid frxxIQ format. See above.")

	def setPol(self, nPol: int = 2):
		self.ds = self.ds.assign_coords(pol=np.arange(nPol))
		self.ds["pol"].attrs = {
			"long_name": "polarized_channels",
			"comment": "In the case of dual-pol, 0 is H and 1 is V"
		}
		self.ds["pol"].encoding = {
			"dtype": "int32",
			"_FillValue": _FILL_VALUES
		}

		self.requiredBools["pol"] = True

	def setNoisedB(self, n0_dB: Union[NDArray[np.floating], Sequence[float]]):
		if not self.requiredBools["pol"]:
			raise RuntimeError("Number of polarizations not set yet.")
		if len(n0_dB) != len(self.ds["pol"]):
			raise RuntimeError("Noise array length should equal number of polarizations.")
		
		self.ds["noise"] = xr.DataArray(
			data = np.array(n0_dB),
			dims = ["pol"],
			attrs = {
				"units": "dB",
				"long_name": "Noise_estimate_from_transmitter",
			}
		)
		self.ds["noise"].encoding = {
			"dtype": "float64",
			"_FillValue": _FILL_VALUES["float64"]
		}

		self.requiredBools["noise"] = True

	def setCal(self, Zcal_db: Union[NDArray[np.floating], Sequence[float]], Dcal_db: float | None = None, Pcal_deg: float | None = None):
		if not self.requiredBools["pol"]:
			raise RuntimeError("Number of polarizations not set yet.")
		nPol = len(self.ds["pol"])
		if len(Zcal_db) != nPol:
			raise RuntimeError("Zcal array length should equal number of polarizations.")
		if nPol == 2 and ((Dcal_db is None) or (Pcal_deg is None)):
			raise RuntimeError("Need Dcal and Pcal for dual pol. If not known, give 0.")
		
		self.ds["Zcal"] = xr.DataArray(
			data = np.array(Zcal_db),
			dims = ["pol"],
			attrs = {
				"units": "dB",
				"long_name": "reflectivity_calibration",
			}
		)
		self.ds["Zcal"].encoding = {
			"dtype": "float64",
			"_FillValue": _FILL_VALUES["float64"]
		}

		if nPol == 2:
			self.ds["Dcal"] = xr.DataArray(
				data = np.array(Dcal_db),
				dims = [],
				attrs = {
					"units": "dB",
					"long_name": "differential_reflectivity_calibration",
				}
			)
			self.ds["Dcal"].encoding = {
				"dtype": "float64",
				"_FillValue": _FILL_VALUES["float64"]
			}
			self.ds["Pcal"] = xr.DataArray(
				data = np.array(Pcal_deg),
				dims = [],
				attrs = {
					"units": "degrees",
					"long_name": "differential_phase_calibration",
				}
			)
			self.ds["Pcal"].encoding = {
				"dtype": "float64",
				"_FillValue": _FILL_VALUES["float64"]
			}

		self.requiredBools["calibration"] = True

	def setSourceFile(self, filename: str | Path):
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime function so "
							   "that number of elements are known.")
		
		sfJson = {
			"files": [
				sourceFile.makeFromPathAndLength(
					filename, 
					len(self.ds["time"])
				).toJson()
			]
		}
		self.ds.attrs["source_files"] = json.dumps(sfJson, indent='\t')

		self.requiredBools["source_file"] = True

	def editPlatformPrefix(self, platform: str, prefix: str):
		if not self.requiredBools["source_file"]:
			raise RuntimeError("Source file needs to be set first!")
		
		sfJson = json.loads(self.ds.attrs["source_files"])
		for i in range(len(sfJson["files"])):
			sfJson["files"][i] = sourceFile.makeFromJson(sfJson["files"][i])\
				.editPlatformPrefix(platform, prefix).toJson()
		self.ds.attrs["source_files"] = json.dumps(sfJson, indent='\t')

	def addDataField(self, name: str, data: NDArray, dims: List[str], attrs, encoding):
		if dims != ["pol", "range", "time", "iqdim"] or name != 'iq':
			raise ValueError("Please only add iq data to this data structure!")
		
		if not all(self.requiredBools.values()):
			raise ValueError("Set all IQ object variables.")

		self.ds[name] = self._constructDataArray(
			data = data,
			dims = dims,
			attrs = attrs,
			encoding = encoding
		)
		self._incDataCounts()

	def constructFilename(self) -> str:
		return super()._constructFilename("frxxIQ")
	
	def checkRequiredFields(self):
		super()._checkRequiredFields()

		#check iqdim
		vars = ["iqdim"]
		self.requiredBools["iqdim"] = self._checkVars(vars)

		#check pol
		vars = ["pol"]
		self.requiredBools["pol"] = self._checkVars(vars)

		#check noise
		vars = ["noise"]
		self.requiredBools["noise"] = self._checkVars(vars)

		#check calibration
		self.requiredBools["calibration"] = (
			self._checkVars(["Zcal"])
			and (self._checkVars(["Dcal", "Pcal"]) if len(self.ds["pol"]) == 2 else True)
		)

		#setSourceFile function
		attrs = ["source_files"]
		self.requiredBools["source_file"] = self._checkAttrs(attrs)

		#check data
		vars = ["iq"]
		self.requiredBools["data"] = self._checkVars(vars, False)

	def validateSelf(self) -> bool:
		base = super()._validateSelf()
		if not base:
			return False
		if not (self.ds.attrs["frxx_data_type"] == 'IQ'):
			return False
		return True
	
	def concat(self, other: "IQ", newSweep: bool = True) -> "IQ":
		merged = self._concat(other, newSweep)
		ret = IQ(merged)
		return ret
	def _alignTime(self, otherDs: xr.Dataset) -> xr.Dataset:
		selfStart = datetime.fromisoformat(self.ds.attrs["start_datetime"])
		otherStart = datetime.fromisoformat(otherDs.attrs["start_datetime"])
		
		offset_seconds = (otherStart - selfStart).total_seconds()
		
		otherDs = otherDs.assign_coords(time=otherDs["time"] + offset_seconds)
		otherDs["time"].attrs["units"] = self.ds["time"].attrs["units"]

		return otherDs
	def _alignSweep(self, otherDs: xr.Dataset) -> xr.Dataset:
		lastSweep = self.ds["sweep"].values[-1]
		otherDs = otherDs.assign_coords(sweep=otherDs["sweep"] + lastSweep+1)
		
		lastSweepNumber = self.ds["sweep_number"].values[-1]
		otherDs["sweep_number"].data = otherDs["sweep_number"] + lastSweepNumber+1
		
		timeLength = len(self.ds["time"])
		otherDs["sweep_start_ray_index"].data = otherDs["sweep_start_ray_index"].values + timeLength
		otherDs["sweep_end_ray_index"].data = otherDs["sweep_end_ray_index"].values + timeLength

		return otherDs
	def _concat(self, other: "frxxData", newSweep: bool = True) -> xr.Dataset:
		if not self.validateSelf():
			raise RuntimeError("First operand seems to be invalid.")
		if not other.validateSelf():
			raise RuntimeError("Second operand seems to be invalid.")
		if self.ds["volume_number"].data.item() != other.ds["volume_number"].data.item():
			warning = ("volume_number in both files should be equal before concatenation "
				 		  "to prevent second volume number from being overwritten")
			warnings.warn(warning)

		selfDsCpy = self.ds.copy(deep=False)
		otherDsCpy = other.ds.copy(deep=False)

		otherDsCpy = self._alignTime(otherDsCpy)
		otherDsCpy = self._alignSweep(otherDsCpy)

		selfDsCpy.attrs["end_datetime"] = otherDsCpy.attrs["end_datetime"]
		selfDsCpy.attrs["time_coverage_end"] = otherDsCpy.attrs["time_coverage_end"]
		selfDsCpy["time_coverage_end"].values = otherDsCpy["time_coverage_end"].values


		selfSfJson = json.loads(selfDsCpy.attrs["source_files"])
		otherSfJson = json.loads(otherDsCpy.attrs["source_files"])

		selfFiles = [sourceFile.makeFromJson(f) for f in selfSfJson["files"]]
		otherFiles = [sourceFile.makeFromJson(f) for f in otherSfJson["files"]]

		for fileToAdd in otherFiles:
			if fileToAdd.isHardware:
				if not selfFiles[-1].isHardware:
					selfFiles.append(fileToAdd)
			else:
				lastFile = selfFiles[-1]
				if not lastFile.isHardware and pathUtils.pathJsonEqual(fileToAdd.pathJson, lastFile.pathJson):
					#if its the last file (we merge)
					selfFiles[-1] += fileToAdd
				else:
					#if we need to append to list of files
					# check if it exists earlier (which would be an error)
					for searchFile in selfFiles[:-1]:
						if not searchFile.isHardware and pathUtils.pathJsonEqual(fileToAdd.pathJson, searchFile.pathJson):
							raise RuntimeError("Source file to add was found in the non-last position. "
											   "Merge is for immediate concatenations!")
					selfFiles.append(fileToAdd)


		selfSfJson["files"] = [f.toJson() for f in selfFiles]
		selfDsCpy.attrs["source_files"] = json.dumps(selfSfJson, indent='\t')

		if set(selfDsCpy.variables) != set(otherDsCpy.variables):
			raise ValueError(f"Variable mismatch: {set(selfDsCpy.variables)} vs {set(otherDsCpy.variables)}")

		merged = xr.concat(
			[selfDsCpy, otherDsCpy], 
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
				merged[var].loc[{"sweep":otherDsCpy["sweep"].values}] = otherDsCpy[var].values
		
		if not newSweep:
			mergedSweep = len(selfDsCpy["sweep"])-1
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

	def __add__(self, other: "IQ") -> "IQ":
		return self.concat(other)
	def __radd__(self, other: "int | IQ") -> "IQ":
		if type(other) is int:
			if other != 0:
				raise ValueError("Cannot add with nonzero.")
			return self
		other = cast("IQ", other)
		if type(other) is not type(self):
			raise ValueError(f"LHS needs to be {type(self)}")
		return other.__add__(self)
	def __iadd__(self, other: "IQ") -> "IQ":
		merged = self._concat(other)
		self.ds = merged
		self.checkRequiredFields()
		if not self.validateSelf():
			raise RuntimeError("Something went wrong merging.")
		return self
	
	def breakAt(self, index: int, newVol: bool = False, newSweep: bool = True) -> Tuple["IQ", "IQ"]:
		pass