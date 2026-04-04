import numpy as np
import xarray as xr

from pathlib import Path
import json

from numpy.typing import NDArray
from typing import List, Tuple, Union, cast
from collections.abc import Sequence
import warnings

from datetime import datetime

from ..utils import pathUtils, sourceFile, cfUtils
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
			"_FillValue": _FILL_VALUES["int32"]
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

	def addDataField(self, name: str, data: NDArray, attrs, encoding):
		if not all(self.requiredBools.values()):
			raise ValueError("Set all IQ object variables.")

		self.ds[name] = self._constructDataArray(
			data = data,
			dims = ["pol", "range", "time", "iqdim"],
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
		# vars = ["iq"]
		# self.requiredBools["data"] = self._checkVars(vars, False)

	def validateSelf(self) -> bool:
		base = super()._validateSelf()
		if not base:
			return False
		if not (self.ds.attrs["frxx_data_type"] == 'IQ'):
			return False
		return True
	
	def concat(self, other: "IQ") -> "IQ":
		merged = cfUtils.cfConcat(self.ds, other.ds, False)

		selfSfJson = json.loads(self.ds.attrs["source_files"])
		otherSfJson = json.loads(other.ds.attrs["source_files"])

		selfFiles = [sourceFile.makeFromJson(f) for f in selfSfJson["files"]]
		otherFiles = [sourceFile.makeFromJson(f) for f in otherSfJson["files"]]

		for fileToAdd in otherFiles:
			if fileToAdd.isHardware:
				if selfFiles[-1].isHardware:
					selfFiles[-1] += fileToAdd
				else:
					selfFiles.append(fileToAdd)
			else:
				fileToAdd.pathJson = cast(dict, fileToAdd.pathJson)
				lastFile = selfFiles[-1]
				if not lastFile.isHardware:
					lastFile.pathJson = cast(dict, lastFile.pathJson) 
					if pathUtils.pathJsonEqual(fileToAdd.pathJson, lastFile.pathJson):
						#if its the last file (we merge)
						selfFiles[-1] += fileToAdd
					else:
						#if we need to append to list of files
						# check if it exists earlier (which would be an error)
						for searchFile in selfFiles[:-1]:
							if not searchFile.isHardware:
								searchFile.pathJson = cast(dict, searchFile.pathJson) 
								if pathUtils.pathJsonEqual(fileToAdd.pathJson, searchFile.pathJson):
									raise RuntimeError("Source file to add was found in the non-last position. "
													"Merge is for immediate concatenations!")
						selfFiles.append(fileToAdd)

		selfSfJson["files"] = [f.toJson() for f in selfFiles]
		merged.attrs["source_files"] = json.dumps(selfSfJson, indent='\t')

		ret = IQ(merged)
		return ret

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
		merged = cfUtils.cfConcat(self.ds, other.ds)
		self.ds = merged
		self.checkRequiredFields()
		if not self.validateSelf():
			raise RuntimeError("Something went wrong merging.")
		return self
	
	def breakAt(self, index: int, newVol: bool = False) -> Tuple["IQ", "IQ"]:
		ds1, ds2 = cfUtils.cfBreakAt(self.ds, index, newVol)
		if (ds1 is None) or (ds2 is None):
			raise ValueError("Index must be 1 to len(ds)-1.")
		
		sfJson = json.loads(self.ds.attrs["source_files"])
		files = [sourceFile.makeFromJson(f) for f in sfJson["files"]]

		ds1Files: List[sourceFile] = []
		ds2Files: List[sourceFile] = []
		accumulated = 0

		for f in files:
			nRays = f.nRays()

			if accumulated + nRays <= index:
				ds1Files.append(f)
			elif accumulated >= index:
				ds2Files.append(f)
			else:
				# Break falls within this file
				remaining = index - accumulated
				if f.isHardware:
					left, right = f.breakAt(remaining)
				else:
					count = 0
					originalIndex = None
					for start, end in f.indices:
						pairSize = end - start + 1
						if count + pairSize >= remaining:
							originalIndex = start + (remaining - count)
							break
						count += pairSize
					left, right = f.breakAt(originalIndex)
				ds1Files.append(left)
				ds2Files.append(right)

			accumulated += nRays

		# Validate ray counts match time dimension lengths
		ds1RayCount = sum(f.nRays() for f in ds1Files)
		ds2RayCount = sum(f.nRays() for f in ds2Files)
		if ds1RayCount != ds1.sizes["time"]:
			raise RuntimeError(f"ds1 source file ray count ({ds1RayCount}) "
							f"does not match time dimension ({ds1.sizes['time']}).")
		if ds2RayCount != ds2.sizes["time"]:
			raise RuntimeError(f"ds2 source file ray count ({ds2RayCount}) "
							f"does not match time dimension ({ds2.sizes['time']}).")

		ds1SfJson = {**sfJson, "files": [f.toJson() for f in ds1Files]}
		ds2SfJson = {**sfJson, "files": [f.toJson() for f in ds2Files]}

		ds1.attrs["source_files"] = json.dumps(ds1SfJson, indent='\t')
		ds2.attrs["source_files"] = json.dumps(ds2SfJson, indent='\t')

		IQ1 = IQ(ds1)
		IQ1.checkRequiredFields()
		if not IQ1.validateSelf():
			raise RuntimeError("Something went wrong breaking ds1.")
		
		IQ2 = IQ(ds2)
		IQ2.checkRequiredFields()
		if not IQ2.validateSelf():
			raise RuntimeError("Something went wrong breaking ds2.")
		
		return IQ1, IQ2
	
	@property
	def iq(self):
		if (len(self.ds["pol"]) == 2):
			return np.ascontiguousarray(self.ds["iq"].data).view(np.complex64).squeeze()
		else:
			return np.ascontiguousarray(self.ds["iq"].data[0]).view(np.complex64).squeeze()
	
	@property
	def iqh(self):
		return np.ascontiguousarray(self.ds["iq"].data[0]).view(np.complex64).squeeze()
	
	@property
	def iqv(self):
		if (len(self.ds["pol"]) != 2):
			raise ValueError("Vertical channel iq only availible for dual-pol data.")
		return np.ascontiguousarray(self.ds["iq"].data[1]).view(np.complex64).squeeze()