import numpy as np
import xarray as xr

from numpy.typing import NDArray
from typing import List, Tuple, Union, cast
from collections.abc import Sequence

from .frxxData import _FILL_VALUES, frxxData

class IQ(frxxData['IQ']):
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
		merged = super()._concat(other, newSweep)
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
		merged = super()._concat(other)
		self.ds = merged
		self.checkRequiredFields()
		if not self.validateSelf():
			raise RuntimeError("Something went wrong merging.")
		return self
	
	def breakAt(self, index: int, newVol: bool = False, newSweep: bool = True) -> Tuple["IQ", "IQ"]:
		pass