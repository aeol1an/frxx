import numpy as np
import xarray as xr

from numpy.typing import NDArray
from typing import List, Union
from collections.abc import Sequence

import json

from .data import _FILL_VALUES, frxxData

class IQ(frxxData):
	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()
		
		self.requiredBoolsIQ = {
			"iqdim": False,
			"pol": False,
			"noise": False,
			"calibration": False
		}

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
			self.requiredBoolsIQ["iqdim"] = True

		else:
			self.ds = ds
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

		self.requiredBoolsIQ["pol"] = True

	def setNoisedB(self, n0_dB: Union[NDArray[np.floating], Sequence[float]]):
		if not self.requiredBoolsIQ["pol"]:
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

		self.requiredBoolsIQ["noise"]

	def setCal(self, Zcal_db: Union[NDArray[np.floating], Sequence[float]], Dcal_db: float | None = None, Pcal_deg: float | None = None):
		if not self.requiredBoolsIQ["pol"]:
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

		self.requiredBoolsIQ["calibration"] = True

	def addDataField(self, name: str, data: NDArray, dims: List[str], attrs, encoding):
		if dims != ["pol", "range", "time", "iqdim"] or name != 'iq':
			raise ValueError("Please only add iq data to this data structure!")
		self.ds["iq"] = self._constructDataArray(
			data = data,
			dims = dims,
			attrs = attrs,
			encoding = encoding
		)

	def constructFilename(self) -> str:
		return super()._constructFilename("frxxIQ")
	
	def checkRequiredFields(self):
		super()._checkRequiredFields()

		#check iqdim
		vars = ["iqdim"]
		self.requiredBoolsIQ["iqdim"] = self._checkVars(vars)

		#check pol
		vars = ["pol"]
		self.requiredBoolsIQ["pol"] = self._checkVars(vars)

		#check noise
		vars = ["noise"]
		self.requiredBoolsIQ["noise"] = self._checkVars(vars)

		#check calibration
		self.requiredBoolsIQ["calibration"] = (
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
		if self.ds.attrs["frxx_data_type"] == 'IQ':
			return False
		if not all(self.requiredBoolsIQ.values()):
			print("Some IQ specific required bools have not been set.")
			return False
		return True