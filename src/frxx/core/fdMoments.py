import numpy as np
import xarray as xr

from typing import List, Union, Sequence
from numpy.typing import NDArray

from .frxxData import _FILL_VALUES, frxxData

class moments(frxxData):
	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()

		self.nonDataVars += [
			"ray_start_end", "pulse_boundaries", 
			"pulse_boundaries", "SNR_threshold", "mask",
		]

		self.requiredBools["ray_start_end"] = False
		self.requiredBools["pulse_boundaries"] = False
		self.requiredBools["SNR_threshold"] = False
		self.requiredBools["mask"] = False

		if ds is None:
			self.ds.attrs["frxx_data_type"] = "moments"
			self.ds.attrs["Conventions"] = "CF/Radial"
			self.ds.attrs["version"] = "CF-Radial-1.3"

			#add pulse bounds dim
			self.ds = self.ds.assign_coords(
				ray_start_end=np.arange(2)
			)
			self.ds["ray_start_end"].encoding = {
				"dtype": "int32",
				"_FillValue": _FILL_VALUES["int32"]
			}
			self.requiredBools["ray_start_end"] = True
		else:
			self.ds = ds.copy(deep=False)
			self.checkRequiredFields()
			valid = self.validateSelf()
			if not valid:
				raise RuntimeError("Invalid format. See above.")

	def setPulseBoundaries(self, boundaries: NDArray[np.int64]):
		if (boundaries.dtype != np.int64):
			raise TypeError("Expected array of np.int64")
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime() before this function.")
		if boundaries.shape != (len(self.ds["time"]), len(self.ds["ray_start_end"])):
			raise RuntimeError("Number of start-end pairs need to match number of rays.")
		
		self.ds["pulse_boundaries"] = xr.DataArray(
			data = boundaries,
			dims = ["time", "ray_start_end"],
			attrs = {
				"long_name": "first_and_last_pulse_indices_in_ray",
                "comment": "First and last pulse index in a ray in corresponding frxxIQ file."
			}
		)
		self.ds["pulse_boundaries"].encoding = {
			"dtype": "int64",
			"_FillValue": _FILL_VALUES["int64"]
		}

		self.requiredBools["pulse_boundaries"] = True

	def setSNRThreshold(self, snrt_db: Union[NDArray[np.floating], Sequence[float]]):
		if not self.requiredBools["pol"]:
			raise RuntimeError("Number of polarizations not set yet.")
		if len(snrt_db) != len(self.ds["pol"]):
			raise ValueError(f"Number of thresholds passed does not match length of pol dimension.")
		
		self.ds["SNR_threshold"] = xr.DataArray(
			data = np.array(snrt_db),
			dims = ["pol"],
			attrs = {
				"long_name": "signal_to_noise_thresholds_for_spectral_masking",
                "comment": "First and last pulse index in a ray in corresponding frxxIQ file."
			}
		)
		self.ds["SNR_threshold"].encoding = {
			"dtype": "float32",
			"_FillValue": _FILL_VALUES["float32"]
		}

		self.requiredBools["SNR_threshold"] = True

	def setMask(self, mask: NDArray[np.bool_], comment: str):
		if mask.shape != (len(self.ds["time"]), len(self.ds["range"])):
			raise ValueError("Mask shape doesnn't match data shape.")
		
		self.ds["mask"] = self._constructDataArray(
			data = mask,
			dims = ["time", "range"],
			encoding = {
				"dtype": "int8",
				"_FillValue": _FILL_VALUES["int8"]
			},
			attrs ={
				"comment": "Boolean data mask. " + comment
			}
		)
		
		self.requiredBools["mask"] = True

	def addDataField(self, name: str, data: NDArray, attrs, encoding):
		if not all(self.requiredBools.values()):
			raise ValueError("Set all moment object variables.")
		if data.shape != (len(self.ds["time"]), len(self.ds["range"])):
			raise ValueError("Data shape shape doesn't match dims.")

		self.ds[name] = self._constructDataArray(
			data = data,
			dims = ["time", "range"],
			attrs = attrs,
			encoding = encoding
		)
		self._incDataCounts()

	def constructFilename(self) -> str:
		return super()._constructFilename("cfradial")
	
	def checkRequiredFields(self):
		super()._checkRequiredFields()

		#check ray_start_end
		vars = ["ray_start_end"]
		self.requiredBools["ray_start_end"] = self._checkVars(vars)

		#check pulse_boundaries
		vars = ["pulse_boundaries"]
		self.requiredBools["pulse_boundaries"] = self._checkVars(vars)

		#check SNR_threshold
		vars = ["SNR_threshold"]
		self.requiredBools["SNR_threshold"] = self._checkVars(vars, False)

		#check mask
		vars = ["mask"]
		self.requiredBools["mask"] = self._checkVars(vars)

	def validateSelf(self) -> bool:
		base = super()._validateSelf()
		if not base:
			return False
		if not (self.ds.attrs["frxx_data_type"] == 'moments'):
			return False
		return True

	@property
	def pb(self) -> NDArray:
		return np.ascontiguousarray(self.ds["pulse_boundaries"].data).astype(np.int64)
	
	@property
	def mask(self) -> NDArray:
		return np.ascontiguousarray(self.ds["mask"].data).astype(bool)
	
	def _getattr__(self, name):
		if name in self.nonDataVars:
			raise ValueError("Non data variables have their own attribute getters.")
		if name not in self.ds:
			raise ValueError("Attribute not found in dataset.")
		return self.ds[name]