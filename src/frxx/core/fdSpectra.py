import numpy as np
import xarray as xr

from typing import List
from numpy.typing import NDArray

from .frxxData import _FILL_VALUES, frxxData

class spectra(frxxData):
	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()

		self.nonDataVars += [
			   "ray_start_end", 
			   "pulse_boundaries",
			   "pol"
			   "SNR_threshold"
			]

		self.requiredBools["ray_start_end"] = False
		self.requiredBools["pulse_boundaries"] = False
		self.requiredBools["pol"] = False
		self.requiredBools["SNR_threshold"] = False

		if ds is None:
			self.ds.attrs["frxx_data_type"] = "spectra"

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

	def setPulseBoundaries(self, boundaries: NDArray[np.int32]):
		if (boundaries.dtype != np.int32):
			raise TypeError("Expected array of np.int32")
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
			"dtype": "int32",
			"_FillValue": _FILL_VALUES["int32"]
		}

		self.requiredBools["pulse_boundaries"] = True

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


	def setSNRThreshold(self, snrh: float, snrv: float | None = None):
		if not self.requiredBools["pol"]:
			raise RuntimeError("Number of polarizations not set yet.")
		if not isinstance(snrh, (int, float)):
			raise TypeError(f"Expected Number for snrh, got {type(snrh).__name__}")
		if not isinstance(snrv, (int, float, type(None))):
			raise TypeError(f"Expected Number or None for snrv, got {type(snrh).__name__}")
		if snrv is None and len(self.ds["pol"]) == 2:
			raise ValueError(f"Dual-pol was set, but only snrh threshold set. Pass 0 if no snrv threshold.")
		
		self.ds["SNR_threshold"] = xr.DataArray(
			data = [snrh, snrv],
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

	def addDataField(self, name: str, data: List[NDArray], dims: List[str], attrs, encoding):
		if not all(self.requiredBools.values()):
			raise ValueError("Set all spectra object variables.")
		if dims != ["time", "range", "velocity"]:
			raise ValueError("Spectral data must be in [\"time\", \"range\", \"velocity\"]!")
		if not isinstance(data, List):
			raise TypeError("Spectra can be jagged so the first dim should be a python list. " \
							"This functon handles filling with NaNs.")
		nRays = len(self.ds["time"])
		if len(data) != nRays:
			raise ValueError("Number of rays passed does not equal length of time coordinate.")
		
		lens = []
		for i, ray in enumerate(data):
			shape = ray.shape
			if len(shape) != 2:
				raise ValueError(f"Counted {len(shape)} dimensions in ray {i}. Need 2!")
			r, v = shape
			if len(self.ds["range"]) != r:
				raise ValueError(f"Range (1st) dim of ray (i={i}, length={r} "
					 			"must match range coordinate!")
			lens.append(v)
		
		maxLen = max(lens)
		lens = np.array(lens, dtype=np.int32)
		if ("velocity" not in self.ds) or (maxLen > self.ds["velocity"]):
			#hopefully this fills with NaNs. Prayge.
			self.ds = self.ds.assign_coords(pol=np.arange(maxLen))
			self.ds["velocity"].attrs = {
				"long_name": "doppler_velocity_spectral_components",
				"units": "integer spectral component indices"
			}
			self.ds["velocity"].encoding = {
				"dtype": "int32",
				"_FillValue": _FILL_VALUES["int32"]
			}

		padLen = len(self.ds["velocity"])
		paddedData = []
		for ray in data:
			_, v = ray.shape
			paddedData.append(np.pad(ray, ((0,0),(0,padLen-v)), constant_values=np.nan))
		paddedData = np.array(paddedData, dtype=np.float32)

		self.ds[name] = self._constructDataArray(
			data = paddedData,
			dims = dims,
			attrs = attrs,
			encoding = encoding
		)
		self._incDataCounts()

	def constructFilename(self) -> str:
		return super()._constructFilename("frxxS")
	
	def checkRequiredFields(self):
		super()._checkRequiredFields()

		#check ray_start_end
		vars = ["ray_start_end"]
		self.requiredBools["iqdim"] = self._checkVars(vars)

		#setSourceFile function
		vars = ["pulse_boundaries"]
		self.requiredBools["pulse_boundaries"] = self._checkVars(vars)

		#check pol
		vars = ["pol"]
		self.requiredBools["pol"] = self._checkVars(vars)

		#check SNR_threshold
		vars = ["SNR_threshold"]
		self.requiredBools["SNR_threshold"] = self._checkVars(vars)

	def validateSelf(self) -> bool:
		base = super()._validateSelf()
		if not base:
			return False
		if not (self.ds.attrs["frxx_data_type"] == 'spectra'):
			return False
		return True