import numpy as np
import xarray as xr

from typing import List
from numpy.typing import NDArray

from .frxxData import _FILL_VALUES, frxxData

class moments(frxxData):
	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()

		self.nonDataVars += ["ray_start_end", "pulse_boundaries", "pol"]

		self.requiredBools["ray_start_end"] = False
		self.requiredBools["pulse_boundaries"] = False
		self.requiredBools["pol"] = False

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

	def addDataField(self, name: str, data: NDArray, dims: List[str], attrs, encoding):
		if dims != ["time", "range"]:
			raise ValueError("Moment data must be in [\"time\", \"range\"]!")
		
		if not all(self.requiredBools.values()):
			raise ValueError("Set all moment object variables.")

		self.ds[name] = self._constructDataArray(
			data = data,
			dims = dims,
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

		#check pol
		vars = ["pol"]
		self.requiredBools["pol"] = self._checkVars(vars)

	def validateSelf(self) -> bool:
		base = super()._validateSelf()
		if not base:
			return False
		if not (self.ds.attrs["frxx_data_type"] == 'moments'):
			return False
		return True

	