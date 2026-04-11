import numpy as np
import xarray as xr

from typing import Tuple, List, Union, Sequence, Annotated
from numpy.typing import NDArray

import dask.array as da

from numba.typed import List as ListType

from .frxxData import _FILL_VALUES, frxxData
from ..utils import stringUtils as su

AvgStrat = Annotated[Tuple[int, int] | None, "None if no averaging, or a Tuple of (K_r, K_az)"]

class spectra(frxxData):
	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()

		self.nonDataVars += [
			   "ray_start_end", 
			   "pulse_boundaries",
			   "SNR_threshold",
			   "mask",
			   "velocity",
			   "vlen"
			]

		self.requiredBools["ray_start_end"] = False
		self.requiredBools["pulse_boundaries"] = False
		self.requiredBools["SNR_threshold"] = False
		self.requiredBools["mask"] = False
		self.requiredBools["beam_spec"] = False
		self.requiredBools["fourier_spec"] = False

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

	def setBeamSpec(self, angleSpacingDeg: float, beamOverlapDeg: float):
		if angleSpacingDeg <= 0 or beamOverlapDeg < 0:
			raise ValueError("Angle Spacing should be greater than 0 and beamOverlap should be 0 or greater.")
		if not np.isclose(1/angleSpacingDeg, np.round(1/angleSpacingDeg)):
			raise ValueError("Angle spacing should fit cleanly into one degree.")
		self.ds.attrs["beam_spec"] = su.constructBeamSpec(angleSpacingDeg, beamOverlapDeg)
		self.requiredBools["beam_spec"] = True

	def setFourierSpec(
		self, 
		pulsesIn: int | None, nFFTOut: int | None, 
		avgStrat: AvgStrat = None, 
		bootstrap: int | None = None
	):
		if (pulsesIn is not None and pulsesIn < 2) or (nFFTOut is not None and nFFTOut < 2):
			raise ValueError("pulsesIn and nFFTOut should be None or 2 or greater.")
		if (avgStrat is not None):
			if (not (isinstance(avgStrat, tuple) and len(avgStrat) == 2)
				and (all(isinstance(x, int) for x in avgStrat))):
				raise TypeError("avgStrat must be a tuple of two ints")
			if avgStrat[0] < 1 or avgStrat[1] < 1:
				raise ValueError("Both values of avgStrat must be one or greater")
		if (bootstrap is not None) and bootstrap < 2:
			raise ValueError("Bootstrap must be minimum 2 or None.")
		
		self.ds.attrs["fourier_spec"] = su.constructFourierSpec(pulsesIn, nFFTOut, avgStrat, bootstrap)
		self.requiredBools["fourier_spec"] = True

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

	def setMask(self, mask: List[NDArray[np.bool_]], comment: str):
		if not (isinstance(mask, List) or isinstance(mask, ListType)):
			raise TypeError("Spectra can be jagged so the first dim should be a python list. " \
							"This functon handles filling with NaNs.")
		nRays = len(self.ds["time"])
		if len(mask) != nRays:
			raise ValueError("Number of rays passed does not equal length of time coordinate.")
		if self.requiredBools["mask"]:
			raise RuntimeError("Mask is already set! Use updateMask. (Notimplemented yet.)")
		
		lens = []
		s = []
		e = []
		curr = 0
		for i, ray in enumerate(mask):
			shape = ray.shape
			if len(shape) != 2:
				raise ValueError(f"Counted {len(shape)} dimensions in ray {i}. Need 2!")
			r, v = shape
			if len(self.ds["range"]) != r:
				raise ValueError(f"Range (1st) dim of ray (i={i}, length={r} "
					 			"must match range coordinate!")
			s.append(curr)
			lens.append(v)
			curr += v
			e.append(curr-1)

		lenSum = sum(lens)
		lens = np.array(lens, dtype=np.int32)
		s = np.array(s, dtype=np.int64).reshape(-1, 1)
		e = np.array(e, dtype=np.int64).reshape(-1, 1)
		spectraBoundaries = np.concatenate([s,e], axis=1)

		self.ds = self.ds.assign_coords(velocity=np.arange(lenSum))
		self.ds["velocity"].attrs = {
			"long_name": "doppler_velocity_spectral_components",
			"units": "integer spectral component indices concatenated along time axis."
		}
		self.ds["velocity"].encoding = {
			"dtype": "int64",
			"_FillValue": _FILL_VALUES["int64"],
		}
		self.ds["vlens"] = self._constructDataArray(
			data = lens,
			dims = ["time"],
			attrs = {
				"comment": f"Lenth of velocity axis for given ray."
			},
			encoding = {
				"dtype": "int32",
				"_FillValue": _FILL_VALUES["int32"]
			}
		)
		self.ds["spectra_boundaries"] = self._constructDataArray(
			data = spectraBoundaries,
			dims = ["time", "ray_start_end"],
			attrs = {
				"comment": f"Start and end indexes of spectra for a given ray."
			},
			encoding = {
				"dtype": "int64",
				"_FillValue": _FILL_VALUES["int64"]
			}
		)
		
		concatData = da.concatenate(mask, axis=1)

		self.ds["mask"] = self._constructDataArray(
			data = concatData,
			dims = ["range", "velocity"],
			encoding = {
				"dtype": "int8",
				"_FillValue": _FILL_VALUES["int8"],
				"chunksizes": (len(self.ds["range"]), int(np.median(self.ds["vlens"].data))),
				"zlib": True,
				"complevel": 4
			},
			attrs ={
				"comment": "Boolean data mask. " + comment
			}
		)
		self.requiredBools["mask"] = True


	def addDataField(self, name: str, data: List[NDArray], attrs, encoding):
		if not all(self.requiredBools.values()):
			raise ValueError("Set all spectra object variables.")
		if not (isinstance(data, List) or isinstance(data, ListType)):
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
		
		lens = np.array(lens, dtype=np.int32)
		if not np.array_equal(lens, self.ds["vlens"].values):
			raise ValueError("All data variables must have the same shape.")

		concatData = da.concatenate(data, axis=1)

		self.ds[name] = self._constructDataArray(
			data = concatData.astype(np.float32),
			dims = ["range", "velocity"],
			attrs = attrs,
			encoding = encoding
		)
		self.ds[name].encoding["chunksizes"] = (len(self.ds["range"]), int(np.median(self.ds["vlens"].data)))
		self.ds[name].encoding["zlib"] = True
		self.ds[name].encoding["complevel"] = 4
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

		#check SNR_threshold
		vars = ["SNR_threshold"]
		self.requiredBools["SNR_threshold"] = self._checkVars(vars, False)

		attrs = ["beam_spec"]
		self.requiredBools["beam_spec"] = self._checkAttrs(attrs)

		attrs = ["fourier_spec"]
		self.requiredBools["fourier_spec"] = self._checkAttrs(attrs)

	def validateSelf(self) -> bool:
		base = super()._validateSelf()
		if not base:
			return False
		if not (self.ds.attrs["frxx_data_type"] == 'spectra'):
			return False
		return True
	
	@property
	def pb(self) -> NDArray:
		return np.ascontiguousarray(self.ds["pulse_boundaries"].data).astype(np.int64)
	
	@property
	def sb(self) -> NDArray:
		return np.ascontiguousarray(self.ds["spectra_boundaries"].data).astype(np.int64)
	
	@property
	def mask(self) -> List[NDArray[np.bool_]]:
		if not self.requiredBools["mask"]:
			raise AttributeError("Mask has not been set.")

		self.ds["spectra_boundaries"].data = np.asarray(self.ds["spectra_boundaries"].data)
		bnd = self.ds["spectra_boundaries"].data
		data = []
		for i in range(len(bnd)):
			data.append(self.ds["mask"].data[:, bnd[i][0]:bnd[i][1]+1])
		return data

	@property
	def vlens(self) -> NDArray:
		return np.ascontiguousarray(self.ds["vlens"].data).astype(np.int64)

	def __getattr__(self, name):
		if name.startswith("m_"):
			masked = True
			name = name[2:]
		else:
			masked = False
		if name in self.nonDataVars:
			raise ValueError("Non data variables have their own attribute getters.")
		if name not in self.ds:
			raise ValueError("Attribute not found in dataset.")
		
		self.ds["spectra_boundaries"].data = np.asarray(self.ds["spectra_boundaries"].data)
		bnd = self.ds["spectra_boundaries"].data	
		if masked:
			array = self.ds[name].where(self.ds["mask"])
		else:
			array = self.ds[name]
		data = []
		for i in range(len(bnd)):
			data.append(array.data[:, bnd[i][0]:bnd[i][1]+1])
		return data