from ...core import spectra
from ...core.frxxData import _FILL_VALUES

from ..algs import fuzzyDCA as DCA

from typing import Tuple, List
from numpy.typing import NDArray

import numpy as np

import dask as d
import dask.array as da


def processRays(
	PSDH: List[NDArray], 
	sZDR: List[NDArray], sRHOHV: List[NDArray], 
	nf: NDArray,
	pts: int, filterStrength: float
) -> Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], List[NDArray]]:
	naz = len(PSDH)
	nr = PSDH[0].shape[0]
	t = PSDH[0].dtype

	def to_single_delayed(dask_arr):
		return dask_arr.rechunk(dask_arr.shape).to_delayed(optimize_graph=False).item()

	rays = [
		d.delayed(DCA.processRay_S)( #type: ignore
			*[to_single_delayed(arr) for arr in (PSDH[az], sZDR[az], sRHOHV[az])], np.int64(pts), t.type(filterStrength)
		)
		for az in range(naz)
	]

	results = []
	for product in range(5):
		chunks = [
			da.from_delayed(
				rays[az][product].astype(np.float32), #type: ignore
				shape=(nr, nf[az]),
				dtype=np.float32,
			)
			for az in range(naz)
		]
		results.append(chunks)

	return tuple(results)


def addFields(s: spectra, pts: int = 9, filterStrength: float = 8.0, delayed = True) -> None:
	sZDRv, sRHOHVv, Arain, Anrain, PSDHF = processRays(
		s.PSDH,
		s.sZDR, s.sRHOHV,
		s.vlens,
		pts, filterStrength
	)
	encoding = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": np.float32(0.01),
		"add_offset": np.float32(0.0)
	}
	encodingSmall = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": np.float32(0.0001),
		"add_offset": np.float32(0.0)
	}

	# PSDHF can contain -inf, or finite values below the range representable by
	# its packed int16 encoding. Letting those values reach the NetCDF encoder
	# can wrap them into large positive powers that dominate the DCA centroid.
	# Reserve the lowest int16 code for the fill value and store every value that
	# cannot be represented safely as NaN, which CF encoding maps to that fill.
	packedMin = np.float32(
		(np.iinfo(np.int16).min + 1) * encoding["scale_factor"] +
		encoding["add_offset"]
	)
	packedMax = np.float32(
		np.iinfo(np.int16).max * encoding["scale_factor"] +
		encoding["add_offset"]
	)
	PSDHF = [
		da.where(
			da.isfinite(ray) & (ray >= packedMin) & (ray <= packedMax),
			ray,
			np.float32(np.nan),
		)
		for ray in PSDHF
	]

	s.addDataField('sZDRv', sZDRv, encoding=encoding,
		attrs={
			"long_name": "spectral_differential_reflectivity_variance",
			"window_length": str(pts),
			"units": "dB^2",
		}
	)
	s.addDataField('sRHOHVv', sRHOHVv, encoding=encodingSmall,
		attrs={
			"long_name": "spectral_correlation_coefficient_variance",
			"window_length": str(pts),
			"units": "unitless",
		}
	)
	s.addDataField('Arain', Arain, encoding=encodingSmall,
		attrs={
			"long_name": "raw_rain_aggregation",
			"units": "unitless",
		}
	)
	s.addDataField('Anrain', Anrain, encoding=encodingSmall,
		attrs={
			"long_name": "normalized_rain_aggregation",
			"comment": "Greater than 0.5 indicates DCA rain classification.",
			"units": "unitless",
		}
	)
	s.addDataField('PSDHF', PSDHF, encoding=encoding,
		attrs={
			"long_name": "DCA_filtered_power_spectral_density",
			"filter_strength": f"{filterStrength:.2f}",
			"units": "dB",
		}
	)

	if not delayed:
		s.load(['sZDRv', 'sRHOHVv', 'Arain', 'Anrain', 'PSDHF'])

	return
