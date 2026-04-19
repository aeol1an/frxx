from ...core import spectra
from ...core.frxxData import _FILL_VALUES

from ..algs import fuzzyDCA as DCA

from ...utils.numbaHelpers import toNumbaList

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

	def processRay_S_precompute(vars, pts, filterStrength):
		PSDH, sZDR, sRHOHV = tuple(vars[n*nr:(n+1)*nr] for n in range(3))
		return DCA.processRay_S(PSDH, sZDR, sRHOHV, pts, filterStrength)


	rays = [
		d.delayed(processRay_S_precompute, nout=5)( #type: ignore
			da.concatenate((PSDH[az], sZDR[az], sRHOHV[az]), axis=0), np.int64(pts), t.type(filterStrength)
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