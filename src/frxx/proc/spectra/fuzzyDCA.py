from ...core import spectra
from ...core.frxxData import _FILL_VALUES

from ..algs import fuzzyDCA as DCA

from typing import Tuple, List
from numpy.typing import NDArray

import numpy as np

from numba import njit, prange
from numba.typed import List as TypedList

@njit(
	[
		'Tuple((ListType(float32[:,::1]), ListType(float32[:,::1]), '
		'ListType(float32[:,::1]), ListType(float32[:,::1]), ListType(float32[:,::1])))'
		'(ListType(float32[:,:]), ListType(float32[:,:]), ListType(float32[:,:]), int64, float32)',

		'Tuple((ListType(float64[:,::1]), ListType(float64[:,::1]), '
		'ListType(float64[:,::1]), ListType(float64[:,::1]), ListType(float64[:,::1])))'
		'(ListType(float64[:,:]), ListType(float64[:,:]), ListType(float64[:,:]), int64, float64)',
	],
	cache=True, parallel=True, nogil=True
)
def _processRays(
	PSDH, 
	sZDR, sRHOHV,
	pts: int, filterStrength: float
) -> Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], List[NDArray]]:
	naz = len(PSDH)

	dummy = np.empty((0, 0), dtype=PSDH[0].dtype)
	sZDRv = TypedList()
	sRHOHVv = TypedList()
	Arain = TypedList()
	Anrain = TypedList()
	PSDHF = TypedList()
	for _ in range(naz):
		sZDRv.append(dummy)
		sRHOHVv.append(dummy)
		Arain.append(dummy)
		Anrain.append(dummy)
		PSDHF.append(dummy)

	for az in prange(naz):
		szv, srv, ar, anr, p = DCA.processRay_S(PSDH[az], sZDR[az], sRHOHV[az], pts, filterStrength)
		sZDRv[az] = np.ascontiguousarray(szv)
		sRHOHVv[az] = np.ascontiguousarray(srv)
		Arain[az] = np.ascontiguousarray(ar)
		Anrain[az] = np.ascontiguousarray(anr)
		PSDHF[az] = np.ascontiguousarray(p)

	return sZDRv, sRHOHVv, Arain, Anrain, PSDHF

def addFields(s: spectra, pts: int = 9, filterStrength: float = 8.0) -> None:
	sZDRv, sRHOHVv, Arain, Anrain, PSDHF = _processRays(
		s.m_PSDH,
		s.m_sZDR, s.m_sRHOHV,
		pts, s.ds.PSDH.dtype.type(filterStrength)
	)
	encoding = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": 0.01,
		"add_offset": 0.0
	}
	encodingSmall = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": 0.0001,
		"add_offset": 0.0
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

	return