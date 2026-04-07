from ...core import moments, spectra
from ...core.frxxData import _FILL_VALUES

from ..algs import fuzzyDCA as DCA

from typing import Tuple, List
from numpy.typing import NDArray

import numpy as np

from numba import njit, prange

@njit(
	[
		'Tuple((float32[:,:], float32[:,:]))'
		'(ListType(float32[:,:]), ListType(float32[:,:]), float32[:,:], float32, boolean)',

		'Tuple((float64[:,:], float64[:,:]))'
		'(ListType(float64[:,:]), ListType(float64[:,:]), float64[:,:], float64, boolean)',
	],
	cache=True, parallel=True
)
def _processRays(PSDHF, PSDH, VEL, va, flipVel):
	naz = len(PSDHF)

	DCAVEL = np.empty(VEL.shape, dtype=VEL.dtype)
	DCAVC = np.empty(VEL.shape, dtype=VEL.dtype)

	for az in prange(naz):
		vdca, corr = DCA.processRay_M(PSDHF[az], PSDH[az], VEL[az], va, flipVel)
		DCAVEL[az,:] = vdca
		DCAVC[az,:] = corr

	return DCAVEL, DCAVC

def addFields(m: moments, s: spectra) -> None:
	DCAVEL, DCAVC = _processRays(s.PSDHF, s.PSDH, m.m_VEL, m.va[0], m.phaseReversed)

	encoding = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": 0.01,
		"add_offset": 0.0
	}

	m.addDataField(
		"DCAVEL", DCAVEL, encoding=encoding,
		attrs={
			"long_name": "DCA_filtered_doppler_velocity",
			"units": "m/s",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField(
		"DCAVC", DCAVC, encoding=encoding,
		attrs={
			"long_name": "DCA_doppler_velocity_correction",
			"units": "m/s",
			"grid_mapping": "grid_mapping",
		}
	)