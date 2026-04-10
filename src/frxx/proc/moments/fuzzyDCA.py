from ...core import moments, spectra
from ...core.frxxData import _FILL_VALUES

from ..algs import fuzzyDCA as DCA

from typing import Tuple, List
from numpy.typing import NDArray

import numpy as np

import dask as d
import dask.array as da

from numba import njit, prange
from numba.typed import List as ListType

# @njit(
# 	[
# 		'Tuple((float32[:,:], float32[:,:]))'
# 		'(ListType(float32[:,:]), ListType(float32[:,:]), float32[:,:], float32, boolean)',

# 		'Tuple((float64[:,:], float64[:,:]))'
# 		'(ListType(float64[:,:]), ListType(float64[:,:]), float64[:,:], float64, boolean)',
# 	],
# 	cache=True, parallel=True, nogil=True
# )
# def _processRaysP(PSDHF, PSDH, VEL, va, flipVel):
# 	naz = len(PSDHF)

# 	DCAVEL = np.empty(VEL.shape, dtype=VEL.dtype)
# 	DCAVC = np.empty(VEL.shape, dtype=VEL.dtype)

# 	for az in prange(naz):
# 		vdca, corr = DCA.processRay_M(PSDHF[az], PSDH[az], VEL[az], va, flipVel)
# 		DCAVEL[az,:] = vdca
# 		DCAVC[az,:] = corr

# 	return DCAVEL, DCAVC

def _processRays(PSDHF, PSDH, VEL, va, flipVel):
	naz = len(PSDHF)
	nr = len(VEL[0])

	rays = [
		d.delayed(DCA.processRay_M)( #type: ignore
			PSDHF[az], PSDH[az], VEL[az], va, flipVel
		)
		for az in range(naz)
	]

	results = []
	for product in range(2):
		chunks = [
			da.from_delayed(
				rays[az][product].reshape(1, -1).astype(np.float32), #type: ignore
				shape=(1, nr),
				dtype=np.float32,
			)
			for az in range(naz)
		]
		results.append(np.concatenate(chunks, axis=0))

	return tuple(results)


def addFields(m: moments, s: spectra) -> None:
	s.load(["PSDH", "PSDHF"])

	DCAVEL, DCAVDIFF = _processRays(s.PSDHF, s.PSDH, m.m_VEL, m.va[0], m.phaseReversed)
	#DCAVEL, DCAVDIFF = _processRaysP(ListType(s.PSDHF), ListType(s.PSDH), m.VEL, m.va[0], m.phaseReversed)

	print("DCAVDIFF NaNs match VEL mask?", np.array_equal(np.isnan(DCAVDIFF), m.mask))
	print("Any DCAVDIFF NaN?", np.any(np.isnan(DCAVDIFF)))
	print("_FILL_VALUES['int16'] =", _FILL_VALUES["int16"])


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
		"DCAVDIFF", DCAVDIFF, encoding=encoding,
		attrs={
			"long_name": "DCA_doppler_velocity_correction",
			"units": "m/s",
			"grid_mapping": "grid_mapping",
		}
	)

	m.load(["DCAVEL", "DCAVDIFF"])