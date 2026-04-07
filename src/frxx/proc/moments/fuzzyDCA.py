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
		'',

		'Tuple((ListType(float64[:,::1]), ListType(float64[:,::1]), '
		'ListType(float64[:,::1]), ListType(float64[:,::1]), ListType(float64[:,::1])))'
		'(ListType(float64[:,:]), ListType(float64[:,:]), ListType(float64[:,:]), int64, float64)',
	],
	cache=True, parallel=True
)