from numba import njit, types
from numba.core.types import FunctionType
import numpy as np

type = FunctionType(types.float64[:](types.int64))

@njit('float64[:](int64)', cache=True, nogil=True)
def rectangular(N):
	return np.ones(N, dtype=np.float64)

@njit('float64[:](int64)', cache=True, nogil=True)
def hanning(N):
	n = np.arange(N)
	return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (N - 1)))

@njit('float64[:](int64)', cache=True, nogil=True)
def hamming(N):
	n = np.arange(N)
	return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (N - 1))

@njit('float64[:](int64)', cache=True, nogil=True)
def blackman(N):
	n = np.arange(N)
	return (0.42
		- 0.50 * np.cos(2.0 * np.pi * n / (N - 1))
		+ 0.08 * np.cos(4.0 * np.pi * n / (N - 1)))

@njit('float64[:](int64)', cache=True, nogil=True)
def bartlett(N):
	n = np.arange(N)
	return 1.0 - np.abs(2.0 * n / (N - 1) - 1.0)