import numpy as np
from numba import njit, prange

@njit('complex128[:](complex64[:,:], complex64[:,:], int32)', nogil=True, parallel=True, cache=True)
def computeRay_M(X1, X2, lag=0):
	if X1.shape != X2.shape:
		raise ValueError("Two array shapes not equal.")
	nr, nt = X1.shape
	result = np.empty(nr, dtype=np.complex128)

	if lag == 0:
		for i in prange(nr):
			acc = np.complex128(0)
			for j in range(nt):
				acc += np.complex128(X1[i, j]) * np.conj(np.complex128(X2[i, j]))
			result[i] = acc / nt
	elif lag > 0:
		for i in prange(nr):
			acc = np.complex128(0)
			for j in range(nt - lag):
				acc += np.complex128(X1[i, j + lag]) * np.conj(np.complex128(X2[i, j]))
			result[i] = acc / nt
	else:
		neg_lag = -lag
		for i in prange(nr):
			acc = np.complex128(0)
			for j in range(nt + lag):
				acc += np.complex128(X1[i, j]) * np.conj(np.complex128(X2[i, j + neg_lag]))
			result[i] = acc / nt

	return result