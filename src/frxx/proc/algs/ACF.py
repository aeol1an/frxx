import numpy as np
from numba import njit, prange

@njit('complex128[:](complex64[:,:], complex64[:,:], int32)', parallel=True, cache=True)
def correlation(X1, X2, lag=0):
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

@njit(
	'Tuple((complex128[:,:,:], complex128[:,:], complex128[:,:]))'
	'(complex64[:,:], complex64[:,:], int64[:,:], int32[:])', 
	parallel=True, cache=True
)
def processRays(iqh, iqv, pulseBoundaries, lags=np.array([0,1], dtype=np.int32)):
	#nRange, nBigTime, nLags
	nRange = iqh.shape[0]
	nBigTime = pulseBoundaries.shape[0]
	nLags = lags.shape[0]

	RH = np.empty((nBigTime, nRange, nLags), dtype=np.complex128)
	RV = np.empty((nBigTime, nRange), dtype=np.complex128)
	RX = np.empty((nBigTime, nRange), dtype=np.complex128)

	for t in range(nBigTime):
		iqhs = iqh[:,pulseBoundaries[t][0]:pulseBoundaries[t][1]]
		iqvs = iqv[:,pulseBoundaries[t][0]:pulseBoundaries[t][1]]
		RV[t,:] = correlation(iqvs, iqvs, 0)
		RX[t,:] = correlation(iqhs, iqvs, 0)
		for l in range(nLags):
			RH[t,:,l] = correlation(iqhs, iqhs, lags[l])

	return RH, RV, RX