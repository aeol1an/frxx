import numpy as np

from typing import Tuple
from numpy.typing import NDArray

from numba import njit, prange
from ...utils.numbaHelpers import unwrap_i64

def averageAlongRange(data: NDArray, gstep: int) -> NDArray:
	if gstep == 1:
		return data
	
	t, r = (data.shape[0], data.shape[1])
	fullGroups = r//gstep
	if fullGroups > 0:
		mainPart = data[:,:fullGroups*gstep,...]\
			.reshape((t, fullGroups, gstep, *data.shape[2:])).mean(axis=2)
	else:
		mainPart = np.array([]).reshape((t, 0, *data.shape[2:]))
		
	if r % gstep != 0:
		remainder = data[:,fullGroups*gstep:,...].mean(axis=1)
		result = np.concatenate((mainPart, remainder), axis=1)
	else:
		result = mainPart
		
	return result


@njit('void(complex64[:,:], complex64[:,:], int32, int32, int32, int32, int64, int64)', parallel=True, cache=True, nogil=True
def _rangeSubsetIQ(iq, result, K, Koffset, NR, startRange, fp, lp):
	ng, _ = iq.shape
	
	for r in prange(NR):
		iK = np.arange(0, K, 1) + (r)*K
		r_set_idx = np.arange(0, K, 1)+r-(K//2-Koffset)+startRange
		r_set_idx[r_set_idx < 0] = 0
		r_set_idx[r_set_idx > (ng-1)] = ng-1
		result[iK,:] = iq[r_set_idx,fp:lp+1]

@njit('void(complex64[:,:], complex64[:,:], int32, int32, int64[:], int64[:])', parallel=True, cache=True, nogil=True)
def _azSubsetIQ(iq, result, NR, startRange, fps, lps):
	naz = len(fps)
	for r in prange(NR):
		for az in range(naz):
			fp = fps[az]
			lp = lps[az]
			iK = r * naz + az
			result[iK,:] = iq[r+startRange, fp:lp+1]

@njit(
	'Tuple((complex64[:,:], int64, int64))'
	'(complex64[:,:,], int64, int64, boolean, int64[:,:], int64[:], '
	'int64, int64, int64, int64)',
	cache=True, nogil=True
)
def subsetIQnumba(
	iq: NDArray,
	iaz, naz, azIncreasing, 
	pulseBoundaries, 
	iranges, 
	swathPulses = -1, 
	K = 1, KOffset = 0, 
	avgStrat = 1
):
	_, ns = iq.shape
	
	if K < 1:
		raise ValueError("K must be greater than 0.")

	if KOffset not in [0, 1]:
		raise ValueError("Valid values for KOffset: {0(low), 1(high)}")
	if KOffset not in [0, 1]:
		raise ValueError("Valid values for i64avgStrat: {0(range), 1(azimuth)}")
	
	NR = iranges[1]+1 - iranges[0]
	
	result = np.empty((0,0), dtype=np.complex64)

	if K > 1:
		if avgStrat == 0:
			pixelBoundaries = pulseBoundaries[iaz]
			
			centerPulse = pixelBoundaries[0] + (pixelBoundaries[1]+1 - pixelBoundaries[0])//2
			if centerPulse < 0 or centerPulse >= ns:
				raise ValueError("Center pulse out of bounds.")
			if swathPulses < 2:
				swathPulses = pixelBoundaries[1]+1 - pixelBoundaries[0]
			firstPulse = centerPulse - swathPulses//2
			lastPulse = centerPulse + swathPulses//2 if swathPulses % 2 != 0 else centerPulse + swathPulses//2 - 1

			if firstPulse < 0:
				firstPulse = 0
			if lastPulse >= ns:
				lastPulse = ns - 1
			swathPulses = lastPulse - firstPulse + 1
			
			result = np.empty((K*NR, swathPulses), dtype=np.complex64)
			_rangeSubsetIQ(iq, result, K, KOffset, NR, iranges[0], firstPulse, lastPulse)
		elif avgStrat == 1:
			if azIncreasing:
				az_set_idx = np.arange(0, K, 1)-(K//2-KOffset)+iaz
			else:
				az_set_idx = np.arange(K-1, -1, -1)-int(np.ceil(K/2)-np.abs(KOffset-1))+iaz
			az_set_idx[az_set_idx < 0] = 0
			az_set_idx[az_set_idx >= naz] = naz - 1
			
			pixelBoundaries = pulseBoundaries[az_set_idx]
			
			centerPulses = pixelBoundaries[:,0] + (pixelBoundaries[:,1]+1 - pixelBoundaries[:,0])//2
			if np.any(centerPulses < 0) or np.any(centerPulses >= ns):
				raise ValueError("A center pulse is out of bounds.")
			if swathPulses < 2:
				swathPulses = np.min(pixelBoundaries[:,1]+1 - pixelBoundaries[:,0])
			firstPulses = centerPulses - swathPulses//2
			lastPulses = centerPulses + swathPulses//2 if swathPulses % 2 != 0 else centerPulses + swathPulses//2 - 1
			
			firstPulses[firstPulses < 0] = 0
			lastPulses[lastPulses >= ns] = ns - 1
			swathPulses = np.min(lastPulses - firstPulses + 1)
			lastPulses = firstPulses + swathPulses - 1
			
			result = np.empty((K*NR, swathPulses), dtype=np.complex64)
			_azSubsetIQ(iq, result, NR, iranges[0], firstPulses, lastPulses)
			
	else:
		pixelBoundaries = pulseBoundaries[iaz]
		
		centerPulse = pixelBoundaries[0] + (pixelBoundaries[1]+1 - pixelBoundaries[0])//2
		if centerPulse < 0 or centerPulse >= ns:
			raise ValueError("Center pulse out of bounds.")
		if swathPulses < 2:
			swathPulses = pixelBoundaries[1]+1 - pixelBoundaries[0]
		firstPulse = centerPulse - swathPulses//2
		lastPulse = centerPulse + swathPulses//2 if swathPulses % 2 != 0 else centerPulse + swathPulses//2 - 1
		
		if firstPulse < 0:
			firstPulse = 0
		if lastPulse >= ns:
			lastPulse = ns - 1
		swathPulses = lastPulse - firstPulse + 1
		
		result = iq[iranges[0]:iranges[1]+1, firstPulse:lastPulse+1]
	
	return result, NR, swathPulses
def _subsetIQStrToInt(KOffset: str | None, avgStrat: str | None) -> Tuple[int, int]:
	if KOffset is None:
		k = 0
	elif KOffset not in ["low", "high"]:
		raise ValueError("KOffset must be 'low' or 'high'.")
	else:
		if KOffset == "low":
			k = 0
		else:
			k = 1

	if avgStrat is None:
		a = 1
	elif avgStrat not in ["r", "az"]:
		raise ValueError("avgStrat must be 'r' or 'az'.")
	else:
		if avgStrat == "r":
			a = 0
		else:
			a = 1

	return k, a

def subsetIQ(
	iq: NDArray,
	iaz, naz, azIncreasing, 
	pulseBoundaries, 
	iranges, 
	swathPulses = None, 
	K = 1, KOffset = None, 
	avgStrat = None
):
	if swathPulses is None:
		swathPulses = -1
	KOffset, avgStrat = _subsetIQStrToInt(KOffset, avgStrat)
	return subsetIQnumba(
		iq, 
		iaz, naz, azIncreasing, 
		pulseBoundaries, iranges, swathPulses, 
		K, KOffset, 
		avgStrat
	)



	

# TODO: Zero-copy refactor for K>1
# Instead of allocating a 2D result matrix with copied IQ data,
# return the original IQ block (or a contiguous slice of it) plus
# a small Nx3 int64 index array of (r, firstPulse, lastPulse).
# Downstream function loops over the index array and accesses
# iq[r, firstPulse:lastPulse+1] directly — no IQ copy needed.
# Index array allocation is negligible (3 int64s per row vs full complex64 row).
# Cache behavior should be fine: az averaging hits one row repeatedly,
# range averaging walks r like [0,0,1,0,1,2,1,2,3,...] — sequential neighbors.
# This also naturally extends to 3D (group index rows by azimuth)
# without requiring the downstream computation to change much.