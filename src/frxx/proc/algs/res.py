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


@njit('void(complex64[:,:], complex64[:,:], int32, int32, int32, int32, int64, int64)', parallel=True, cache=True)
def _rangeSubsetIQ(iq, result, K, Koffset, NR, startRange, fp, lp):
    ng, _ = iq.shape
    
    for r in prange(NR):
        iK = np.arange(0, K, 1) + (r)*K
        r_set_idx = np.arange(0, K, 1)+r-(K//2-Koffset)+startRange
        r_set_idx[r_set_idx < 0] = 0
        r_set_idx[r_set_idx > (ng-1)] = ng-1
        result[iK,:] = iq[r_set_idx,fp:lp+1]

@njit('void(complex64[:,:], complex64[:,:], int32, int32, int64[:], int64[:])', parallel=True, cache=True)
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
    '(complex64[:,:,], int64, float32[:], int64[:,:], int64[:], optional(int64), int64, optional(int64), optional(int64))',
    cache=True
)
def subsetIQ(
    iq: NDArray,
    iaz, azVals, 
    pulseBoundaries, 
    iranges, 
    swathPulses = None, 
    K = 1, KOffset = None, 
    avgStrat = None
):
    _, ns = iq.shape
    
    if K < 1:
        raise ValueError("K must be greater than 0.")

    i64KOffset = unwrap_i64(KOffset, 0)
    if i64KOffset not in [0, 1]:
        raise ValueError("Valid values for KOffset: {0(low), 1(high)}")
    i64avgStrat = unwrap_i64(avgStrat, 1)
    if i64avgStrat not in [0, 1]:
        raise ValueError("Valid values for i64avgStrat: {0(range), 1(azimuth)}")
    
    NR = iranges[1]+1 - iranges[0]
    
    result = np.empty((0,0), dtype=np.complex64)
    i64swathPulses = 0

    if K > 1:
        if i64avgStrat == 0:
            pixelBoundaries = pulseBoundaries[iaz]
            
            centerPulse = pixelBoundaries[0] + (pixelBoundaries[1]+1 - pixelBoundaries[0])//2
            if centerPulse < 0 or centerPulse >= ns:
                raise ValueError("Center pulse out of bounds.")
            i64swathPulses = unwrap_i64(swathPulses, pixelBoundaries[1]+1 - pixelBoundaries[0])
            firstPulse = centerPulse - i64swathPulses//2
            lastPulse = centerPulse + i64swathPulses//2 if i64swathPulses % 2 != 0 else centerPulse + i64swathPulses//2 - 1

            if firstPulse < 0:
                firstPulse = 0
            if lastPulse >= ns:
                lastPulse = ns - 1
            i64swathPulses = lastPulse - firstPulse + 1
            
            result = np.empty((K*NR, i64swathPulses), dtype=np.complex64)
            _rangeSubsetIQ(iq, result, K, i64KOffset, NR, iranges[0], firstPulse, lastPulse)
        elif i64avgStrat == 1:
            naz = len(azVals)
            if np.mean(np.sign(azVals[1:]-azVals[:-1])) > 0:
                az_set_idx = np.arange(0, K, 1)-(K//2-i64KOffset)+iaz
            else:
                az_set_idx = np.arange(K-1, -1, -1)-int(np.ceil(K/2)-np.abs(i64KOffset-1))+iaz
            az_set_idx[az_set_idx < 0] = 0
            az_set_idx[az_set_idx >= naz] = naz - 1
            
            pixelBoundaries = pulseBoundaries[az_set_idx]
            
            centerPulses = pixelBoundaries[:,0] + (pixelBoundaries[:,1]+1 - pixelBoundaries[:,0])//2
            if np.any(centerPulses < 0) or np.any(centerPulses >= ns):
                raise ValueError("A center pulse is out of bounds.")
            i64swathPulses = np.min(pixelBoundaries[:,1]+1 - pixelBoundaries[:,0])
            i64swathPulses = unwrap_i64(swathPulses, i64swathPulses)
            firstPulses = centerPulses - i64swathPulses//2
            lastPulses = centerPulses + i64swathPulses//2 if i64swathPulses % 2 != 0 else centerPulses + i64swathPulses//2 - 1
            
            firstPulses[firstPulses < 0] = 0
            lastPulses[lastPulses >= ns] = ns - 1
            i64swathPulses = np.min(lastPulses - firstPulses + 1)
            lastPulses = firstPulses + i64swathPulses - 1
            
            result = np.empty((K*NR, i64swathPulses), dtype=np.complex64)
            _azSubsetIQ(iq, result, NR, iranges[0], firstPulses, lastPulses)
            
    else:
        pixelBoundaries = pulseBoundaries[iaz]
        
        centerPulse = pixelBoundaries[0] + (pixelBoundaries[1]+1 - pixelBoundaries[0])//2
        if centerPulse < 0 or centerPulse >= ns:
            raise ValueError("Center pulse out of bounds.")
        i64swathPulses = unwrap_i64(swathPulses, pixelBoundaries[1]+1 - pixelBoundaries[0])
        firstPulse = centerPulse - i64swathPulses//2
        lastPulse = centerPulse + i64swathPulses//2 if i64swathPulses % 2 != 0 else centerPulse + i64swathPulses//2 - 1
        
        if firstPulse < 0:
            firstPulse = 0
        if lastPulse >= ns:
            lastPulse = ns - 1
        i64swathPulses = lastPulse - firstPulse + 1
        
        result = iq[iranges[0]:iranges[1]+1, firstPulse:lastPulse+1]
    
    return result, NR, i64swathPulses