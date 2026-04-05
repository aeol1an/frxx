import numpy as np

from typing import Tuple
from numpy.typing import NDArray

from numba import njit, prange

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


@njit('void(complex64[:,:,:], complex64[:,:,:], int32, int32, int32, int32, int64, int64)', parallel=True, cache=True)
def _rangeSubsetIQ(iq, result, K, Koffset, NR, startRange, fp, lp):
    _, ng, n_ = iq.shape
    
    for r in prange(NR):
        iK = np.arange(0, K, 1) + (r)*K
        r_set_idx = np.arange(0, K, 1)+r-(K//2-Koffset)+startRange
        r_set_idx[r_set_idx < 0] = 0
        r_set_idx[r_set_idx > (ng-1)] = ng-1
        result[:,iK,:] = iq[:,r_set_idx,fp:lp+1]
@njit('void(complex64[:,:,:], complex64[:,:,:], int32, int32, int64[:], int64[:])', parallel=True, cache=True)
def _azSubsetIQ(iq, result, NR, startRange, fps, lps):
    iK = 0
    naz = len(fps)
    for r in prange(NR):
        for az in range(naz):
            fp = fps[az]
            lp = lps[az]
            result[:,iK,:] = iq[:, r+startRange,fp:lp+1]
            iK+=1
def subsetIQ(
    iq: Tuple[NDArray],
    iaz, azVals, 
    pulseBoundaries, 
    iranges, 
    swathPulses = None, 
    K = 1, KOffset = 'low', 
    avgStrat = 'az'
):
    ng, ns = iq[0].shape
    
    if K % 2 == 1:
        KOffset = 0
    else:
        if KOffset == 'low':
            KOffset = 0
        elif KOffset == 'high':
            KOffset = 1
        else:
            raise ValueError("Valid values for KOffset: {'low', 'high'}")
    
    NR = iranges[1]+1 - iranges[0]
    
    result = np.array([], dtype=iq[0].dtype)

    if K > 1:
        if avgStrat == 'r':
            pixelBoundaries = pulseBoundaries[iaz]
            
            centerPulse = pixelBoundaries[0] + (pixelBoundaries[1]+1 - pixelBoundaries[0])//2
            if swathPulses is None:
                swathPulses = pixelBoundaries[1]+1 - pixelBoundaries[0]
            firstPulse = centerPulse - swathPulses//2
            lastPulse = centerPulse + swathPulses//2 if swathPulses % 2 != 0 else centerPulse + swathPulses//2 - 1

            if firstPulse < 0 or lastPulse >= ns:
                raise ValueError("Swath too large and pulse out of bounds.")
            
            result = np.empty((len(iq), K*NR, swathPulses), dtype=iq[0].dtype)
            _rangeSubsetIQ(np.array(iq), result, K, KOffset, NR, iranges[0], firstPulse, lastPulse)
        elif avgStrat == 'az':
            if np.mean(np.sign(np.diff(azVals))) > 0:
                az_set_idx = np.arange(0, K, 1)-(K//2-KOffset)+iaz
            else:
                az_set_idx = np.arange(K-1, -1, -1)-int(np.ceil(K/2)-np.abs(KOffset-1))+iaz
            if np.any(az_set_idx < 0) or np.any(az_set_idx >= len(azVals)):
                raise ValueError("Some azimuths being averaged over do not exist. Lower K or move target azimuth away from edge.")
            
            pixelBoundaries = pulseBoundaries[az_set_idx]
            
            centerPulses = pixelBoundaries[:,0] + (pixelBoundaries[:,1]+1 - pixelBoundaries[:,0])//2
            if swathPulses is None:
                swathPulses = np.min(pixelBoundaries[:,1]+1 - pixelBoundaries[:,0])
            firstPulses = centerPulses - swathPulses//2
            lastPulses = centerPulses + swathPulses//2 if swathPulses % 2 != 0 else centerPulses + swathPulses//2 - 1
            
            result = np.empty((len(iq), K*NR, swathPulses), dtype=iq[0].dtype)
            _azSubsetIQ(np.array(iq), result, NR, iranges[0], firstPulses, lastPulses)
            
    else:
        pixelBoundaries = pulseBoundaries[iaz]
        
        centerPulse = pixelBoundaries[0] + (pixelBoundaries[1]+1 - pixelBoundaries[0])//2
        if swathPulses is None:
            swathPulses = pixelBoundaries[1]+1 - pixelBoundaries[0]
        firstPulse = centerPulse - swathPulses//2
        lastPulse = centerPulse + swathPulses//2 if swathPulses % 2 != 0 else centerPulse + swathPulses//2 - 1
        
        result = tuple(iqi[iranges[0]:iranges[1]+1,firstPulse:lastPulse+1] for iqi in iq)
    
    return tuple(result), NR, swathPulses