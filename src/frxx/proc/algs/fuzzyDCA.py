import numpy as np

from typing import Tuple
from numpy.typing import NDArray

from numba import njit, prange
from ...utils.numbaHelpers import get_masked_float2d, set_masked_float2d, nanargmax, nanargmin
from ...utils.freqResolution import velocityAxis

@njit(
    [
        'float32[:,:](float32[:,:], int64)',
        'float64[:,:](float64[:,:], int64)'
    ],
    cache=True, parallel=True
)
def calcVariance(field: NDArray, pts: int = 9):
    nr, nv = field.shape
    fieldResult = np.empty(field.shape, dtype=field.dtype)
    for idx in prange(nv):
        lowval = idx - (pts//2)
        highval = idx + (pts//2 if pts%2 == 0 else pts//2+1)
        if lowval < 0:
            lowval = 0
        if highval > nv:
            highval = nv

        for r in range(nr):
            fieldResult[r,idx] = np.nanvar(field[r,lowval:highval])
    return fieldResult

_classIDX = ['rain', 'debris']
_fieldIDX = ['sZDR', 'sRHOHV', 'sZDRvar', 'sRHOHVvar']
_sideIDX = ['left', 'full', 'right']
_membershipThresholds = {
    'rain' : {
        'ZDR': ('full', (-1.5, 1.0, 2.0, 4.0)),
        'rhoHV': ('right', (0.79, 0.98)),
        'ZDRvar': ('left', (0.6, 5.0)),
        'rhoHVvar': ('left', (0.00025,  0.027))
    },
    'debris': {
        'ZDR': ('full', (-19.0, -7.4, 1.7, 10.6)),
        'rhoHV': ('full', (0.0, 0.3, 0.94, 0.99)),
        'ZDRvar': ('right', (0.4, 7.1)),
        'rhoHVvar': ('right', (0.0001, 0.027))
    }
}
_numbaMembershipThresholds = (
    (
        (1, (-1.5, 1.0, 2.0, 4.0)),
        (2, (0.79, 0.98, -9999.0, -9999.0)),
        (0, (0.6, 5.0, -9999.0, -9999.0)),
        (0, (0.00025,  0.027, -9999.0, -9999.0))
    ),
    (
        (1, (-19.0, -7.4, 1.7, 10.6)),
        (1, (0.0, 0.3, 0.94, 0.99)),
        (2, (0.4, 7.1, -9999.0, -9999.0)),
        (2, (0.0001, 0.027, -9999.0, -9999.0))
    )
)

@njit(
    [
        'float32[:,:](float32[:,:], float32, float32, int64)',
        'float64[:,:](float64[:,:], float64, float64, int64)'
    ],
    inline='always', cache=True
)
def _membershipFnLine(x, x1, x2, sign):
    t = x.dtype
    m = t.type(sign) * (t.type(1.0)/(x2-x1))
    return m * (x-x1) + (t.type(0.0) if sign>0 else t.type(1.0))

@njit(
    [
        'float32[:,:](float32[:,:], int64, int64)',
        'float64[:,:](float64[:,:], int64, int64)'
    ]
    , cache=True, inline='always'
)
def _membership(x, scattererClass: int, field: int):
    side, thresholds = _numbaMembershipThresholds[scattererClass][field]
    ret=np.empty(x.shape, dtype=x.dtype)
    if side == 1:
        X1, X2, X3, X4 = thresholds
        set_masked_float2d(ret, x < X1, 0.0)
        m1 = (x >= X1) & (x < X2)
        m2 = (x >= X3) & (x < X4)
        set_masked_float2d(ret, m1, _membershipFnLine(get_masked_float2d(x, m1), X1, X2, 1))
        set_masked_float2d(ret, (x >= X2) & (x < X3), 1.0)
        set_masked_float2d(ret, m2, _membershipFnLine(get_masked_float2d(x, m2), X3, X4, -1))
        set_masked_float2d(ret, x >= X4, 0.0)
    elif side == 0:
        X3, X4, _, _ = thresholds
        set_masked_float2d(ret, x < X3, 1.0)
        m1 = (x >= X3) & (x < X4)
        set_masked_float2d(ret, m1, _membershipFnLine(get_masked_float2d(x, m1), X3, X4, -1))
        set_masked_float2d(ret, x >= X4, 0.0)
    else:
        X1, X2, _, _ = thresholds
        set_masked_float2d(ret, x < X1, 0.0)
        m1 = (x >= X1) & (x < X2)
        set_masked_float2d(ret, m1, _membershipFnLine(get_masked_float2d(x, m1), X1, X2, 1))
        set_masked_float2d(ret, x >= X2, 1.0)
    return ret
        
@njit(
    [
        'Tuple((float32[:,:], float32[:,:], float32[:,:]))'
        '(float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:], int64)',
        
        'Tuple((float64[:,:], float64[:,:], float64[:,:]))'
        '(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], int64)',        
    ],
    cache=True
)
def processRay(sZDR, sRHOHV, sZDRvar, sRHOHVvar, PSDH, filterStrength=8):
    t = sZDR.dtype
    Arain = \
        _membership(sZDR, 0, 0) * t.type(0.25) + \
        _membership(sRHOHV, 0, 1) * t.type(0.25) + \
        _membership(sZDRvar, 0, 2) * t.type(0.25) + \
        _membership(sRHOHVvar, 0, 3) * t.type(0.25)
    
    Adebris = \
        _membership(sZDR, 1, 0) * t.type(0.10)+ \
        _membership(sRHOHV, 1, 1) * t.type(0.25)+ \
        _membership(sZDRvar, 1, 2) * t.type(0.40)+ \
        _membership(sRHOHVvar, 1, 3) * t.type(0.25)
        
    return Arain, Arain/(Arain+Adebris), (10**(PSDH/10))*(Arain**filterStrength)

@njit(
    [
        'Tuple((float32[:], float32[:]))(float32[:,:], float32[:], float32, boolean)',
        'Tuple((float64[:], float64[:]))(float64[:,:], float64[:], float64, boolean)'
    ],
    cache=True, parallel=True
)
def calcVelocity(PSDH_f, vACF, va, flipVel):
    t = PSDH_f.dtype
    nr, nv = PSDH_f.shape

    vAxis = velocityAxis(nv, va, flipVel)

    vDCA = np.empty((nr,), dtype=t)
    correction = np.empty((nr,), dtype=t)
    for r in prange(nr):
        #find vDCA
        if np.isnan(PSDH_f[r]).all():
            vDCA[r] = np.nan
            correction[r] = 0.0
            continue
        P = np.nansum(PSDH_f[r])
        km = nanargmax(PSDH_f[r])
        vMax = vAxis[km]
        delV = vAxis - vMax
        vDCA[r] = vMax + (1/P)*np.nansum(delV*PSDH_f[r])

        #calculate correction
        if np.isnan(vACF[r]):
            correction[r] = 0.0
            continue
        iACF = np.argmin(np.abs(vAxis-vACF[r]))
        iDCA = np.argmin(np.abs(vAxis-vDCA[r]))

        #case equal
        if iACF == iDCA:
            correction[r] = vDCA[r] - vACF[r]
            continue

        lower = iACF if iACF < iDCA else iDCA
        higher = iACF if iACF > iDCA else iDCA
        betweenMean = np.nanmean(PSDH_f[r][lower:higher+1])

        outsideSum = 0.0
        outsideCount = 0
        for i in range(higher, nv):
            v = PSDH_f[r][i]
            if not np.isnan(v):
                outsideSum += v
                outsideCount += 1
        for i in range(0, lower + 1):
            v = PSDH_f[r][i]
            if not np.isnan(v):
                outsideSum += v
                outsideCount += 1
        outsideMean = 0.0 if outsideCount is 0 else outsideSum/outsideCount

        #if between is greater, assume no aliaing and standard correction
        if betweenMean > outsideMean:
            correction[r] = vDCA[r] - vACF[r]
            continue
        
        #otherwise, we have to fold one of them to be on the same "segment"
        #fold the lower one to the right
        if vACF[r] < vDCA[r]:
            vACFUnfolded = vACF[r] + 2*va
            correction[r] = vDCA[r] - vACFUnfolded
        else:
            vDCAUnfolded = vDCA[r] + 2*va
            correction[r] = vDCAUnfolded - vACF[r]

    return vDCA, correction

