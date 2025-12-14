import numpy as np

from typing import Tuple
import numpy.typing as npt

def calcVariance(fields: Tuple[npt.NDArray], pts: int):
    result = []
    for field in fields:
        fieldResult = np.empty(field.shape, dtype=field.dtype)
        for idx in range(field.shape[-1]):
            lowval = idx - (pts//2)
            highval = idx + (pts//2 if pts%2 == 0 else pts//2+1)
            if lowval < 0:
                lowval = 0
            if highval > field.shape[-1]:
                highval = field.shape[-1]

            fieldResult[...,idx] = np.nanvar(field[...,lowval:highval], axis=-1)
        result.append(fieldResult)
    return tuple(result)

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

def _membershipFnLine(x, x1, x2, sign):
    m = sign * (1/(x2-x1))
    return m*(x-x1) + (0 if sign>0 else 1)

def _membership(x, scattererClass: str, variable: str):
    side, thresholds = _membershipThresholds[scattererClass][variable]
    x = np.array(x)
    ret = np.array(x)
    if side == 'full':
        X1, X2, X3, X4 = thresholds
        ret[x < X1] = 0
        ret[(x >= X1) & (x < X2)] = _membershipFnLine(x[(x >= X1) & (x < X2)], X1, X2, 1)
        ret[(x >= X2) & (x < X3)] = 1
        ret[(x >= X3) & (x < X4)] = _membershipFnLine(x[(x >= X3) & (x < X4)], X3, X4, -1)
        ret[x >= X4] = 0
    elif side == 'left':
        X3, X4 = thresholds
        ret[x < X3] = 1
        ret[(x >= X3) & (x < X4)] = _membershipFnLine(x[(x >= X3) & (x < X4)], X3, X4, -1)
        ret[x >= X4] = 0
    else:
        X1, X2 = thresholds
        ret[x < X1] = 0
        ret[(x >= X1) & (x < X2)] = _membershipFnLine(x[(x >= X1) & (x < X2)], X1, X2, 1)
        ret[x >= X2] = 1
    return ret
        

def calcAggregation(ZDR, rhoHV, ZDRvar, rhoHVvar):
    currClass = 'rain'
    Arain = \
        0.25*_membership(ZDR, currClass, 'ZDR') + \
        0.25*_membership(rhoHV, currClass, 'rhoHV') + \
        0.25*_membership(ZDRvar, currClass, 'ZDRvar') + \
        0.25*_membership(rhoHVvar, currClass, 'rhoHVvar')
    
    currClass = 'debris'
    Adebris = \
        0.1*_membership(ZDR, currClass, 'ZDR') + \
        0.25*_membership(rhoHV, currClass, 'rhoHV') + \
        0.4*_membership(ZDRvar, currClass, 'ZDRvar') + \
        0.25*_membership(rhoHVvar, currClass, 'rhoHVvar')
        
    return Arain, Adebris

def calcVelocityWithVAxis(HREF, vAxis):
    weakEcho = np.isnan(HREF).all(axis=1)
    HREF = HREF[~weakEcho,:]
    P = np.nansum(HREF, axis=1)
    km = np.nanargmax(HREF, axis=1)
    vMax = vAxis[km]
    delV = np.tile(vAxis, (len(km),1)) - vMax.reshape(-1, 1)
    vDCAPre = vMax + (1/P)*np.nansum(delV*HREF, axis=1)
    vDCA = np.empty(shape=len(weakEcho))
    vDCA[weakEcho] = np.nan
    vDCA[~weakEcho] = vDCAPre
    return vDCA