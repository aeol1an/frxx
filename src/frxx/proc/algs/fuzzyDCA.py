import numpy as np

from typing import Tuple
from numpy.typing import NDArray

from numba import njit, prange
from ...utils.numbaHelpers import \
	get_masked_float2d, \
	set_masked_float2d_scalar, set_masked_float2d_array, \
	nanargmax
from ...utils.freqResolution import velocityAxis

@njit(
	[
		'float32[:,:](float32[:,:], int64)',
		'float64[:,:](float64[:,:], int64)'
	],
	parallel=True
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
_fieldIDX = ['sZDR', 'sRHOHV', 'sZDRv', 'sRHOHVv']
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
		'float32[:](float32[:], float32, float32, int64)',
		'float32[:,:](float32[:,:], float32, float32, int64)',
		'float64[:](float64[:], float64, float64, int64)',
		'float64[:,:](float64[:,:], float64, float64, int64)'
	],
	cache=True
)
def _membershipFnLine(x, x1, x2, sign):
	t = x.dtype
	m = t.type(sign) * (t.type(1.0)/(x2-x1))
	return m * (x-x1) + (t.type(0.0) if sign>0 else t.type(1.0))

@njit(
	[
		'float32[:,:](float32[:,:], int64, int64)',
		'float64[:,:](float64[:,:], int64, int64)'
	], cache=True,
)
def _membership(x, scattererClass: int, field: int):
	side, thresholds = _numbaMembershipThresholds[scattererClass][field]
	ret = np.full(x.shape, np.nan, dtype=x.dtype)
	if side == 1:
		X1, X2, X3, X4 = thresholds
		set_masked_float2d_scalar(ret, x < X1, 0.0)
		m1 = (x >= X1) & (x < X2)
		m2 = (x >= X3) & (x < X4)
		set_masked_float2d_array(ret, m1, _membershipFnLine(get_masked_float2d(x, m1), X1, X2, 1))
		set_masked_float2d_scalar(ret, (x >= X2) & (x < X3), 1.0)
		set_masked_float2d_array(ret, m2, _membershipFnLine(get_masked_float2d(x, m2), X3, X4, -1))
		set_masked_float2d_scalar(ret, x >= X4, 0.0)
	elif side == 0:
		X3, X4, _, _ = thresholds
		set_masked_float2d_scalar(ret, x < X3, 1.0)
		m1 = (x >= X3) & (x < X4)
		set_masked_float2d_array(ret, m1, _membershipFnLine(get_masked_float2d(x, m1), X3, X4, -1))
		set_masked_float2d_scalar(ret, x >= X4, 0.0)
	else:
		X1, X2, _, _ = thresholds
		set_masked_float2d_scalar(ret, x < X1, 0.0)
		m1 = (x >= X1) & (x < X2)
		set_masked_float2d_array(ret, m1, _membershipFnLine(get_masked_float2d(x, m1), X1, X2, 1))
		set_masked_float2d_scalar(ret, x >= X2, 1.0)
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
def calcAggregation(sZDR, sRHOHV, sZDRv, sRHOHVv, PSDH, filterStrength: float = 8):
	t = sZDR.dtype
	Arain = np.clip(
		_membership(sZDR, 0, 0) * t.type(0.25) + 
		_membership(sRHOHV, 0, 1) * t.type(0.25) + 
		_membership(sZDRv, 0, 2) * t.type(0.25) + 
		_membership(sRHOHVv, 0, 3) * t.type(0.25),
		0, 1
	)
	
	Adebris = np.clip(
		_membership(sZDR, 1, 0) * t.type(0.10)+ 
		_membership(sRHOHV, 1, 1) * t.type(0.25)+ 
		_membership(sZDRv, 1, 2) * t.type(0.40)+ 
		_membership(sRHOHVv, 1, 3) * t.type(0.25),
		0, 1
	)
	
	return Arain, Arain/(Arain+Adebris), 10*np.log10((10**(PSDH/10))*(Arain**filterStrength))

@njit(
	[
		'Tuple((float32[:,:], float32[:,:], '
		'float32[:,:], float32[:,:], float32[:,:]))'
		'(float32[:,:], float32[:,:], float32[:,:], int64, float32)',

		'Tuple((float64[:,:], float64[:,:], '
		'float64[:,:], float64[:,:], float64[:,:]))'
		'(float64[:,:], float64[:,:], float64[:,:], int64, float64)',
	],
	cache=True
)
def processRay_S(
	PSDH, 
	sZDR, sRHOHV,
	pts: int = 9, filterStrength: float = 8
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
	sZDRv = calcVariance(sZDR, pts)
	sRHOHVv = calcVariance(sRHOHV, pts)
	Arain, Anrain, PSDHF = calcAggregation(sZDR, sRHOHV, sZDRv, sRHOHVv, PSDH, filterStrength)
	return sZDRv, sRHOHVv, Arain, Anrain, PSDHF

@njit(
	[
		'Tuple((float32[:], float32[:]))(float32[:,:], float32[:,:], float32[:], float32, boolean)',
		'Tuple((float64[:], float64[:]))(float64[:,:], float64[:,:], float64[:], float64, boolean)'
	],
	cache=True, parallel=True
)
def processRay_M(PSDHF, PSDH, vACF, va, flipVel):
	t = PSDHF.dtype
	nr, nv = PSDHF.shape
	PSDHF = 10**(PSDHF/10)
	PSDH = 10**(PSDH/10)

	vAxis = velocityAxis(nv, va, flipVel, 0, 0)

	vDCA = np.empty((nr,), dtype=t)
	correction = np.empty((nr,), dtype=t)
	for r in prange(nr):
		#find vDCA
		if np.isnan(PSDHF[r]).all() or np.isnan(vACF[r]):
			vDCA[r] = np.nan
			correction[r] = 0.0
			continue
		P = np.nansum(PSDHF[r])
		km = nanargmax(PSDHF[r])
		vMax = vAxis[km]
		delV = vAxis - vMax
		Vn = 2 * va
		delV = ((delV + va) % Vn) - va  # wrap into [-va, +va]
		vDCA[r] = vMax + (1/P) * np.nansum(delV * PSDHF[r])

		#calculate correction
		iACF = np.argmin(np.abs(vAxis-vACF[r]))
		iDCA = np.argmin(np.abs(vAxis-vDCA[r]))

		#case equal
		if iACF == iDCA:
			correction[r] = vDCA[r] - vACF[r]
			continue

		lower = iACF if iACF < iDCA else iDCA
		higher = iACF if iACF > iDCA else iDCA

		# find nanmin of unfiltered spectrum in between region [lower, higher]
		betweenMin = np.inf
		for i in range(lower, higher + 1):
			v = PSDH[r][i]
			if not np.isnan(v) and v < betweenMin:
				betweenMin = v

		# find nanmin of unfiltered spectrum in outside region [0, lower] u [higher, nv)
		outsideMin = np.inf
		for i in range(0, lower + 1):
			v = PSDH[r][i]
			if not np.isnan(v) and v < outsideMin:
				outsideMin = v
		for i in range(higher, nv):
			v = PSDH[r][i]
			if not np.isnan(v) and v < outsideMin:
				outsideMin = v

		# if global min is in outside region, no aliasing
		if outsideMin <= betweenMin:
			correction[r] = vDCA[r] - vACF[r]
			continue

		# global min is in between region — possible aliasing
		# count NaNs in filtered spectrum to confirm
		betweenNaN = 0
		betweenTotal = 0
		for i in range(lower, higher + 1):
			betweenTotal += 1
			if np.isnan(PSDHF[r][i]):
				betweenNaN += 1

		outsideNaN = 0
		outsideTotal = 0
		for i in range(0, lower + 1):
			outsideTotal += 1
			if np.isnan(PSDHF[r][i]):
				outsideNaN += 1
		for i in range(higher, nv):
			outsideTotal += 1
			if np.isnan(PSDHF[r][i]):
				outsideNaN += 1

		betweenNaNFrac = 0.0 if betweenTotal == 0 else betweenNaN / betweenTotal
		outsideNaNFrac = 0.0 if outsideTotal == 0 else outsideNaN / outsideTotal

		# if between has more NaNs (more filtered out = more noise/no signal),
		# confirms aliasing — the valley between is just noise
		if betweenNaNFrac >= outsideNaNFrac:
			if vACF[r] < vDCA[r]:
				vACFUnfolded = vACF[r] + 2 * va
				correction[r] = vDCA[r] - vACFUnfolded
			else:
				vDCAUnfolded = vDCA[r] + 2 * va
				correction[r] = vDCAUnfolded - vACF[r]
		else:
			# between has fewer NaNs (more signal) — not aliased after all
			correction[r] = vDCA[r] - vACF[r]

	return vDCA, correction

