import numpy as np

from typing import TYPE_CHECKING, Tuple
from numpy.typing import NDArray

if TYPE_CHECKING:
	ComplexArray = NDArray[np.complex64] | NDArray[np.complex128]
	Int64Array = NDArray[np.int64]

	def _rangeSubsetIQ(
		iq: ComplexArray,
		result: ComplexArray,
		K: int,
		Koffset: int,
		NR: int,
		startRange: int,
		fp: int,
		lp: int,
	) -> None: ...

	def _azSubsetIQ(
		iq: ComplexArray,
		result: ComplexArray,
		NR: int,
		startRange: int,
		fps: Int64Array,
		lps: Int64Array,
	) -> None: ...

	def subsetIQcpp(
		iq: ComplexArray,
		iaz: int,
		naz: int,
		azIncreasing: bool,
		pulseBoundaries: Int64Array,
		iranges: Int64Array,
		swathPulses: int = -1,
		K: int = 1,
		KOffset: int = 0,
		avgStrat: int = 1,
		shapeOnly: bool = False,
	) -> tuple[ComplexArray, int, int]: ...
else:
	from ._res import _azSubsetIQ, _rangeSubsetIQ, subsetIQcpp

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
	return subsetIQcpp(
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
