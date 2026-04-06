from ...core import IQ, moments, spectra
from ...core.frxxData import _FILL_VALUES

from ...utils import findPulseBoundaries
from .. import algs

from typing import Tuple

import numpy as np
from ...utils import numbaWindows as wn

from numba import njit, prange

import warnings

@njit(
	'Tuple((List(float64[:,:]), List(float64[:,:]), List(complex128[:,:]), '
	'List(float64[:,:]), List(float64[:,:]), List(float64[:,:]), List(float64[:,:])))'
	'(complex64[:,:], complex64[:,:], int64[:,:], boolean, int64, optional(int64), '
	'float64, float64, int64, int64, optional(int64), optional(int64), optional(int64))',
	parallel=True, cache=True
)
def _computeBootstrapDPSD(
	iqh, iqv, 
	pulseBoundaries, azIncreasing, 
	window, 
	swathPulses, 
	noiseh, noisev, 
	nBootstraps, 
	K = 1, KOffset = None, 
	avgStrat = None,
	NFT = None
):
	naz = len(pulseBoundaries)
	iranges = np.array([0, iqh.shape[0]-1], dtype=np.int64)

	if window == 0:
		w = wn.rectangular
	elif window == 1:
		w = wn.hanning
	elif window == 2:
		w = wn.hamming
	elif window == 3:
		w = wn.blackman
	elif window == 4:
		w = wn.bartlett
	elif window == 5:
		w = wn.tukey
	else:
		raise ValueError("Bad window selection.")

	result = ([],[],[],[],[],[],[])
	for az in prange(naz):
		iqhs, _, nSAZ = algs.res.subsetIQnumba(
			iqh, 
			az, naz, azIncreasing, 
			pulseBoundaries, iranges, swathPulses, 
			K, KOffset, 
			avgStrat
		)
		iqvs, _, _ = algs.res.subsetIQnumba(
			iqv, 
			az, naz, azIncreasing, 
			pulseBoundaries, iranges, swathPulses, 
			K, KOffset, 
			avgStrat
		)
		wValues = w(nSAZ)
		R = algs.bootstrapDPSD.computeRay(iqhs, iqvs, wValues, noiseh, noisev, nBootstraps, K, NFT)
		for i in range(len(R)):
			result[i].append(R[i])

	return result

def calculatePPIDPSD(
	iq: IQ, m: moments | None = None, 
	azSpacingDeg: float | None = None, beamOverlapDeg: float | None = None,
	SNRthresholddB: Tuple[float,float] | None = None, 
	nBootstraps: int = 50, 
	swathPulses: int | None = None, NFT: int | None = None, window: str = "blackman",
	K: int = 1, KOffset: str | None = None, avgStrat: str | None = None
) -> spectra:
	if m is None:
		if azSpacingDeg is None:
			azSpacingDeg = 1.0
		if beamOverlapDeg is None:
			beamOverlapDeg = 0.0
	else:
		if azSpacingDeg is not None:
			warnings.warn("Overriding azSpacingDeg: using pulse grouping from passed moments.")
		if beamOverlapDeg is not None:
			warnings.warn("Overriding beamOverlapDeg: using pulse grouping from passed moments.")

	if nBootstraps < 1:
		raise ValueError("nBootstraps must be greater than 1, and the larger, the better.")

	if swathPulses is not None and swathPulses < 2:
		raise ValueError("swathPulses must be greater than 2.")
	
	if NFT is not None and NFT < 2:
		raise ValueError("NFT must be greater than 2.")

	if window == "rectangular":
		w = 0
	elif window == "hanning":
		w = 1
	elif window == "hamming":
		w = 2
	elif window == "blackman":
		w = 3
	elif window == "bartlett":
		w = 4
	elif window == "tukey":
		w = 5
	else:
		raise ValueError("Unsupported Window.")
	
	if K < 1:
		raise ValueError("K must be an int greater than one.")
	if K > 1:
		if avgStrat is None:
			raise ValueError("Need to pick an axis (r, az) to average along.")
		elif avgStrat not in ["r", "az"]:
			raise ValueError("avgStrat can only be 'r' or 'az'.")
	if K % 2 == 0:
		if KOffset is None:
			raise ValueError("If K is even, need to specify if \"high\"er or "
							"\"low\"er indices get grouped first.")
		elif KOffset not in ["high", "Low"]:
			raise ValueError("KOffset can only be 'high' or 'low'.")
	
	iqh, iqv = (iq.iqh, iq.iqv)

	if m is not None:
		time = m.time
		az = m.az
		el = m.el
		pw = m.pw
		prt = m.prt
		wavelength = m.wavelength
		pulseBoundaries = m.pb
	else:
		pulseBoundaries, azUnique = findPulseBoundaries(iq.az, azSpacingDeg, beamOverlapDeg)
		middlePulses = np.rint(pulseBoundaries.mean(axis=1)).astype(np.int32)
		time = iq.time[middlePulses]
		az = azUnique
		el = iq.el[middlePulses]
		pw = iq.pw[middlePulses]
		prt = iq.prt[middlePulses]
		wavelength = iq.wavelength[middlePulses]

	azIncreasing = np.mean(np.sign(np.diff(az)))
	KOffset, avgStrat = algs.res._subsetIQStrToInt(KOffset, avgStrat)
	PSDH, PSDV, COV, sSNRH, sSNRV, sZDR, sRHOHV = _computeBootstrapDPSD(
		iqh, iqv, pulseBoundaries, azIncreasing, w, swathPulses,
		10**(0.1*iq.N0H), 10**(0.1*iq.N0V), nBootstraps,
		K, KOffset,
		avgStrat, NFT
	)

	s = spectra()

	s.setInstrument(
		name = iq.ds.attrs["instrument_name"],
		institution = iq.ds.attrs["institution"],
		source = "frxx"
	)
	s.setVolume(iq.vol)
	s._cpyTime(iq, time)
	s.setSweep(iq.sweep)
	s.setRange(iq.rm, True)
	s.setPosition(*iq.pos.values())
	s.setScanningStrategy("ppi", iq.fixedAngle)
	s.setAzimuth(az)
	s.setElevation(el)
	s.setPulseWidthSeconds(pw)
	s.setPrtSeconds(prt)
	s.setWavelengthMeters(wavelength)
	s.setPol(2)
	s.setSNRThreshold([-np.inf, -np.inf] if SNRthresholddB is None else SNRthresholddB)
	s.setPulseBoundaries(pulseBoundaries)

	return s