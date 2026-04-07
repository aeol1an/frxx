from ...core import IQ, moments, spectra
from ...core.frxxData import _FILL_VALUES

from ...utils import findPulseBoundaries
from .. import algs

from typing import Tuple, List
from numpy.typing import NDArray

import numpy as np
from ...utils import numbaWindows as wn

from numba import jit, njit, prange
from numba.typed import List as ListType

import warnings

@njit(
	'Tuple((ListType(float64[:,::1]), ListType(float64[:,::1]), '
	'ListType(float64[:,::1]), ListType(float64[:,::1])))'
	'(complex64[:,:], complex64[:,:], int64[:,:], boolean, int64, int64, '
	'int64, int64, int64, int64, int64)',
	parallel=True, cache=True
)
def _computeBootstrapDPSD(
	iqh, iqv, 
	pulseBoundaries, azIncreasing, 
	window, 
	swathPulses = -1, 
	nBootstraps = 50, 
	K = 1, KOffset = 0, 
	avgStrat = 1,
	NFT = -1
):
	naz = len(pulseBoundaries)
	iranges = np.array([0, iqh.shape[0]-1], dtype=np.int64)

	# Pre-allocate typed lists so numba knows it's List(float64[:,:])
	dummy = np.empty((0, 0), dtype=np.float64)
	PSDH = ListType()
	PSDV = ListType()
	sZDR = ListType()
	sRHOHV = ListType()
	for _ in range(naz):
		PSDH.append(dummy)
		PSDV.append(dummy)
		sZDR.append(dummy)
		sRHOHV.append(dummy)

	#ugly code, sorry! Numba does not like function pointers, and I don't want branching in my loop.
	if window == 0:
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
			wValues = wn.rectangular(nSAZ)
			h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
			PSDH[az] = np.ascontiguousarray(h)
			PSDV[az] = np.ascontiguousarray(v)
			sZDR[az] = np.ascontiguousarray(d)
			sRHOHV[az] = np.ascontiguousarray(r)
	elif window == 1:
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
			wValues = wn.hanning(nSAZ)
			h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
			PSDH[az] = np.ascontiguousarray(h)
			PSDV[az] = np.ascontiguousarray(v)
			sZDR[az] = np.ascontiguousarray(d)
			sRHOHV[az] = np.ascontiguousarray(r)
	elif window == 2:
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
			wValues = wn.hamming(nSAZ)
			h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
			PSDH[az] = np.ascontiguousarray(h)
			PSDV[az] = np.ascontiguousarray(v)
			sZDR[az] = np.ascontiguousarray(d)
			sRHOHV[az] = np.ascontiguousarray(r)
	elif window == 3:
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
			wValues = wn.blackman(nSAZ)
			h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
			PSDH[az] = np.ascontiguousarray(h)
			PSDV[az] = np.ascontiguousarray(v)
			sZDR[az] = np.ascontiguousarray(d)
			sRHOHV[az] = np.ascontiguousarray(r)
	elif window == 4:
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
			wValues = wn.bartlett(nSAZ)
			h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
			PSDH[az] = np.ascontiguousarray(h)
			PSDV[az] = np.ascontiguousarray(v)
			sZDR[az] = np.ascontiguousarray(d)
			sRHOHV[az] = np.ascontiguousarray(r)
	# elif window == 5:
		# for az in range(naz):
		# 	iqhs, _, nSAZ = algs.res.subsetIQnumba(
		# 		iqh, 
		# 		az, naz, azIncreasing, 
		# 		pulseBoundaries, iranges, swathPulses, 
		# 		K, KOffset, 
		# 		avgStrat
		# 	)
		# 	iqvs, _, _ = algs.res.subsetIQnumba(
		# 		iqv, 
		# 		az, naz, azIncreasing, 
		# 		pulseBoundaries, iranges, swathPulses, 
		# 		K, KOffset, 
		# 		avgStrat
		# 	)
		# 	wValues = wn.tukey(nSAZ)
		# 	h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
		# 	PSDH[az] = np.ascontiguousarray(h)
		# 	PSDV[az] = np.ascontiguousarray(v)
		# 	sZDR[az] = np.ascontiguousarray(d)
		# 	sRHOHV[az] = np.ascontiguousarray(r)
	else:
		raise ValueError("Bad window selection.")

	return (PSDH, PSDV, sZDR, sRHOHV)

def calculatePPIDPSD(
	iq: IQ, m: moments | None = None, 
	azSpacingDeg: float | None = None, beamOverlapDeg: float | None = None,
	SNRthresholddB: Tuple[float,float] = (-np.inf, -np.inf), 
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
	if swathPulses is None:
		swathPulses = -1
	
	if NFT is not None and NFT < 2:
		raise ValueError("NFT must be greater than 2.")
	if NFT is None:
		NFT = -1

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
	# elif window == "tukey":
	# 	w = 5
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
	k, a = algs.res._subsetIQStrToInt(KOffset, avgStrat)
	PSDH, PSDV, sZDR, sRHOHV = _computeBootstrapDPSD(
		iqh, iqv, pulseBoundaries, azIncreasing, w, swathPulses,
		nBootstraps,
		K, k,
		a, NFT
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
	s.setNoisedB(iq.N0)
	s.setSNRThreshold(SNRthresholddB)
	s.setPulseBoundaries(pulseBoundaries)

	mask = []
	for i in range(len(PSDH)):
		imask = ((PSDH[i]-iq.N0H) > SNRthresholddB[0]) & \
				((PSDV[i]-iq.N0H) > SNRthresholddB[1]) & \
				~np.isnan(PSDH[i]) & \
				~np.isnan(PSDV[i]) & \
				~np.isnan(sZDR[i])
				
		mask.append(imask)
	s.setMask(mask, "True if SNR below threshold or linear PSDs, ZDR below 0.")

	encoding = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": 0.01,
		"add_offset": 0.0
	}

	s.addDataField('PSDH', PSDH, encoding=encoding,
		attrs={
			"long_name": "horizontal_power_spectral_density",
			"units": "dB",
		}
	)
	s.addDataField('PSDV', PSDV, encoding=encoding,
		attrs={
			"long_name": "vertical_power_spectral_density",
			"units": "dB",
		}
	)
	s.addDataField('sZDR', sZDR, encoding=encoding,
		attrs={
			"long_name": "spectral_differential_reflectivity",
			"units": "dB",
		}
	)
	s.addDataField('sRHOHV', sRHOHV, encoding=encoding,
		attrs={
			"long_name": "spectral_correlation_coefficient",
			"units": "unitless",
		}
	)

	return s