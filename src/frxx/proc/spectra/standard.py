from ...core import IQ, moments, spectra
from ...core.frxxData import _FILL_VALUES

from ...utils import findPulseBoundaries
from ...utils import stringUtils as su
from .. import algs

from typing import Tuple, cast
from numpy.typing import NDArray

import numpy as np
from ...utils import numbaWindows as wn

from numba import njit, prange

import dask as d
import dask.array as da

import warnings

@njit(
	'Tuple((float64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1], boolean[:,::1]))'
	'(complex64[:,:], complex64[:,:], int64[:,:], boolean, int64, int64, '
	'int64[:], int64, int64, int64, int64, int64, int64, int64, float64, float64, float64, float64)',
	cache=True, nogil=True
)
def _processRay(
	iqh, iqv, 
	pulseBoundaries, azIncreasing, 
	az, naz, iranges,
	window, 
	swathPulses, nBootstraps, 
	K, KOffset, avgStrat, NFT,
	noisehDB, noisevDB,
	SNRHThreshold, SNRVThreshold
):
	iqhs, _, nSAZ = algs.res.subsetIQnumba(
		iqh, az, naz, azIncreasing, 
		pulseBoundaries, iranges, swathPulses, 
		K, KOffset, avgStrat, False
	)
	iqvs, _, _ = algs.res.subsetIQnumba(
		iqv, az, naz, azIncreasing, 
		pulseBoundaries, iranges, swathPulses, 
		K, KOffset, avgStrat, False
	)

	if window == 0:
		wValues = wn.rectangular(nSAZ)
	elif window == 1:
		wValues = wn.hanning(nSAZ)
	elif window == 2:
		wValues = wn.hamming(nSAZ)
	elif window == 3:
		wValues = wn.blackman(nSAZ)
	elif window == 4:
		wValues = wn.bartlett(nSAZ)
	else:
		raise ValueError("Bad window selection.")

	h, v, d, r = algs.bootstrapDPSD.processRay_S(iqhs, iqvs, wValues, nBootstraps, K, NFT)
	m = ((h-noisehDB) > SNRHThreshold) & \
		((v-noisevDB) > SNRVThreshold) & \
		~np.isnan(h) & \
		~np.isnan(h) & \
		~np.isnan(h)
	return (np.ascontiguousarray(h), np.ascontiguousarray(v),
			np.ascontiguousarray(d), np.ascontiguousarray(r), np.ascontiguousarray(m))

def processRays(
	iqh, iqv,
	pulseBoundaries, azIncreasing,
	window,
	swathPulses=-1,
	nBootstraps=50,
	K=1, KOffset=0,
	avgStrat=1,
	NFT=-1,
	noisehDB = -np.inf, noisevDB = -np.inf,
	SNRHThreshold = -np.inf, SNRVThreshold = -np.inf
):
	naz = len(pulseBoundaries)
	nr = iqh.shape[0]
	iranges = np.array([0, nr - 1], dtype=np.int64)
	nf = [
		algs.res.subsetIQnumba(
			iqh, az, naz, azIncreasing, 
			pulseBoundaries, iranges, swathPulses, 
			K, KOffset, avgStrat, shapeOnly=True
		)[2] for az in range(naz)
	]

	rays = [
		d.delayed(_processRay)( #type: ignore
			iqh, iqv, pulseBoundaries, azIncreasing,
			az, naz, iranges, window,
			swathPulses, nBootstraps, K, KOffset, avgStrat, NFT,
			noisehDB, noisevDB, SNRHThreshold, SNRVThreshold
		)
		for az in range(naz)
	]

	results = []
	for product in range(5):
		dt = np.float32 if product != 4 else np.bool_
		chunks = [
			da.from_delayed(
				rays[az][product].astype(dt), #type: ignore
				shape=(nr, nf[az]),
				dtype=dt,
			)
			for az in range(naz)
		]
		results.append(chunks)

	return tuple(results)



def calculatePPIDPSD(
	iq: IQ, m: moments | None = None, 
	azSpacingDeg: float | None = None, beamOverlapDeg: float | None = None,
	SNRthresholddB: Tuple[float,float] = (-np.inf, -np.inf), 
	nBootstraps: int = 50, 
	swathPulses: int | None = None, NFT: int | None = None, window: str = "blackman",
	K: int = 1, KOffset: str | None = None, avgStrat: str | None = None,
	delayed = True
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
	else:
		raise ValueError("Unsupported Window.")
	
	if K < 1:
		raise ValueError("K must be an int greater than one.")
	if K > 1:
		if avgStrat is None:
			raise ValueError("Need to pick an axis (r,TypedList az) to average along.")
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
		bs = su.parseBeamSpec(m.beamSpec)
		if bs is None:
			raise ValueError("Invalid beamspec.")
	else:
		pulseBoundaries, azUnique = findPulseBoundaries(iq.az, azSpacingDeg, beamOverlapDeg)
		middlePulses = np.rint(pulseBoundaries.mean(axis=1)).astype(np.int32)
		time = iq.time[middlePulses]
		az = azUnique
		el = iq.el[middlePulses]
		pw = iq.pw[middlePulses]
		prt = iq.prt[middlePulses]
		wavelength = iq.wavelength[middlePulses]
		bs = (azSpacingDeg, beamOverlapDeg)
	bs = cast(Tuple[float, float], bs)
	azIncreasing = np.mean(np.sign(np.diff(az))) > 0
	k, a = algs.res._subsetIQStrToInt(KOffset, avgStrat)
	PSDH, PSDV, sZDR, sRHOHV, mask = processRays(
		iqh, iqv, pulseBoundaries, azIncreasing, w, swathPulses,
		nBootstraps,
		K, k,
		a, NFT,
		iq.N0H, iq.N0V, SNRHThreshold=SNRthresholddB[0], SNRVThreshold=SNRthresholddB[1]
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
	s.setPhaseDirection(iq.ds.attrs["phase_direction"])
	s.setPol(2)
	s.setNoisedB(iq.N0)
	s.setSNRThreshold(SNRthresholddB)
	s.setBeamSpec(*bs)
	s.setFourierSpec(None, None, None, nBootstraps)
	s.setPulseBoundaries(pulseBoundaries)

	s.setMask(mask, "True if SNR below threshold or linear PSDs, ZDR below 0.")

	encoding = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": np.float32(0.01),
		"add_offset": np.float32(0.0)
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

	if not delayed:
		s.ds.load()

	return s