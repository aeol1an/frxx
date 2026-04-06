from ...core import IQ, moments
from ...core.frxxData import _FILL_VALUES

from ...utils import findPulseBoundaries
from .. import algs

from typing import Tuple

import numpy as np

def calculateDualPolPPIACF(
		iq: IQ, 
		azSpacingDeg: float = 1.0, beamOverlapDeg: float = 0.0, gstep: int = 1,
		SNRthresholddB: Tuple[float,float] = (-np.inf, -np.inf), 
		subtractNoiseEstimate: bool = True, flipVel: bool = False
) -> moments:
	if not ("iq" in iq.ds):
		raise ValueError("IQ data structure is not complete yet.")
	if len(iq.ds["sweep"]) != 1:
		raise ValueError("IQ data should have one sweep per Dataset.")
	
	va = iq.va[0]

	iqh, iqv = (iq.iqh, iq.iqv)
	az = iq.az
	el = iq.el

	time = iq.time
	rkm = iq.rkm

	zcalh, zcalv = tuple(iq.ds["Zcal"].values)
	zcalh = zcalh+8
	dcal = float(iq.ds["Dcal"].values)
	pcal = float(iq.ds["Pcal"].values)

	N0h, N0v = tuple(iq.ds["noise"].values)

	if (len(np.unique(np.rint(el))) > len(np.unique(np.rint(az))))\
		and (len(np.unique(np.rint(el))) > 5):
		
		raise ValueError("Elevation varies by more than 5 degrees. This might be an RHI")
	
	azSpacing = azSpacingDeg

	pulseBoundaries, azUnique = findPulseBoundaries(az, azSpacing, beamOverlapDeg)
	middlePulses = np.rint(pulseBoundaries.mean(axis=1)).astype(np.int32)

	mAz = azUnique
	mEl = el[middlePulses]
	mTime = time[middlePulses]

	R = algs.ACF.processRays(iqh, iqv, pulseBoundaries, np.array([0, 1], dtype=np.int32))

	Rh, Rv, Rx = tuple(algs.res.averageAlongRange(r, gstep) for r in R)


	N0hLin = 10**(0.1 * N0h)
	N0vLin = 10**(0.1 * N0v)

	Sh = np.abs(Rh[:,:,0]) - (N0hLin if subtractNoiseEstimate else 0)
	Sv = np.abs(Rv) - (N0vLin if subtractNoiseEstimate else 0)

	Sh[Sh <= 0] = np.nan
	Sv[Sh <= 0] = np.nan

	DBZ = 10*np.log10(Sh*(rkm**2)) + zcalh

	if flipVel:
		s = -1
	else:
		s = 1
	VEL = s * -va / np.pi * np.angle(Rh[...,1])

	WIDTH = np.sqrt(2) * va / np.pi * np.sqrt(np.abs((np.log(Sh / np.abs(Rh[:,:,1])))))

	ZDR = 10*np.log10(Sh/Sv) + dcal

	PHIDP = np.angle(Rx)/np.pi*180 + pcal
	PHIDP[PHIDP < -180] += 360
	PHIDP[PHIDP > 180] -= 360

	RHOHV = np.abs(Rx) / np.sqrt(Sh*Sv)

	SNRH = 10*np.log10(Sh / N0hLin)
	SNRV = 10*np.log10(Sv / N0vLin)

	m = moments()

	m.setInstrument(
		name = iq.ds.attrs["instrument_name"],
		institution = iq.ds.attrs["institution"],
		source = "frxx"
	)
	m.setVolume(iq.vol)
	m._cpyTime(iq, mTime)
	m.setSweep(iq.sweep)
	m.setRange(iq.rm, True)
	m.setPosition(*iq.pos.values())
	m.setScanningStrategy("ppi", iq.fixedAngle)
	m.setAzimuth(mAz)
	m.setElevation(mEl)
	m.setPulseWidthSeconds(iq.pw[middlePulses])
	m.setPrtSeconds(iq.prt[middlePulses])
	m.setWavelengthMeters(iq.wavelength[middlePulses])
	m.setPol(2)
	m.setSNRThreshold(SNRthresholddB)
	m.setPulseBoundaries(pulseBoundaries)

	mask =  (SNRH > SNRthresholddB[0]) & \
			(SNRV > SNRthresholddB[1]) & \
			~np.isnan(Sh) & \
			~np.isnan(Sv)
	
	m.setMask(mask, "True if SNR below threshold or raw signal below 0.")

	encoding = {
		"dtype": "int16",
		"_FillValue": _FILL_VALUES["int16"],
		"scale_factor": 0.01,
		"add_offset": 0.0
	}

	m.addDataField('DBZ', DBZ, encoding=encoding,
		attrs={
			"long_name": "reflectivity",
			"standard_name": "equivalent_reflectivity_factor",
			"units": "dBZ",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('VEL', VEL, encoding=encoding,
		attrs={
			"long_name": "doppler_velocity",
			"standard_name": "radial_velocity_of_scatterers_away_from_instrument",
			"units": "m/s",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('WIDTH', WIDTH, encoding=encoding,
		attrs={
			"long_name": "spectrum_width",
			"standard_name": "doppler_spectrum_width",
			"units": "m/s",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('ZDR', ZDR, encoding=encoding,
		attrs={
			"long_name": "differential_reflectivity",
			"standard_name": "log_differential_reflectivity_hv",
			"units": "dB",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('PHIDP', PHIDP, encoding=encoding,
		attrs={
			"long_name": "differential_phase",
			"standard_name": "differential_phase_hv",
			"units": "degrees",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('RHOHV', RHOHV, encoding=encoding,
		attrs={
			"long_name": "cross_correlation_ratio",
			"standard_name": "cross_correlation_ratio_hv",
			"units": "unitless",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('SNRH', SNRH, encoding=encoding,
		attrs={
			"long_name": "horizontal_channel_signal_to_noise_ratio",
			"standard_name": "signal_to_noise_ratio_h",
			"units": "dB",
			"grid_mapping": "grid_mapping",
		}
	)
	m.addDataField('SNRV', SNRV, encoding=encoding,
		attrs={
			"long_name": "vertical_channel_signal_to_noise_ratio",
			"standard_name": "signal_to_noise_ratio_v",
			"units": "dB",
			"grid_mapping": "grid_mapping",
		}
	)

	return m