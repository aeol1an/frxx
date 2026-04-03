from ...core import IQ, moments
from ...core.frxxData import _FILL_VALUES

from ...utils import findPulseBoundaries

import numpy as np

import sys

def calculateDualPolPPI(iq: IQ, azSpacingDeg: float = 1.0, beamOverlapDeg: float = 0.0) -> moments | None:
	if not ("iq" in iq.ds):
		raise ValueError("IQ data structure is not complete yet.")
	if len(iq.ds["sweep"]) != 1:
		raise ValueError("IQ data should have one sweep per Dataset.")
	
	c = 299792458.0

	iqh, iqv = (iq.iqh, iq.iqv)
	az = iq.az
	el = iq.el

	r = iq.range
	rkm = r/1000.
	nr = len(r)

	if (len(np.unique(np.rint(el))) > len(np.unique(np.rint(az))))\
		and (len(np.unique(np.rint(el))) > 5):
		
		print("Elevation varies by more than 5 degrees. This might be an RHI. Returning None.", file=sys.stderr)
		return
	
	azSpacing = azSpacingDeg
	azSwath = azSpacing + (2 * beamOverlapDeg)

	pulseBoundaries, azUnique = findPulseBoundaries(az, azSpacing, beamOverlapDeg)

	