from ...core import IQ, moments, spectra
from ...core.frxxData import _FILL_VALUES

from ...utils import findPulseBoundaries
from .. import algs

from typing import Tuple

import numpy as np
import scipy.signal.windows as wn

def calculateDPSD(
    iq: IQ, m: moments | None = None, 
    nBootstraps: int = 50, 
    swathPulses: int | None = None, NFT: int | None = None, window: str = "blackman",
    K: int = 1, KOffset: str | None = None, avgStrat: str | None = None
) -> spectra:
    if nBootstraps < 1:
        raise ValueError("nBootstraps must be greater than 1, and the larger, the better.")

    if swathPulses is not None and swathPulses < 2:
        raise ValueError("swathPulses must be greater than 2.")
    if swathPulses is None:
        swathPulsesFromBoundaries = True
    
    if NFT is not None and NFT < 2:
        raise ValueError("NFT must be greater than 2.")

    if window == "rectangular":
        w = wn.boxcar
    elif window == "hanning":
        w = wn.hann
    elif window == "hamming":
        w = wn.hamming
    elif window == "blackman":
        w = wn.blackman
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

    

    s = spectra()

    return s