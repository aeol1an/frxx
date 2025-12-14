import numpy as np

def velResolution(nPulses, prf = 4000, wavelength = 0.0308):
    delta_fd = prf / nPulses
    delta_v = delta_fd * wavelength / 2.0
    return delta_v

def velResolutionTonPulses(delta_v, prf = 4000, wavelength = 0.0308):
    delta_fd = delta_v * 2.0 / wavelength
    nPulses = prf / delta_fd
    return nPulses