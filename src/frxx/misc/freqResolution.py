import numpy as np

def velResolution(nPulses, prf = 4000, wavelength = 0.0308):
    delta_fd = prf / nPulses
    delta_v = delta_fd * wavelength / 2.0
    return delta_v

def velResolutionTonPulses(delta_v, prf = 4000, wavelength = 0.0308):
    delta_fd = delta_v * 2.0 / wavelength
    nPulses = prf / delta_fd
    return nPulses

def velSpanToNumBins(delta_v, nFFT, prf=4000, wavelength=0.0308):
    bin_width = prf * wavelength / (2.0 * nFFT)   # m/s per bin
    nBins = int(round(delta_v / bin_width))
    return max(1, nBins)  # a window should be at least 1 bin wide