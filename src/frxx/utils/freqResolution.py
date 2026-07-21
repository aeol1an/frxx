import numpy as np

from numba import njit

def velResolution(nPulses, prf = 4000, wavelength = 0.0308):
    delta_fd = prf / nPulses
    delta_v = delta_fd * wavelength / 2.0
    return delta_v

def velResolutionTonPulses(delta_v, prf = 4000, wavelength = 0.0308):
    delta_fd = delta_v * 2.0 / wavelength
    nPulses = prf / delta_fd
    return nPulses

@njit(
    [
        'float32[:](int64, float32, boolean, int64, int64)',
        'float64[:](int64, float64, boolean, int64, int64)'
    ],
    cache=True, nogil=True
)
def velocityAxis(NFT: int, va: np.floating, flipVel: bool, leftUnfolds = 0, rightUnfolds = 0):
    t = np.array([va]).dtype
    if flipVel:
        return np.linspace(-va-(2*leftUnfolds*va), va+(2*rightUnfolds*va), NFT).astype(t)
    else:
        return np.linspace(va+(2*rightUnfolds*va), -va-(2*leftUnfolds*va), NFT).astype(t)
    
def velSpanToNumBins(delta_v, nFFT, prf=4000, wavelength=0.0308):
    bin_width = prf * wavelength / (2.0 * nFFT)   # m/s per bin
    nBins = int(round(delta_v / bin_width))
    return max(1, nBins)  # a window should be at least 1 bin wide