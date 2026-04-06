from numba import njit
import numpy as np

@njit('float64[:](int64)', cache=True)
def rectangular(N):
	return np.ones(N, dtype=np.float64)

@njit('float64[:](int64)', cache=True)
def hanning(N):
	n = np.arange(N)
	return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (N - 1)))

@njit('float64[:](int64)', cache=True)
def hamming(N):
	n = np.arange(N)
	return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (N - 1))

@njit('float64[:](int64)', cache=True)
def blackman(N):
	n = np.arange(N)
	return (0.42
		- 0.50 * np.cos(2.0 * np.pi * n / (N - 1))
		+ 0.08 * np.cos(4.0 * np.pi * n / (N - 1)))

@njit('float64[:](int64)', cache=True)
def bartlett(N):
	n = np.arange(N)
	return 1.0 - np.abs(2.0 * n / (N - 1) - 1.0)

@njit('float64[:](int64, float64)', cache=True)
def tukey(N, alpha=0.5):
	w = np.ones(N)
	if alpha <= 0.0:
		return w
	if alpha >= 1.0:
		return hanning(N)
	left = int(alpha * (N - 1) / 2.0)
	for i in range(left + 1):
		w[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (alpha * (N - 1))))
		w[N - 1 - i] = w[i]
	return w