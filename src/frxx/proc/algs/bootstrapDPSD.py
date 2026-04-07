import numpy as np
from numba import njit, prange

from typing import Tuple
from numpy.typing import NDArray

from ...utils.numbaHelpers import unwrap_i64

@njit(
    'Tuple((float64[:], float64[:], complex128[:]))'
    '(complex64[:], complex64[:], float64[:], int64, int64, int64, float64)',
    parallel=False, cache=True, nogil=True
)
def _computeSingleSpectrum(VH, VV, w, M, NFT, B, r):
    # Guard CX
    if abs(VH[-1]) < 1e-30 or abs(VH[0]) < 1e-30 or abs(VV[-1]) < 1e-30 or abs(VV[0]) < 1e-30:
        CX_left = np.complex64(1.0)
        CX_right = np.complex64(1.0)
    else:
        CX_left = 0.5 * (np.complex128(VH[0]) / np.complex128(VH[-1]) +
                        np.complex128(VV[0]) / np.complex128(VV[-1]))
        CX_right = 0.5 * (np.complex128(VH[-1]) / np.complex128(VH[0]) +
                        np.complex128(VV[-1]) / np.complex128(VV[0]))

    nr = int(round(M * r))
    negnr = -nr
    XH = np.concatenate((
        VH[negnr:-1] * CX_left,
        VH,
        VH[1:nr] * CX_right
    ))
    XV = np.concatenate((
        VV[negnr:-1] * CX_left,
        VV,
        VV[1:nr] * CX_right
    ))

    # R0 with no temporaries
    accH = 0.0
    accV = 0.0
    for j in range(M):
        vh = VH[j]
        vv = VV[j]
        accH += vh.real * vh.real + vh.imag * vh.imag
        accV += vv.real * vv.real + vv.imag * vv.imag
    R0H = accH / M
    R0V = accV / M

    Mx = len(XH)

    # Fused: bootstrap + R0 + rescale + window in one parallel loop
    WH = np.empty((B, M), dtype=np.complex64)
    WV = np.empty((B, M), dtype=np.complex64)

    for i in range(B):
        boot_idx = np.random.randint(0, Mx - M + 1)

        # Pass 1: extract block and accumulate |x|^2 for R0
        accH = 0.0
        accV = 0.0
        for j in range(M):
            vh = np.complex128(XH[boot_idx + j])
            vv = np.complex128(XV[boot_idx + j])
            WH[i, j] = np.complex64(vh)  # store back as 64
            WV[i, j] = np.complex64(vv)
            accH += vh.real * vh.real + vh.imag * vh.imag
            accV += vv.real * vv.real + vv.imag * vv.imag


        # Guard bootstrap scale
        if accH < 1e-30:
            accH = 1e-30
        if accV < 1e-30:
            accV = 1e-30
        # Pass 2: rescale + window in-place (row is hot in L1)
        scaleH = np.sqrt(R0H * M / accH)
        scaleV = np.sqrt(R0V * M / accV)
        for j in range(M):
            WH[i, j] *= scaleH * w[j]
            WV[i, j] *= scaleV * w[j]

    # FFT
    zH = np.fft.fft(WH, n=NFT, axis=1)
    zV = np.fft.fft(WV, n=NFT, axis=1)

    # Spectral averages — parallelize over frequency bins
    alpha = np.mean(np.abs(w) ** 2)
    norm = M * alpha * B

    SHi = np.empty(NFT, dtype=np.float64)
    SVi = np.empty(NFT, dtype=np.float64)
    SXi = np.empty(NFT, dtype=np.complex128)

    half = NFT // 2
    for j in prange(NFT):
        sh = np.float64(0.0)
        sv = np.float64(0.0)
        sx = np.complex128(0)
        for i in range(B):
            zh = np.complex128(zH[i, j])
            zv = np.complex128(zV[i, j])
            sh += zh.real * zh.real + zh.imag * zh.imag
            sv += zv.real * zv.real + zv.imag * zv.imag
            sx += zh * np.conj(zv)
        k = (j + half) % NFT
        SHi[k] = sh / norm
        SVi[k] = sv / norm
        SXi[k] = sx / norm

    return SHi, SVi, SXi

@njit(
    'Tuple((float64[:,:], float64[:,:], complex128[:,:]))'
    '(complex64[:,:], complex64[:,:], float64[:], int64, int64, int64, int64, float64)',
    parallel=True, cache=True, nogil=True
)
def _computeMultipleSpectra(
    VH: np.ndarray, VV: np.ndarray, w: np.ndarray,
    NK: int, M: int, NFT: int, B: int, r: float
):
    SH = np.empty((NK, NFT), dtype=np.float64)
    SV = np.empty((NK, NFT), dtype=np.float64)
    SX = np.empty((NK, NFT), dtype=np.complex128)
    
    for i in prange(NK):
        SHi, SVi, SXi = _computeSingleSpectrum(
            VH[i,:], VV[i,:], w,
            M, NFT, B, r
        )
        SH[i,:] = SHi
        SV[i,:] = SVi
        SX[i,:] = SXi
        
    return SH, SV, SX

# @njit(
#     'Tuple((float64[:,:], float64[:,:], complex128[:,:], float64[:,:], float64[:,:]))'
#     '(complex64[:,:], complex64[:,:], float64[:],  int64, int64, int64)',
#     cache=True, nogil=True
# )
@njit(
    'Tuple((float64[:,:], float64[:,:], float64[:,:], float64[:,:]))'
    '(complex64[:,:], complex64[:,:], float64[:],  int64, int64, int64)',
    cache=True, nogil=True
)
def processRay_S(
    iqh: NDArray, iqv: NDArray, window, nBootstraps, K = 1, NFT = 1
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r = 0.5 - np.sqrt(np.mean(np.power(window, 2)))*0.5

    NK = iqh.shape[0]
    M = iqh.shape[1]

    N = NK//K
    
    if NFT <= 1:
        NFT = M

    SH, SV, SX = _computeMultipleSpectra(
        iqh, iqv, window,
        NK, M, NFT, nBootstraps, r
    )

    tsh = np.empty((N, NFT), dtype=np.float64)
    tsv = np.empty((N, NFT), dtype=np.float64)
    tsx = np.empty((N, NFT), dtype=np.complex128)
    td = np.empty((N, NFT), dtype=np.float64)
    tr = np.empty((N, NFT), dtype=np.float64)

    for i in range(N):
        for j in range(NFT):
            sh = 0.0
            sv = 0.0
            sx = np.complex128(0)
            for k in range(K):
                sh += SH[i * K + k, j]
                sv += SV[i * K + k, j]
                sx += SX[i * K + k, j]
            tsh[i, j] = sh / K
            tsv[i, j] = sv / K
            tsx[i, j] = sx / K
            if tsv[i, j] < 1e-30:
                td[i, j] = np.nan
                tr[i, j] = np.nan
            else:
                td[i, j] = tsh[i, j] / tsv[i, j]
                denom = np.sqrt(tsh[i, j] * tsv[i, j])
                if denom < 1e-30:
                    tr[i, j] = np.nan
                else:
                    tr[i, j] = np.abs(tsx[i, j]) / denom

    if K == 1:
        beta = (1-r)**(-3.3) - 2*((1-r)**1.1)
    else:
        beta = (1-r)**(-4.5) - (1-r)**(-2.1)
    
    PSDH = tsh
    PSDV = tsv
    #COV = tsx

    trsquared = np.power(tr, 2)
    for i in range(N):
        for j in range(NFT):
            if trsquared[i, j] < 1e-30:
                trsquared[i, j] = 1e-30

    sZDR = td * (1 - (1 / (beta * K) * (1 - trsquared)))
    sRHOHV = tr * (1 - (1 / (beta * K) * ((np.power(1 - trsquared, 2)) / (4 * trsquared))))

    for i in range(N):
        for j in range(NFT):
            if PSDH[i, j] < 0:
                PSDH[i, j] = np.nan
            if PSDV[i, j] < 0:
                PSDV[i, j] = np.nan
            if sZDR[i, j] < 0:
                sZDR[i, j] = np.nan
            if sRHOHV[i, j] < 0:
                sRHOHV[i, j] = 0

    #return 10*np.log10(PSDH), 10*np.log10(PSDV), COV, 10*np.log10(sZDR), sRHOHV
    return 10*np.log10(PSDH), 10*np.log10(PSDV), 10*np.log10(sZDR), sRHOHV

