import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _computeSingleSpectrum(VH, VV, w, M, NFT, B, r):
    CX_left = 0.5 * (VH[0]/VH[-1] + VV[0]/VV[-1])
    CX_right = 0.5 * (VH[-1]/VH[0] + VV[-1]/VV[0])

    XH = np.concatenate((
        VH[-round(M * r):-1] * CX_left,
        VH,
        VH[1:round(M * r)] * CX_right
    ))
    XV = np.concatenate((
        VV[-round(M * r):-1] * CX_left,
        VV,
        VV[1:round(M * r)] * CX_right
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
    WH = np.empty((B, M), dtype=XH.dtype)
    WV = np.empty((B, M), dtype=XV.dtype)

    for i in prange(B):
        boot_idx = np.random.randint(0, Mx - M + 1)

        # Pass 1: extract block and accumulate |x|^2 for R0
        accH = 0.0
        accV = 0.0
        for j in range(M):
            vh = XH[boot_idx + j]
            vv = XV[boot_idx + j]
            WH[i, j] = vh
            WV[i, j] = vv
            accH += vh.real * vh.real + vh.imag * vh.imag
            accV += vv.real * vv.real + vv.imag * vv.imag

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

    for j in prange(NFT):
        sh = 0.0
        sv = 0.0
        sx = np.complex128(0)
        for i in range(B):
            zh = zH[i, j]
            zv = zV[i, j]
            sh += zh.real * zh.real + zh.imag * zh.imag
            sv += zv.real * zv.real + zv.imag * zv.imag
            sx += zh * np.conj(zv)
        SHi[j] = sh / norm
        SVi[j] = sv / norm
        SXi[j] = sx / norm

    return SHi, SVi, SXi

@njit(parallel=True, cache=True)
def _computeMultpleSpectra(
    VH: np.ndarray, VV: np.ndarray, w: np.ndarray,
    NK: int, M: int, NFT: int, B: int, r: int
):
    SH = np.full((NK, NFT), np.nan)
    SV = np.full((NK, NFT), np.nan)
    SX = np.full((NK, NFT), np.nan + np.nan * 1j)
    
    for i in prange(NK):
        SHi, SVi, SXi = _computeSingleSpectrum(
            VH[i,:], VV[i,:], w,
            M, NFT, B, r
        )
        SH[i,:] = SHi
        SV[i,:] = SVi
        SX[i,:] = SXi
        
    return SH, SV, SX


def bootstrapDPSD(V, w, N0, NFT, B, K, N):
    r = 0.5 - np.sqrt(np.mean(np.power(w, 2)))*0.5

    NK = V['H'].shape[0]
    M = V['H'].shape[1]

    if NFT is None:
        NFT = M

    SH, SV, SX = _computeMultpleSpectra(
        V['H'], V['V'], w,
        NK, M, NFT, B, r
    )
    
    S = {
        'H': np.fft.fftshift(SH,axes=1),
        'V': np.fft.fftshift(SV,axes=1),
        'X': np.fft.fftshift(SX,axes=1)
    }

    tsh = np.full((N, NFT), np.nan)
    tsv = np.full((N, NFT), np.nan)
    tsx = np.full((N, NFT), np.nan + np.nan * 1j)
    td = np.full((N, NFT), np.nan)
    tr = np.full((N, NFT), np.nan)

    for i in range(N):
        iK = np.arange(0, K, 1) + (i)*K
        
        tsh[i,:] = np.mean(S['H'][iK,:], axis=0)
        tsv[i,:] = np.mean(S['V'][iK,:], axis=0)
        tsx[i,:] = np.mean(S['X'][iK,:], axis=0)

        td[i,:] = tsh[i,:] / tsv[i,:]
        tr[i,:] = np.abs(tsx[i,:]) / np.sqrt(tsh[i,:] * tsv[i,:])

    if K == 1:
        beta = (1-r)**(-3.3) - 2*((1-r)**1.1)
    else:
        beta = (1-r)**(-4.5) - (1-r)**(-2.1)

    E = {}
    E['sS'] = {
        'H': tsh,
        'V': tsv,
        'X': tsx
    }
    E['sSNR'] = {
        'H': tsh / N0['H'],
        'V': tsv / N0['V']
    }
    E['sD'] = td * (1 - (1 / (beta * K) * (1 - np.power(tr, 2))))
    E['sR'] = tr * (1 - (1 / (beta * K) * ((np.power(1 - np.power(tr, 2), 2)) / (4 * np.power(tr, 2)))))

    E['sD'][E['sD'] < 0] = np.nan
    E['sR'][E['sR'] < 0] = 0

    return E

