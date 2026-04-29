import numpy as np
from numba import njit, prange

from typing import Tuple
from numpy.typing import NDArray

from frxx import BACKEND
from frxx._backend import Backend

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
    for j in range(NFT):
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
def processRay_S_numba(
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

import threading
_thread_lock = threading.Lock()

def processRay_S_torch(
    iqh: np.ndarray,          # (NK, M) complex64
    iqv: np.ndarray,          # (NK, M) complex64
    window: np.ndarray,       # (M,)
    nBootstraps: int,
    K: int = 1,
    NFT: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch reimplementation of processRay_S.

    Accepts numpy arrays, returns numpy arrays.
    Runs entirely in float32/complex64 on the target GPU device.
    """
    import torch

    # --- Resolve device from backend ---
    match BACKEND:
        case Backend.TORCH_CUDA:
            device = torch.device("cuda")
        case Backend.TORCH_MPS:
            device = torch.device("mps")
        case _:
            raise ValueError(f"Unsupported backend for PyTorch path: {BACKEND}")

    fdtype = torch.float32
    cdtype = torch.complex64

    NK, M = iqh.shape
    N = NK // K

    if NFT <= 1:
        NFT = M
    global allocation_count

    with _thread_lock:
        # --- Move inputs to device ---
        VH = torch.from_numpy(iqh).to(dtype=cdtype, device=device)
        VV = torch.from_numpy(iqv).to(dtype=cdtype, device=device)
        win = torch.from_numpy(window).to(dtype=fdtype, device=device)

        # --- Compute r and nr ---
        alpha = torch.mean(win ** 2).item()
        r = 0.5 - np.sqrt(alpha) * 0.5
        nr = int(round(M * r))

        # --- Compute CX_left, CX_right for each spectrum (NK,) ---
        VH_first = VH[:, 0]
        VH_last = VH[:, -1]
        VV_first = VV[:, 0]
        VV_last = VV[:, -1]

        guard = ((torch.abs(VH_first) < 1e-20) | (torch.abs(VH_last) < 1e-20) |
                (torch.abs(VV_first) < 1e-20) | (torch.abs(VV_last) < 1e-20))

        CX_left = 0.5 * (VH_first / VH_last + VV_first / VV_last)
        CX_right = 0.5 * (VH_last / VH_first + VV_last / VV_first)
        CX_left[guard] = 1.0 + 0j
        CX_right[guard] = 1.0 + 0j

        # --- Build extended signals XH, XV: (NK, Mx) ---
        left_h = VH[:, -nr:-1] * CX_left[:, None]
        right_h = VH[:, 1:nr] * CX_right[:, None]
        XH = torch.cat([left_h, VH, right_h], dim=1)

        left_v = VV[:, -nr:-1] * CX_left[:, None]
        right_v = VV[:, 1:nr] * CX_right[:, None]
        XV = torch.cat([left_v, VV, right_v], dim=1)

        Mx = XH.shape[1]

        # Free original tensors early
        del VH, VV

        # --- Compute R0 of original signals (from the middle of extended) ---
        R0H = torch.mean(torch.abs(XH[:, nr - 1:nr - 1 + M]) ** 2, dim=1)  # (NK,)
        R0V = torch.mean(torch.abs(XV[:, nr - 1:nr - 1 + M]) ** 2, dim=1)  # (NK,)

        # --- Accumulate spectra across bootstraps ---
        SH_acc = torch.zeros((NK, NFT), dtype=fdtype, device=device)
        SV_acc = torch.zeros((NK, NFT), dtype=fdtype, device=device)
        SX_acc = torch.zeros((NK, NFT), dtype=cdtype, device=device)

        offsets = torch.arange(M, device=device)  # (M,)

        # --- Auto-calculate bootstrap_batch ---
        if device.type == "cuda":
            torch.cuda.synchronize()
            free_mem, _ = torch.cuda.mem_get_info(device)
        else:
            # MPS has no reliable free-memory query; fall back to a conservative estimate.
            # torch.mps.recommended_max_memory() exists but isn't a "free" query.
            import psutil
            free_mem = int(psutil.virtual_memory().available * 0.5)  # rough heuristic

        # Peak in-loop memory per unit of batch_size:
        # All of these coexist simultaneously at peak (before the del at loop end):
        #   indices_flat : NK * B * M * 8   (int64)
        #   blockH       : NK * B * M * 8   (complex64)
        #   blockV       : NK * B * M * 8   (complex64)
        #   fftH         : NK * B * NFT * 8 (complex64)
        #   fftV         : NK * B * NFT * 8 (complex64)
        #   temp from SX : NK * B * NFT * 8 (fftH * conj(fftV))
        #   boot_idx     : NK * B * 8       (int64, negligible)
        bytes_per_unit = NK * (2 * M * 8 + 3 * NFT * 8 + 8)

        # Target 75% of free memory, with 1.15x safety factor on the estimate
        bootstrap_batch = max(1, int((free_mem * 0.75) / (bytes_per_unit * 1.15)))
        bootstrap_batch = min(bootstrap_batch, nBootstraps)

        for batch_start in range(0, nBootstraps, bootstrap_batch):
            batch_size = min(bootstrap_batch, nBootstraps - batch_start)

            # Random start indices: (NK, batch_size)
            boot_idx = torch.randint(0, Mx - M + 1, (NK, batch_size), device=device)

            # Gather indices: (NK, batch_size * M)
            indices = boot_idx.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)
            indices_flat = indices.reshape(NK, batch_size * M)

            # Gather blocks: (NK, batch_size, M)
            blockH = torch.gather(XH, 1, indices_flat).reshape(NK, batch_size, M)
            blockV = torch.gather(XV, 1, indices_flat).reshape(NK, batch_size, M)
            del indices, indices_flat, boot_idx  # free before FFT

            # Compute R0 of each bootstrap block: (NK, batch_size)
            r0h_block = torch.mean(torch.abs(blockH) ** 2, dim=2).clamp(min=1e-20)
            r0v_block = torch.mean(torch.abs(blockV) ** 2, dim=2).clamp(min=1e-20)

            # Scale: (NK, batch_size)
            scaleH = torch.sqrt(R0H[:, None] / r0h_block)
            scaleV = torch.sqrt(R0V[:, None] / r0v_block)

            # Apply scale and window: (NK, batch_size, M)
            blockH = blockH * (scaleH.unsqueeze(2) * win.unsqueeze(0).unsqueeze(0))
            blockV = blockV * (scaleV.unsqueeze(2) * win.unsqueeze(0).unsqueeze(0))

            # FFT along last dim: (NK, batch_size, NFT)
            fftH = torch.fft.fft(blockH, n=NFT, dim=2)
            fftV = torch.fft.fft(blockV, n=NFT, dim=2)

            # Accumulate power and cross-spectrum, summing over batch dim
            SH_acc += (fftH.real ** 2 + fftH.imag ** 2).sum(dim=1)
            SV_acc += (fftV.real ** 2 + fftV.imag ** 2).sum(dim=1)
            SX_acc += (fftH * fftV.conj()).sum(dim=1)

            del blockH, blockV, fftH, fftV  # if still in scope

        # --- Normalize ---
        norm = M * alpha * nBootstraps
        SH_acc /= norm
        SV_acc /= norm
        SX_acc /= norm

        # --- fftshift ---
        SH_acc = torch.fft.fftshift(SH_acc, dim=1)
        SV_acc = torch.fft.fftshift(SV_acc, dim=1)
        SX_acc = torch.fft.fftshift(SX_acc, dim=1)

        # --- Average over K looks ---
        SH_avg = SH_acc.reshape(N, K, NFT).mean(dim=1)  # (N, NFT)
        SV_avg = SV_acc.reshape(N, K, NFT).mean(dim=1)
        SX_avg = SX_acc.reshape(N, K, NFT).mean(dim=1)

        # --- Bias correction (ZDR, RHOHV) ---
        if K == 1:
            beta = (1 - r) ** (-3.3) - 2 * ((1 - r) ** 1.1)
        else:
            beta = (1 - r) ** (-4.5) - (1 - r) ** (-2.1)

        td = SH_avg / SV_avg.clamp(min=1e-20)

        denom = torch.sqrt((SH_avg * SV_avg).clamp(min=0)).clamp(min=1e-20)
        tr = torch.abs(SX_avg) / denom

        trsquared = (tr ** 2).clamp(min=1e-20)

        sZDR = td * (1 - (1 / (beta * K)) * (1 - trsquared))
        sRHOHV = tr * (1 - (1 / (beta * K)) * ((1 - trsquared) ** 2 / (4 * trsquared)))

        # --- NaN / clamp handling ---
        bad = (SV_avg < 1e-20) | (denom < 1e-20)

        PSDH = SH_avg.clone()
        PSDV = SV_avg.clone()
        PSDH[PSDH < 0] = float('nan')
        PSDV[PSDV < 0] = float('nan')
        sZDR[bad] = float('nan')
        sZDR[sZDR < 0] = float('nan')
        sRHOHV[bad] = float('nan')
        sRHOHV[sRHOHV < 0] = 0.0

        # --- Convert to dB, move to CPU, return as numpy ---
        psdh_db = (10 * torch.log10(PSDH)).cpu().numpy()
        psdv_db = (10 * torch.log10(PSDV)).cpu().numpy()
        szdr_db = (10 * torch.log10(sZDR)).cpu().numpy()
        rhohv = sRHOHV.cpu().numpy()

        # Explicitly free ALL GPU tensors before releasing the lock
        del XH, XV, SH_acc, SV_acc, SX_acc
        del SH_avg, SV_avg, SX_avg
        del PSDH, PSDV, sZDR, sRHOHV
        del td, tr, trsquared, denom, bad
        del R0H, R0V, win
        #torch.cuda.empty_cache()

    return psdh_db, psdv_db, szdr_db, rhohv

if BACKEND == Backend.NUMBA:
    processRay_S = processRay_S_numba
else:
    processRay_S = processRay_S_torch