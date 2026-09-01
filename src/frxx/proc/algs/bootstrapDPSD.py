from typing import TYPE_CHECKING, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray


Complex64Array: TypeAlias = NDArray[np.complex64]
Complex128Array: TypeAlias = NDArray[np.complex128]
Float64Array: TypeAlias = NDArray[np.float64]

if TYPE_CHECKING:
    def _computeSingleSpectrum(
        VH: Complex64Array,
        VV: Complex64Array,
        w: Float64Array,
        M: int,
        NFT: int,
        B: float,
        r: float,
    ) -> tuple[Float64Array, Float64Array, Complex128Array]: ...

    def _computeMultipleSpectra(
        VH: Complex64Array,
        VV: Complex64Array,
        w: Float64Array,
        NK: int,
        M: int,
        NFT: int,
        B: float,
        r: float,
    ) -> tuple[Float64Array, Float64Array, Complex128Array]: ...

    def processRay_S_cpp(
        iqh: Complex64Array,
        iqv: Complex64Array,
        window: Float64Array,
        nBootstraps: float,
        K: int = 1,
        NFT: int = 1,
    ) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]: ...
else:
    from ._bootstrapDPSD import (
        _computeMultipleSpectra,
        _computeSingleSpectrum,
        processRay_S as processRay_S_cpp,
    )

import threading
_thread_lock = threading.Lock()

def processRay_S_torch(
    iqh: np.ndarray,          # (NK, M) complex64
    iqv: np.ndarray,          # (NK, M) complex64
    window: np.ndarray,       # (M,)
    nBootstraps: float,
    K: int = 1,
    NFT: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch reimplementation of processRay_S.

    Accepts numpy arrays, returns numpy arrays.
    Runs entirely in float32/complex64 on the target GPU device.
    """
    from frxx import BACKEND, Backend
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

        allBootstraps = nBootstraps == 0 or (
            np.isinf(nBootstraps) and nBootstraps > 0
        )
        if allBootstraps:
            nBootstraps = Mx - M + 1
        elif (
            not np.isfinite(nBootstraps)
            or nBootstraps < 1
            or int(nBootstraps) != nBootstraps
        ):
            raise ValueError(
                "nBootstraps must be a positive integer, zero, or positive infinity"
            )
        else:
            nBootstraps = int(nBootstraps)

        # Free original tensors early
        del VH, VV

        # --- Compute R0 of original signals (from the middle of extended) ---
        middleStart = M - 1 if nr == 0 else nr - 1
        R0H = torch.mean(torch.abs(XH[:, middleStart:middleStart + M]) ** 2, dim=1)  # (NK,)
        R0V = torch.mean(torch.abs(XV[:, middleStart:middleStart + M]) ** 2, dim=1)  # (NK,)

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

            # Bootstrap start indices: (NK, batch_size)
            if allBootstraps:
                boot_idx = torch.arange(
                    batch_start, batch_start + batch_size, device=device
                ).expand(NK, -1)
            else:
                boot_idx = torch.randint(
                    0, Mx - M + 1, (NK, batch_size), device=device
                )

            # Gather indices: (NK, batch_size * M)
            indices = boot_idx.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)
            indices_flat = indices.reshape(NK, batch_size * M)

            def gather_complex(x, dim, index):
                if x.is_complex() and x.device.type == "mps":
                    real = torch.gather(x.real, dim, index)
                    imag = torch.gather(x.imag, dim, index)
                    return torch.complex(real, imag)
                else:
                    return torch.gather(x, dim, index)


            blockH = gather_complex(XH, 1, indices_flat).reshape(NK, batch_size, M)
            blockV = gather_complex(XV, 1, indices_flat).reshape(NK, batch_size, M)

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

def processRay_S(
    iqh: np.ndarray,          # (NK, M) complex64
    iqv: np.ndarray,          # (NK, M) complex64
    window: np.ndarray,       # (M,)
    nBootstraps: float,
    K: int = 1,
    NFT: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from frxx import BACKEND, Backend
    if BACKEND == Backend.NUMBA:
        return processRay_S_cpp(iqh, iqv, window, nBootstraps, K, NFT)
    else:
        return processRay_S_torch(iqh, iqv, window, nBootstraps, K, NFT)
