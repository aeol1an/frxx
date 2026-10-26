import operator
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
        B: int,
        r: float,
    ) -> tuple[Float64Array, Float64Array, Complex128Array]: ...

    def _computeMultipleSpectra(
        VH: Complex64Array,
        VV: Complex64Array,
        w: Float64Array,
        NK: int,
        M: int,
        NFT: int,
        B: int,
        r: float,
    ) -> tuple[Float64Array, Float64Array, Complex128Array]: ...

    def processRay_S_cpp(
        iqh: Complex64Array,
        iqv: Complex64Array,
        window: int,
        nBootstraps: int,
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

def _add_power_torch(total, correction, value):
    """Compensated signed power addition, also when a strong sample leaves."""
    import torch
    updated = total + value
    error = torch.where(total.abs() >= value.abs(),
                        (total - updated) + value, (value - updated) + total)
    return updated, correction + error


def _initial_power_torch(XH, XV, start, M):
    """Reduce the first block with compensation without per-sample GPU launches."""
    import torch
    total = torch.stack((
        XH[:, start:start + M].abs().square(),
        XV[:, start:start + M].abs().square(),
    ))
    nonzero = (total > 0).sum(dim=-1)
    correction = torch.zeros_like(total)
    while total.shape[-1] > 1:
        if total.shape[-1] % 2:
            total = torch.cat((total, torch.zeros_like(total[..., :1])), dim=-1)
            correction = torch.cat((correction, torch.zeros_like(correction[..., :1])), dim=-1)
        left, right = total[..., ::2], total[..., 1::2]
        low = correction[..., ::2] + correction[..., 1::2]
        total, correction = _add_power_torch(left, low, right)
    return total.squeeze(-1), correction.squeeze(-1), nonzero


def _accumulate_sdft_torch(XH, XV, R0H, R0V, window, M):
    """Sum all starts using rectangular modulated state and output windowing.

    Like sdft.hpp, seed with one FFT and use periodic phase lookup plus
    compensated integration. No windowed/scaled output feeds back into state.
    Storage grows with gates and pulses, without a bootstrap dimension.
    """
    import torch
    from ...utils import windows as wn

    coefficient_functions = (
        wn.rectangular_sdft_coefficients,
        wn.hanning_sdft_coefficients,
        wn.hamming_sdft_coefficients,
        wn.blackman_sdft_coefficients,
        wn.blackman_nuttall_sdft_coefficients,
    )
    coefficients = coefficient_functions[window]()
    original_power = torch.stack((R0H, R0V))
    # Stack only the initial blocks, without duplicating both extended signals.
    accumulator = torch.fft.fft(torch.stack((XH[:, :M], XV[:, :M])), dim=-1)
    compensation = torch.zeros_like(accumulator)
    rectangular = accumulator

    # Build the bounded phase table once in double on the host, then transfer
    # float32 components. MPS need not support float64 or complex indexing.
    table = np.exp(-2j * np.pi * np.arange(M) / M).astype(np.complex64)
    table_real = torch.tensor(table.real, dtype=XH.real.dtype, device=XH.device)
    table_imag = torch.tensor(table.imag, dtype=XH.real.dtype, device=XH.device)
    bins = torch.arange(M, device=XH.device)
    phase = torch.zeros_like(bins)
    modulation = accumulator.new_ones(M)

    SH = torch.zeros_like(accumulator[0].real)
    SV = torch.zeros_like(SH)
    SX = torch.zeros_like(accumulator[0])
    half = len(coefficients) // 2
    for start in range(XH.shape[-1] - M + 1):
        if start:
            delta = torch.stack((
                XH[:, start + M - 1] - XH[:, start - 1],
                XV[:, start + M - 1] - XV[:, start - 1],
            ))
            corrected = modulation * delta.unsqueeze(-1) - compensation
            updated = accumulator + corrected
            compensation = (updated - accumulator) - corrected
            accumulator = updated
            phase = (phase + bins) % M
            modulation = torch.complex(table_real[phase], table_imag[phase])
            rectangular = accumulator * modulation.conj()

        windowed = torch.zeros_like(rectangular)
        for j, coefficient in enumerate(coefficients):
            windowed += float(coefficient) * torch.roll(rectangular, j - half, dims=-1)

        # Keep running sums per polarization/gate, not an M-sample scan per slide.
        # Rebase once per M slides; no device-to-host synchronization is needed.
        if start % M == 0:
            power_sum, power_correction, nonzero = _initial_power_torch(XH, XV, start, M)
        else:
            outgoing = torch.stack((XH[:, start - 1], XV[:, start - 1])).abs().square()
            incoming = torch.stack((XH[:, start + M - 1], XV[:, start + M - 1])).abs().square()
            power_sum, power_correction = _add_power_torch(power_sum, power_correction, -outgoing)
            power_sum, power_correction = _add_power_torch(power_sum, power_correction, incoming)
            nonzero += (incoming > 0).to(nonzero.dtype) - (outgoing > 0).to(nonzero.dtype)
            # A completely empty block must have exactly zero power.
            power_sum = torch.where(nonzero == 0, 0.0, power_sum)
            power_correction = torch.where(nonzero == 0, 0.0, power_correction)
        block_power = (power_sum + power_correction) / M
        scale = torch.sqrt(original_power / block_power.clamp(min=1e-20))
        windowed *= scale.unsqueeze(-1)
        zh, zv = windowed[0], windowed[1]
        SH += zh.real.square() + zh.imag.square()
        SV += zv.real.square() + zv.imag.square()
        SX += zh * zv.conj()

    return SH, SV, SX

def processRay_S_torch(
    iqh: np.ndarray,          # (NK, M) complex64
    iqv: np.ndarray,          # (NK, M) complex64
    window: int,             # 0 rectangular, 1 Hann, 2 Hamming, 3 Blackman, 4 Blackman-Nuttall
    nBootstraps: int,
    K: int = 1,
    NFT: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch reimplementation of processRay_S.

    Accepts numpy arrays, returns numpy arrays.
    Runs entirely in float32/complex64 on the target GPU device.
    Zero bootstraps use SDFT when NFT equals the pulse count; otherwise use FFT.
    """
    nBootstraps = operator.index(nBootstraps)
    if nBootstraps < 0:
        raise ValueError("nBootstraps must be nonnegative")
    window = operator.index(window)
    if window < 0 or window > 4:
        raise ValueError("window must be an integer from 0 to 4")

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
        # Generate periodic windows only after entering the selected backend.
        if window == 0:
            win = torch.ones(M, dtype=fdtype, device=device)
        elif window == 1:
            win = torch.hann_window(M, periodic=True, dtype=fdtype, device=device)
        elif window == 2:
            win = torch.hamming_window(M, periodic=True, dtype=fdtype, device=device)
        elif window == 3:
            win = torch.blackman_window(M, periodic=True, dtype=fdtype, device=device)
        else:
            angle = torch.arange(M, dtype=fdtype, device=device) * (2.0 * np.pi / M)
            win = (0.3635819 - 0.4891775 * torch.cos(angle)
                   + 0.1365995 * torch.cos(2.0 * angle)
                   - 0.0106411 * torch.cos(3.0 * angle))
        # Match the existing C++ formulas for a one-sample window as well.
        if M == 1 and window in (1, 2, 3):
            win.fill_(0.08 if window == 2 else 0.0)

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

        # Zero evaluates every valid bootstrap start exactly once.
        allBootstraps = nBootstraps == 0
        if allBootstraps:
            nBootstraps = Mx - M + 1

        # Free original tensors early
        del VH, VV

        # --- Compute R0 of original signals (from the middle of extended) ---
        middleStart = M - 1 if nr == 0 else nr - 1
        R0H = torch.mean(torch.abs(XH[:, middleStart:middleStart + M]) ** 2, dim=1)  # (NK,)
        R0V = torch.mean(torch.abs(XV[:, middleStart:middleStart + M]) ** 2, dim=1)  # (NK,)

        # --- Accumulate spectra across bootstraps ---
        if allBootstraps and NFT == M:
            SH_acc, SV_acc, SX_acc = _accumulate_sdft_torch(
                XH, XV, R0H, R0V, window, M)
        else:
            SH_acc = torch.zeros((NK, NFT), dtype=fdtype, device=device)
            SV_acc = torch.zeros((NK, NFT), dtype=fdtype, device=device)
            SX_acc = torch.zeros((NK, NFT), dtype=cdtype, device=device)
            offsets = torch.arange(M, device=device)  # (M,)

            # --- Auto-calculate bootstrap_batch ---
            if device.type == "cuda":
                torch.cuda.synchronize()
                free_mem, _ = torch.cuda.mem_get_info(device)
            else:
                # MPS has no reliable free-memory query.
                import psutil
                free_mem = int(psutil.virtual_memory().available * 0.5)

            # Peak storage per bootstrap: two sample blocks and three spectra.
            bytes_per_unit = NK * (2 * M * 8 + 3 * NFT * 8 + 8)
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
                del indices, indices_flat, boot_idx

                r0h_block = torch.mean(torch.abs(blockH) ** 2, dim=2).clamp(min=1e-20)
                r0v_block = torch.mean(torch.abs(blockV) ** 2, dim=2).clamp(min=1e-20)
                scaleH = torch.sqrt(R0H[:, None] / r0h_block)
                scaleV = torch.sqrt(R0V[:, None] / r0v_block)

                blockH = blockH * (scaleH.unsqueeze(2) * win.unsqueeze(0).unsqueeze(0))
                blockV = blockV * (scaleV.unsqueeze(2) * win.unsqueeze(0).unsqueeze(0))
                fftH = torch.fft.fft(blockH, n=NFT, dim=2)
                fftV = torch.fft.fft(blockV, n=NFT, dim=2)

                SH_acc += (fftH.real ** 2 + fftH.imag ** 2).sum(dim=1)
                SV_acc += (fftV.real ** 2 + fftV.imag ** 2).sum(dim=1)
                SX_acc += (fftH * fftV.conj()).sum(dim=1)
                del blockH, blockV, fftH, fftV

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
    window: int,             # Built-in window ID, passed unchanged to the backend
    nBootstraps: int,
    K: int = 1,
    NFT: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from frxx import BACKEND, Backend
    if BACKEND == Backend.CPU:
        return processRay_S_cpp(iqh, iqv, window, nBootstraps, K, NFT)
    else:
        return processRay_S_torch(iqh, iqv, window, nBootstraps, K, NFT)
