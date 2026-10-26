#pragma once

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::utils {

/// Return `size` float64 samples of a rectangular (all-ones) window.
frxx::eigen::Array1D<double> rectangular(i64 size);

/// Return `size` float64 samples of a periodic Hann window (denominator `size`).
frxx::eigen::Array1D<double> hanning(i64 size);

/// Return `size` float64 samples of a periodic Hamming window (denominator `size`).
frxx::eigen::Array1D<double> hamming(i64 size);

/// Return `size` float64 samples of a periodic Blackman window (denominator `size`).
frxx::eigen::Array1D<double> blackman(i64 size);

/// Return `size` float64 samples of a periodic Blackman-Nuttall (Nuttall4c) window.
/// Cosine amplitudes: [0.3635819, -0.4891775, 0.1365995, -0.0106411].
/// Matches scipy.signal.windows.nuttall(size, sym=False) for size >= 2.
frxx::eigen::Array1D<double> blackman_nuttall(i64 size);

/// Frequency-domain window kernels for an M-point DFT of M samples.
///
/// For a kernel c of length 2*P+1, apply
/// Y[k] = sum_{q=-P}^{P} c[q+P] * X[(k-q) mod M].
/// Coefficients are ordered by offsets -P through +P, with DC in the middle;
/// repeated bin indices for small M are added. No further normalization is needed.
/// These are length-independent kernels of the periodic windows above, not
/// time-domain cosine amplitudes. With zero padding, adjacent-bin offsets no
/// longer apply: the harmonic spacing is NFT/M bins instead.
/// Return the rectangular kernel [1].
frxx::eigen::Array1D<double> rectangular_sdft_coefficients();

/// Return the periodic Hann kernel [-0.25, 0.5, -0.25].
frxx::eigen::Array1D<double> hanning_sdft_coefficients();

/// Return the periodic Hamming kernel [-0.23, 0.54, -0.23].
frxx::eigen::Array1D<double> hamming_sdft_coefficients();

/// Return the periodic Blackman kernel [0.04, -0.25, 0.42, -0.25, 0.04].
frxx::eigen::Array1D<double> blackman_sdft_coefficients();

/// Return the seven-tap periodic Blackman-Nuttall kernel at offsets -3..+3.
frxx::eigen::Array1D<double> blackman_nuttall_sdft_coefficients();

}  // namespace frxx::utils
