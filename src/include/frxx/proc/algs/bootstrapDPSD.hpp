#pragma once

#include <complex>

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::proc::algs::bootstrap_dpsd {

struct SingleSpectrumResult {
    frxx::eigen::Array1D<double> SHi;
    frxx::eigen::Array1D<double> SVi;
    frxx::eigen::Array1D<std::complex<double>> SXi;
};

struct MultipleSpectraResult {
    frxx::eigen::Array2D<double> SH;
    frxx::eigen::Array2D<double> SV;
    frxx::eigen::Array2D<std::complex<double>> SX;
};

struct ProcessRayResult {
    frxx::eigen::Array2D<double> PSDH;
    frxx::eigen::Array2D<double> PSDV;
    frxx::eigen::Array2D<double> sZDR;
    frxx::eigen::Array2D<double> sRHOHV;
};

/// Compute one dual-polarization bootstrap spectrum.
///
/// `nBootstraps == 0` evaluates every valid bootstrap start exactly once.
/// A positive integer samples starts with replacement.
SingleSpectrumResult computeSingleSpectrum(
    frxx::eigen::ConstArray1DRef<std::complex<float>> VH,
    frxx::eigen::ConstArray1DRef<std::complex<float>> VV,
    frxx::eigen::ConstArray1DRef<double> w,
    frxx::utils::i64 M,
    frxx::utils::i64 NFT,
    frxx::utils::i64 nBootstraps,
    double r);

/// Compute a bootstrap spectrum for every row of VH and VV.
MultipleSpectraResult computeMultipleSpectra(
    frxx::eigen::ConstArray2DRef<std::complex<float>> VH,
    frxx::eigen::ConstArray2DRef<std::complex<float>> VV,
    frxx::eigen::ConstArray1DRef<double> w,
    frxx::utils::i64 NK,
    frxx::utils::i64 M,
    frxx::utils::i64 NFT,
    frxx::utils::i64 nBootstraps,
    double r);

/// Calculate PSDH, PSDV, spectral ZDR, and spectral RHOHV for one ray.
/// Window IDs: 0 rectangular, 1 Hann, 2 Hamming, 3 Blackman, 4 Blackman-Nuttall.
/// With nBootstraps == 0 and NFT equal to the pulse count (including its default),
/// use SDFT to evaluate all starts. Other cases use the FFT bootstrap path.
ProcessRayResult processRay_S(
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqh,
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqv,
    frxx::utils::i64 window,
    frxx::utils::i64 nBootstraps,
    frxx::utils::i64 K = 1,
    frxx::utils::i64 NFT = 1);

}  // namespace frxx::proc::algs::bootstrap_dpsd
