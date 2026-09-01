#include <frxx/proc/algs/bootstrapDPSD.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pocketfft/pocketfft.hpp>

#include <frxx/utils/workerPool.hpp>

namespace frxx::proc::algs::bootstrap_dpsd {

namespace {

using frxx::utils::i64;
using Complex64Array2D = frxx::eigen::Array2D<std::complex<float>>;

struct BootstrapSetup {
    i64 B;
    bool allBootstraps;
};

BootstrapSetup resolve_bootstraps(double nBootstraps, i64 nPossibleBootstraps) {
    if (nBootstraps == 0.0 ||
        (std::isinf(nBootstraps) && nBootstraps > 0.0)) {
        return {nPossibleBootstraps, true};
    }
    if (!std::isfinite(nBootstraps) || nBootstraps < 1.0 ||
        std::floor(nBootstraps) != nBootstraps ||
        nBootstraps > static_cast<double>(std::numeric_limits<i64>::max())) {
        throw std::invalid_argument(
            "nBootstraps must be a positive integer, zero, or positive infinity");
    }
    return {static_cast<i64>(nBootstraps), false};
}

void fft_rows(Complex64Array2D& W, i64 B, i64 NFT) {
    const pocketfft::shape_t shape{
        static_cast<std::size_t>(B), static_cast<std::size_t>(NFT)};
    const pocketfft::stride_t stride{
        static_cast<std::ptrdiff_t>(NFT * sizeof(std::complex<float>)),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
    const pocketfft::shape_t axes{1};
    pocketfft::c2c(
        shape, stride, stride, axes, true,
        W.data(), W.data(), 1.0F, 1);
}

void require_single_spectrum_inputs(
    frxx::eigen::ConstArray1DRef<std::complex<float>> VH,
    frxx::eigen::ConstArray1DRef<std::complex<float>> VV,
    frxx::eigen::ConstArray1DRef<double> w,
    i64 M,
    i64 NFT
) {
    if (M <= 0) {
        throw std::invalid_argument("M must be positive");
    }
    if (NFT <= 0) {
        throw std::invalid_argument("NFT must be positive");
    }
    if (VH.size() != M || VV.size() != M || w.size() != M) {
        throw std::invalid_argument("VH, VV, and w must have length M");
    }
}

}  // namespace

SingleSpectrumResult computeSingleSpectrum(
    frxx::eigen::ConstArray1DRef<std::complex<float>> VH,
    frxx::eigen::ConstArray1DRef<std::complex<float>> VV,
    frxx::eigen::ConstArray1DRef<double> w,
    i64 M,
    i64 NFT,
    double nBootstraps,
    double r
) {
    require_single_spectrum_inputs(VH, VV, w, M, NFT);

    // Guard CX
    std::complex<double> CX_left;
    std::complex<double> CX_right;
    if (std::abs(VH(M - 1)) < 1e-30 || std::abs(VH(0)) < 1e-30 ||
        std::abs(VV(M - 1)) < 1e-30 || std::abs(VV(0)) < 1e-30) {
        CX_left = std::complex<float>(1.0F);
        CX_right = std::complex<float>(1.0F);
    } else {
        CX_left = 0.5 * (
            static_cast<std::complex<double>>(VH(0)) /
                static_cast<std::complex<double>>(VH(M - 1)) +
            static_cast<std::complex<double>>(VV(0)) /
                static_cast<std::complex<double>>(VV(M - 1)));
        CX_right = 0.5 * (
            static_cast<std::complex<double>>(VH(M - 1)) /
                static_cast<std::complex<double>>(VH(0)) +
            static_cast<std::complex<double>>(VV(M - 1)) /
                static_cast<std::complex<double>>(VV(0)));
    }

    const i64 nr = static_cast<i64>(std::nearbyint(static_cast<double>(M) * r));
    const i64 negnr = -nr;
    if (nr < 0 || nr > M) {
        throw std::invalid_argument("round(M * r) must be between zero and M");
    }

    const i64 leftStart = negnr == 0 ? 0 : M + negnr;
    const i64 leftSize = M - 1 - leftStart;
    const i64 rightSize = std::max<i64>(0, nr - 1);
    const i64 Mx = leftSize + M + rightSize;
    frxx::eigen::Array1D<std::complex<double>> XH(Mx);
    frxx::eigen::Array1D<std::complex<double>> XV(Mx);

    i64 x = 0;
    for (i64 j = leftStart; j < M - 1; ++j, ++x) {
        XH(x) = static_cast<std::complex<double>>(VH(j)) * CX_left;
        XV(x) = static_cast<std::complex<double>>(VV(j)) * CX_left;
    }
    for (i64 j = 0; j < M; ++j, ++x) {
        XH(x) = static_cast<std::complex<double>>(VH(j));
        XV(x) = static_cast<std::complex<double>>(VV(j));
    }
    for (i64 j = 1; j < nr; ++j, ++x) {
        XH(x) = static_cast<std::complex<double>>(VH(j)) * CX_right;
        XV(x) = static_cast<std::complex<double>>(VV(j)) * CX_right;
    }

    // R0 with no temporaries
    double accH = 0.0;
    double accV = 0.0;
    for (i64 j = 0; j < M; ++j) {
        const auto vh = VH(j);
        const auto vv = VV(j);
        accH += static_cast<double>(
            vh.real() * vh.real() + vh.imag() * vh.imag());
        accV += static_cast<double>(
            vv.real() * vv.real() + vv.imag() * vv.imag());
    }
    const double R0H = accH / static_cast<double>(M);
    const double R0V = accV / static_cast<double>(M);

    const i64 nPossibleBootstraps = Mx - M + 1;
    const auto bootstrapSetup = resolve_bootstraps(
        nBootstraps, nPossibleBootstraps);
    const i64 B = bootstrapSetup.B;

    // Fused: bootstrap + R0 + rescale + window in one parallel loop
    // The original implementation used range, rather than prange, here.
    //
    // WH and WV include the zero padding required by NFT. pocketfft transforms
    // every bootstrap row in-place, avoiding separate zH and zV allocations.
    Complex64Array2D WH(B, NFT);
    Complex64Array2D WV(B, NFT);
    WH.setZero();
    WV.setZero();

    std::optional<std::mt19937_64> randomEngine;
    if (!bootstrapSetup.allBootstraps) {
        randomEngine.emplace(std::random_device{}());
    }
    std::uniform_int_distribution<i64> randomBootIdx(
        0, nPossibleBootstraps - 1);

    for (i64 i = 0; i < B; ++i) {
        const i64 boot_idx = bootstrapSetup.allBootstraps
            ? i
            : randomBootIdx(*randomEngine);

        // Pass 1: extract block and accumulate |x|^2 for R0
        accH = 0.0;
        accV = 0.0;
        for (i64 j = 0; j < M; ++j) {
            const auto vh = XH(boot_idx + j);
            const auto vv = XV(boot_idx + j);
            accH += vh.real() * vh.real() + vh.imag() * vh.imag();
            accV += vv.real() * vv.real() + vv.imag() * vv.imag();
        }

        // Guard bootstrap scale
        if (accH < 1e-30) {
            accH = 1e-30;
        }
        if (accV < 1e-30) {
            accV = 1e-30;
        }
        // Pass 2: rescale + window in-place (row is hot in L1)
        const double scaleH = std::sqrt(R0H * M / accH);
        const double scaleV = std::sqrt(R0V * M / accV);
        for (i64 j = 0; j < std::min(M, NFT); ++j) {
            // Match the original complex64 WH/WV storage before the FFT.
            const auto wh = static_cast<std::complex<float>>(XH(boot_idx + j));
            const auto wv = static_cast<std::complex<float>>(XV(boot_idx + j));
            WH(i, j) = static_cast<std::complex<float>>(
                static_cast<std::complex<double>>(wh) * (scaleH * w(j)));
            WV(i, j) = static_cast<std::complex<float>>(
                static_cast<std::complex<double>>(wv) * (scaleV * w(j)));
        }
    }

    // FFT
    fft_rows(WH, B, NFT);
    fft_rows(WV, B, NFT);

    // Spectral averages — parallelize over frequency bins
    // The original implementation used range, rather than prange, here too.
    double alpha = 0.0;
    for (i64 j = 0; j < M; ++j) {
        alpha += w(j) * w(j);
    }
    alpha /= static_cast<double>(M);
    const double norm = static_cast<double>(M) * alpha * B;

    SingleSpectrumResult result{
        frxx::eigen::Array1D<double>(NFT),
        frxx::eigen::Array1D<double>(NFT),
        frxx::eigen::Array1D<std::complex<double>>(NFT)};

    const i64 half = NFT / 2;
    for (i64 j = 0; j < NFT; ++j) {
        double sh = 0.0;
        double sv = 0.0;
        std::complex<double> sx{0.0, 0.0};
        for (i64 i = 0; i < B; ++i) {
            const auto zh = static_cast<std::complex<double>>(WH(i, j));
            const auto zv = static_cast<std::complex<double>>(WV(i, j));
            sh += zh.real() * zh.real() + zh.imag() * zh.imag();
            sv += zv.real() * zv.real() + zv.imag() * zv.imag();
            sx += zh * std::conj(zv);
        }
        const i64 k = (j + half) % NFT;
        result.SHi(k) = sh / norm;
        result.SVi(k) = sv / norm;
        result.SXi(k) = sx / norm;
    }

    return result;
}

MultipleSpectraResult computeMultipleSpectra(
    frxx::eigen::ConstArray2DRef<std::complex<float>> VH,
    frxx::eigen::ConstArray2DRef<std::complex<float>> VV,
    frxx::eigen::ConstArray1DRef<double> w,
    i64 NK,
    i64 M,
    i64 NFT,
    double nBootstraps,
    double r
) {
    if (VH.rows() != NK || VV.rows() != NK ||
        VH.cols() != M || VV.cols() != M) {
        throw std::invalid_argument("VH and VV must have shape (NK, M)");
    }
    if (w.size() != M) {
        throw std::invalid_argument("w must have length M");
    }

    MultipleSpectraResult result{
        frxx::eigen::Array2D<double>(NK, NFT),
        frxx::eigen::Array2D<double>(NK, NFT),
        frxx::eigen::Array2D<std::complex<double>>(NK, NFT)};

    frxx::utils::WorkerPool pool;
    pool.pfor(0, NK, [&](i64 i) {
        auto VH_view = VH.row(i).transpose();
        auto VV_view = VV.row(i).transpose();
        const auto spectrum = computeSingleSpectrum(
            VH_view, VV_view, w,
            M, NFT, nBootstraps, r);
        result.SH.row(i) = spectrum.SHi.transpose();
        result.SV.row(i) = spectrum.SVi.transpose();
        result.SX.row(i) = spectrum.SXi.transpose();
    });

    return result;
}

ProcessRayResult processRay_S(
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqh,
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqv,
    frxx::eigen::ConstArray1DRef<double> window,
    double nBootstraps,
    i64 K,
    i64 NFT
) {
    if (iqh.rows() != iqv.rows() || iqh.cols() != iqv.cols()) {
        throw std::invalid_argument("iqh and iqv must have equal shapes");
    }
    if (K <= 0) {
        throw std::invalid_argument("K must be positive");
    }
    if (iqh.cols() <= 0 || window.size() != iqh.cols()) {
        throw std::invalid_argument("window must have one value per IQ pulse");
    }

    double meanWindowPower = 0.0;
    for (i64 j = 0; j < window.size(); ++j) {
        meanWindowPower += window(j) * window(j);
    }
    meanWindowPower /= static_cast<double>(window.size());
    const double r = 0.5 - std::sqrt(meanWindowPower) * 0.5;

    const i64 NK = iqh.rows();
    const i64 M = iqh.cols();

    const i64 N = NK / K;

    if (NFT <= 1) {
        NFT = M;
    }

    auto spectra = computeMultipleSpectra(
        iqh, iqv, window,
        NK, M, NFT, nBootstraps, r);
    auto& SH = spectra.SH;
    auto& SV = spectra.SV;
    auto& SX = spectra.SX;

    frxx::eigen::Array2D<double> tsh(N, NFT);
    frxx::eigen::Array2D<double> tsv(N, NFT);
    frxx::eigen::Array2D<std::complex<double>> tsx(N, NFT);
    frxx::eigen::Array2D<double> td(N, NFT);
    frxx::eigen::Array2D<double> tr(N, NFT);

    for (i64 i = 0; i < N; ++i) {
        for (i64 j = 0; j < NFT; ++j) {
            double sh = 0.0;
            double sv = 0.0;
            std::complex<double> sx{0.0, 0.0};
            for (i64 k = 0; k < K; ++k) {
                sh += SH(i * K + k, j);
                sv += SV(i * K + k, j);
                sx += SX(i * K + k, j);
            }
            tsh(i, j) = sh / K;
            tsv(i, j) = sv / K;
            tsx(i, j) = sx / static_cast<double>(K);
            if (tsv(i, j) < 1e-30) {
                td(i, j) = std::numeric_limits<double>::quiet_NaN();
                tr(i, j) = std::numeric_limits<double>::quiet_NaN();
            } else {
                td(i, j) = tsh(i, j) / tsv(i, j);
                const double denom = std::sqrt(tsh(i, j) * tsv(i, j));
                if (denom < 1e-30) {
                    tr(i, j) = std::numeric_limits<double>::quiet_NaN();
                } else {
                    tr(i, j) = std::abs(tsx(i, j)) / denom;
                }
            }
        }
    }

    double beta;
    if (K == 1) {
        beta = std::pow(1 - r, -3.3) - 2 * std::pow(1 - r, 1.1);
    } else {
        beta = std::pow(1 - r, -4.5) - std::pow(1 - r, -2.1);
    }

    auto& PSDH = tsh;
    auto& PSDV = tsv;
    // COV = tsx

    frxx::eigen::Array2D<double> sZDR(N, NFT);
    frxx::eigen::Array2D<double> sRHOHV(N, NFT);
    for (i64 i = 0; i < N; ++i) {
        for (i64 j = 0; j < NFT; ++j) {
            double trsquared = std::pow(tr(i, j), 2);
            if (trsquared < 1e-30) {
                trsquared = 1e-30;
            }

            sZDR(i, j) = td(i, j) *
                (1 - (1 / (beta * K) * (1 - trsquared)));
            sRHOHV(i, j) = tr(i, j) *
                (1 - (1 / (beta * K) *
                    (std::pow(1 - trsquared, 2) / (4 * trsquared))));

            if (PSDH(i, j) < 0) {
                PSDH(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
            if (PSDV(i, j) < 0) {
                PSDV(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
            if (sZDR(i, j) < 0) {
                sZDR(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
            if (sRHOHV(i, j) < 0) {
                sRHOHV(i, j) = 0;
            }

            PSDH(i, j) = 10 * std::log10(PSDH(i, j));
            PSDV(i, j) = 10 * std::log10(PSDV(i, j));
            sZDR(i, j) = 10 * std::log10(sZDR(i, j));
        }
    }

    // return 10*log10(PSDH), 10*log10(PSDV), COV, 10*log10(sZDR), sRHOHV
    return {
        std::move(PSDH), std::move(PSDV),
        std::move(sZDR), std::move(sRHOHV)};
}

}  // namespace frxx::proc::algs::bootstrap_dpsd
