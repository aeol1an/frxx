/**
 * Adapted from https://github.com/jurihock/sdft
 * Upstream: cpp/src/sdft/sdft.h
 * Copyright (c) 2022 Juergen Hock
 * SPDX-License-Identifier: MIT
 * License text: LICENSES/sdft/LICENSE
 *
 * Modulated Sliding DFT according to:
 * [1] Krzysztof Duda, Accurate, Guaranteed Stable, Sliding Discrete Fourier
 *     Transform, IEEE Signal Processing Magazine (2010).
 *     https://ieeexplore.ieee.org/document/5563098
 * [2] Russell Bradford, Richard Dobson and John ffitch, Sliding is Smoother
 *     than Jumping, International Computer Music Conference (2005).
 *     http://hdl.handle.net/2027/spo.bbp2372.2005.086
 *
 * frxx adaptation: Eigen arrays, complex IQ input, full unnormalized M-bin
 * spectra, bounded phase lookup, and compensated sums. Windowing is external.
 * The upstream real-input half-spectrum and inverse synthesis API is omitted.
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <pocketfft/pocketfft.hpp>

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::utils::sdft {

/// Modulated sliding DFT of the most recent M complex samples.
///
/// Output equals FFT(x) with negative exponential sign, no scaling,
/// and no fftshift. Construct from the first complete block, ordered oldest to
/// newest; one unwindowed FFT initializes the internal state. spectrum() returns
/// the first rectangular spectrum; slide(sample) replaces the oldest sample and
/// returns the next rectangular spectrum. Returned arrays own their data.
/// For allocation-free streaming, call advance(sample) and read spectrum_view().
/// M is both the window and FFT length; zero padding and a separate NFT are not
/// supported. Apply window kernels to returned spectra outside this class.
///
/// T is the input component type; F is the accumulator/output component type.
/// Double accumulation is recommended. Phase factors come from a periodic table
/// rather than a recursive oscillator. Compensated sums reduce remaining
/// accumulation roundoff; finite-precision output is not exact indefinitely.
/// Each plan owns mutable stream state and should be used by one worker at a time.
template <typename T = float, typename F = double>
class SDFT {
    static_assert(std::is_floating_point_v<T> && std::is_floating_point_v<F>,
        "SDFT component types must be floating point");

public:
    using Input = std::complex<T>;
    using Complex = std::complex<F>;

    explicit SDFT(frxx::eigen::ConstArray1DRef<Input> first_samples)
        : size_(require_size(first_samples.size())),
          input_(first_samples),
          accumulator_(2, size_),
          compensation_(frxx::eigen::Array2D<F>::Zero(2, size_)),
          phases_(frxx::eigen::Array1D<i64>::Zero(size_)),
          twiddles_(2, size_), spectrum_(first_samples.template cast<Complex>()) {
        const F omega = F(-2) * std::acos(F(-1)) / static_cast<F>(size_);
        for (i64 k = 0; k < size_; ++k) {
            const auto twiddle = std::polar(F(1), omega * static_cast<F>(k));
            twiddles_(0, k) = twiddle.real();
            twiddles_(1, k) = twiddle.imag();
        }
        const pocketfft::shape_t shape{static_cast<std::size_t>(size_)};
        const pocketfft::stride_t stride{
            static_cast<std::ptrdiff_t>(sizeof(Complex))};
        pocketfft::c2c(shape, stride, stride, {0}, true,
            spectrum_.data(), spectrum_.data(), F(1), 1);
        accumulator_.row(0) = spectrum_.real().transpose();
        accumulator_.row(1) = spectrum_.imag().transpose();
    }

    i64 size() const noexcept { return size_; }

    /// Replace the oldest sample and return the next rectangular spectrum.
    frxx::eigen::Array1D<Complex> slide(Input sample) {
        advance(sample);
        return spectrum();
    }

    /// Replace the oldest sample, reusing the plan's spectrum buffer.
    void advance(Input sample) {
        const Complex delta = static_cast<Complex>(sample) -
            static_cast<Complex>(input_(cursor_));
        input_(cursor_) = sample;

        // Work in small fixed-size Eigen arrays: arithmetic can use each
        // platform's SIMD packets without full-spectrum temporary arrays.
        i64 k = 0;
        for (; k <= size_ - block_size; k += block_size) {
            update_block<block_size>(k, delta);
        }
        for (; k < size_; ++k) {
            update_block<1>(k, delta);
        }
        cursor_ = cursor_ == size_ - 1 ? 0 : cursor_ + 1;
    }

    /// Return the current rectangular spectrum without advancing the sample block.
    frxx::eigen::Array1D<Complex> spectrum() const {
        return spectrum_;
    }

    /// Borrow the current spectrum without allocating or copying.
    /// The buffer belongs to this plan and its contents change on the next update.
    /// The reference must not outlive the plan or be used across a move of it.
    const frxx::eigen::Array1D<Complex>& spectrum_view() const & noexcept {
        return spectrum_;
    }
    const frxx::eigen::Array1D<Complex>& spectrum_view() const && = delete;

private:
    static constexpr int block_size = 8;

    template <int Count>
    void update_block(i64 first, Complex delta) {
        using Block = Eigen::Array<F, Count, 1>;
        Block modulationR, modulationI, demodulationR, demodulationI;
        // Gather table entries independently of the SIMD arithmetic. This also
        // works on CPUs without hardware gather instructions.
        for (int j = 0; j < Count; ++j) {
            const i64 k = first + j;
            const i64 phase = phases_(k);
            modulationR(j) = twiddles_(0, phase);
            modulationI(j) = twiddles_(1, phase);
            const i64 distance = size_ - k;
            const i64 next = phase >= distance ? phase - distance : phase + k;
            phases_(k) = next;
            demodulationR(j) = twiddles_(0, next);
            demodulationI(j) = twiddles_(1, next);
        }

        Eigen::Map<Block> accR(accumulator_.row(0).data() + first);
        Eigen::Map<Block> accI(accumulator_.row(1).data() + first);
        Eigen::Map<Block> compR(compensation_.row(0).data() + first);
        Eigen::Map<Block> compI(compensation_.row(1).data() + first);
        // Materialize each stage so compensated summation keeps its ordering.
        const Block correctedR = (modulationR * delta.real() -
            modulationI * delta.imag()) - compR;
        const Block correctedI = (modulationR * delta.imag() +
            modulationI * delta.real()) - compI;
        const Block updatedR = accR + correctedR;
        const Block updatedI = accI + correctedI;
        compR = (updatedR - accR) - correctedR;
        compI = (updatedI - accI) - correctedI;
        accR = updatedR;
        accI = updatedI;

        const Block resultR = updatedR * demodulationR + updatedI * demodulationI;
        const Block resultI = updatedI * demodulationR - updatedR * demodulationI;
        for (int j = 0; j < Count; ++j) {
            spectrum_(first + j) = Complex(resultR(j), resultI(j));
        }
    }

    static i64 require_size(i64 size) {
        if (size <= 0) {
            throw std::invalid_argument("M must be positive");
        }
        return size;
    }

    const i64 size_;
    frxx::eigen::Array1D<Input> input_;
    frxx::eigen::Array2D<F> accumulator_;
    frxx::eigen::Array2D<F> compensation_;
    frxx::eigen::Array1D<i64> phases_;
    frxx::eigen::Array2D<F> twiddles_;
    frxx::eigen::Array1D<Complex> spectrum_;
    i64 cursor_ = 0;
};

}  // namespace frxx::utils::sdft
