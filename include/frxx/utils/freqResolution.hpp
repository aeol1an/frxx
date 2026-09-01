#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <frxx/utils/integer.hpp>

namespace frxx::utils {

inline double velocity_resolution(
    double pulse_count, double prf = 4000.0, double wavelength = 0.0308
) {
    return prf / pulse_count * wavelength / 2.0;
}

inline double velocity_resolution_to_pulses(
    double delta_velocity, double prf = 4000.0, double wavelength = 0.0308
) {
    return prf / (delta_velocity * 2.0 / wavelength);
}

template <typename T>
std::vector<T> velocity_axis(
    i64 size, T nyquist_velocity, bool flip_velocity,
    i64 left_unfolds = 0, i64 right_unfolds = 0
) {
    if (size < 0) {
        throw std::invalid_argument("Number of samples, " + std::to_string(size) +
            ", must be non-negative.");
    }
    if (size == 0) {
        return {};
    }
    const T typed_start = flip_velocity
        ? -nyquist_velocity - static_cast<T>(2 * left_unfolds) * nyquist_velocity
        : nyquist_velocity + static_cast<T>(2 * right_unfolds) * nyquist_velocity;
    const T typed_stop = flip_velocity
        ? nyquist_velocity + static_cast<T>(2 * right_unfolds) * nyquist_velocity
        : -nyquist_velocity - static_cast<T>(2 * left_unfolds) * nyquist_velocity;
    if (size == 1) {
        return {typed_start};
    }
    std::vector<T> output(static_cast<std::size_t>(size));
    const double start = static_cast<double>(typed_start);
    const double stop = static_cast<double>(typed_stop);
    const double step = (stop - start) / static_cast<double>(size - 1);
    for (i64 index = 0; index < size; ++index) {
        // NumPy performs linspace multiplication and addition as separate
        // operations. Prevent contraction so the converted axis is bit-identical.
        volatile double scaled = static_cast<double>(index) * step;
        output[static_cast<std::size_t>(index)] =
            static_cast<T>(start + scaled);
    }
    output.back() = typed_stop;
    return output;
}

inline i64 velocity_span_to_bins(
    double delta_velocity, i64 fft_size,
    double prf = 4000.0, double wavelength = 0.0308
) {
    const double bin_width = prf * wavelength /
        (2.0 * static_cast<double>(fft_size));
    return std::max<i64>(1, static_cast<i64>(std::nearbyint(delta_velocity / bin_width)));
}

}  // namespace frxx::utils
