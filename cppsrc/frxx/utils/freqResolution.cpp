#include <frxx/utils/freqResolution.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace frxx::utils {

double velocity_resolution(double pulse_count, double prf, double wavelength) {
    return prf / pulse_count * wavelength / 2.0;
}

double velocity_resolution_to_pulses(
    double delta_velocity, double prf, double wavelength
) {
    return prf / (delta_velocity * 2.0 / wavelength);
}

namespace {

template <typename T>
frxx::eigen::Array1D<T> make_velocity_axis(
    i64 size,
    T nyquist_velocity,
    bool flip_velocity,
    i64 left_unfolds,
    i64 right_unfolds
) {
    if (size < 0) {
        throw std::invalid_argument("Number of samples, " + std::to_string(size) +
            ", must be non-negative.");
    }
    frxx::eigen::Array1D<T> output(size);
    if (size == 0) {
        return output;
    }
    const T typed_start = flip_velocity
        ? -nyquist_velocity - static_cast<T>(2 * left_unfolds) * nyquist_velocity
        : nyquist_velocity + static_cast<T>(2 * right_unfolds) * nyquist_velocity;
    const T typed_stop = flip_velocity
        ? nyquist_velocity + static_cast<T>(2 * right_unfolds) * nyquist_velocity
        : -nyquist_velocity - static_cast<T>(2 * left_unfolds) * nyquist_velocity;
    if (size == 1) {
        output(0) = typed_start;
        return output;
    }
    const double start = static_cast<double>(typed_start);
    const double stop = static_cast<double>(typed_stop);
    const double step = (stop - start) / static_cast<double>(size - 1);
    for (i64 index = 0; index < size; ++index) {
        // Match NumPy linspace's separate multiplication and addition.
        volatile double scaled = static_cast<double>(index) * step;
        output(index) = static_cast<T>(start + scaled);
    }
    output(size - 1) = typed_stop;
    return output;
}

}  // namespace

frxx::eigen::Array1D<float> velocity_axis(
    i64 size, float nyquist_velocity, bool flip_velocity,
    i64 left_unfolds, i64 right_unfolds
) {
    return make_velocity_axis(
        size, nyquist_velocity, flip_velocity, left_unfolds, right_unfolds);
}

frxx::eigen::Array1D<double> velocity_axis(
    i64 size, double nyquist_velocity, bool flip_velocity,
    i64 left_unfolds, i64 right_unfolds
) {
    return make_velocity_axis(
        size, nyquist_velocity, flip_velocity, left_unfolds, right_unfolds);
}

i64 velocity_span_to_bins(
    double delta_velocity, i64 fft_size, double prf, double wavelength
) {
    const double bin_width = prf * wavelength /
        (2.0 * static_cast<double>(fft_size));
    return std::max<i64>(1, static_cast<i64>(std::nearbyint(delta_velocity / bin_width)));
}

}  // namespace frxx::utils
