#pragma once

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::utils {

/// Compute Doppler velocity resolution.
///
/// @param pulse_count Number of pulses contributing to the transform.
/// @param prf Pulse repetition frequency in hertz.
/// @param wavelength Radar wavelength in meters.
/// @return Velocity resolution in meters per second.
double velocity_resolution(
    double pulse_count, double prf = 4000.0, double wavelength = 0.0308);

/// Compute the pulse count needed for a requested velocity resolution.
///
/// @param delta_velocity Requested velocity resolution in meters per second.
/// @param prf Pulse repetition frequency in hertz.
/// @param wavelength Radar wavelength in meters.
/// @return Required pulse count.
double velocity_resolution_to_pulses(
    double delta_velocity, double prf = 4000.0, double wavelength = 0.0308);

/// Construct a float32 Doppler velocity axis.
///
/// @param size Number of bins in the axis.
/// @param nyquist_velocity Nyquist velocity in meters per second.
/// @param flip_velocity Whether the axis should increase from negative to positive.
/// @param left_unfolds Number of additional Nyquist intervals on the left.
/// @param right_unfolds Number of additional Nyquist intervals on the right.
/// @return Float32 velocity coordinates.
frxx::eigen::Array1D<float> velocity_axis(
    i64 size,
    float nyquist_velocity,
    bool flip_velocity,
    i64 left_unfolds = 0,
    i64 right_unfolds = 0
);

/// Construct a float64 Doppler velocity axis. Arguments match the float32 overload.
frxx::eigen::Array1D<double> velocity_axis(
    i64 size,
    double nyquist_velocity,
    bool flip_velocity,
    i64 left_unfolds = 0,
    i64 right_unfolds = 0
);

/// Convert a velocity span to the nearest non-zero FFT bin count.
///
/// @param delta_velocity Velocity span in meters per second.
/// @param fft_size Number of FFT bins.
/// @param prf Pulse repetition frequency in hertz.
/// @param wavelength Radar wavelength in meters.
/// @return Nearest bin count, with a minimum of one.
i64 velocity_span_to_bins(
    double delta_velocity,
    i64 fft_size,
    double prf = 4000.0,
    double wavelength = 0.0308
);

}  // namespace frxx::utils
