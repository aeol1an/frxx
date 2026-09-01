#pragma once

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::utils {

/// Pulse index ranges and center angles produced by pulse grouping.
struct PulseBoundaries {
    /// Inclusive start and end pulse indices, one row per group.
    frxx::eigen::Array2D<i64> indices;
    /// Center angle in degrees for each pulse group.
    frxx::eigen::Array1D<float> angles;
};

/// Test whether an angle lies strictly inside a possibly wrapped degree range.
///
/// @param value Angle to test, in degrees.
/// @param low Exclusive lower bound, in degrees.
/// @param high Exclusive upper bound, in degrees.
/// @return True when `value` is inside the range.
bool in_degree_range(double value, double low, double high);

/// Group a scan's pulse angles and find each group's inclusive index bounds.
///
/// @param angle Pulse angles stored as float32 degrees.
/// @param pixel_width_degrees Requested pixel width in degrees.
/// @param beam_overlap_degrees Beam overlap on each side in degrees.
/// @return Index pairs and their corresponding group-center angles.
PulseBoundaries find_pulse_boundaries(
    frxx::eigen::ConstArray1DRef<float> angle,
    float pixel_width_degrees,
    float beam_overlap_degrees
);

}  // namespace frxx::utils
