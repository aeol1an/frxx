#pragma once

#include <complex>

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::proc::algs::acf {

/// Write the range-wise cross-correlation of two complex64 IQ matrices.
///
/// Both inputs are indexed by range and time and must have equal shapes.
/// The result contains one complex128 value per range. Positive lags shift
/// `X1`; negative lags shift `X2`. Normalization always uses the full
/// time dimension, matching the original implementation.
void compute_ray_m(
    frxx::eigen::ConstArray2DRef<std::complex<float>> X1,
    frxx::eigen::ConstArray2DRef<std::complex<float>> X2,
    frxx::eigen::Array1DRef<std::complex<double>> result,
    frxx::utils::i64 lag = 0);

/// Owning-result overload of `compute_ray_m`.
frxx::eigen::Array1D<std::complex<double>> compute_ray_m(
    frxx::eigen::ConstArray2DRef<std::complex<float>> X1,
    frxx::eigen::ConstArray2DRef<std::complex<float>> X2,
    frxx::utils::i64 lag = 0);

}  // namespace frxx::proc::algs::acf
