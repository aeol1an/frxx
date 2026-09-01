#pragma once

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::utils {

/// Return `size` float64 samples of a rectangular (all-ones) window.
frxx::eigen::Array1D<double> rectangular(i64 size);

/// Return `size` float64 samples of a Hann window.
frxx::eigen::Array1D<double> hanning(i64 size);

/// Return `size` float64 samples of a Hamming window.
frxx::eigen::Array1D<double> hamming(i64 size);

/// Return `size` float64 samples of a Blackman window.
frxx::eigen::Array1D<double> blackman(i64 size);

/// Return `size` float64 samples of a Bartlett window.
frxx::eigen::Array1D<double> bartlett(i64 size);

}  // namespace frxx::utils
