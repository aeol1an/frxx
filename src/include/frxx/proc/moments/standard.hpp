#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::proc::moments::standard {

using Complex128Array2DRef =
    frxx::eigen::Array2DRef<std::complex<double>>;

/// Calculate RH for each requested lag plus lag-zero RV and RX,
/// lag-zero correlations, writing directly into caller-owned output arrays.
///
/// Each RH output has shape `(time_group, range)` and corresponds to the lag at
/// the same index. RV and RX have that same shape.
void process_rays(
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqh,
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqv,
    frxx::eigen::ConstArray2DRef<frxx::utils::i64> pulseBoundaries,
    frxx::eigen::ConstArray1DRef<std::int32_t> lags,
    std::vector<Complex128Array2DRef>& RH,
    Complex128Array2DRef RV,
    Complex128Array2DRef RX);

}  // namespace frxx::proc::moments::standard
