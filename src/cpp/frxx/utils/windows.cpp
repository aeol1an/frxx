#include <frxx/utils/windows.hpp>

#include <cmath>
#include <stdexcept>

namespace frxx::utils {

namespace {

template <typename Function>
frxx::eigen::Array1D<double> make_window(i64 size, Function function) {
    if (size <= 0) {
        return frxx::eigen::Array1D<double>(0);
    }
    frxx::eigen::Array1D<double> output(size);
    for (i64 index = 0; index < size; ++index) {
        output(index) = function(index, size);
    }
    return output;
}

}  // namespace

frxx::eigen::Array1D<double> rectangular(i64 size) {
    if (size < 0) {
        throw std::invalid_argument("negative dimensions not allowed");
    }
    return frxx::eigen::Array1D<double>::Ones(size);
}

frxx::eigen::Array1D<double> hanning(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        return 0.5 * (1.0 - std::cos(
            2.0 * pi * static_cast<double>(index) / static_cast<double>(count)));
    });
}

frxx::eigen::Array1D<double> hamming(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        return 0.54 - 0.46 * std::cos(
            2.0 * pi * static_cast<double>(index) / static_cast<double>(count));
    });
}

frxx::eigen::Array1D<double> blackman(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        const double position = static_cast<double>(index) /
            static_cast<double>(count);
        return 0.42 - 0.50 * std::cos(2.0 * pi * position) +
            0.08 * std::cos(4.0 * pi * position);
    });
}

frxx::eigen::Array1D<double> blackman_nuttall(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        const double angle = 2.0 * pi * static_cast<double>(index) /
            static_cast<double>(count);
        // Nuttall4c, also used by scipy.signal.windows.nuttall.
        // https://www.mathworks.com/help/signal/ref/nuttallwin.html
        return 0.3635819 - 0.4891775 * std::cos(angle) +
            0.1365995 * std::cos(2.0 * angle) -
            0.0106411 * std::cos(3.0 * angle);
    });
}

frxx::eigen::Array1D<double> rectangular_sdft_coefficients() {
    return frxx::eigen::Array1D<double>::Ones(1);
}

frxx::eigen::Array1D<double> hanning_sdft_coefficients() {
    frxx::eigen::Array1D<double> coefficients(3);
    coefficients << -0.25, 0.5, -0.25;
    return coefficients;
}

frxx::eigen::Array1D<double> hamming_sdft_coefficients() {
    frxx::eigen::Array1D<double> coefficients(3);
    coefficients << -0.23, 0.54, -0.23;
    return coefficients;
}

frxx::eigen::Array1D<double> blackman_sdft_coefficients() {
    frxx::eigen::Array1D<double> coefficients(5);
    coefficients << 0.04, -0.25, 0.42, -0.25, 0.04;
    return coefficients;
}

frxx::eigen::Array1D<double> blackman_nuttall_sdft_coefficients() {
    frxx::eigen::Array1D<double> coefficients(7);
    coefficients << -0.00532055, 0.06829975, -0.24458875, 0.3635819,
        -0.24458875, 0.06829975, -0.00532055;
    return coefficients;
}

}  // namespace frxx::utils
