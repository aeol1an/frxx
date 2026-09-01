#include <frxx/utils/numbaWindows.hpp>

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
            2.0 * pi * static_cast<double>(index) / static_cast<double>(count - 1)));
    });
}

frxx::eigen::Array1D<double> hamming(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        return 0.54 - 0.46 * std::cos(
            2.0 * pi * static_cast<double>(index) / static_cast<double>(count - 1));
    });
}

frxx::eigen::Array1D<double> blackman(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        const double position = static_cast<double>(index) /
            static_cast<double>(count - 1);
        return 0.42 - 0.50 * std::cos(2.0 * pi * position) +
            0.08 * std::cos(4.0 * pi * position);
    });
}

frxx::eigen::Array1D<double> bartlett(i64 size) {
    return make_window(size, [](i64 index, i64 count) {
        return 1.0 - std::abs(
            2.0 * static_cast<double>(index) / static_cast<double>(count - 1) - 1.0);
    });
}

}  // namespace frxx::utils
