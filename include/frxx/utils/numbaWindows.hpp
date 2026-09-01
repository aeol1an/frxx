#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <frxx/utils/integer.hpp>

namespace frxx::utils {

inline std::vector<double> rectangular(i64 size) {
    if (size < 0) {
        throw std::invalid_argument("negative dimensions not allowed");
    }
    return std::vector<double>(static_cast<std::size_t>(size), 1.0);
}

template <typename Function>
std::vector<double> make_window(i64 size, Function&& function) {
    if (size <= 0) {
        return {};
    }
    std::vector<double> output(static_cast<std::size_t>(size));
    for (i64 index = 0; index < size; ++index) {
        output[static_cast<std::size_t>(index)] = function(index, size);
    }
    return output;
}

inline std::vector<double> hanning(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        return 0.5 * (1.0 - std::cos(
            2.0 * pi * static_cast<double>(index) / static_cast<double>(count - 1)));
    });
}

inline std::vector<double> hamming(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        return 0.54 - 0.46 * std::cos(
            2.0 * pi * static_cast<double>(index) / static_cast<double>(count - 1));
    });
}

inline std::vector<double> blackman(i64 size) {
    const double pi = std::acos(-1.0);
    return make_window(size, [pi](i64 index, i64 count) {
        const double position = static_cast<double>(index) /
            static_cast<double>(count - 1);
        return 0.42 - 0.50 * std::cos(2.0 * pi * position) +
            0.08 * std::cos(4.0 * pi * position);
    });
}

inline std::vector<double> bartlett(i64 size) {
    return make_window(size, [](i64 index, i64 count) {
        return 1.0 - std::abs(
            2.0 * static_cast<double>(index) / static_cast<double>(count - 1) - 1.0);
    });
}

}  // namespace frxx::utils
