#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <frxx/utils/integer.hpp>

namespace frxx::utils {

template <typename T, typename Array, typename Mask>
std::vector<T> get_masked_float2d(
    const Array& array, const Mask& mask, i64 rows, i64 columns
) {
    std::vector<T> output;
    for (i64 row = 0; row < rows; ++row) {
        for (i64 column = 0; column < columns; ++column) {
            if (mask(row, column)) {
                output.push_back(array(row, column));
            }
        }
    }
    return output;
}

template <typename T, typename Array, typename Mask>
void set_masked_float2d_scalar(
    Array& array, const Mask& mask, i64 rows, i64 columns, T value
) {
    for (i64 row = 0; row < rows; ++row) {
        for (i64 column = 0; column < columns; ++column) {
            if (mask(row, column)) {
                array(row, column) = value;
            }
        }
    }
}

template <typename Array, typename Mask, typename Values>
void set_masked_float2d_array(
    Array& array, const Mask& mask, const Values& values,
    i64 rows, i64 columns
) {
    i64 index = 0;
    for (i64 row = 0; row < rows; ++row) {
        for (i64 column = 0; column < columns; ++column) {
            if (mask(row, column)) {
                array(row, column) = values(index++);
            }
        }
    }
}

template <typename Array>
i64 nanargmax(const Array& array, i64 size) {
    i64 index = 0;
    bool found = false;
    for (i64 current = 0; current < size; ++current) {
        const auto value = array(current);
        if (!std::isnan(value) && (!found || value > array(index))) {
            index = current;
            found = true;
        }
    }
    if (!found) {
        throw std::invalid_argument("All-NaN slice encountered");
    }
    return index;
}

template <typename Array>
i64 nanargmin(const Array& array, i64 size) {
    i64 index = 0;
    bool found = false;
    for (i64 current = 0; current < size; ++current) {
        const auto value = array(current);
        if (!std::isnan(value) && (!found || value < array(index))) {
            index = current;
            found = true;
        }
    }
    if (!found) {
        throw std::invalid_argument("All-NaN slice encountered");
    }
    return index;
}

}  // namespace frxx::utils
