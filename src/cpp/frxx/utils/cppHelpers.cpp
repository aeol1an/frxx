#include <frxx/utils/cppHelpers.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>

namespace frxx::utils {

namespace {

template <typename Array, typename Mask>
void require_same_shape(const Array& array, const Mask& mask) {
    if (array.rows() != mask.rows() || array.cols() != mask.cols()) {
        throw std::invalid_argument("arr and mask must have the same shape");
    }
}

template <typename T, typename Array, typename Mask>
frxx::eigen::Array1D<T> get_masked_impl(
    const Array& array, const Mask& mask
) {
    require_same_shape(array, mask);
    i64 selected = 0;
    for (Eigen::Index row = 0; row < mask.rows(); ++row) {
        for (Eigen::Index column = 0; column < mask.cols(); ++column) {
            selected += mask(row, column) ? 1 : 0;
        }
    }

    frxx::eigen::Array1D<T> output(selected);
    i64 index = 0;
    for (Eigen::Index row = 0; row < array.rows(); ++row) {
        for (Eigen::Index column = 0; column < array.cols(); ++column) {
            if (mask(row, column)) {
                output(index++) = array(row, column);
            }
        }
    }
    return output;
}

template <typename Array, typename Mask, typename T>
void set_masked_scalar_impl(Array array, const Mask& mask, T value) {
    require_same_shape(array, mask);
    for (Eigen::Index row = 0; row < array.rows(); ++row) {
        for (Eigen::Index column = 0; column < array.cols(); ++column) {
            if (mask(row, column)) {
                array(row, column) = value;
            }
        }
    }
}

template <typename Array, typename Mask, typename Values>
void set_masked_array_impl(Array array, const Mask& mask, const Values& values) {
    require_same_shape(array, mask);
    i64 selected = 0;
    for (Eigen::Index row = 0; row < mask.rows(); ++row) {
        for (Eigen::Index column = 0; column < mask.cols(); ++column) {
            selected += mask(row, column) ? 1 : 0;
        }
    }
    if (values.size() < selected) {
        throw std::invalid_argument("val does not contain enough elements");
    }

    i64 index = 0;
    for (Eigen::Index row = 0; row < array.rows(); ++row) {
        for (Eigen::Index column = 0; column < array.cols(); ++column) {
            if (mask(row, column)) {
                array(row, column) = values(index++);
            }
        }
    }
}

template <typename Array, typename Compare>
i64 nanarg_impl(const Array& array, Compare compare) {
    i64 index = 0;
    bool found = false;
    for (Eigen::Index current = 0; current < array.size(); ++current) {
        const auto value = array(current);
        if (!std::isnan(value) && (!found || compare(value, array(index)))) {
            index = static_cast<i64>(current);
            found = true;
        }
    }
    if (!found) {
        throw std::invalid_argument("All-NaN slice encountered");
    }
    return index;
}

}  // namespace

frxx::eigen::Array1D<float> get_masked_float2d(
    frxx::eigen::ConstArray2DRef<float> array,
    frxx::eigen::ConstArray2DRef<bool> mask
) {
    return get_masked_impl<float>(array, mask);
}

frxx::eigen::Array1D<double> get_masked_float2d(
    frxx::eigen::ConstArray2DRef<double> array,
    frxx::eigen::ConstArray2DRef<bool> mask
) {
    return get_masked_impl<double>(array, mask);
}

void set_masked_float2d_scalar(
    frxx::eigen::Array2DRef<float> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    float value
) {
    set_masked_scalar_impl(array, mask, value);
}

void set_masked_float2d_scalar(
    frxx::eigen::Array2DRef<double> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    double value
) {
    set_masked_scalar_impl(array, mask, value);
}

void set_masked_float2d_array(
    frxx::eigen::Array2DRef<float> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    frxx::eigen::ConstArray1DRef<float> values
) {
    set_masked_array_impl(array, mask, values);
}

void set_masked_float2d_array(
    frxx::eigen::Array2DRef<double> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    frxx::eigen::ConstArray1DRef<double> values
) {
    set_masked_array_impl(array, mask, values);
}

i64 nanargmax(frxx::eigen::ConstArray1DRef<float> array) {
    return nanarg_impl(array, std::greater<float>{});
}

i64 nanargmax(frxx::eigen::ConstArray1DRef<double> array) {
    return nanarg_impl(array, std::greater<double>{});
}

i64 nanargmin(frxx::eigen::ConstArray1DRef<float> array) {
    return nanarg_impl(array, std::less<float>{});
}

i64 nanargmin(frxx::eigen::ConstArray1DRef<double> array) {
    return nanarg_impl(array, std::less<double>{});
}

}  // namespace frxx::utils
