#pragma once

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::utils {

/// Copy masked float32 elements from a two-dimensional Eigen array.
///
/// @param array Source values.
/// @param mask Boolean selector with the same shape as `array`.
/// @return Selected elements in row-major mask traversal order.
frxx::eigen::Array1D<float> get_masked_float2d(
    frxx::eigen::ConstArray2DRef<float> array,
    frxx::eigen::ConstArray2DRef<bool> mask
);

/// Float64 overload of `get_masked_float2d`.
frxx::eigen::Array1D<double> get_masked_float2d(
    frxx::eigen::ConstArray2DRef<double> array,
    frxx::eigen::ConstArray2DRef<bool> mask
);

/// Assign a scalar to selected elements of a float32 Eigen array.
///
/// @param array Mutable destination values.
/// @param mask Boolean selector with the same shape as `array`.
/// @param value Scalar assigned wherever the mask is true.
void set_masked_float2d_scalar(
    frxx::eigen::Array2DRef<float> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    float value
);

/// Float64 overload of `set_masked_float2d_scalar`.
void set_masked_float2d_scalar(
    frxx::eigen::Array2DRef<double> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    double value
);

/// Assign consecutive float32 values to selected elements of an Eigen array.
///
/// @param array Mutable destination values.
/// @param mask Boolean selector with the same shape as `array`.
/// @param values Replacement values in row-major mask-selection order.
void set_masked_float2d_array(
    frxx::eigen::Array2DRef<float> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    frxx::eigen::ConstArray1DRef<float> values
);

/// Float64 overload of `set_masked_float2d_array`.
void set_masked_float2d_array(
    frxx::eigen::Array2DRef<double> array,
    frxx::eigen::ConstArray2DRef<bool> mask,
    frxx::eigen::ConstArray1DRef<double> values
);

/// Return the index of the greatest non-NaN float32 value.
i64 nanargmax(frxx::eigen::ConstArray1DRef<float> array);

/// Float64 overload of `nanargmax`.
i64 nanargmax(frxx::eigen::ConstArray1DRef<double> array);

/// Return the index of the least non-NaN float32 value.
i64 nanargmin(frxx::eigen::ConstArray1DRef<float> array);

/// Float64 overload of `nanargmin`.
i64 nanargmin(frxx::eigen::ConstArray1DRef<double> array);

}  // namespace frxx::utils
