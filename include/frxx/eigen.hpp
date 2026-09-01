#pragma once

#include <Eigen/Core>

namespace frxx::eigen {

using DynamicStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using DynamicInnerStride = Eigen::InnerStride<Eigen::Dynamic>;

template <typename Scalar>
using Array1D = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Array2D = Eigen::Array<
    Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Scalar>
using ConstArray1DRef = Eigen::Ref<
    const Array1D<Scalar>, 0, DynamicInnerStride>;

template <typename Scalar>
using Array1DRef = Eigen::Ref<
    Array1D<Scalar>, 0, DynamicInnerStride>;

template <typename Scalar>
using ConstArray2DRef = Eigen::Ref<
    const Array2D<Scalar>, 0, DynamicStride>;

template <typename Scalar>
using Array2DRef = Eigen::Ref<
    Array2D<Scalar>, 0, DynamicStride>;

}  // namespace frxx::eigen
