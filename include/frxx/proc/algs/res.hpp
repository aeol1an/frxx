#pragma once

#include <Eigen/Core>

#include <complex>
#include <stdexcept>
#include <string>

#include <frxx/utils/integer.hpp>

namespace frxx::proc::algs::res {

/// Invalid configuration, shape, or argument supplied to an IQ subset operation.
class ArgumentError final : public std::invalid_argument {
public:
    explicit ArgumentError(const std::string& message);
};

/// Index or range outside the available IQ data.
class BoundsError final : public std::out_of_range {
public:
    explicit BoundsError(const std::string& message);
};

using DynamicStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

using Complex64Matrix = Eigen::Matrix<
    std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Complex128Matrix = Eigen::Matrix<
    std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Int64Matrix = Eigen::Matrix<
    frxx::utils::i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Int64Vector = Eigen::Matrix<frxx::utils::i64, Eigen::Dynamic, 1>;

using ConstComplex64MatrixRef = Eigen::Ref<
    const Complex64Matrix, 0, DynamicStride>;
using Complex64MatrixRef = Eigen::Ref<
    Complex64Matrix, 0, DynamicStride>;
using ConstComplex128MatrixRef = Eigen::Ref<
    const Complex128Matrix, 0, DynamicStride>;
using Complex128MatrixRef = Eigen::Ref<
    Complex128Matrix, 0, DynamicStride>;
using ConstInt64MatrixRef = Eigen::Ref<
    const Int64Matrix, 0, DynamicStride>;
using ConstInt64VectorRef = Eigen::Ref<
    const Int64Vector, 0, Eigen::InnerStride<Eigen::Dynamic>>;

struct Complex64SubsetResult {
    Complex64Matrix values;
    frxx::utils::i64 range_count;
    frxx::utils::i64 pulse_count;
};

struct Complex128SubsetResult {
    Complex128Matrix values;
    frxx::utils::i64 range_count;
    frxx::utils::i64 pulse_count;
};

/// Copy a range-averaging IQ subset into an existing complex64 matrix.
///
/// @param iq Source matrix indexed by range gate and pulse.
/// @param result Destination with at least `K * range_count` rows.
/// @param K Number of neighboring range gates represented per output range.
/// @param Koffset Selects the low or high side for even-sized neighborhoods.
/// @param range_count Number of source range gates to process.
/// @param start_range First source range gate.
/// @param first_pulse First source pulse, inclusive.
/// @param last_pulse Last source pulse, inclusive.
void range_subset(
    ConstComplex64MatrixRef iq,
    Complex64MatrixRef result,
    frxx::utils::i64 K,
    frxx::utils::i64 Koffset,
    frxx::utils::i64 range_count,
    frxx::utils::i64 start_range,
    frxx::utils::i64 first_pulse,
    frxx::utils::i64 last_pulse
);

/// Complex128 overload of `range_subset`.
void range_subset(
    ConstComplex128MatrixRef iq,
    Complex128MatrixRef result,
    frxx::utils::i64 K,
    frxx::utils::i64 Koffset,
    frxx::utils::i64 range_count,
    frxx::utils::i64 start_range,
    frxx::utils::i64 first_pulse,
    frxx::utils::i64 last_pulse
);

/// Copy an azimuth-averaging IQ subset into an existing complex64 matrix.
///
/// @param iq Source matrix indexed by range gate and pulse.
/// @param result Destination arranged as range-major azimuth rows.
/// @param range_count Number of source range gates to process.
/// @param start_range First source range gate.
/// @param first_pulses Inclusive first pulse for each azimuth.
/// @param last_pulses Inclusive last pulse for each azimuth.
void azimuth_subset(
    ConstComplex64MatrixRef iq,
    Complex64MatrixRef result,
    frxx::utils::i64 range_count,
    frxx::utils::i64 start_range,
    ConstInt64VectorRef first_pulses,
    ConstInt64VectorRef last_pulses
);

/// Complex128 overload of `azimuth_subset`.
void azimuth_subset(
    ConstComplex128MatrixRef iq,
    Complex128MatrixRef result,
    frxx::utils::i64 range_count,
    frxx::utils::i64 start_range,
    ConstInt64VectorRef first_pulses,
    ConstInt64VectorRef last_pulses
);

/// Build an IQ subset for a range- or azimuth-averaging request.
///
/// @param iq Source matrix indexed by range gate and pulse.
/// @param azimuth_index Requested azimuth index; negative indices are supported.
/// @param azimuth_count Number of available azimuth groups.
/// @param azimuth_increasing Whether scan angles increase with pulse order.
/// @param pulse_boundaries Inclusive first/last pulse columns per azimuth.
/// @param ranges Inclusive first/last source range gates.
/// @param swath_pulses Requested pulse count, or less than two for automatic sizing.
/// @param K Neighborhood size used for range or azimuth averaging.
/// @param Koffset Selects the low or high side for even-sized neighborhoods.
/// @param average_strategy Zero for range averaging, one for azimuth averaging.
/// @param shape_only Skip matrix construction and return only output dimensions.
Complex64SubsetResult subset_iq(
    ConstComplex64MatrixRef iq,
    frxx::utils::i64 azimuth_index,
    frxx::utils::i64 azimuth_count,
    bool azimuth_increasing,
    ConstInt64MatrixRef pulse_boundaries,
    ConstInt64VectorRef ranges,
    frxx::utils::i64 swath_pulses = -1,
    frxx::utils::i64 K = 1,
    frxx::utils::i64 Koffset = 0,
    frxx::utils::i64 average_strategy = 1,
    bool shape_only = false
);

/// Complex128 overload of `subset_iq`.
Complex128SubsetResult subset_iq(
    ConstComplex128MatrixRef iq,
    frxx::utils::i64 azimuth_index,
    frxx::utils::i64 azimuth_count,
    bool azimuth_increasing,
    ConstInt64MatrixRef pulse_boundaries,
    ConstInt64VectorRef ranges,
    frxx::utils::i64 swath_pulses = -1,
    frxx::utils::i64 K = 1,
    frxx::utils::i64 Koffset = 0,
    frxx::utils::i64 average_strategy = 1,
    bool shape_only = false
);

}  // namespace frxx::proc::algs::res
