#include <frxx/proc/algs/res.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <frxx/utils/integer.hpp>

namespace frxx::proc::algs::res {

ArgumentError::ArgumentError(const std::string& message)
    : std::invalid_argument(message) {}

BoundsError::BoundsError(const std::string& message)
    : std::out_of_range(message) {}

namespace {

using frxx::utils::floor_div;
using frxx::utils::i64;

i64 normalize_index(i64 index, Eigen::Index size, const char* name) {
    if (index < 0) {
        index += static_cast<i64>(size);
    }
    if (index < 0 || index >= static_cast<i64>(size)) {
        throw BoundsError(std::string(name) + " is out of bounds");
    }
    return index;
}

template <typename Input, typename Output>
void range_subset_impl(
    const Input& iq,
    Output&& result,
    i64 K,
    i64 Koffset,
    i64 range_count,
    i64 start_range,
    i64 first_pulse,
    i64 last_pulse
) {
    if (K < 0 || range_count < 0) {
        throw ArgumentError("K and NR must be non-negative");
    }

    const i64 range_gates = static_cast<i64>(iq.rows());
    const i64 pulses = static_cast<i64>(iq.cols());
    const i64 pulse_count = last_pulse - first_pulse + 1;
    if (range_gates == 0 || first_pulse < 0 ||
        last_pulse >= pulses || pulse_count < 0) {
        throw BoundsError("IQ subset indices are out of bounds");
    }
    if (result.rows() < K * range_count || result.cols() != pulse_count) {
        throw ArgumentError("result has an incompatible shape");
    }

    for (i64 range = 0; range < range_count; ++range) {
        for (i64 neighbor = 0; neighbor < K; ++neighbor) {
            i64 source_range = neighbor + range - (K / 2 - Koffset) + start_range;
            source_range = std::clamp(source_range, i64{0}, range_gates - 1);
            const i64 output_range = range * K + neighbor;
            result.row(output_range) = iq.row(source_range).segment(
                first_pulse, pulse_count);
        }
    }
}

template <typename Input, typename Output>
void azimuth_subset_impl(
    const Input& iq,
    Output&& result,
    i64 range_count,
    i64 start_range,
    ConstInt64VectorRef first_pulses,
    ConstInt64VectorRef last_pulses
) {
    if (first_pulses.size() != last_pulses.size()) {
        throw ArgumentError("fps and lps must have the same length");
    }

    const i64 azimuth_count = static_cast<i64>(first_pulses.size());
    const i64 pulse_count = static_cast<i64>(result.cols());
    if (range_count < 0 || result.rows() < range_count * azimuth_count) {
        throw ArgumentError("result has an incompatible shape");
    }
    if (start_range < 0 || start_range + range_count > iq.rows()) {
        throw BoundsError("range indices are out of bounds");
    }

    const i64 available_pulses = static_cast<i64>(iq.cols());
    for (i64 azimuth = 0; azimuth < azimuth_count; ++azimuth) {
        if (first_pulses(azimuth) < 0 ||
            last_pulses(azimuth) >= available_pulses ||
            last_pulses(azimuth) - first_pulses(azimuth) + 1 != pulse_count) {
            throw ArgumentError("pulse bounds are incompatible with result");
        }
    }

    for (i64 range = 0; range < range_count; ++range) {
        for (i64 azimuth = 0; azimuth < azimuth_count; ++azimuth) {
            const i64 output_range = range * azimuth_count + azimuth;
            result.row(output_range) = iq.row(range + start_range).segment(
                first_pulses(azimuth), pulse_count);
        }
    }
}

template <typename Matrix>
struct NativeSubsetResult {
    Matrix values;
    i64 range_count;
    i64 pulse_count;
};

template <typename Matrix, typename Input>
NativeSubsetResult<Matrix> subset_iq_impl(
    const Input& iq,
    i64 azimuth_index,
    i64 azimuth_count,
    bool azimuth_increasing,
    ConstInt64MatrixRef pulse_boundaries,
    ConstInt64VectorRef ranges,
    i64 swath_pulses,
    i64 K,
    i64 Koffset,
    i64 average_strategy,
    bool shape_only
) {
    if (K < 1) {
        throw ArgumentError("K must be greater than 0.");
    }
    if (Koffset != 0 && Koffset != 1) {
        throw ArgumentError("Valid values for KOffset: {0(low), 1(high)}");
    }
    if (average_strategy != 0 && average_strategy != 1) {
        throw ArgumentError(
            "Valid values for avgStrat: {0(range), 1(azimuth)}");
    }
    if (pulse_boundaries.cols() < 2) {
        throw ArgumentError("pulseBoundaries must have at least two columns");
    }
    if (ranges.size() < 2) {
        throw ArgumentError("iranges must contain a start and end index");
    }

    const i64 pulse_count = static_cast<i64>(iq.cols());
    const i64 start_range = ranges(0);
    const i64 range_count = ranges(1) + 1 - start_range;
    if (range_count < 0) {
        throw ArgumentError("negative dimensions are not allowed");
    }

    NativeSubsetResult<Matrix> result{Matrix(0, 0), range_count, swath_pulses};

    if (K > 1 && average_strategy == 0) {
        const i64 boundary_index = normalize_index(
            azimuth_index, pulse_boundaries.rows(), "iaz");
        const i64 boundary_start = pulse_boundaries(boundary_index, 0);
        const i64 boundary_end = pulse_boundaries(boundary_index, 1);
        const i64 center_pulse = boundary_start + floor_div(
            boundary_end + 1 - boundary_start, 2);
        if (center_pulse < 0 || center_pulse >= pulse_count) {
            throw ArgumentError("Center pulse out of bounds.");
        }
        if (swath_pulses < 2) {
            swath_pulses = boundary_end + 1 - boundary_start;
        }

        i64 first_pulse = center_pulse - floor_div(swath_pulses, 2);
        i64 last_pulse = swath_pulses % 2 != 0
            ? center_pulse + floor_div(swath_pulses, 2)
            : center_pulse + floor_div(swath_pulses, 2) - 1;
        first_pulse = std::max(i64{0}, first_pulse);
        last_pulse = std::min(pulse_count - 1, last_pulse);
        swath_pulses = last_pulse - first_pulse + 1;

        if (!shape_only) {
            result.values.resize(K * range_count, swath_pulses);
            range_subset_impl(
                iq, result.values, K, Koffset, range_count,
                start_range, first_pulse, last_pulse);
        }
    } else if (K > 1) {
        if (azimuth_count < 1) {
            throw ArgumentError("naz must be greater than 0");
        }

        std::vector<i64> azimuth_indices(static_cast<std::size_t>(K));
        const i64 decreasing_shift = (K + 1) / 2 - std::abs(Koffset - 1);
        for (i64 index = 0; index < K; ++index) {
            const i64 azimuth = azimuth_increasing
                ? index - (K / 2 - Koffset) + azimuth_index
                : (K - 1 - index) - decreasing_shift + azimuth_index;
            azimuth_indices[static_cast<std::size_t>(index)] =
                std::clamp(azimuth, i64{0}, azimuth_count - 1);
        }

        Int64Vector first_pulses(K);
        Int64Vector last_pulses(K);
        if (swath_pulses < 2) {
            swath_pulses = std::numeric_limits<i64>::max();
        }

        for (i64 index = 0; index < K; ++index) {
            const i64 boundary_index = normalize_index(
                azimuth_indices[static_cast<std::size_t>(index)],
                pulse_boundaries.rows(), "azimuth index");
            azimuth_indices[static_cast<std::size_t>(index)] = boundary_index;
            const i64 boundary_start = pulse_boundaries(boundary_index, 0);
            const i64 boundary_end = pulse_boundaries(boundary_index, 1);
            const i64 center_pulse = boundary_start + floor_div(
                boundary_end + 1 - boundary_start, 2);
            if (center_pulse < 0 || center_pulse >= pulse_count) {
                throw ArgumentError("A center pulse is out of bounds.");
            }
            if (swath_pulses == std::numeric_limits<i64>::max()) {
                first_pulses(index) = boundary_end + 1 - boundary_start;
            }
        }

        if (swath_pulses == std::numeric_limits<i64>::max()) {
            swath_pulses = first_pulses.minCoeff();
        }

        i64 common_pulses = std::numeric_limits<i64>::max();
        for (i64 index = 0; index < K; ++index) {
            const i64 boundary_index =
                azimuth_indices[static_cast<std::size_t>(index)];
            const i64 boundary_start = pulse_boundaries(boundary_index, 0);
            const i64 boundary_end = pulse_boundaries(boundary_index, 1);
            const i64 center_pulse = boundary_start + floor_div(
                boundary_end + 1 - boundary_start, 2);
            i64 first_pulse = center_pulse - floor_div(swath_pulses, 2);
            i64 last_pulse = swath_pulses % 2 != 0
                ? center_pulse + floor_div(swath_pulses, 2)
                : center_pulse + floor_div(swath_pulses, 2) - 1;
            first_pulse = std::max(i64{0}, first_pulse);
            last_pulse = std::min(pulse_count - 1, last_pulse);
            first_pulses(index) = first_pulse;
            last_pulses(index) = last_pulse;
            common_pulses = std::min(common_pulses, last_pulse - first_pulse + 1);
        }
        swath_pulses = common_pulses;
        for (i64 index = 0; index < K; ++index) {
            last_pulses(index) = first_pulses(index) + swath_pulses - 1;
        }

        if (!shape_only) {
            result.values.resize(K * range_count, swath_pulses);
            azimuth_subset_impl(
                iq, result.values, range_count, start_range,
                first_pulses, last_pulses);
        }
    } else {
        const i64 boundary_index = normalize_index(
            azimuth_index, pulse_boundaries.rows(), "iaz");
        const i64 boundary_start = pulse_boundaries(boundary_index, 0);
        const i64 boundary_end = pulse_boundaries(boundary_index, 1);
        const i64 center_pulse = boundary_start + floor_div(
            boundary_end + 1 - boundary_start, 2);
        if (center_pulse < 0 || center_pulse >= pulse_count) {
            throw ArgumentError("Center pulse out of bounds.");
        }
        if (swath_pulses < 2) {
            swath_pulses = boundary_end + 1 - boundary_start;
        }

        i64 first_pulse = center_pulse - floor_div(swath_pulses, 2);
        i64 last_pulse = swath_pulses % 2 != 0
            ? center_pulse + floor_div(swath_pulses, 2)
            : center_pulse + floor_div(swath_pulses, 2) - 1;
        first_pulse = std::max(i64{0}, first_pulse);
        last_pulse = std::min(pulse_count - 1, last_pulse);
        swath_pulses = last_pulse - first_pulse + 1;

        if (!shape_only) {
            if (start_range < 0 || start_range + range_count > iq.rows()) {
                throw BoundsError("range indices are out of bounds");
            }
            result.values = iq.block(
                start_range, first_pulse, range_count, swath_pulses);
        }
    }

    result.pulse_count = swath_pulses;
    return result;
}

}  // namespace

void range_subset(
    ConstComplex64MatrixRef iq, Complex64MatrixRef result,
    i64 K, i64 Koffset, i64 range_count, i64 start_range,
    i64 first_pulse, i64 last_pulse
) {
    range_subset_impl(
        iq, result, K, Koffset, range_count, start_range,
        first_pulse, last_pulse);
}

void range_subset(
    ConstComplex128MatrixRef iq, Complex128MatrixRef result,
    i64 K, i64 Koffset, i64 range_count, i64 start_range,
    i64 first_pulse, i64 last_pulse
) {
    range_subset_impl(
        iq, result, K, Koffset, range_count, start_range,
        first_pulse, last_pulse);
}

void azimuth_subset(
    ConstComplex64MatrixRef iq, Complex64MatrixRef result,
    i64 range_count, i64 start_range,
    ConstInt64VectorRef first_pulses, ConstInt64VectorRef last_pulses
) {
    azimuth_subset_impl(
        iq, result, range_count, start_range, first_pulses, last_pulses);
}

void azimuth_subset(
    ConstComplex128MatrixRef iq, Complex128MatrixRef result,
    i64 range_count, i64 start_range,
    ConstInt64VectorRef first_pulses, ConstInt64VectorRef last_pulses
) {
    azimuth_subset_impl(
        iq, result, range_count, start_range, first_pulses, last_pulses);
}

Complex64SubsetResult subset_iq(
    ConstComplex64MatrixRef iq,
    i64 azimuth_index,
    i64 azimuth_count,
    bool azimuth_increasing,
    ConstInt64MatrixRef pulse_boundaries,
    ConstInt64VectorRef ranges,
    i64 swath_pulses,
    i64 K,
    i64 Koffset,
    i64 average_strategy,
    bool shape_only
) {
    auto result = subset_iq_impl<Complex64Matrix>(
        iq, azimuth_index, azimuth_count, azimuth_increasing,
        pulse_boundaries, ranges, swath_pulses, K, Koffset,
        average_strategy, shape_only);
    return {std::move(result.values), result.range_count, result.pulse_count};
}

Complex128SubsetResult subset_iq(
    ConstComplex128MatrixRef iq,
    i64 azimuth_index,
    i64 azimuth_count,
    bool azimuth_increasing,
    ConstInt64MatrixRef pulse_boundaries,
    ConstInt64VectorRef ranges,
    i64 swath_pulses,
    i64 K,
    i64 Koffset,
    i64 average_strategy,
    bool shape_only
) {
    auto result = subset_iq_impl<Complex128Matrix>(
        iq, azimuth_index, azimuth_count, azimuth_increasing,
        pulse_boundaries, ranges, swath_pulses, K, Koffset,
        average_strategy, shape_only);
    return {std::move(result.values), result.range_count, result.pulse_count};
}

}  // namespace frxx::proc::algs::res
