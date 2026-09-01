#include <frxx/proc/algs/fuzzyDCA.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <frxx/utils/freqResolution.hpp>
#include <frxx/utils/workerPool.hpp>

namespace frxx::proc::algs::fuzzy_dca {

namespace {

using frxx::utils::i64;

enum class Side : i64 { Left = 0, Full = 1, Right = 2 };

struct MembershipDefinition {
    Side side;
    std::array<double, 4> thresholds;
};

constexpr MembershipDefinition membership_definitions[2][4] = {
    {
        {Side::Full, {-1.5, 1.0, 2.0, 4.0}},
        {Side::Right, {0.79, 0.98, -9999.0, -9999.0}},
        {Side::Left, {0.6, 5.0, -9999.0, -9999.0}},
        {Side::Left, {0.00025, 0.027, -9999.0, -9999.0}},
    },
    {
        {Side::Full, {-19.0, -7.4, 1.7, 10.6}},
        {Side::Full, {0.0, 0.3, 0.94, 0.99}},
        {Side::Right, {0.4, 7.1, -9999.0, -9999.0}},
        {Side::Right, {0.0001, 0.027, -9999.0, -9999.0}},
    },
};

template <typename... Arrays>
void require_same_shape(const Arrays&... arrays) {
    const std::array<Eigen::Index, sizeof...(Arrays)> rows{arrays.rows()...};
    const std::array<Eigen::Index, sizeof...(Arrays)> cols{arrays.cols()...};
    if (!std::all_of(rows.begin(), rows.end(), [&](Eigen::Index value) {
            return value == rows.front();
        }) ||
        !std::all_of(cols.begin(), cols.end(), [&](Eigen::Index value) {
            return value == cols.front();
        })) {
        throw std::invalid_argument("all input arrays must have the same shape");
    }
}

template <typename T>
class RollingNanVariance {
public:
    void add(T value) {
        if (std::isnan(value)) {
            return;
        }
        ++count_;
        const T delta = value - mean_;
        mean_ += delta / static_cast<T>(count_);
        const T adjusted_delta = value - mean_;
        m2_ += delta * adjusted_delta;
    }

    void remove(T value) {
        if (std::isnan(value)) {
            return;
        }
        if (count_ == 1) {
            count_ = 0;
            mean_ = T{0};
            m2_ = T{0};
            return;
        }
        const T old_mean = mean_;
        --count_;
        mean_ = (old_mean * static_cast<T>(count_ + 1) - value) /
            static_cast<T>(count_);
        m2_ -= (value - old_mean) * (value - mean_);
        if (m2_ < T{0}) {
            m2_ = T{0};
        }
    }

    T variance() const {
        return count_ == 0
            ? std::numeric_limits<T>::quiet_NaN()
            : m2_ / static_cast<T>(count_);
    }

private:
    i64 count_ = 0;
    T mean_ = T{0};
    T m2_ = T{0};
};

template <typename T>
frxx::eigen::Array2D<T> calc_variance_impl(
    frxx::eigen::ConstArray2DRef<T> field,
    i64 points,
    frxx::utils::WorkerPool& pool
) {
    frxx::eigen::Array2D<T> output(field.rows(), field.cols());
    if (points <= 0 || field.cols() == 0) {
        output.setConstant(std::numeric_limits<T>::quiet_NaN());
        return output;
    }
    const i64 width = static_cast<i64>(field.cols());
    const i64 left_width = points / 2;
    const i64 right_width = (points % 2 == 0) ? left_width : left_width + 1;
    pool.pfor(0, field.rows(), [&](i64 row) {
        RollingNanVariance<T> variance;
        i64 low = 0;
        i64 high = std::min(width, right_width);
        for (i64 index = low; index < high; ++index) {
            variance.add(field(row, index));
        }
        output(row, 0) = variance.variance();

        for (i64 column = 1; column < width; ++column) {
            const i64 next_low = std::max<i64>(0, column - left_width);
            const i64 next_high = std::min(width, column + right_width);
            while (low < next_low) {
                variance.remove(field(row, low));
                ++low;
            }
            while (high < next_high) {
                variance.add(field(row, high));
                ++high;
            }
            output(row, column) = variance.variance();
        }
    });
    return output;
}

template <typename Output, typename Input, typename T>
Output membership_fn_line_impl(const Input& values, T x1, T x2, i64 sign) {
    Output output(values.rows(), values.cols());
    const T slope = static_cast<T>(sign) * (T{1} / (x2 - x1));
    const T intercept = sign > 0 ? T{0} : T{1};
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            output(row, column) =
                slope * (values(row, column) - x1) + intercept;
        }
    }
    return output;
}

template <typename T>
frxx::eigen::Array2D<T> membership_impl(
    frxx::eigen::ConstArray2DRef<T> values, i64 scatterer_class, i64 field
) {
    if (scatterer_class < 0 || scatterer_class >= 2 || field < 0 || field >= 4) {
        throw std::out_of_range("membership class or field index is out of range");
    }
    const auto& definition = membership_definitions[scatterer_class][field];
    const T x1 = static_cast<T>(definition.thresholds[0]);
    const T x2 = static_cast<T>(definition.thresholds[1]);
    const T x3 = static_cast<T>(definition.thresholds[2]);
    const T x4 = static_cast<T>(definition.thresholds[3]);
    frxx::eigen::Array2D<T> output(values.rows(), values.cols());
    output.setConstant(std::numeric_limits<T>::quiet_NaN());

    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            const T value = values(row, column);
            if (std::isnan(value)) {
                continue;
            }
            if (definition.side == Side::Full) {
                if (value < x1 || value >= x4) {
                    output(row, column) = T{0};
                } else if (value < x2) {
                    output(row, column) = (value - x1) / (x2 - x1);
                } else if (value < x3) {
                    output(row, column) = T{1};
                } else {
                    output(row, column) = T{1} - (value - x3) / (x4 - x3);
                }
            } else if (definition.side == Side::Left) {
                if (value < x1) {
                    output(row, column) = T{1};
                } else if (value < x2) {
                    output(row, column) = T{1} - (value - x1) / (x2 - x1);
                } else {
                    output(row, column) = T{0};
                }
            } else {
                if (value < x1) {
                    output(row, column) = T{0};
                } else if (value < x2) {
                    output(row, column) = (value - x1) / (x2 - x1);
                } else {
                    output(row, column) = T{1};
                }
            }
        }
    }
    return output;
}

template <typename T>
T clip_unit(T value) {
    if (std::isnan(value)) {
        return value;
    }
    return std::max(T{0}, std::min(T{1}, value));
}

template <typename T>
AggregationResult<T> calc_aggregation_impl(
    frxx::eigen::ConstArray2DRef<T> zdr,
    frxx::eigen::ConstArray2DRef<T> rhohv,
    frxx::eigen::ConstArray2DRef<T> zdr_variance,
    frxx::eigen::ConstArray2DRef<T> rhohv_variance,
    frxx::eigen::ConstArray2DRef<T> psd,
    T filter_strength
) {
    require_same_shape(zdr, rhohv, zdr_variance, rhohv_variance, psd);
    const auto rain_zdr = membership_impl<T>(zdr, 0, 0);
    const auto rain_rhohv = membership_impl<T>(rhohv, 0, 1);
    const auto rain_zdrv = membership_impl<T>(zdr_variance, 0, 2);
    const auto rain_rhohvv = membership_impl<T>(rhohv_variance, 0, 3);
    const auto debris_zdr = membership_impl<T>(zdr, 1, 0);
    const auto debris_rhohv = membership_impl<T>(rhohv, 1, 1);
    const auto debris_zdrv = membership_impl<T>(zdr_variance, 1, 2);
    const auto debris_rhohvv = membership_impl<T>(rhohv_variance, 1, 3);

    AggregationResult<T> result{
        frxx::eigen::Array2D<T>(zdr.rows(), zdr.cols()),
        frxx::eigen::Array2D<T>(zdr.rows(), zdr.cols()),
        frxx::eigen::Array2D<T>(zdr.rows(), zdr.cols()),
    };
    for (Eigen::Index row = 0; row < zdr.rows(); ++row) {
        for (Eigen::Index column = 0; column < zdr.cols(); ++column) {
            const T rain = clip_unit<T>(
                rain_zdr(row, column) * T{0.25} +
                rain_rhohv(row, column) * T{0.25} +
                rain_zdrv(row, column) * T{0.25} +
                rain_rhohvv(row, column) * T{0.25});
            const T debris = clip_unit<T>(
                debris_zdr(row, column) * T{0.10} +
                debris_rhohv(row, column) * T{0.25} +
                debris_zdrv(row, column) * T{0.40} +
                debris_rhohvv(row, column) * T{0.25});
            result.rain(row, column) = rain;
            result.normalized_rain(row, column) = rain / (rain + debris);
            result.filtered_psd(row, column) = T{10} * std::log10(
                std::pow(T{10}, psd(row, column) / T{10}) *
                std::pow(rain, filter_strength));
        }
    }
    return result;
}

template <typename T>
SpectralRayResult<T> process_ray_s_impl(
    frxx::eigen::ConstArray2DRef<T> psd,
    frxx::eigen::ConstArray2DRef<T> zdr,
    frxx::eigen::ConstArray2DRef<T> rhohv,
    i64 points,
    T filter_strength,
    frxx::utils::WorkerPool& pool
) {
    require_same_shape(psd, zdr, rhohv);
    auto zdr_variance = calc_variance_impl<T>(zdr, points, pool);
    auto rhohv_variance = calc_variance_impl<T>(rhohv, points, pool);
    auto aggregation = calc_aggregation_impl<T>(
        zdr, rhohv, zdr_variance, rhohv_variance, psd, filter_strength);
    return {
        std::move(zdr_variance),
        std::move(rhohv_variance),
        std::move(aggregation.rain),
        std::move(aggregation.normalized_rain),
        std::move(aggregation.filtered_psd),
    };
}

template <typename T>
frxx::eigen::Array2D<double> db_to_linear_impl(
    frxx::eigen::ConstArray2DRef<T> values,
    frxx::utils::WorkerPool& pool
) {
    frxx::eigen::Array2D<double> output(values.rows(), values.cols());
    pool.pfor(0, values.rows(), [&](i64 row) {
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            output(row, column) = std::pow(
                10.0, static_cast<double>(values(row, column)) / 10.0);
        }
    });
    return output;
}

template <typename T>
T wrap_nyquist(T value, T nyquist_velocity) {
    const T span = T{2} * nyquist_velocity;
    const T shifted = value + nyquist_velocity;
    return shifted - std::floor(shifted / span) * span - nyquist_velocity;
}

template <typename T>
MomentRayResult<T> process_ray_m_impl(
    frxx::eigen::ConstArray2DRef<T> filtered_psd_db,
    frxx::eigen::ConstArray2DRef<T> psd_db,
    frxx::eigen::ConstArray1DRef<T> acf_velocity,
    T nyquist_velocity,
    bool flip_velocity,
    frxx::utils::WorkerPool& pool
) {
    require_same_shape(filtered_psd_db, psd_db);
    if (acf_velocity.size() != filtered_psd_db.rows()) {
        throw std::invalid_argument(
            "vACF length must equal the number of spectrum rows");
    }
    const auto filtered_psd = db_to_linear_impl<T>(filtered_psd_db, pool);
    const auto psd = db_to_linear_impl<T>(psd_db, pool);
    const auto velocity_axis = frxx::utils::velocity_axis(
        static_cast<i64>(filtered_psd_db.cols()), nyquist_velocity,
        flip_velocity, 0, 0);
    MomentRayResult<T> result{
        frxx::eigen::Array1D<T>(filtered_psd_db.rows()),
        frxx::eigen::Array1D<T>(filtered_psd_db.rows()),
    };
    const Eigen::Index bins = filtered_psd_db.cols();

    pool.pfor(0, filtered_psd_db.rows(), [&](i64 row) {
        bool all_nan = true;
        double power = 0.0;
        Eigen::Index maximum_index = 0;
        double maximum = 0.0;
        bool found_maximum = false;
        for (Eigen::Index bin = 0; bin < bins; ++bin) {
            const double value = filtered_psd(row, bin);
            if (!std::isnan(value)) {
                all_nan = false;
                power += value;
                if (!found_maximum || value > maximum) {
                    maximum = value;
                    maximum_index = bin;
                    found_maximum = true;
                }
            }
        }
        if (all_nan || std::isnan(acf_velocity(row)) || power < 1e-10) {
            result.velocity(row) = std::numeric_limits<T>::quiet_NaN();
            result.correction(row) = T{0};
            return;
        }

        const T maximum_velocity = velocity_axis(maximum_index);
        double weighted_offset = 0.0;
        for (Eigen::Index bin = 0; bin < bins; ++bin) {
            const double value = filtered_psd(row, bin);
            if (!std::isnan(value)) {
                const T offset = wrap_nyquist<T>(
                    velocity_axis(bin) - maximum_velocity, nyquist_velocity);
                const double product = static_cast<double>(offset) * value;
                if (!std::isnan(product)) {
                    weighted_offset += product;
                }
            }
        }
        result.velocity(row) = wrap_nyquist<T>(
            maximum_velocity + static_cast<T>(weighted_offset / power),
            nyquist_velocity);
        if (std::isnan(result.velocity(row))) {
            result.correction(row) = T{0};
            return;
        }

        Eigen::Index acf_index = 0;
        Eigen::Index dca_index = 0;
        T acf_distance = std::abs(velocity_axis(0) - acf_velocity(row));
        T dca_distance = std::abs(velocity_axis(0) - result.velocity(row));
        for (Eigen::Index bin = 1; bin < bins; ++bin) {
            const T current_acf = std::abs(velocity_axis(bin) - acf_velocity(row));
            const T current_dca = std::abs(velocity_axis(bin) - result.velocity(row));
            if (current_acf < acf_distance) {
                acf_distance = current_acf;
                acf_index = bin;
            }
            if (current_dca < dca_distance) {
                dca_distance = current_dca;
                dca_index = bin;
            }
        }
        if (acf_index == dca_index) {
            result.correction(row) = result.velocity(row) - acf_velocity(row);
            return;
        }

        const Eigen::Index lower = std::min(acf_index, dca_index);
        const Eigen::Index higher = std::max(acf_index, dca_index);
        double between_min = std::numeric_limits<double>::infinity();
        double outside_min = std::numeric_limits<double>::infinity();
        for (Eigen::Index bin = lower; bin <= higher; ++bin) {
            if (!std::isnan(psd(row, bin))) {
                between_min = std::min(between_min, psd(row, bin));
            }
        }
        for (Eigen::Index bin = 0; bin <= lower; ++bin) {
            if (!std::isnan(psd(row, bin))) {
                outside_min = std::min(outside_min, psd(row, bin));
            }
        }
        for (Eigen::Index bin = higher; bin < bins; ++bin) {
            if (!std::isnan(psd(row, bin))) {
                outside_min = std::min(outside_min, psd(row, bin));
            }
        }
        if (outside_min <= between_min) {
            result.correction(row) = result.velocity(row) - acf_velocity(row);
            return;
        }

        i64 between_nan = 0;
        for (Eigen::Index bin = lower; bin <= higher; ++bin) {
            between_nan += std::isnan(filtered_psd(row, bin)) ? 1 : 0;
        }
        i64 outside_nan = 0;
        for (Eigen::Index bin = 0; bin <= lower; ++bin) {
            outside_nan += std::isnan(filtered_psd(row, bin)) ? 1 : 0;
        }
        for (Eigen::Index bin = higher; bin < bins; ++bin) {
            outside_nan += std::isnan(filtered_psd(row, bin)) ? 1 : 0;
        }
        const double between_fraction = static_cast<double>(between_nan) /
            static_cast<double>(higher - lower + 1);
        const double outside_fraction = static_cast<double>(outside_nan) /
            static_cast<double>((lower + 1) + (bins - higher));
        if (between_fraction >= outside_fraction) {
            if (acf_velocity(row) < result.velocity(row)) {
                result.correction(row) = result.velocity(row) -
                    (acf_velocity(row) + T{2} * nyquist_velocity);
            } else {
                result.correction(row) =
                    (result.velocity(row) + T{2} * nyquist_velocity) -
                    acf_velocity(row);
            }
        } else {
            result.correction(row) = result.velocity(row) - acf_velocity(row);
        }
        });
    return result;
}

}  // namespace

#define FRXX_FUZZY_DCA_OVERLOADS(T) \
frxx::eigen::Array2D<T> calc_variance( \
    frxx::eigen::ConstArray2DRef<T> field, i64 points) { \
    frxx::utils::WorkerPool pool; \
    return calc_variance_impl<T>(field, points, pool); \
} \
frxx::eigen::Array1D<T> membership_fn_line( \
    frxx::eigen::ConstArray1DRef<T> values, T x1, T x2, i64 sign) { \
    return membership_fn_line_impl<frxx::eigen::Array1D<T>>(values, x1, x2, sign); \
} \
frxx::eigen::Array2D<T> membership_fn_line( \
    frxx::eigen::ConstArray2DRef<T> values, T x1, T x2, i64 sign) { \
    return membership_fn_line_impl<frxx::eigen::Array2D<T>>(values, x1, x2, sign); \
} \
frxx::eigen::Array2D<T> membership( \
    frxx::eigen::ConstArray2DRef<T> values, i64 scatterer_class, i64 field) { \
    return membership_impl<T>(values, scatterer_class, field); \
} \
AggregationResult<T> calc_aggregation( \
    frxx::eigen::ConstArray2DRef<T> zdr, \
    frxx::eigen::ConstArray2DRef<T> rhohv, \
    frxx::eigen::ConstArray2DRef<T> zdr_variance, \
    frxx::eigen::ConstArray2DRef<T> rhohv_variance, \
    frxx::eigen::ConstArray2DRef<T> psd, T filter_strength) { \
    return calc_aggregation_impl<T>( \
        zdr, rhohv, zdr_variance, rhohv_variance, psd, filter_strength); \
} \
SpectralRayResult<T> process_ray_s( \
    frxx::eigen::ConstArray2DRef<T> psd, \
    frxx::eigen::ConstArray2DRef<T> zdr, \
    frxx::eigen::ConstArray2DRef<T> rhohv, i64 points, T filter_strength) { \
    frxx::utils::WorkerPool pool; \
    return process_ray_s_impl<T>( \
        psd, zdr, rhohv, points, filter_strength, pool); \
} \
frxx::eigen::Array2D<double> db_to_linear_2d( \
    frxx::eigen::ConstArray2DRef<T> values) { \
    frxx::utils::WorkerPool pool; \
    return db_to_linear_impl<T>(values, pool); \
} \
MomentRayResult<T> process_ray_m( \
    frxx::eigen::ConstArray2DRef<T> filtered_psd_db, \
    frxx::eigen::ConstArray2DRef<T> psd_db, \
    frxx::eigen::ConstArray1DRef<T> acf_velocity, \
    T nyquist_velocity, bool flip_velocity) { \
    frxx::utils::WorkerPool pool; \
    return process_ray_m_impl<T>( \
        filtered_psd_db, psd_db, acf_velocity, nyquist_velocity, \
        flip_velocity, pool); \
}

FRXX_FUZZY_DCA_OVERLOADS(float)
FRXX_FUZZY_DCA_OVERLOADS(double)

#undef FRXX_FUZZY_DCA_OVERLOADS

}  // namespace frxx::proc::algs::fuzzy_dca
