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

constexpr MembershipDefinition numbaMembershipThresholds[2][4] = {
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
    i64 pts,
    frxx::utils::WorkerPool& pool
) {
    const i64 nr = static_cast<i64>(field.rows());
    const i64 nv = static_cast<i64>(field.cols());
    frxx::eigen::Array2D<T> fieldResult(nr, nv);
    if (pts <= 0 || nv == 0) {
        fieldResult.setConstant(std::numeric_limits<T>::quiet_NaN());
        return fieldResult;
    }

    const i64 leftWidth = pts / 2;
    const i64 rightWidth = (pts % 2 == 0) ? leftWidth : leftWidth + 1;
    pool.pfor(0, nr, [&](i64 r) {
        RollingNanVariance<T> variance;
        i64 lowval = 0;
        i64 highval = std::min(nv, rightWidth);
        for (i64 idx = lowval; idx < highval; ++idx) {
            variance.add(field(r, idx));
        }
        fieldResult(r, 0) = variance.variance();

        for (i64 idx = 1; idx < nv; ++idx) {
            const i64 nextLowval = std::max<i64>(0, idx - leftWidth);
            const i64 nextHighval = std::min(nv, idx + rightWidth);
            while (lowval < nextLowval) {
                variance.remove(field(r, lowval));
                ++lowval;
            }
            while (highval < nextHighval) {
                variance.add(field(r, highval));
                ++highval;
            }
            fieldResult(r, idx) = variance.variance();
        }
    });
    return fieldResult;
}

template <typename Output, typename Input, typename T>
Output membership_fn_line_impl(const Input& x, T x1, T x2, i64 sign) {
    Output ret(x.rows(), x.cols());
    const T m = static_cast<T>(sign) * (T{1} / (x2 - x1));
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        for (Eigen::Index j = 0; j < x.cols(); ++j) {
            ret(i, j) = m * (x(i, j) - x1) + (sign > 0 ? T{0} : T{1});
        }
    }
    return ret;
}

template <typename T>
frxx::eigen::Array2D<T> membership_impl(
    frxx::eigen::ConstArray2DRef<T> x, i64 scattererClass, i64 field
) {
    if (scattererClass < 0 || scattererClass >= 2 || field < 0 || field >= 4) {
        throw std::out_of_range("membership class or field index is out of range");
    }
    const auto& definition = numbaMembershipThresholds[scattererClass][field];
    const T X1 = static_cast<T>(definition.thresholds[0]);
    const T X2 = static_cast<T>(definition.thresholds[1]);
    const T X3 = static_cast<T>(definition.thresholds[2]);
    const T X4 = static_cast<T>(definition.thresholds[3]);
    frxx::eigen::Array2D<T> ret(x.rows(), x.cols());
    ret.setConstant(std::numeric_limits<T>::quiet_NaN());

    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        for (Eigen::Index j = 0; j < x.cols(); ++j) {
            const T value = x(i, j);
            if (std::isnan(value)) {
                continue;
            }

            if (definition.side == Side::Full) {
                if (value < X1 || value >= X4) {
                    ret(i, j) = T{0};
                } else if (value < X2) {
                    ret(i, j) = (value - X1) / (X2 - X1);
                } else if (value < X3) {
                    ret(i, j) = T{1};
                } else {
                    ret(i, j) = T{1} - (value - X3) / (X4 - X3);
                }
            } else if (definition.side == Side::Left) {
                if (value < X1) {
                    ret(i, j) = T{1};
                } else if (value < X2) {
                    ret(i, j) = T{1} - (value - X1) / (X2 - X1);
                } else {
                    ret(i, j) = T{0};
                }
            } else {
                if (value < X1) {
                    ret(i, j) = T{0};
                } else if (value < X2) {
                    ret(i, j) = (value - X1) / (X2 - X1);
                } else {
                    ret(i, j) = T{1};
                }
            }
        }
    }
    return ret;
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
    frxx::eigen::ConstArray2DRef<T> sZDR,
    frxx::eigen::ConstArray2DRef<T> sRHOHV,
    frxx::eigen::ConstArray2DRef<T> sZDRv,
    frxx::eigen::ConstArray2DRef<T> sRHOHVv,
    frxx::eigen::ConstArray2DRef<T> PSDH,
    T filterStrength
) {
    require_same_shape(sZDR, sRHOHV, sZDRv, sRHOHVv, PSDH);
    const auto rainZDR = membership_impl<T>(sZDR, 0, 0);
    const auto rainRHOHV = membership_impl<T>(sRHOHV, 0, 1);
    const auto rainZDRv = membership_impl<T>(sZDRv, 0, 2);
    const auto rainRHOHVv = membership_impl<T>(sRHOHVv, 0, 3);
    const auto debrisZDR = membership_impl<T>(sZDR, 1, 0);
    const auto debrisRHOHV = membership_impl<T>(sRHOHV, 1, 1);
    const auto debrisZDRv = membership_impl<T>(sZDRv, 1, 2);
    const auto debrisRHOHVv = membership_impl<T>(sRHOHVv, 1, 3);

    AggregationResult<T> result{
        frxx::eigen::Array2D<T>(sZDR.rows(), sZDR.cols()),
        frxx::eigen::Array2D<T>(sZDR.rows(), sZDR.cols()),
        frxx::eigen::Array2D<T>(sZDR.rows(), sZDR.cols()),
    };
    auto& Arain = result.rain;
    auto& Anrain = result.normalized_rain;
    auto& PSDHF = result.filtered_psd;

    for (Eigen::Index i = 0; i < sZDR.rows(); ++i) {
        for (Eigen::Index j = 0; j < sZDR.cols(); ++j) {
            Arain(i, j) = clip_unit<T>(
                rainZDR(i, j) * T{0.25} +
                rainRHOHV(i, j) * T{0.25} +
                rainZDRv(i, j) * T{0.25} +
                rainRHOHVv(i, j) * T{0.25});

            const T Adebris = clip_unit<T>(
                debrisZDR(i, j) * T{0.10} +
                debrisRHOHV(i, j) * T{0.25} +
                debrisZDRv(i, j) * T{0.40} +
                debrisRHOHVv(i, j) * T{0.25});

            Anrain(i, j) = Arain(i, j) / (Arain(i, j) + Adebris);
            PSDHF(i, j) = T{10} * std::log10(
                std::pow(T{10}, PSDH(i, j) / T{10}) *
                std::pow(Arain(i, j), filterStrength));
        }
    }
    return result;
}

template <typename T>
SpectralRayResult<T> process_ray_s_impl(
    frxx::eigen::ConstArray2DRef<T> PSDH,
    frxx::eigen::ConstArray2DRef<T> sZDR,
    frxx::eigen::ConstArray2DRef<T> sRHOHV,
    i64 pts,
    T filterStrength,
    frxx::utils::WorkerPool& pool
) {
    require_same_shape(PSDH, sZDR, sRHOHV);
    auto sZDRv = calc_variance_impl<T>(sZDR, pts, pool);
    auto sRHOHVv = calc_variance_impl<T>(sRHOHV, pts, pool);
    auto aggregation = calc_aggregation_impl<T>(
        sZDR, sRHOHV, sZDRv, sRHOHVv, PSDH, filterStrength);
    return {
        std::move(sZDRv),
        std::move(sRHOHVv),
        std::move(aggregation.rain),
        std::move(aggregation.normalized_rain),
        std::move(aggregation.filtered_psd),
    };
}

template <typename T>
frxx::eigen::Array2D<double> db_to_linear_impl(
    frxx::eigen::ConstArray2DRef<T> arr,
    frxx::utils::WorkerPool& pool
) {
    frxx::eigen::Array2D<double> out(arr.rows(), arr.cols());
    pool.pfor(0, arr.rows(), [&](i64 i) {
        for (Eigen::Index j = 0; j < arr.cols(); ++j) {
            out(i, j) = std::pow(
                10.0, static_cast<double>(arr(i, j)) / 10.0);
        }
    });
    return out;
}

template <typename T>
T wrap_nyquist(T value, T va) {
    const T Vn = T{2} * va;
    const T shifted = value + va;
    return shifted - std::floor(shifted / Vn) * Vn - va;
}

template <typename T>
MomentRayResult<T> process_ray_m_impl(
    frxx::eigen::ConstArray2DRef<T> PSDHFdb,
    frxx::eigen::ConstArray2DRef<T> PSDHdb,
    frxx::eigen::ConstArray1DRef<T> vACF,
    T va,
    bool flipVel,
    frxx::utils::WorkerPool& pool
) {
    require_same_shape(PSDHFdb, PSDHdb);
    if (vACF.size() != PSDHFdb.rows()) {
        throw std::invalid_argument(
            "vACF length must equal the number of spectrum rows");
    }

    const i64 nr = static_cast<i64>(PSDHFdb.rows());
    const i64 nv = static_cast<i64>(PSDHFdb.cols());

    // Work in linear power and float64 when estimating the DCA velocity.
    // Keep PSDH in dB: the original spectrum is only used to compare valleys,
    // and differences between valley depths are most meaningful in dB.
    const auto PSDHF = db_to_linear_impl<T>(PSDHFdb, pool);

    const auto vAxis = frxx::utils::velocity_axis(nv, va, flipVel, 0, 0);

    MomentRayResult<T> result{
        frxx::eigen::Array1D<T>(nr),
        frxx::eigen::Array1D<T>(nr),
    };
    auto& vDCA = result.velocity;
    auto& correction = result.correction;

    pool.pfor(0, nr, [&](i64 r) {
        // Find vDCA. Using the ordinary linear centroid directly would fail
        // when a peak straddles -va/+va. We therefore anchor offsets at the
        // strongest bin, wrap those offsets into one Nyquist interval, and
        // compute the power-weighted centroid in that local coordinate system.
        bool allNaN = true;
        double P = 0.0;
        i64 km = 0;
        double maxPower = 0.0;
        bool foundMaximum = false;
        for (i64 i = 0; i < nv; ++i) {
            const double v = PSDHF(r, i);
            if (!std::isnan(v)) {
                allNaN = false;
                P += v;
                if (!foundMaximum || v > maxPower) {
                    maxPower = v;
                    km = i;
                    foundMaximum = true;
                }
            }
        }
        if (allNaN || !std::isfinite(vACF(r)) ||
            !std::isfinite(va) || va <= T{0}) {
            vDCA(r) = std::numeric_limits<T>::quiet_NaN();
            correction(r) = T{0};
            return;
        }
        if (!std::isfinite(P) || P < 1e-10) {
            vDCA(r) = std::numeric_limits<T>::quiet_NaN();
            correction(r) = T{0};
            return;
        }
        const T vMax = vAxis(km);
        const T Vn = T{2} * va;
        double weightedDelV = 0.0;
        for (i64 i = 0; i < nv; ++i) {
            const double v = PSDHF(r, i);
            if (!std::isnan(v)) {
                const T delV = wrap_nyquist<T>(vAxis(i) - vMax, va);
                const double product = static_cast<double>(delV) * v;
                if (!std::isnan(product)) {
                    weightedDelV += product;
                }
            }
        }
        vDCA(r) = wrap_nyquist<T>(
            vMax + static_cast<T>(weightedDelV / P),
            va);

        if (!std::isfinite(vDCA(r))) {
            correction(r) = T{0};
            return;
        }

        // The likely correction is the shortest signed distance around the
        // Nyquist circle. The unlikely correction takes the complementary path
        // and is considered only when the original PSDH rejects the short path.
        const T rawCorrection = vDCA(r) - vACF(r);
        const T likelyCorrection = wrap_nyquist<T>(rawCorrection, va);
        const T unlikelyCorrection = likelyCorrection > T{0}
            ? likelyCorrection - Vn
            : likelyCorrection < T{0}
                ? likelyCorrection + Vn
                : (rawCorrection < T{0} ? -Vn : Vn);

        // Map the ACF and DCA velocities onto the spectrum so PSDH can be
        // inspected along the likely and unlikely circular connections.
        i64 iACF = 0;
        i64 iDCA = 0;
        T acfDistance = std::abs(vAxis(0) - vACF(r));
        T dcaDistance = std::abs(vAxis(0) - vDCA(r));
        for (i64 i = 1; i < nv; ++i) {
            const T currentACF = std::abs(vAxis(i) - vACF(r));
            const T currentDCA = std::abs(vAxis(i) - vDCA(r));
            if (currentACF < acfDistance) {
                acfDistance = currentACF;
                iACF = i;
            }
            if (currentDCA < dcaDistance) {
                dcaDistance = currentDCA;
                iDCA = i;
            }
        }

        // There is no resolvable path between estimates in the same bin.
        if (iACF == iDCA) {
            correction(r) = likelyCorrection;
            return;
        }

        // Follow the sign of the circular correction through vAxis. This
        // identifies the likely arc correctly even when it crosses index 0 or
        // when flipVel reverses the velocity axis.
        i64 likelyStep;
        if (likelyCorrection == T{0}) {
            const i64 forwardSteps = (iDCA - iACF + nv) % nv;
            const i64 backwardSteps = (iACF - iDCA + nv) % nv;
            likelyStep = forwardSteps <= backwardSteps ? 1 : -1;
        } else {
            const bool axisAscending = nv == 1 || vAxis(1) > vAxis(0);
            likelyStep = (likelyCorrection > T{0}) == axisAscending ? 1 : -1;
        }

        // The endpoint bins belong to both hypotheses and provide no evidence
        // for choosing one, so only the interiors of the two arcs are scored.
        double likelyMin = std::numeric_limits<double>::infinity();
        i64 likelyNaN = 0;
        i64 likelyTotal = 0;
        for (i64 i = (iACF + likelyStep + nv) % nv;
             i != iDCA;
             i = (i + likelyStep + nv) % nv) {
            ++likelyTotal;
            const double v = static_cast<double>(PSDHdb(r, i));
            if (std::isnan(v)) {
                ++likelyNaN;
            } else if (v < likelyMin) {
                likelyMin = v;
            }
        }

        const i64 unlikelyStep = -likelyStep;
        double unlikelyMin = std::numeric_limits<double>::infinity();
        i64 unlikelyNaN = 0;
        i64 unlikelyTotal = 0;
        for (i64 i = (iACF + unlikelyStep + nv) % nv;
             i != iDCA;
             i = (i + unlikelyStep + nv) % nv) {
            ++unlikelyTotal;
            const double v = static_cast<double>(PSDHdb(r, i));
            if (std::isnan(v)) {
                ++unlikelyNaN;
            } else if (v < unlikelyMin) {
                unlikelyMin = v;
            }
        }

        // Keep the short circular correction unless both PSDH tests reject its
        // path: its valley must be lower and its missing-data fraction higher.
        // An all-NaN likely path is a strong valley only when the other path has
        // finite data; an empty path provides no evidence either way.
        const bool likelyHasFinite = likelyMin !=
            std::numeric_limits<double>::infinity();
        const bool unlikelyHasFinite = unlikelyMin !=
            std::numeric_limits<double>::infinity();
        const bool likelyPathHasLowerValley = likelyTotal > 0 &&
            unlikelyHasFinite &&
            (!likelyHasFinite || likelyMin < unlikelyMin);
        const double likelyNaNFrac = likelyTotal == 0
            ? 0.0
            : static_cast<double>(likelyNaN) / static_cast<double>(likelyTotal);
        const double unlikelyNaNFrac = unlikelyTotal == 0
            ? 0.0
            : static_cast<double>(unlikelyNaN) / static_cast<double>(unlikelyTotal);
        const bool likelyPathHasMoreNaNs = likelyTotal > 0 &&
            unlikelyTotal > 0 && likelyNaNFrac > unlikelyNaNFrac;

        correction(r) = likelyPathHasLowerValley && likelyPathHasMoreNaNs
            ? unlikelyCorrection
            : likelyCorrection;
    });
    return result;
}

}  // namespace

#define FRXX_FUZZY_DCA_OVERLOADS(T) \
frxx::eigen::Array2D<T> calc_variance( \
    frxx::eigen::ConstArray2DRef<T> field, i64 pts) { \
    frxx::utils::WorkerPool pool; \
    return calc_variance_impl<T>(field, pts, pool); \
} \
frxx::eigen::Array1D<T> membership_fn_line( \
    frxx::eigen::ConstArray1DRef<T> x, T x1, T x2, i64 sign) { \
    return membership_fn_line_impl<frxx::eigen::Array1D<T>>(x, x1, x2, sign); \
} \
frxx::eigen::Array2D<T> membership_fn_line( \
    frxx::eigen::ConstArray2DRef<T> x, T x1, T x2, i64 sign) { \
    return membership_fn_line_impl<frxx::eigen::Array2D<T>>(x, x1, x2, sign); \
} \
frxx::eigen::Array2D<T> membership( \
    frxx::eigen::ConstArray2DRef<T> x, i64 scattererClass, i64 field) { \
    return membership_impl<T>(x, scattererClass, field); \
} \
AggregationResult<T> calc_aggregation( \
    frxx::eigen::ConstArray2DRef<T> sZDR, \
    frxx::eigen::ConstArray2DRef<T> sRHOHV, \
    frxx::eigen::ConstArray2DRef<T> sZDRv, \
    frxx::eigen::ConstArray2DRef<T> sRHOHVv, \
    frxx::eigen::ConstArray2DRef<T> PSDH, T filterStrength) { \
    return calc_aggregation_impl<T>( \
        sZDR, sRHOHV, sZDRv, sRHOHVv, PSDH, filterStrength); \
} \
SpectralRayResult<T> process_ray_s( \
    frxx::eigen::ConstArray2DRef<T> PSDH, \
    frxx::eigen::ConstArray2DRef<T> sZDR, \
    frxx::eigen::ConstArray2DRef<T> sRHOHV, i64 pts, T filterStrength) { \
    frxx::utils::WorkerPool pool; \
    return process_ray_s_impl<T>( \
        PSDH, sZDR, sRHOHV, pts, filterStrength, pool); \
} \
frxx::eigen::Array2D<double> db_to_linear_2d( \
    frxx::eigen::ConstArray2DRef<T> arr) { \
    frxx::utils::WorkerPool pool; \
    return db_to_linear_impl<T>(arr, pool); \
} \
MomentRayResult<T> process_ray_m( \
    frxx::eigen::ConstArray2DRef<T> PSDHFdb, \
    frxx::eigen::ConstArray2DRef<T> PSDHdb, \
    frxx::eigen::ConstArray1DRef<T> vACF, \
    T va, bool flipVel) { \
    frxx::utils::WorkerPool pool; \
    return process_ray_m_impl<T>( \
        PSDHFdb, PSDHdb, vACF, va, flipVel, pool); \
}

FRXX_FUZZY_DCA_OVERLOADS(float)
FRXX_FUZZY_DCA_OVERLOADS(double)

#undef FRXX_FUZZY_DCA_OVERLOADS

}  // namespace frxx::proc::algs::fuzzy_dca
