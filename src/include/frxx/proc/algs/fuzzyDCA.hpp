#pragma once

#include <frxx/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace frxx::proc::algs::fuzzy_dca {

template <typename T>
struct AggregationResult {
    frxx::eigen::Array2D<T> rain;
    frxx::eigen::Array2D<T> normalized_rain;
    frxx::eigen::Array2D<T> filtered_psd;
};

template <typename T>
struct SpectralRayResult {
    frxx::eigen::Array2D<T> zdr_variance;
    frxx::eigen::Array2D<T> rhohv_variance;
    frxx::eigen::Array2D<T> rain;
    frxx::eigen::Array2D<T> normalized_rain;
    frxx::eigen::Array2D<T> filtered_psd;
};

template <typename T>
struct MomentRayResult {
    frxx::eigen::Array1D<T> velocity;
    frxx::eigen::Array1D<T> correction;
};

/// Calculate a moving, NaN-ignoring population variance along each matrix row.
frxx::eigen::Array2D<float> calc_variance(
    frxx::eigen::ConstArray2DRef<float> field,
    frxx::utils::i64 points = 9);
frxx::eigen::Array2D<double> calc_variance(
    frxx::eigen::ConstArray2DRef<double> field,
    frxx::utils::i64 points = 9);

/// Evaluate the linear portion of a fuzzy membership function.
frxx::eigen::Array1D<float> membership_fn_line(
    frxx::eigen::ConstArray1DRef<float> values,
    float x1, float x2, frxx::utils::i64 sign);
frxx::eigen::Array1D<double> membership_fn_line(
    frxx::eigen::ConstArray1DRef<double> values,
    double x1, double x2, frxx::utils::i64 sign);
frxx::eigen::Array2D<float> membership_fn_line(
    frxx::eigen::ConstArray2DRef<float> values,
    float x1, float x2, frxx::utils::i64 sign);
frxx::eigen::Array2D<double> membership_fn_line(
    frxx::eigen::ConstArray2DRef<double> values,
    double x1, double x2, frxx::utils::i64 sign);

/// Calculate fuzzy membership for a scatterer class and spectral field.
frxx::eigen::Array2D<float> membership(
    frxx::eigen::ConstArray2DRef<float> values,
    frxx::utils::i64 scatterer_class,
    frxx::utils::i64 field);
frxx::eigen::Array2D<double> membership(
    frxx::eigen::ConstArray2DRef<double> values,
    frxx::utils::i64 scatterer_class,
    frxx::utils::i64 field);

/// Combine spectral memberships and apply the rain filter to a PSD.
AggregationResult<float> calc_aggregation(
    frxx::eigen::ConstArray2DRef<float> zdr,
    frxx::eigen::ConstArray2DRef<float> rhohv,
    frxx::eigen::ConstArray2DRef<float> zdr_variance,
    frxx::eigen::ConstArray2DRef<float> rhohv_variance,
    frxx::eigen::ConstArray2DRef<float> psd,
    float filter_strength = 8.0F);
AggregationResult<double> calc_aggregation(
    frxx::eigen::ConstArray2DRef<double> zdr,
    frxx::eigen::ConstArray2DRef<double> rhohv,
    frxx::eigen::ConstArray2DRef<double> zdr_variance,
    frxx::eigen::ConstArray2DRef<double> rhohv_variance,
    frxx::eigen::ConstArray2DRef<double> psd,
    double filter_strength = 8.0);

/// Calculate the five fuzzy-DCA spectral products for one ray.
SpectralRayResult<float> process_ray_s(
    frxx::eigen::ConstArray2DRef<float> psd,
    frxx::eigen::ConstArray2DRef<float> zdr,
    frxx::eigen::ConstArray2DRef<float> rhohv,
    frxx::utils::i64 points = 9,
    float filter_strength = 8.0F);
SpectralRayResult<double> process_ray_s(
    frxx::eigen::ConstArray2DRef<double> psd,
    frxx::eigen::ConstArray2DRef<double> zdr,
    frxx::eigen::ConstArray2DRef<double> rhohv,
    frxx::utils::i64 points = 9,
    double filter_strength = 8.0);

/// Convert a dB matrix to linear power in float64.
frxx::eigen::Array2D<double> db_to_linear_2d(
    frxx::eigen::ConstArray2DRef<float> values);
frxx::eigen::Array2D<double> db_to_linear_2d(
    frxx::eigen::ConstArray2DRef<double> values);

/// Calculate DCA velocity and the correction relative to the ACF velocity.
MomentRayResult<float> process_ray_m(
    frxx::eigen::ConstArray2DRef<float> filtered_psd_db,
    frxx::eigen::ConstArray2DRef<float> psd_db,
    frxx::eigen::ConstArray1DRef<float> acf_velocity,
    float nyquist_velocity,
    bool flip_velocity);
MomentRayResult<double> process_ray_m(
    frxx::eigen::ConstArray2DRef<double> filtered_psd_db,
    frxx::eigen::ConstArray2DRef<double> psd_db,
    frxx::eigen::ConstArray1DRef<double> acf_velocity,
    double nyquist_velocity,
    bool flip_velocity);

}  // namespace frxx::proc::algs::fuzzy_dca
