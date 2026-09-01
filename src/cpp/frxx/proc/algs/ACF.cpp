#include <frxx/proc/algs/ACF.hpp>

#include <algorithm>
#include <cstdlib>
#include <complex>
#include <stdexcept>

#include <frxx/utils/workerPool.hpp>

namespace frxx::proc::algs::acf {

namespace {

using frxx::utils::i64;

void require_shapes(
    frxx::eigen::ConstArray2DRef<std::complex<float>> X1,
    frxx::eigen::ConstArray2DRef<std::complex<float>> X2,
    frxx::eigen::Array1DRef<std::complex<double>> result
) {
    if (X1.rows() != X2.rows() || X1.cols() != X2.cols()) {
        throw std::invalid_argument("Two array shapes not equal.");
    }
    if (result.size() != X1.rows()) {
        throw std::invalid_argument(
            "result length must equal the number of input ranges");
    }
    if (X1.cols() == 0) {
        throw std::invalid_argument("IQ matrices must contain at least one time sample");
    }
}

}  // namespace

void compute_ray_m(
    frxx::eigen::ConstArray2DRef<std::complex<float>> X1,
    frxx::eigen::ConstArray2DRef<std::complex<float>> X2,
    frxx::eigen::Array1DRef<std::complex<double>> result,
    i64 lag
) {
    require_shapes(X1, X2, result);
    const i64 time_count = static_cast<i64>(X1.cols());
    const i64 sample_count = std::max<i64>(0, time_count - std::abs(lag));
    frxx::utils::WorkerPool pool;
    pool.pfor(0, X1.rows(), [&](i64 range) {
        std::complex<double> accumulator{0.0, 0.0};
        if (lag >= 0) {
            for (i64 time = 0; time < sample_count; ++time) {
                accumulator += static_cast<std::complex<double>>(
                    X1(range, time + lag)) *
                    std::conj(static_cast<std::complex<double>>(
                        X2(range, time)));
            }
        } else {
            const i64 negative_lag = -lag;
            for (i64 time = 0; time < sample_count; ++time) {
                accumulator += static_cast<std::complex<double>>(
                    X1(range, time)) *
                    std::conj(static_cast<std::complex<double>>(
                        X2(range, time + negative_lag)));
            }
        }
        result(range) = accumulator / static_cast<double>(time_count);
    });
}

frxx::eigen::Array1D<std::complex<double>> compute_ray_m(
    frxx::eigen::ConstArray2DRef<std::complex<float>> X1,
    frxx::eigen::ConstArray2DRef<std::complex<float>> X2,
    i64 lag
) {
    frxx::eigen::Array1D<std::complex<double>> result(X1.rows());
    compute_ray_m(X1, X2, result, lag);
    return result;
}

}  // namespace frxx::proc::algs::acf
