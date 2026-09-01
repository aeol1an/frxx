#include <frxx/proc/algs/ACF.hpp>

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
    const i64 nr = static_cast<i64>(X1.rows());
    const i64 nt = static_cast<i64>(X1.cols());
    frxx::utils::WorkerPool pool;
    pool.pfor(0, nr, [&](i64 i) {
        std::complex<double> acc{0.0, 0.0};
        if (lag == 0) {
            for (i64 j = 0; j < nt; ++j) {
                acc += static_cast<std::complex<double>>(X1(i, j)) *
                    std::conj(static_cast<std::complex<double>>(
                        X2(i, j)));
            }
        } else if (lag > 0) {
            for (i64 j = 0; j < nt - lag; ++j) {
                acc += static_cast<std::complex<double>>(X1(i, j + lag)) *
                    std::conj(static_cast<std::complex<double>>(
                        X2(i, j)));
            }
        } else {
            const i64 neg_lag = -lag;
            for (i64 j = 0; j < nt + lag; ++j) {
                acc += static_cast<std::complex<double>>(X1(i, j)) *
                    std::conj(static_cast<std::complex<double>>(
                        X2(i, j + neg_lag)));
            }
        }
        result(i) = acc / static_cast<double>(nt);
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
