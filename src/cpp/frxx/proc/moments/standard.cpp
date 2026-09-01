#include <frxx/proc/moments/standard.hpp>

#include <complex>
#include <stdexcept>
#include <string>

#include <frxx/proc/algs/ACF.hpp>
#include <frxx/utils/workerPool.hpp>

namespace frxx::proc::moments::standard {

namespace {

using frxx::utils::i64;
using Complex128Array1D = frxx::eigen::Array1D<std::complex<double>>;
using Complex128RowMap = Eigen::Map<
    Complex128Array1D, 0, frxx::eigen::DynamicInnerStride>;

void require_output_shape(
    const Complex128Array2DRef& output,
    Eigen::Index time_count,
    Eigen::Index range_count,
    const char* name
) {
    if (output.rows() != time_count || output.cols() != range_count) {
        throw std::invalid_argument(
            std::string(name) + " has an incorrect output shape");
    }
}

}  // namespace

void process_rays(
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqh,
    frxx::eigen::ConstArray2DRef<std::complex<float>> iqv,
    frxx::eigen::ConstArray2DRef<i64> pulseBoundaries,
    frxx::eigen::ConstArray1DRef<std::int32_t> lags,
    std::vector<Complex128Array2DRef>& RH,
    Complex128Array2DRef RV,
    Complex128Array2DRef RX
) {
    if (iqh.rows() != iqv.rows() || iqh.cols() != iqv.cols()) {
        throw std::invalid_argument("Two array shapes not equal.");
    }
    if (pulseBoundaries.cols() < 2) {
        throw std::invalid_argument("pulseBoundaries must have at least two columns");
    }
    if (RH.size() != static_cast<std::size_t>(lags.size())) {
        throw std::invalid_argument("RH output count must equal lag count");
    }
    const Eigen::Index time_count = pulseBoundaries.rows();
    const Eigen::Index range_count = iqh.rows();
    require_output_shape(RV, time_count, range_count, "RV");
    require_output_shape(RX, time_count, range_count, "RX");
    for (const auto& output : RH) {
        require_output_shape(output, time_count, range_count, "RH");
    }

    frxx::utils::WorkerPool pool;
    pool.pfor(0, time_count, [&](i64 time_group) {
        const i64 first_pulse = pulseBoundaries(time_group, 0);
        const i64 last_pulse = pulseBoundaries(time_group, 1);
        if (first_pulse < 0 || last_pulse < first_pulse ||
            last_pulse > iqh.cols()) {
            throw std::out_of_range("pulse boundary is outside the IQ matrix");
        }
        const i64 pulse_count = last_pulse - first_pulse;
        auto iqhs = iqh.middleCols(first_pulse, pulse_count);
        auto iqvs = iqv.middleCols(first_pulse, pulse_count);

        Complex128RowMap RV_view(
            RV.data() + time_group * RV.outerStride(),
            range_count,
            frxx::eigen::DynamicInnerStride(RV.innerStride()));
        Complex128RowMap RX_view(
            RX.data() + time_group * RX.outerStride(),
            range_count,
            frxx::eigen::DynamicInnerStride(RX.innerStride()));
        frxx::proc::algs::acf::compute_ray_m(
            iqvs, iqvs, RV_view, 0);
        frxx::proc::algs::acf::compute_ray_m(
            iqhs, iqvs, RX_view, 0);

        for (i64 lag_index = 0; lag_index < lags.size(); ++lag_index) {
            auto& RH_lag = RH[lag_index];
            Complex128RowMap RH_view(
                RH_lag.data() + time_group * RH_lag.outerStride(),
                range_count,
                frxx::eigen::DynamicInnerStride(RH_lag.innerStride()));
            frxx::proc::algs::acf::compute_ray_m(
                iqhs, iqhs, RH_view,
                static_cast<i64>(lags(lag_index)));
        }
    });
}

}  // namespace frxx::proc::moments::standard
