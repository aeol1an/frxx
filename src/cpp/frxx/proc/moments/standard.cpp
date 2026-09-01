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
    Eigen::Index nBigTime,
    Eigen::Index nRange,
    const char* name
) {
    if (output.rows() != nBigTime || output.cols() != nRange) {
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
    const Eigen::Index nBigTime = pulseBoundaries.rows();
    const Eigen::Index nRange = iqh.rows();
    require_output_shape(RV, nBigTime, nRange, "RV");
    require_output_shape(RX, nBigTime, nRange, "RX");
    for (const auto& output : RH) {
        require_output_shape(output, nBigTime, nRange, "RH");
    }

    frxx::utils::WorkerPool pool;
    pool.pfor(0, nBigTime, [&](i64 t) {
        const i64 firstPulse = pulseBoundaries(t, 0);
        const i64 lastPulse = pulseBoundaries(t, 1);
        if (firstPulse < 0 || lastPulse < firstPulse ||
            lastPulse > iqh.cols()) {
            throw std::out_of_range("pulse boundary is outside the IQ matrix");
        }
        const i64 nPulses = lastPulse - firstPulse;
        auto iqhs = iqh.middleCols(firstPulse, nPulses);
        auto iqvs = iqv.middleCols(firstPulse, nPulses);

        Complex128RowMap RV_view(
            RV.data() + t * RV.outerStride(),
            nRange,
            frxx::eigen::DynamicInnerStride(RV.innerStride()));
        Complex128RowMap RX_view(
            RX.data() + t * RX.outerStride(),
            nRange,
            frxx::eigen::DynamicInnerStride(RX.innerStride()));
        frxx::proc::algs::acf::compute_ray_m(
            iqvs, iqvs, RV_view, 0);
        frxx::proc::algs::acf::compute_ray_m(
            iqhs, iqvs, RX_view, 0);

        for (i64 l = 0; l < lags.size(); ++l) {
            auto& RHl = RH[l];
            Complex128RowMap RH_view(
                RHl.data() + t * RHl.outerStride(),
                nRange,
                frxx::eigen::DynamicInnerStride(RHl.innerStride()));
            frxx::proc::algs::acf::compute_ray_m(
                iqhs, iqhs, RH_view,
                static_cast<i64>(lags(l)));
        }
    });
}

}  // namespace frxx::proc::moments::standard
