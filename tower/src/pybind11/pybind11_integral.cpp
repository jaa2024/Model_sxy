#include "integral/integral.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace integral;

// 定义一个初始化函数，供主模块调用
void init_integral(py::module& m)
{
    // 获取或创建子模块
    py::module integral_module = m.def_submodule("integral", "Integral functions");

    // 将函数绑定到子模块中
    integral_module.def("get_bp_hso2e", [](Eigen::MatrixXd& dm, int nao, const std::vector<int>& atm, const std::vector<int>& bas, const std::vector<double>& env) {
        auto [vj, vk] = get_bp_hso2e(
            dm, nao,
            const_cast<int*>(atm.data()),
            atm.size() / ATM_SLOTS,
            const_cast<int*>(bas.data()),
            bas.size() / BAS_SLOTS,
            const_cast<double*>(env.data())
        );

        const auto& dims = vj.dimensions();
        std::vector<long int> shape = {dims[0], dims[1], dims[2]};
        std::vector<size_t> strides = {
            sizeof(double),
            sizeof(double) * dims[0],
            sizeof(double) * dims[0] * dims[1]
        };

        size_t total_size = vj.size();
        double* vj_copy = new double[total_size];
        double* vk_copy = new double[total_size];
        std::memcpy(vj_copy, vj.data(), sizeof(double) * total_size);
        std::memcpy(vk_copy, vk.data(), sizeof(double) * total_size);

        py::array_t<double> vj_array(
            shape, strides, vj_copy,
            py::capsule(vj_copy, [](void* p) { delete[] static_cast<double*>(p); })
        );

        py::array_t<double> vk_array(
            shape, strides, vk_copy,
            py::capsule(vk_copy, [](void* p) { delete[] static_cast<double*>(p); })
        );

        return std::make_tuple(vj_array, vk_array); }, py::arg("dm").noconvert(), py::arg("nao"), py::arg("atm"), py::arg("bas"), py::arg("env"));
}