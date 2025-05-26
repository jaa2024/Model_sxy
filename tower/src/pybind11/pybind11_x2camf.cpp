#include "x2camf/atom_hf.hpp"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace atom_hf;

void init_x2camf(py::module& m)
{
    py::module x2camf_module = m.def_submodule("x2camf", "X2CAMF functions");

    x2camf_module.def("kernel", [](std::vector<int>& atm, std::vector<int>& bas, std::vector<double>& env) -> py::array_t<std::complex<double>> {
        // 调用 C++ 函数，返回列优先的 Eigen 矩阵
        Eigen::MatrixXcd result = kernel<>(atm, bas, env);

        // 将列优先转为行优先（深拷贝）
        Eigen::Matrix<std::complex<double>, 
                      Eigen::Dynamic, 
                      Eigen::Dynamic, 
                      Eigen::RowMajor> row_major_result = result;

        // 获取行数和列数
        const long rows = row_major_result.rows();
        const long cols = row_major_result.cols();

        // 分配新内存并拷贝数据（确保Python管理内存）
        auto* data_copy = new std::complex<double>[rows * cols];
        std::memcpy(data_copy, row_major_result.data(), 
                   rows * cols * sizeof(std::complex<double>));

        // 定义 NumPy 数组的形状和步幅（行优先）
        std::vector<size_t> shape = {static_cast<size_t>(rows), 
                                    static_cast<size_t>(cols)};
        std::vector<size_t> strides = {
            sizeof(std::complex<double>) * cols,  // 行步幅（相邻行间隔）
            sizeof(std::complex<double>)           // 列步幅（相邻元素间隔）
        };

        // 创建并返回 NumPy 数组
        return py::array_t<std::complex<double>>(
            shape,
            strides,
            data_copy,
            py::capsule(data_copy, [](void* p) { 
                delete[] static_cast<std::complex<double>*>(p); 
            })
        ); }, py::arg("atm"), py::arg("bas"), py::arg("env"));
}