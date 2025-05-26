// Tower.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_integral(py::module& m);
void init_x2camf(py::module& m);

PYBIND11_MODULE(Tower, m)
{
    m.doc() = "Tower main module";

    // 创建子模块 integral（实际实现在 integral.cpp 中）
    init_integral(m);
    init_x2camf(m);
}