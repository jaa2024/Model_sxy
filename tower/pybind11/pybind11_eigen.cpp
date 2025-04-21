// #include "../src/matrix.hpp"
// #include <pybind11/eigen.h>
// #include <pybind11/pybind11.h>

// namespace py = pybind11;
// using namespace linalg;

// namespace linalg {
// template Eigen::MatrixXd Mdot<double>(const Eigen::Ref<const Eigen::MatrixXd>&, const Eigen::Ref<const Eigen::MatrixXd>&);
// template Eigen::MatrixXf Mdot<float>(const Eigen::Ref<const Eigen::MatrixXf>&, const Eigen::Ref<const Eigen::MatrixXf>&);
// template Eigen::MatrixXcd Mdot<std::complex<double>>(
//     const Eigen::Ref<const Eigen::MatrixXcd>&, const Eigen::Ref<const Eigen::MatrixXcd>&);
// template Eigen::MatrixXcf Mdot<std::complex<float>>(
//     const Eigen::Ref<const Eigen::MatrixXcf>&, const Eigen::Ref<const Eigen::MatrixXcf>&);
// } // namespace linalg

// PYBIND11_MODULE(pyTower, m)
// {
//     m.doc() = "Eigen matrix multiplication module";

//     m.def("Mdot",
//         py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd>&,
//             const Eigen::Ref<const Eigen::MatrixXd>&>(&Mdot<double>),
//         py::call_guard<py::gil_scoped_release>(),
//         "double precision matrix multiplication");

//     m.def("Mdot",
//         py::overload_cast<const Eigen::Ref<const Eigen::MatrixXf>&,
//             const Eigen::Ref<const Eigen::MatrixXf>&>(&Mdot<float>),
//         py::call_guard<py::gil_scoped_release>(),
//         "single precision matrix multiplication");
//     m.def("Mdot",
//         py::overload_cast<const Eigen::Ref<const Eigen::MatrixXcd>&,
//             const Eigen::Ref<const Eigen::MatrixXcd>&>(&Mdot<std::complex<double>>),
//         py::call_guard<py::gil_scoped_release>(),
//         "complex double precision matrix multiplication");
//     m.def("Mdot",
//         py::overload_cast<const Eigen::Ref<const Eigen::MatrixXcf>&,
//             const Eigen::Ref<const Eigen::MatrixXcf>&>(&Mdot<std::complex<float>>),
//         py::call_guard<py::gil_scoped_release>(),
//         "complex single precision matrix multiplication");
// }