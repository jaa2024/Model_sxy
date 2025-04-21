// #pragma once
// #include <Eigen/Core>
// #include <fmt/core.h>

// namespace linalg {
// template <typename T>
// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mdot(
//     const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& a,
//     const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& b)
// {
//     auto rows = a.rows();
//     auto cols = b.cols();
//     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(rows, cols);
//     result.noalias() = a * b;
//     return result;
// }

// } // namespace Matrix
