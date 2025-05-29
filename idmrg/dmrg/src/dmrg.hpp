#pragma once
#ifndef DMRG_HPP
#define DMRG_HPP

#include "mpo.hpp"
#include "mps.hpp"
#include <Eigen/Dense>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
namespace DMRG {

template <typename T = double>
inline std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::Vector<T, Eigen::Dynamic>, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
svd_compress(const Eigen::Tensor<double, 3>& tensor, const std::string& direction, const int maxM)
{
    const int left = tensor.dimension(0);
    const int phy_dim = tensor.dimension(1);
    const int right = tensor.dimension(2);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
    if (direction == "left" || direction == "l") {
        mat = Eigen::Map<const Eigen::MatrixXd>(
            tensor.data(),
            left * phy_dim,
            right);
    }
    else if (direction == "right" || direction == "r") {
        mat = Eigen::Map<const Eigen::MatrixXd>(
            tensor.data(),
            left,
            phy_dim * right);
    }
    else {
        throw std::invalid_argument("Invalid direction: " + direction);
    }

    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
        mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const int actualM = std::min(svd.singularValues().size(), maxM);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> U = svd.matrixU().leftCols(actualM);
    Eigen::Vector<T, Eigen::Dynamic> S = svd.singularValues().head(actualM);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V = svd.matrixV().leftCols(actualM).transpose();

    return std::make_tuple(std::move(U), std::move(S), std::move(V));
}

template <typename T = double>
struct DMRG {
    int max_bond_dim_ = 0; // Maximum bond dimension
    int max_sweeps_ = 20; // Maximum number of sweeps
    double error_threshold_ = 1e-7; // Error threshold for convergence
    int site_num_ = 0; // Total number of sites
    std::vector<Eigen::Tensor<T, 3>> mps_list_; // List of MPS tensors
    std::vector<Eigen::Tensor<T, 4>> mpo_list_; // List of MPO tensors
    std::vector<std::optional<Eigen::Tensor<T, 6>>> F_list; // List of F tensors for DMRG
    std::vector<std::optional<Eigen::Tensor<T, 6>>> L_list; // List of L tensors for DMRG
    std::vector<std::optional<Eigen::Tensor<T, 6>>> R_list; // List of R tensors for DMRG

    DMRG() = delete;
    DMRG(const std::vector<Eigen::Tensor<T, 4>>& mpo, const std::vector<Eigen::Tensor<T, 3>>& mps, const int max_bond_dim = 0, const int max_sweeps = 20, const T error_threshold = 1e-7);

}; // struct DMRG

template <typename T>
DMRG<T>::DMRG(const std::vector<Eigen::Tensor<T, 4>>& mpo,
    const std::vector<Eigen::Tensor<T, 3>>& mps,
    int max_bond_dim,
    int max_sweeps,
    T error_threshold)
    : max_bond_dim_(max_bond_dim), max_sweeps_(max_sweeps), error_threshold_(error_threshold)
{
    // 参数校验
    assert(max_bond_dim_ > 0);
    assert(!mpo.empty() && mpo.size() == mps.size());
    const size_t site_num = mpo.size();

    auto create_boundary_mpo = [] {
        Eigen::Tensor<T, 4> boundary(1, 1, 1, 1);
        boundary.setZero();
        return boundary;
    };

    auto create_boundary_mps = [] {
        Eigen::Tensor<T, 3> boundary(1, 1, 1);
        boundary.setZero();
        return boundary;
    };
    auto create_boundary_F = [] {
        Eigen::Tensor<T, 6> boundary(1, 1, 1, 1, 1, 1);
        boundary.setZero();
        return boundary;
    };

    mpo_list_.reserve(site_num + 2);
    mpo_list_.emplace_back(create_boundary_mpo());
    mpo_list_.insert(mpo_list_.end(), mpo.begin(), mpo.end());
    mpo_list_.emplace_back(create_boundary_mpo());

    mps_list_.reserve(site_num + 2);
    mps_list_.emplace_back(create_boundary_mps());
    mps_list_.insert(mps_list_.end(), mps.begin(), mps.end());
    mps_list_.emplace_back(create_boundary_mps());

    F_list.reserve(site_num + 2);
    F_list.emplace_back(create_boundary_F());
    F_list.insert(F_list.end(), site_num, std::nullopt);
    F_list.emplace_back(create_boundary_F());

    L_list = F_list;
    R_list = F_list;
}
} // namespace DMRG
#endif // DMRG_HPP