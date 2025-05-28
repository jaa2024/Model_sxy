#pragma once
#ifndef MPO_HPP
#define MPO_HPP

#include "toolkit.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

namespace DMRG {
template <typename T = double>
struct MPO {
    /*
    build MPO list for DMRG.
    :param single_mpo: a numpy ndarray with ndim=4.
    The first 2 dimensions reprsents the square shape of the MPO and the last 2 dimensions are physical dimensions.
    :param site_num: the total number of sites
    :return MPO list
    The bond order of the local operator is [left, right, up, down], as
                                    | 2
                                0 --#-- 1
                                    | 3
    */
    MPO(Eigen::Tensor<T, 4>& single_mpo, const int size_num); // Default constructor
    std::vector<Eigen::Tensor<T, 4>> mpo_list_; // List of MPO tensors
    int site_num_ = 0; // Total number of sites
}; // Class MPO

template <typename T>
MPO<T>::MPO(Eigen::Tensor<T, 4>& single_mpo, const int size_num)
    : site_num_(size_num)
{
    // Extract dimensions from input MPO
    const int left_dim = single_mpo.dimension(0);
    const int right_dim = single_mpo.dimension(1);
    const int phys_up = single_mpo.dimension(2);
    const int phys_down = single_mpo.dimension(3);

    // First MPO: last row of input tensor
    Eigen::array<Eigen::Index, 4> first_start { left_dim - 1, 0, 0, 0 };
    Eigen::array<Eigen::Index, 4> first_size { 1, right_dim, phys_up, phys_down };
    Eigen::Tensor<T, 4> first_mpo = single_mpo.slice(first_start, first_size);

    // Last MPO: first column of input tensor
    Eigen::array<Eigen::Index, 4> last_start { 0, 0, 0, 0 };
    Eigen::array<Eigen::Index, 4> last_size { left_dim, 1, phys_up, phys_down };
    Eigen::Tensor<T, 4> last_mpo = single_mpo.slice(last_start, last_size);

    // Build MPO list
    mpo_list_.reserve(site_num_);
    mpo_list_.emplace_back(std::move(first_mpo));

    // Add middle MPOs (original tensor copies)
    for (int i = 0; i < site_num_ - 2; ++i) {
        mpo_list_.push_back(single_mpo);
    }

    mpo_list_.emplace_back(std::move(last_mpo));
    fmt::print("mpo_first:\n");
    Toolkit::print_tensor(mpo_list_[0]); // Print first MPO
    fmt::print("mpo_last:\n");
    Toolkit::print_tensor(mpo_list_[site_num_ - 1]); // Print last MPO
    fmt::print("\n");
}
// Constructor implementation
} // namespace DMRG
#endif // MPO_HPP