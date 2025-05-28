#include "src/mpo.hpp"
#include "src/operators.hpp"
#include "src/toolkit.hpp"
#include <iostream>
constexpr auto MAX_BOND_DIMENSION = 50;
constexpr auto MAX_SWEEPS = 20;
constexpr auto PHY_DIM = 2;
constexpr auto SITE_NUM = 5;
constexpr auto ERROR_THRESHOLD = 1e-7;
int main()
{
    const auto id = DMRG::Operator::Identity; // Identity operator for 2x2 matrices
    const auto zero = DMRG::Operator::Zero; // Zero operator for 2x2 matrices
    const auto sz = DMRG::Operator::Sz; // Pauli Z operator
    const auto sp = DMRG::Operator::Splus; // Pauli S+ operator
    const auto sm = DMRG::Operator::Sminus; // Pauli S- operator

    Eigen::Tensor<double, 4> single_mpo(5, 5, 2, 2);
    single_mpo.setZero();
    DMRG::Toolkit::assign_block<double>(single_mpo, id, 0, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, sp, 1, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, sm, 2, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, sz, 3, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, 0.5 * sm, 4, 1);
    DMRG::Toolkit::assign_block<double>(single_mpo, 0.5 * sp, 4, 2);
    DMRG::Toolkit::assign_block<double>(single_mpo, sz, 4, 3);
    DMRG::Toolkit::assign_block<double>(single_mpo, id, 4, 4);
    // DMRG::Toolkit::print_tensor(single_mpo);
    DMRG::MPO<> mpo(single_mpo, SITE_NUM);
    return 0;
}