#pragma once
#ifndef ATOM_HF_HPP
#define ATOM_HF_HPP

#include "integral/integral.hpp"
#include "linalg/einsum.hpp"
namespace atom_hf {

template <typename T = std::complex<double>, int N = 2>
using NTensor = Eigen::Tensor<T, N, Eigen::ColMajor>;
template <typename T = std::complex<double>>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

const int nao_2c(std::vector<int>& bas)
{
    const auto nbas = bas.size() / BAS_SLOTS;
    int total = 0;
    for (int i = 0; i < nbas; ++i) {
        const int basis_offset = i * BAS_SLOTS;

        const int l = bas[basis_offset + ANG_OF];
        const int kappa = bas[basis_offset + KAPPA_OF];
        const int nctr = bas[basis_offset + NCTR_OF];

        int dim;
        if (kappa < 0) {
            dim = (l * 2 + 2) * nctr; // Kappa < 0 case
        }
        else if (kappa > 0) {
            dim = l * 2 * nctr; // Kappa > 0 case
        }
        else {
            dim = (l * 4 + 2) * nctr; // Default case (kappa == 0)
        }

        total += dim;
    }
    return total;
}

template <typename T = std::complex<double>, int N = 2>
NTensor<T, 2> get_hcore(const int nao, std::vector<int>& atm, std::vector<int>& bas, std::vector<double>& env)
{
    const auto nbas = bas.size() / BAS_SLOTS;
    const auto natm = atm.size() / ATM_SLOTS;

    if constexpr (std::is_same_v<T, std::complex<double>>) {
        auto t = integral::compute_unc_spsp_spinor(nao, atm.data(), natm, bas.data(), nbas, env.data());
        YXTensor::print_tensor(t);
    }
    else if constexpr (std::is_same_v<T, double>) {
    }
    else {
        static_assert(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, double>,
            "Unsupported type. Use std::complex<double> or double.");
    }
    NTensor<T, 2> hcore(nao, nao);
    hcore.setZero();

    return hcore;
}

template <typename T = std::complex<double>>
Matrix<T> kernel(std::vector<int>& atm, std::vector<int>& bas, std::vector<double>& env)
{
    int nao { 0 };
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        fmt::print("Using complex double precision\n");
        const auto n2c = nao_2c(bas);
        nao = n2c;
        fmt::println("nao_2c: {}", n2c);
    }
    else if constexpr (std::is_same_v<T, double>) {
        fmt::print("Using double precision\n");
    }
    else {
        static_assert(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, double>,
            "Unsupported type. Use std::complex<double> or double.");
    }

    auto hcore = get_hcore(nao, atm, bas, env);
    Matrix<T> hcore_2c = Matrix<T>::Zero(nao, nao);
    return hcore_2c;
}
} // namespace atom_hf
#endif