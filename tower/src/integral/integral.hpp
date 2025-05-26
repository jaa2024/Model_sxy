#pragma once
#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#include "linalg/einsum.hpp"
#include <Eigen/Dense>
#include <fmt/core.h>
#include <unsupported/Eigen/CXX11/Tensor>

extern "C" {
#include <cint.h>
int cint1e_ovlp_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_nuc_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_kin_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_pnucp_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env); // x2c1e spin-free nuc
int cint1e_pnucxp_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env); // x2c1e spin-depended nuc
int cint2e_p1vxp1_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt); // breit-pauli 2e

// 4C Dirac-HF integrals
void calc_int1e_spsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env);
void calc_int1e_nuc_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env);
void calc_int1e_spnucsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env);
void calc_int1e_ovlp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env);
void calc_int2e_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void calc_int2e_spsp1_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void calc_int2e_spsp1spsp2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void calc_int2e_ssp1ssp2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void calc_int2e_ssp1sps2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void calc_int2e_breit_ssp1ssp2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void calc_int2e_breit_ssp1sps2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
}
namespace integral {
inline const Eigen::Tensor<double, 3> compute_x2c1e_sd_sph(const int nao, int* atm, int natm, int* bas, int nbas, double* env)
{
    Eigen::Tensor<double, 3> result(3, nao, nao);
    result.setZero();

    for (int ipr = 0; ipr < nbas; ++ipr) {
        auto di = CINTcgto_spheric(ipr, bas);
        auto x = CINTtot_cgto_spheric(bas, ipr);

        for (int jpr = 0; jpr < nbas; ++jpr) {
            auto dj = CINTcgto_spheric(jpr, bas);
            auto y = CINTtot_cgto_spheric(bas, jpr);

            int shls[] { ipr, jpr };

            std::vector<double> buf(3 * di * dj);

            if (cint1e_pnucxp_sph(buf.data(), shls, atm, natm, bas, nbas, env)) {
                for (int xyz = 0; xyz < 3; ++xyz) {
                    for (int jbf = 0; jbf < dj; ++jbf) {
                        for (int ibf = 0; ibf < di; ++ibf) {
                            result(xyz, x + ibf, y + jbf) = buf[xyz * di * dj + jbf * di + ibf];
                        }
                    }
                }
            }
            else {
                throw std::runtime_error(fmt::format("Error: Failed to compute integral for shells {} and {}", ipr, jpr));
            }
        }
    }
    YXTensor::print_tensor(result);
    return result;
}

inline const Eigen::Tensor<double, 5> compute_bp_2e_sph(const int nao, int* atm, int natm, int* bas, int nbas, double* env)
{
    Eigen::Tensor<double, 5> result(3, nao, nao, nao, nao);
    result.setZero();

    CINTOpt* opt = nullptr;

    for (int ipr = 0; ipr < nbas; ++ipr) {
        auto di = CINTcgto_spheric(ipr, bas);
        auto x = CINTtot_cgto_spheric(bas, ipr);
        for (int jpr = 0; jpr < nbas; ++jpr) {
            auto dj = CINTcgto_spheric(jpr, bas);
            auto y = CINTtot_cgto_spheric(bas, jpr);

            for (int kpr = 0; kpr < nbas; ++kpr) {
                auto dk = CINTcgto_spheric(kpr, bas);
                auto z = CINTtot_cgto_spheric(bas, kpr);

                for (int lpr = 0; lpr < nbas; ++lpr) {
                    auto dl = CINTcgto_spheric(lpr, bas);
                    auto w = CINTtot_cgto_spheric(bas, lpr);
                    int shls[] { ipr, jpr, kpr, lpr };

                    std::vector<double> buf(3 * di * dj * dk * dl);
                    if (cint2e_p1vxp1_sph(buf.data(), shls, atm, natm, bas, nbas, env, nullptr)) {
                        for (int xyz = 0; xyz < 3; ++xyz) {
                            for (int ibf = 0; ibf < di; ++ibf) {
                                for (int jbf = 0; jbf < dj; ++jbf) {
                                    for (int kbf = 0; kbf < dk; ++kbf) {
                                        for (int lbf = 0; lbf < dl; ++lbf) {
                                            result(xyz, x + ibf, y + jbf, z + kbf, w + lbf) = buf[xyz * di * dj * dk * dl + lbf * di * dj * dk + kbf * di * dj + jbf * di + ibf];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    else {
                        throw std::runtime_error(fmt::format("Error: Failed to compute integral for shells {} {} {} {}", ipr, jpr, kpr, lpr));
                    }
                }
            }
        }
    }
    return result;
}

inline const std::pair<Eigen::Tensor<double, 3>, Eigen::Tensor<double, 3>> get_bp_hso2e(Eigen::MatrixXd& dm, const int nao, int* atm, int natm, int* bas, int nbas, double* env)
{
    auto hso2e = compute_bp_2e_sph(nao, atm, natm, bas, nbas, env);
    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::ColMajor>> dm_ao(dm.data(), nao, nao);
    auto vj = YXTensor::einsum<2, double, 5, 2, 3>("yijkl,lk->yij", hso2e, dm_ao);
    auto vk = YXTensor::einsum<2, double, 5, 2, 3>("yijkl,jk->yil", hso2e, dm_ao);
    vk += YXTensor::einsum<2, double, 5, 2, 3>("yijkl,li->ykj", hso2e, dm_ao);
    return { vj, vk };
}

inline const Eigen::Tensor<std::complex<double>, 2> compute_unc_spsp_spinor(int nao, int* atm, int natm, int* bas, int nbas, double* env)
{
    Eigen::Tensor<std::complex<double>, 2> T(nao, nao);
    T.setZero();

    for (int jdx = 0; jdx < nbas; jdx++) {
        for (int idx = jdx; idx < nbas; idx++) {
            int shls[] { idx, jdx };
            auto di = CINTcgto_spinor(idx, bas);
            auto dj = CINTcgto_spinor(jdx, bas);
            auto x = CINTtot_cgto_spinor(bas, idx);
            auto y = CINTtot_cgto_spinor(bas, jdx);

            auto total = di * dj;

            std::vector<double> real_part(total);
            std::vector<double> imag_part(total);

            calc_int1e_spsp_spinor(real_part.data(), imag_part.data(), total, shls, atm, natm, bas, nbas, env);
            for (int fj = 0, fij = 0; fj < dj; ++fj) {
                for (int fi = 0; fi < di; ++fi, ++fij) {
                    auto val = std::complex<double>(real_part[fij], imag_part[fij]);
                    T(x + fi, y + fj) = val;
                    T(y + fj, x + fi) = std::conj(val);
                }
            }
        }
    }
    return T;
}
} // namespace integral

#endif // INTEGRAL_HPP