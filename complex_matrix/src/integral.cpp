#include "integral.h"
#include <format>
#include <cassert>
extern "C" {
#include "cint.h"

void calc_int1e_spsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env);
void calc_int2e_spsp1_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
void cint2e_spsp1_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
}
using namespace cint;

Integral::Integral()
    : _spsp(14, 14, false)
{
    // 初始化 _atm
    _atm = {
        8, 20, 1, 23, 0, 0,
        1, 24, 1, 27, 0, 0,
        1, 28, 1, 31, 0, 0
    };

    // 初始化 _bas
    _bas = {
        0, 0, 3, 1, 0, 38, 41, 0,
        0, 0, 3, 1, 0, 44, 47, 0,
        0, 1, 3, 1, 0, 50, 53, 0,
        1, 0, 3, 1, 0, 32, 35, 0,
        2, 0, 3, 1, 0, 32, 35, 0
    };

    // 初始化 _env
    _env = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        -1.43052268, 1.10926924, 0.0, 0.0, 1.43052268,
        1.10926924, 0.0, 3.42525091, 0.62391373, 0.1688554,
        0.98170675, 0.94946401, 0.29590645, 130.70932, 23.808861,
        6.4436083, 15.07274649, 14.57770167, 4.54323359, 5.0331513,
        1.1695961, 0.380389, -0.848697, 1.13520079, 0.85675304,
        5.0331513, 1.1695961, 0.380389, 3.42906571, 2.15628856,
        0.34159239
    };
}

void Integral::calc_spsp()
{

    CINTOpt* opt = nullptr;
    cint2e_spsp1_optimizer(&opt, _atm.data(), _atm.size() / ATM_SLOTS, _bas.data(), _bas.size() / BAS_SLOTS, _env.data());

    std::vector<int> shls(4);
    auto natm = _atm.size() / ATM_SLOTS;
    auto nbas = _bas.size() / BAS_SLOTS;
    const int n2c_ = 14;
    std::vector<std::complex<double>> _LLLL;
    std::vector<std::complex<double>> _SSLL;
    _LLLL.resize(n2c_ * n2c_ * n2c_ * n2c_);
    _SSLL.resize(n2c_ * n2c_ * n2c_ * n2c_);

    std::vector<double> real_part_llll;
    std::vector<double> imag_part_llll;
    std::vector<double> real_part_ssll;
    std::vector<double> imag_part_ssll;

    for (auto i = 0; i < nbas; i++) {
        shls[0] = i;
        auto di = CINTcgtos_spinor(i, _bas.data());
        auto x = CINTtot_cgto_spinor(_bas.data(), i);

        for (auto j = 0; j < nbas; j++) {
            shls[1] = j;
            auto dj = CINTcgtos_spinor(j, _bas.data());
            auto y = CINTtot_cgto_spinor(_bas.data(), j);

            for (auto k = 0; k < nbas; k++) {
                shls[2] = k;
                auto dk = CINTcgtos_spinor(k, _bas.data());
                auto z = CINTtot_cgto_spinor(_bas.data(), k);
                for (auto l = 0; l < nbas; l++) {
                    shls[3] = l;
                    auto dl = CINTcgtos_spinor(l, _bas.data());
                    auto w = CINTtot_cgto_spinor(_bas.data(), l);

                    int tot = di * dj * dk * dl;
                    // std::cout << "tot = " << tot << std::endl;

                    real_part_llll.resize(tot, 0.0);
                    imag_part_llll.resize(tot, 0.0);
                    real_part_ssll.resize(tot, 0.0);
                    imag_part_ssll.resize(tot, 0.0);

                    std::cout << "i = " << i << " j = " << j << " k = " << k << " l = " << l << std::endl;

                    // calc_int2e_spinor(real_part_llll.data(), imag_part_llll.data(), tot, shls, atm, natm, bas, nbas, env, opt1);
                    // std::cout << "FUCK!" << std::endl;
                    calc_int2e_spsp1_spinor(real_part_ssll.data(), imag_part_ssll.data(), tot, shls.data(), _atm.data(), natm, _bas.data(), nbas, _env.data(), opt);

                    auto fijkl { 0 };
                    for (auto fi = 0; fi < di; fi++) {
                        assert(x+fi < n2c_);
                        for (auto fj = 0; fj < dj; fj++) {
                            assert(y+fj < n2c_);
                            for (auto fk = 0; fk < dk; fk++) {
                                assert(z+fk < n2c_);
                                for (auto fl = 0; fl < dl; fl++, fijkl++) {
                                    assert(w+fl < n2c_);

                                    auto val = std::complex<double>(real_part_llll[fijkl], imag_part_llll[fijkl]);
                                    auto idx = ((x + fi) * n2c_ * n2c_ * n2c_ + (y + fj) * n2c_ * n2c_ + (z + fk) * n2c_ + w + fl);
                                    _LLLL[idx] = val;

                                    val = std::complex<double>(real_part_ssll[fijkl], imag_part_ssll[fijkl]);
                                    _SSLL[idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CINTdel_optimizer(&opt);

    // for (auto i = 0; i < _spsp.getRows(); i++) {
    //     for (auto j = 0; j < _spsp.getCols(); j++) {
    //         std::complex<double> val = _spsp(i, j);
    //         // 使用 std::format 格式化复数，显示实部和虚部
    //         std::cout << std::format("{: .8f} {: .8f}i ", val.real(), val.imag());

    //        // 每两个元素换一行
    //        if ((j + 1) % 2 == 0) {
    //            std::cout << std::endl; // 换行
    //        }
    //    }
    //    std::cout << std::endl; // 每行结束后再换行
    //}
}