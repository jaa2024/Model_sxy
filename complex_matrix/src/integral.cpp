#include "integral.h"
#include <format>
extern "C" {
#include "cint.h"

void calc_int1e_spsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env);
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
    std::vector<int> shls(2);
    auto natm = _atm.size() / ATM_SLOTS;
    auto nbas = _bas.size() / BAS_SLOTS;

    for (auto j = 0; j < 5; j++) {
        shls[1] = j;
        auto dj = CINTcgtos_spinor(shls[1], _bas.data());
        auto y = CINTtot_cgto_spinor(_bas.data(), shls[1]);

        for (auto i = j; i < 5; i++) {
            shls[0] = i;
            auto di = CINTcgtos_spinor(shls[0], _bas.data());
            auto x = CINTtot_cgto_spinor(_bas.data(), shls[0]);

            auto tot = di * dj;
            std::vector<double> real_part(di * dj);
            std::vector<double> imag_part(di * dj);

            calc_int1e_spsp_spinor(real_part.data(), imag_part.data(), tot, shls.data(), _atm.data(), natm, _bas.data(), nbas, _env.data());
            std::vector<std::complex<double>> buf(tot);
            for (int i = 0; i < tot; ++i) {
                buf[i] = std::complex<double>(real_part[i], imag_part[i]);
            }
            auto fij { 0 };
            for (auto fj = 0; fj < dj; fj++) {
                for (auto fi = 0; fi < di; fi++, fij++) {
                    _spsp(x + fi, y + fj) = buf[fij];
                    _spsp(y + fj, x + fi) = std::conj(buf[fij]);
                }
            }
        }
    }

    for (auto i = 0; i < _spsp.getRows(); i++) {
        for (auto j = 0; j < _spsp.getCols(); j++) {
            std::complex<double> val = _spsp(i, j);
            // 使用 std::format 格式化复数，显示实部和虚部
            std::cout << std::format("{: .8f} {: .8f}i ", val.real(), val.imag());

            // 每两个元素换一行
            if ((j + 1) % 2 == 0) {
                std::cout << std::endl; // 换行
            }
        }
        std::cout << std::endl; // 每行结束后再换行
    }
}