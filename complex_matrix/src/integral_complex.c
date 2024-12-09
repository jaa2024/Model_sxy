#include "cint.h"
#include <complex.h>

int cint1e_spsp(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint2e_spsp1(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
void calc_int1e_spsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env)
{
    double complex buf[num_elements];

    cint1e_spsp(buf, shls, atm, natm, bas, nbas, env);

    // 将 buf 中的实部和虚部分别存入 real_part 和 imag_part 数组
    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]); // 提取实部
        imag_part[i] = cimag(buf[i]); // 提取虚部
    }
}

void calc_int2e_spsp1_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double complex buf[num_elements];

    cint2e_spsp1(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}