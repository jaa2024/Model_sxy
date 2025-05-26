#include "cint.h"
#include <complex.h>
#include <omp.h>

// int1e
int cint1e_spsp(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_nuc(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_spnucsp(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ovlp(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
// int2e
// int cint2e(double complex* buf, int* shls,
//     int* atm, int natm, int* bas, int nbas, double* env,
//     CINTOpt* opt);
int cint2e_spsp1(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
int cint2e_spsp1spsp2(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
int cint2e_ssp1ssp2(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
int cint2e_ssp1sps2(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
int cint2e_breit_ssp1ssp2_spinor(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
int cint2e_breit_ssp1sps2_spinor(double complex* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env,
    CINTOpt* opt);
// int2e_opt
void cint2e_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
void cint2e_spsp1_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
void cint2e_spsp1spsp2_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
void cint2e_ssp1ssp2_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
void cint2e_ssp1sps2_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
void cint2e_breit_ssp1ssp2_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);
void cint2e_breit_ssp1sps2_optimizer(CINTOpt** opt, int* atm, int natm,
    int* bas, int nbas, double* env);

void calc_int1e_spsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env)
{
    double complex buf[num_elements];

    cint1e_spsp(buf, shls, atm, natm, bas, nbas, env);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}
void calc_int1e_nuc_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env)
{
    double complex buf[num_elements];

    cint1e_nuc(buf, shls, atm, natm, bas, nbas, env);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}
void calc_int1e_spnucsp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env)
{
    double complex buf[num_elements];

    cint1e_spnucsp(buf, shls, atm, natm, bas, nbas, env);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}
void calc_int1e_ovlp_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env)
{
    double complex buf[num_elements];

    cint1e_ovlp(buf, shls, atm, natm, bas, nbas, env);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}

void calc_int2e_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double buf[num_elements * 2];

    cint2e(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; i += 2) {
        real_part[i] = buf[i];
        imag_part[i] = buf[i + 1];
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
void calc_int2e_spsp1spsp2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double complex buf[num_elements];

    cint2e_spsp1spsp2(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}
void calc_int2e_ssp1ssp2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double complex buf[num_elements];

    cint2e_ssp1ssp2(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}
void calc_int2e_ssp1sps2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double complex buf[num_elements];

    cint2e_ssp1sps2(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; ++i) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}

void calc_int2e_breit_ssp1ssp2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double complex buf[num_elements];

    cint2e_breit_ssp1ssp2_spinor(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; i++) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}

void calc_int2e_breit_ssp1sps2_spinor(double* real_part, double* imag_part, int num_elements,
    int* shls, int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt)
{
    double complex buf[num_elements];
    cint2e_breit_ssp1sps2_spinor(buf, shls, atm, natm, bas, nbas, env, opt);

    for (int i = 0; i < num_elements; i++) {
        real_part[i] = creal(buf[i]);
        imag_part[i] = cimag(buf[i]);
    }
}