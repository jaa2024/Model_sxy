#ifndef __INTEGRAL_H__
#define __INTEGRAL_H__
#include "cmatrix.h"
#include <complex>
#include <iostream>
#include <vector>

namespace cint {
class Integral {
private:
    std::vector<int> _atm;
    std::vector<int> _bas;
    std::vector<double> _env;

    cmatrix::CMatrix _spsp;

public:
    Integral();
    void calc_spsp();
};
}

#endif // INTEGRAL_H
