#include "ci/kh.hpp"
#include "ci/sc.hpp"
#include "integral/integral.hpp"
#include <iostream>

int main()
{

    integral::Integral<> MO_INTEGRAL("../example/n2_fci/fcidump.example1");
    ci::SlaterCondon<double> SC("../example/n2_fci/electron_configurations.txt");
    SC.kernel(MO_INTEGRAL, 6, 6);
    return 0;
}