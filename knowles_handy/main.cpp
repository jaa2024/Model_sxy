#include "ci/kh.hpp"
#include "integral/integral.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <integral_file>\n";
    std::cerr << "Example: ./main int.txt\n";
    return 1;
  }
  integral::Integral<> MO_INTEGRAL(argv[1]);

  linalg::Matrix<double> S = linalg::Matrix<double>::random(2, 2);
  S.print();
  S.conservativeResize(2, 3);
  S.setCol(2, {1.0, 2.0});
  S.print();
  return 0;
}