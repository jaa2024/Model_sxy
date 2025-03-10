#include "linalg/matrix.hpp"

int main() {
  linalg::Matrix<std::complex<double>> A =
      linalg::Matrix<std::complex<double>>::zero(3, 3);
  linalg::Matrix<std::complex<double>> B =
      linalg::Matrix<std::complex<double>>::random(3, 3);
  auto C = B.transpose();

  fmt::println("A:");
  A.print();
  fmt::println("B:");
  B.print();
  fmt::println("C = B.transpose()");
  C.print();

  fmt::println("D = B:");
  auto D = std::move(B);
  D.print();
  return 0;
}