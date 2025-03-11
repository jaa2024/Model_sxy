#include "linalg/matrix.hpp"
using namespace linalg;

void test_blas() {
  linalg::Matrix<std::complex<double>> A =
      linalg::Matrix<std::complex<double>>::zero(3, 3);
  fmt::println("A:");
  A.print();

  linalg::Matrix<std::complex<double>> B =
      linalg::Matrix<std::complex<double>>::random(3, 3);
  fmt::println("B:");
  B.print();

  auto C = B.conjugate();
  fmt::println("C = B.conjugate():");
  C.print();

  auto D = B.adjoint();
  fmt::println("D = B.adjoint():");
  D.print();
}

void test_real_general() {
  // 普通实数特征值问题
  Matrix<double> A(3, 3);
  A(0, 0) = 4.0;
  A(0, 1) = 2.0;
  A(0, 2) = 0.0;
  A(1, 0) = 2.0;
  A(1, 1) = 4.0;
  A(1, 2) = 2.0;
  A(2, 0) = 0.0;
  A(2, 1) = 2.0;
  A(2, 2) = 4.0;

  auto [vals, vecs] = eigh(A);
  fmt::print("Real general eigenvalues:\n");
  for (double v : vals)
    fmt::print("{:>10.6f}", v, " ");
  fmt::println("\nExpected: [1.171573 4.       6.828427]\n");
}

void test_real_generalized() {
  // 广义实数特征值问题 (A, B)
  Matrix<double> A(3, 3), B(3, 3);
  // 对称矩阵A
  A(0, 0) = 2;
  A(0, 1) = 1;
  A(0, 2) = 0;
  A(1, 0) = 1;
  A(1, 1) = 2;
  A(1, 2) = 1;
  A(2, 0) = 0;
  A(2, 1) = 1;
  A(2, 2) = 2;

  // 正定对称矩阵B
  B(0, 0) = 1;
  B(0, 1) = 0;
  B(0, 2) = 0;
  B(1, 0) = 0;
  B(1, 1) = 1;
  B(1, 2) = 0;
  B(2, 0) = 0;
  B(2, 1) = 0;
  B(2, 2) = 1;

  auto [vals, vecs] = eigh(A, B);
  fmt::println("Real generalized eigenvalues:");
  for (double v : vals)
    fmt::print("{:>10.6f}", v, " ");
  fmt::println("\nExpected: [0.585786 2.       3.414214]\n");
}

void test_complex_general() {
  // 普通复数特征值问题
  using namespace std::complex_literals;
  Matrix<std::complex<double>> H(3, 3);
  H(0, 0) = 3.0;
  H(0, 1) = -1.0i;
  H(0, 2) = 0.0;
  H(1, 0) = 1.0i;
  H(1, 1) = 4.0;
  H(1, 2) = -1.0i;
  H(2, 0) = 0.0;
  H(2, 1) = 1.0i;
  H(2, 2) = 5.0;

  auto [vals, vecs] = eigh(H);
  fmt::print("Complex general eigenvalues:\n");
  for (double v : vals)
    fmt::print("{:>10.6f}", v, " ");
  fmt::println("\nExpected: [2.267949 4.       5.732051]\n");
}

void test_complex_generalized() {
  // 广义复数特征值问题 (A, B)
  using namespace std::complex_literals;
  Matrix<std::complex<double>> A(3, 3), B(3, 3);

  // Hermitian矩阵A
  A(0, 0) = 2.0;
  A(0, 1) = 1.0 - 1.0i;
  A(0, 2) = 0.0;
  A(1, 0) = 1.0 + 1.0i;
  A(1, 1) = 4.0;
  A(1, 2) = 1.0 - 1.0i;
  A(2, 0) = 0.0;
  A(2, 1) = 1.0 + 1.0i;
  A(2, 2) = 6.0;

  // Hermitian正定矩阵B
  B(0, 0) = 1.0;
  B(0, 1) = 0.0;
  B(0, 2) = 0.0;
  B(1, 0) = 0.0;
  B(1, 1) = 2.0;
  B(1, 2) = 0.0;
  B(2, 0) = 0.0;
  B(2, 1) = 0.0;
  B(2, 2) = 1.0;

  auto [vals, vecs] = eigh(A, B);
  fmt::print("Complex generalized eigenvalues:\n");
  for (double v : vals)
    fmt::print("{:>10.6f}", v, " ");
  fmt::println("\nExpected: [0.897225 2.853635 6.249141]\n");
}

int main() {
  // test_blas();
  test_real_general();
  test_real_generalized();
  test_complex_general();
  test_complex_generalized();

  return 0;
}