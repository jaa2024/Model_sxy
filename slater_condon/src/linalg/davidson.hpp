#pragma once
#ifndef DAVIDSON_HPP
#define DAVIDSON_HPP

#include "linalg/matrix.hpp"
#include "linalg/sparse_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>

namespace linalg {

// help functions
template <typename T = double> // returns \sqrt(\sum (v * v))
double norm_(const std::vector<T> &v) {
  double result = 0.0;
  if constexpr (std::is_same_v<T, double>) {
    result = cblas_dnrm2(v.size(), v.data(), 1);
  } else if constexpr (std::is_same_v<T, float>) {
    result = cblas_snrm2(v.size(), v.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    result = cblas_dznrm2(v.size(), v.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    result = cblas_scnrm2(v.size(), v.data(), 1);
  }
  return result;
}
template <typename T = double> // returns \sum (v1 * v2)
T dot_(const std::vector<T> &v1, const std::vector<T> &v2) {
  T result = T(0);
  if constexpr (std::is_same_v<T, double>) {
    result = cblas_ddot(v1.size(), v1.data(), 1, v2.data(), 1);
  } else if constexpr (std::is_same_v<T, float>) {
    result = cblas_sdot(v1.size(), v1.data(), 1, v2.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    std::complex<double> blas_result;
    cblas_zdotc_sub(v1.size(), v1.data(), 1, v2.data(), 1, &blas_result);
    result = blas_result;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    std::complex<float> blas_result;
    cblas_cdotc_sub(v1.size(), v1.data(), 1, v2.data(), 1, &blas_result);
    result = blas_result;
  }
  return result;
}
template <typename T = double> // v1 = v1 * scalar
void dot_(std::vector<T> &v1, T scalar) {
  auto len = v1.size();
  if constexpr (std::is_same_v<T, double>) {
    cblas_dscal(len, scalar, v1.data(), 1);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_sscal(len, scalar, v1.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    std::complex<double> alpha{scalar.real(), scalar.imag()};
    cblas_zscal(len, &alpha, v1.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    std::complex<float> alpha{scalar.real(), scalar.imag()};
    cblas_cscal(len, &alpha, v1.data(), 1);
  }
}
template <typename T = double> // v1 = v1 + v2 * scalar
void add_(std::vector<T> &v1, const std::vector<T> &v2, const T scalar) {
  auto len = v1.size();
  if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(len, scalar, v2.data(), 1, v1.data(), 1);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(len, scalar, v2.data(), 1, v1.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    const std::complex<double> alpha{scalar.real(), scalar.imag()};
    cblas_zaxpy(len, &alpha, v2.data(), 1, v1.data(), 1);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    const std::complex<float> alpha{scalar.real(), scalar.imag()};
    cblas_caxpy(len, &alpha, v2.data(), 1, v1.data(), 1);
  }
}

template <typename T = double> Matrix<T> gramschmidt(const Matrix<T> &X) {
  Matrix<T> orthonormal = X;
  const int cols = X.cols();
  const int rows = X.rows();

  for (int i = 0; i < cols; ++i) {
    std::vector<T> col_i = orthonormal.col(i);

    for (int j = 0; j < i; ++j) {
      const std::vector<T> &col_j = orthonormal.col(j);

      auto proj = dot_(col_i, col_j);
      auto denom = dot_(col_j, col_j);

      T coeff = T(0);
      if (std::abs(denom) > 1e-14) {
        coeff = proj / denom;
      }

      // col_i = col_i - coeff * col_j
      add_(col_i, col_j, -coeff);
    }

    double norm_d = norm_(col_i);

    if (norm_d > 1e-7) {
      const T scale = T(1.0 / norm_d);
      dot_(col_i, scale);
    }
    orthonormal.setCol(i, col_i);
  }
  return orthonormal;
}

template <typename Transformer, typename T = double>
const double davidson_solver(Transformer transformer, const T *diagonal,
                             std::size_t n_dim, std::size_t start_dim = 5,
                             std::size_t max_iter = 100,
                             double residue_tol = 1e-6) {

  // initial guess
  Matrix<T> search_space = Matrix<T>::identity(n_dim, start_dim) +
                           0.01 * Matrix<T>::ones(n_dim, start_dim);

  // start iteration
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    auto M = search_space.cols();
    Matrix<T> orthonormal_subspace = gramschmidt(search_space);
    Matrix<T> Ab_i = Matrix<T>::zero(n_dim, M);
    for (std::size_t j = 0; j < M; j++) {
      auto vec = orthonormal_subspace.col(j);
      Ab_i.setCol(j, transformer(vec));
    }

    Matrix<double> B = orthonormal_subspace.transpose().conjugate() * Ab_i;
    auto [eigenvalues, eigenvectors] = eigh(B);
    // find the index of the smallest eigenvalue
    auto min_it = std::min_element(eigenvalues.begin(), eigenvalues.end());
    double theta = *min_it;
    std::size_t minIndex = std::distance(eigenvalues.begin(), min_it);
    fmt::println("davidson diagonalization iter: {:>2}, theta: {:10.10f}",
                 iter + 1, theta);

    // check the residue
    std::vector<T> s = eigenvectors.col(minIndex);
    std::vector<T> residue = Ab_i * s;
    add_(residue, orthonormal_subspace * s, -theta);
    // theta * orthonormal_subspace * s;
    double residue_norm = norm_(residue);
    if (residue_norm < residue_tol) {
      fmt::println("davidson diagonalization converged in {:>2} iterations",
                   iter + 1);
      return theta;
    }
    // update the search space
    std::vector<T> xi(n_dim);
    for (int i = 0; i < n_dim; ++i) {
      xi[i] = residue[i] / (theta - diagonal[i]);
    }
    auto xi_norm = norm_(xi);
    std::transform(xi.begin(), xi.end(), xi.begin(),
                   [xi_norm](T &val) { return val / xi_norm; });

    search_space = orthonormal_subspace;
    search_space.conservativeResize(n_dim, M + 1);
    search_space.setCol(M, xi);
  }
  throw std::runtime_error("Davidson diagonalization failed");
}

// TEST
using CSRMatrix =
    std::tuple<std::vector<double>, std::vector<int>, std::vector<int>,
               std::vector<double>, int, int, int>;

CSRMatrix readCSRFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Failed to open file");
  }

  int rows, cols, nnz;
  std::vector<double> values, diag;
  std::vector<int> columns, rowIndex;
  std::string line;

  file >> rows >> cols >> nnz;

  while (file >> line) {
    if (line == "values") {
      values.resize(nnz);
      for (double &val : values)
        file >> val;
    } else if (line == "columns") {
      columns.resize(nnz);
      for (int &col : columns)
        file >> col;
    } else if (line == "rowIndex") {
      rowIndex.resize(rows + 1);
      for (int &row : rowIndex)
        file >> row;
    } else if (line == "diag") {
      diag.resize(rows);
      for (double &d : diag)
        file >> d;
    }
  }

  return {values, columns, rowIndex, diag, rows, cols, nnz};
}

void debug() {
  auto [values, columns, rowIndex, diag, rows, cols, nnz] =
      readCSRFromFile("../sparse_matrix.txt");

  linalg::SparseMatrix<double> A(linalg::MatrixFillMode::UPPER, values, columns,
                                 rowIndex, rows, cols);

  auto transformer = [&](const std::vector<double> &v) { return A * v; };
  auto start = std::chrono::high_resolution_clock::now();
  linalg::davidson_solver(transformer, diag.data(), rows);
  fmt::println("Time taken: {} ms",
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count());
}

} // namespace linalg
#endif