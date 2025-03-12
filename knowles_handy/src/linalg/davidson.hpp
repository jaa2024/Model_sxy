#pragma once
#ifndef DAVIDSON_HPP
#define DAVIDSON_HPP

#include "linalg/matrix.hpp"
#include <algorithm>

namespace linalg {

// help functions
template <typename T = double> double norm(const std::vector<T> &v) {
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
template <typename T = double> Matrix<T> gramschmidt(const Matrix<T> &X) {
  Matrix<T> orthonormal = X;
  const int cols = X.cols();
  const int rows = X.rows();

  for (int i = 0; i < cols; ++i) {
    std::vector<T> col_i = orthonormal.col(i);

    for (int j = 0; j < i; ++j) {
      const std::vector<T> &col_j = orthonormal.col(j);
      T proj = T(0);
      T denom = T(0);

      if constexpr (std::is_same_v<T, double>) {
        proj = cblas_ddot(rows, col_j.data(), 1, col_i.data(), 1);
        denom = cblas_ddot(rows, col_j.data(), 1, col_j.data(), 1);
      } else if constexpr (std::is_same_v<T, float>) {
        proj = cblas_sdot(rows, col_j.data(), 1, col_i.data(), 1);
        denom = cblas_sdot(rows, col_j.data(), 1, col_j.data(), 1);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::complex<double> blas_proj, blas_denom;
        cblas_zdotc_sub(rows, col_j.data(), 1, col_i.data(), 1, &blas_proj);
        cblas_zdotc_sub(rows, col_j.data(), 1, col_j.data(), 1, &blas_denom);
        proj = blas_proj;
        denom = blas_denom;
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::complex<float> blas_proj, blas_denom;
        cblas_cdotc_sub(rows, col_j.data(), 1, col_i.data(), 1, &blas_proj);
        cblas_cdotc_sub(rows, col_j.data(), 1, col_j.data(), 1, &blas_denom);
        proj = blas_proj;
        denom = blas_denom;
      }

      T coeff = T(0);
      if (std::abs(denom) > 1e-14) {
        coeff = proj / denom;
      }

      // col_i = col_i - coeff * col_j
      if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(rows, -coeff, col_j.data(), 1, col_i.data(), 1);
      } else if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(rows, -coeff, col_j.data(), 1, col_i.data(), 1);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        const std::complex<double> alpha{-coeff.real(), -coeff.imag()};
        cblas_zaxpy(rows, &alpha, col_j.data(), 1, col_i.data(), 1);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        const std::complex<float> alpha{-coeff.real(), -coeff.imag()};
        cblas_caxpy(rows, &alpha, col_j.data(), 1, col_i.data(), 1);
      }
    }

    double norm_d = norm(col_i);

    if (norm_d > 1e-7) {
      const T scale = T(1.0 / norm_d);
      if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(rows, scale, col_i.data(), 1);
      } else if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(rows, scale, col_i.data(), 1);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        const std::complex<double> alpha{scale.real(), scale.imag()};
        cblas_zscal(rows, &alpha, col_i.data(), 1);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        const std::complex<float> alpha{scale.real(), scale.imag()};
        cblas_cscal(rows, &alpha, col_i.data(), 1);
      }
    }

    orthonormal.setCol(i, col_i);
  }

  return orthonormal;
}

template <typename Transformer, typename T = double>
double davidson_solver(Transformer transformer, const T *diagonal,
                       std::size_t n_dim, std::size_t start_dim = 2,
                       std::size_t max_iter = 100, double residue_tol = 1e-6) {

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

    Matrix<double> B = orthonormal_subspace.transpose() * Ab_i;
    auto [eigenvalues, eigenvectors] = eigh(B);
    // find the index of the smallest eigenvalue
    auto min_it = std::min_element(eigenvalues.begin(), eigenvalues.end());
    double theta = *min_it;
    std::size_t minIndex = std::distance(eigenvalues.begin(), min_it);
    fmt::println("davidson diagonalization iter: {}, theta: {:>10:10f}",
                 iter + 1, theta);

    // check the residue
    std::vector<T> s = eigenvectors.col(minIndex);
    std::vector<T> residue = Ab_i * s - theta * orthonormal_subspace * s;
    double residue_norm = norm(residue);
    if (residue_norm < residue_tol) {
      fmt::println("davidson diagonalization converged in {} iterations",
                   iter + 1);
      return theta;
    }
    // update the search space
    std::vector<T> xi(n_dim);
    for (int i = 0; i < n_dim; ++i) {
      xi[i] = residue[i] / (theta - diagonal[i]);
    }
    auto xi_norm = norm(xi);
    std::transform(xi.begin(), xi.end(), xi.begin(),
                   [xi_norm](T &val) { return val / xi_norm; });

    search_space = orthonormal_subspace;
    search_space.conservativeResize(n_dim, M + 1);
    search_space.setCol(M, xi);
  }
  throw std::runtime_error("Davidson diagonalization failed");
}
} // namespace linalg
#endif