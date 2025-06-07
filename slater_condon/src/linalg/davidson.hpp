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
double norm_(const std::vector<T>& v)
{
    double result = 0.0;
    if constexpr (std::is_same_v<T, double>) {
        result = cblas_dnrm2(v.size(), v.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        result = cblas_snrm2(v.size(), v.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        result = cblas_dznrm2(v.size(), v.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        result = cblas_scnrm2(v.size(), v.data(), 1);
    }
    return result;
}
template <typename T = double> // returns \sum (v1 * v2)
T dot_(const std::vector<T>& v1, const std::vector<T>& v2)
{
    T result = T(0);
    if constexpr (std::is_same_v<T, double>) {
        result = cblas_ddot(v1.size(), v1.data(), 1, v2.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        result = cblas_sdot(v1.size(), v1.data(), 1, v2.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::complex<double> blas_result;
        cblas_zdotc_sub(v1.size(), v1.data(), 1, v2.data(), 1, &blas_result);
        result = blas_result;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::complex<float> blas_result;
        cblas_cdotc_sub(v1.size(), v1.data(), 1, v2.data(), 1, &blas_result);
        result = blas_result;
    }
    return result;
}
template <typename T = double> // v1 = v1 * scalar
void dot_(std::vector<T>& v1, T scalar)
{
    auto len = v1.size();
    if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(len, scalar, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(len, scalar, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::complex<double> alpha { scalar.real(), scalar.imag() };
        cblas_zscal(len, &alpha, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::complex<float> alpha { scalar.real(), scalar.imag() };
        cblas_cscal(len, &alpha, v1.data(), 1);
    }
}
template <typename T = double> // v1 = v1 + v2 * scalar
void add_(std::vector<T>& v1, const std::vector<T>& v2, const T scalar)
{
    auto len = v1.size();
    if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(len, scalar, v2.data(), 1, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(len, scalar, v2.data(), 1, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        const std::complex<double> alpha { scalar.real(), scalar.imag() };
        cblas_zaxpy(len, &alpha, v2.data(), 1, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        const std::complex<float> alpha { scalar.real(), scalar.imag() };
        cblas_caxpy(len, &alpha, v2.data(), 1, v1.data(), 1);
    }
}

template <typename T = double>
Matrix<T> gramschmidt(const Matrix<T>& X)
{
    Matrix<T> orthonormal = X;
    const int cols = X.cols();
    const int rows = X.rows();

    for (int i = 0; i < cols; ++i) {
        std::vector<T> col_i = orthonormal.col(i);

        for (int j = 0; j < i; ++j) {
            const std::vector<T>& col_j = orthonormal.col(j);

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
const std::vector<double> davidson_solver(Transformer transformer, const T* diagonal,
    std::size_t n_dim, std::size_t n_roots = 1,
    std::size_t start_dim = 5,
    std::size_t max_iter = 100,
    double residue_tol = 1e-6)
{

    if (start_dim < n_roots) {
        throw std::runtime_error(
            "start_dim should be greater than or equal to n_roots");
    }
    // initial guess
    Matrix<T> search_space = Matrix<T>::identity(n_dim, start_dim) + 0.01 * Matrix<T>::ones(n_dim, start_dim);

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

        std::vector<std::vector<T>> xi_n(n_roots, std::vector<T>(n_dim));
        std::vector<double> theta_n(n_roots);
        std::vector<std::uint8_t> has_converged(n_roots, 0);

        fmt::println("davidson diagonalization iter: {:>2} ", iter + 1);

        // eig_pairs[0..n_roots-1]
        for (int n = 0; n < n_roots; ++n) {
            theta_n[n] = eigenvalues[n]; // the nth smallest eigenvalue

            std::vector<T> s = eigenvectors.col(n);
            std::vector<T> residue_n = Ab_i * s;
            add_(residue_n, orthonormal_subspace * s, -theta_n[n]);
            double residue_norm = norm_(residue_n);

            fmt::println("  root {:>2}: theta = {:10.10f}  |residue| = {:10.10e}", n + 1, theta_n[n], residue_norm);

            if (residue_norm < residue_tol) {
                has_converged[n] = 1;
            }

            // update the search space
            for (int i = 0; i < n_dim; ++i) {
                xi_n[n][i] = residue_n[i] / (theta_n[n] - diagonal[i]);
            }
            auto xi_norm = norm_(xi_n[n]);
            std::transform(xi_n[n].begin(), xi_n[n].end(), xi_n[n].begin(),
                [xi_norm](T& val) { return val / xi_norm; });
        }
        search_space = orthonormal_subspace;
        search_space.conservativeResize(n_dim, M + n_roots);
        for (int n = 0; n < n_roots; ++n) {
            search_space.setCol(M + n, xi_n[n]);
        }
        if (std::all_of(has_converged.begin(), has_converged.end(),
                [](auto c) { return c; })) {
            fmt::println("davidson diagonalization converged in {:>2} iterations",
                iter + 1);
            return theta_n;
        }
    }
    throw std::runtime_error("Davidson diagonalization failed");
}

// The vector corresponding to the convergent root is not updated in each round
// maybe it is not friendly to a lower limit of convergence
template <typename Transformer, typename T = double>
const std::vector<double> davidson_solver_improved(Transformer transformer, const T* diagonal,
    std::size_t n_dim, std::size_t n_roots = 1,
    std::size_t start_dim = 5,
    std::size_t max_iter = 100,
    double residue_tol = 1e-6)
{

    if (start_dim < n_roots) {
        throw std::runtime_error(
            "start_dim should be greater than or equal to n_roots");
    }
    // initial guess
    Matrix<T> search_space = Matrix<T>::identity(n_dim, start_dim) + 0.01 * Matrix<T>::ones(n_dim, start_dim);

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

        std::vector<std::vector<T>> xi_n(n_roots, std::vector<T>(n_dim));
        std::vector<double> theta_n(n_roots);
        std::vector<bool> has_converged(n_roots, false);

        // find the index of n_roots smallest eigenvalues
        std::vector<std::pair<double, std::size_t>> eig_pairs;
        eig_pairs.reserve(eigenvalues.size());
        for (std::size_t i = 0; i < eigenvalues.size(); ++i) {
            eig_pairs.emplace_back(eigenvalues[i], i);
        }
        // sort the eigenvalues
        std::partial_sort(
            eig_pairs.begin(),
            eig_pairs.begin() + n_roots,
            eig_pairs.end(),
            [](auto const& a, auto const& b) {
                return a.first < b.first;
            });

        fmt::println("davidson diagonalization iter: {:>2} ", iter + 1);

        // eig_pairs[0..n_roots-1]
        for (int n = 0; n < n_roots; ++n) {
            theta_n[n] = eig_pairs[n].first; // the nth smallest eigenvalue
            std::size_t idx_n = eig_pairs[n].second; // corresponding column index

            std::vector<T> s = eigenvectors.col(idx_n);
            std::vector<T> residue_n = Ab_i * s;
            add_(residue_n, orthonormal_subspace * s, -theta_n[n]);
            double residue_norm = norm_(residue_n);

            fmt::println("  root {:>2}: theta = {:10.10f}  |residue| = {:10.10e}", n + 1, theta_n[n], residue_norm);

            if (residue_norm < residue_tol) {
                has_converged[n] = true;
                continue;
            }
            else {
                // update the search space
                for (int i = 0; i < n_dim; ++i) {
                    xi_n[n][i] = residue_n[i] / (theta_n[n] - diagonal[i]);
                }
                auto xi_norm = norm_(xi_n[n]);
                std::transform(xi_n[n].begin(), xi_n[n].end(), xi_n[n].begin(),
                    [xi_norm](T& val) { return val / xi_norm; });
            }
        }
        search_space = orthonormal_subspace;
        int add = std::count(has_converged.begin(), has_converged.end(), false);
        search_space.conservativeResize(n_dim, M + add);

        int new_col_index = 0;
        for (int n = 0; n < n_roots; ++n) {
            if (!has_converged[n]) {
                search_space.setCol(M + new_col_index, xi_n[n]);
                new_col_index++;
            }
        }
        if (add == 0) {
            fmt::println("davidson diagonalization converged in {:>2} iterations", iter + 1);
            return theta_n;
        }
    }
    throw std::runtime_error("Davidson diagonalization failed");
}

} // namespace linalg
#endif