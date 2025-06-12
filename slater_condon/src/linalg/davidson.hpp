#pragma once
#include <vector>
#ifndef DAVIDSON_HPP
#define DAVIDSON_HPP

#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include <algorithm>
#include <chrono>

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

// The first freeze_cols column keeps the input values unchanged
// while the subsequent ones have been orthogonized
template <typename T>
Matrix<T> gramschmidt_incremental(const Matrix<T>& X, int freeze_cols)
{
    Matrix<T> Y = X;
    const int cols = Y.cols();
    const int rows = Y.rows();

    // only handle the vectors after the freeze_cols column
    for (int i = freeze_cols; i < cols; ++i) {
        std::vector<T> col_i = Y.col(i); // take out the i-th column

        // project to all columns earlier than it, including the frozen 0... column (i-1)
        for (int j = 0; j < i; ++j) {
            const std::vector<T>& col_j = Y.col(j);

            T denom = dot_(col_j, col_j);
            if (std::abs(denom) < 1e-14)
                continue;

            T coeff = dot_(col_i, col_j) / denom;
            add_(col_i, col_j, -coeff);
        }

        // normalization
        double nrm = norm_(col_i);
        if (nrm > 1e-7) {
            dot_(col_i, T(1.0 / nrm));
        }

        Y.setCol(i, col_i);
    }

    return Y;
}

template <typename T>
struct p_space {
    std::size_t n_front;
    // TODO: sparse matrix only need idx
    // std::vector<std::size_t> idx;
    Matrix<T> pspace;
};

template <typename T>
p_space<T> make_initial_pspace(const std::vector<T>& diag, std::size_t n_pspace, double front_ratio = 0.5)
{
    const std::size_t N = diag.size();
    if (n_pspace == 0 || n_pspace > N)
        throw std::invalid_argument("n_roots out of range");
    if (front_ratio < 0.0 || front_ratio > 1.0)
        throw std::invalid_argument("front_ratio must be in [0,1]");

    // determine the front and back
    std::size_t n_front = static_cast<std::size_t>(std::round(n_pspace * front_ratio));
    if (n_front > n_pspace)
        n_front = n_pspace;
    std::size_t n_back = n_pspace - n_front;

    // initialize the matrix V with the first n_front columns of the identity matrix
    Matrix<T> V = Matrix<T>::zero(N, n_pspace);
    for (std::size_t k = 0; k < n_front; ++k)
        V(k, k) = T(1);

    // find the smallest n_roots subscript
    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    idx.erase(std::remove_if(idx.begin(), idx.end(),
                  [n_front](std::size_t i) { return i < n_front; }),
        idx.end());

    std::partial_sort(idx.begin(), idx.begin() + n_pspace, idx.end(),
        [&](std::size_t a, std::size_t b) { return diag[a] < diag[b]; });

    // print the initial guess for test
    // for (std::size_t i = 0; i < n_pspace; ++i) {
    //     fmt::println("initial guess {:>2}: {:>2}  {:>10.10f}", i + 1, idx[i], diag[idx[i]]);
    // }

    // generate the back unit basis matrix
    for (std::size_t k = 0; k < n_back; ++k)
        V(idx[k], n_front + k) = T(1);
    fmt::println("{:>3}, {:>3} ", n_front, n_back);
    return { n_front, V };
}

template <typename Transformer, typename T = double>
const std::vector<double> davidson_solver(Transformer transformer, const std::vector<T>& diagonal,
    std::size_t n_dim, std::size_t n_roots = 1,
    std::size_t n_pspace = 5,
    std::size_t max_iter = 1000,
    double residue_tol = 1e-10)
{

    if (n_pspace < n_roots) {
        throw std::runtime_error(
            "start_dim should be greater than or equal to n_roots");
    }
    // initial guess
    const p_space<T> p_space = make_initial_pspace(diagonal, n_pspace, 0.9);
    Matrix<T> search_space = p_space.pspace;
    // Matrix<T> search_space = Matrix<T>::identity(n_dim, n_pspace);
    Matrix<T> Ab_i = Matrix<T>::zero(n_dim, n_pspace);
    double residue_norm_n = n_roots;
    double residue_norm_npre = n_roots;
    // start iteration
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // project dim
        auto M = n_pspace + iter * n_roots;
        // const bool do_full_ortho = (iter == 0) || (iter % 10 == 0) || (residue_norm_n > residue_norm_npre);
        const bool do_full_ortho = (iter == 0) || (residue_norm_n > residue_norm_npre);
        residue_norm_npre = residue_norm_n;
        residue_norm_n = 0.0;
        Matrix<T> orthonormal_subspace;
        Ab_i.conservativeResize(n_dim, M);
        if (do_full_ortho) {
            if (iter == 0) {
                orthonormal_subspace = search_space;
                for (std::size_t j = 0; j < M; j++) {
                    auto vec = orthonormal_subspace.col(j);
                    Ab_i.setCol(j, transformer(vec));
                }
            }
            else {
                orthonormal_subspace = gramschmidt_incremental(search_space, n_pspace);
                for (std::size_t j = n_pspace; j < M; j++) {
                    auto vec = orthonormal_subspace.col(j);
                    Ab_i.setCol(j, transformer(vec));
                }
            }
            // orthonormal_subspace = gramschmidt(search_space);
            // for (std::size_t j = 0; j < M; j++) {
            //     auto vec = orthonormal_subspace.col(j);
            //     Ab_i.setCol(j, transformer(vec));
            // }
        }
        else {
            orthonormal_subspace = gramschmidt_incremental(search_space, M - n_roots);
            for (std::size_t j = M - n_roots; j < M; j++) {
                auto vec = orthonormal_subspace.col(j);
                Ab_i.setCol(j, transformer(vec));
            }
        }

        Matrix<double> B = orthonormal_subspace.transpose().conjugate() * Ab_i;
        auto [eigenvalues, eigenvectors] = eigh(B);

        std::vector<std::vector<T>> xi_n(n_roots, std::vector<T>(n_dim, T(0)));
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
            residue_norm_n += residue_norm;

            fmt::println("  root {:>2}: theta = {:10.10f}  |residue| = {:10.10e}", n + 1, theta_n[n], residue_norm);

            if (residue_norm < residue_tol) {
                has_converged[n] = 1;
            }

            // update the search space
            for (int i = p_space.n_front; i < n_dim; ++i) {
                xi_n[n][i] = residue_n[i] / (theta_n[n] - diagonal[i]);
            }
            // for (int i = 0; i < n_dim; ++i) {
            //     xi_n[n][i] = residue_n[i] / (theta_n[n] - diagonal[i]);
            // }
            auto xi_norm = norm_(xi_n[n]);
            std::transform(xi_n[n].begin(), xi_n[n].end(), xi_n[n].begin(),
                [xi_norm](T& val) { return val / xi_norm; });
        }

        if (std::all_of(has_converged.begin(), has_converged.end(),
                [](auto c) { return c; })) {
            fmt::println("davidson diagonalization converged in {:>2} iterations",
                iter + 1);
            return theta_n;
        }

        search_space = orthonormal_subspace;
        search_space.conservativeResize(n_dim, M + n_roots);
        for (int n = 0; n < n_roots; ++n) {
            search_space.setCol(M + n, xi_n[n]);
        }
    }
    throw std::runtime_error("Davidson diagonalization failed");
}

template <typename Transformer, typename T = double>
const std::vector<double> davidson_solver_s(Transformer transformer, const std::vector<T>& diagonal,
    std::size_t n_dim, std::size_t n_roots = 1,
    std::size_t n_pspace = 5,
    std::size_t max_iter = 1000,
    double residue_tol = 1e-10)
{

    if (n_pspace < n_roots) {
        throw std::runtime_error(
            "start_dim should be greater than or equal to n_roots");
    }
    // initial guess
    Matrix<T> search_space = make_initial_pspace(diagonal, n_pspace, 0.5);
    Matrix<T> Ab_i = Matrix<T>::zero(n_dim, n_pspace);
    double residue_norm_n = n_roots;
    double residue_norm_npre = n_roots;
    // start iteration
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // project dim
        auto M = n_pspace + iter;
        // const bool do_full_ortho = (iter == 0) || (iter % 10 == 0) || (residue_norm_n > residue_norm_npre);
        const bool do_full_ortho = (iter == 0) || residue_norm_n > residue_norm_npre;
        residue_norm_npre = residue_norm_n;
        residue_norm_n = 0.0;

        Matrix<T> orthonormal_subspace;
        Ab_i.conservativeResize(n_dim, M);
        if (do_full_ortho) {
            if (iter == 0) {
                orthonormal_subspace = search_space;
                for (std::size_t j = 0; j < M; j++) {
                    auto vec = orthonormal_subspace.col(j);
                    Ab_i.setCol(j, transformer(vec));
                }
            }
            else {
                orthonormal_subspace = gramschmidt_incremental(search_space, n_pspace);
                for (std::size_t j = n_pspace; j < M; j++) {
                    auto vec = orthonormal_subspace.col(j);
                    Ab_i.setCol(j, transformer(vec));
                }
            }
            // orthonormal_subspace = gramschmidt(search_space);
            // for (std::size_t j = 0; j < M; j++) {
            //     auto vec = orthonormal_subspace.col(j);
            //     Ab_i.setCol(j, transformer(vec));
            // }
        }
        else {
            orthonormal_subspace = gramschmidt_incremental(search_space, M - 1);
            auto vec = orthonormal_subspace.col(M - 1);
            Ab_i.setCol(M - 1, transformer(vec));
        }

        Matrix<double> B = orthonormal_subspace.transpose().conjugate() * Ab_i;
        auto [eigenvalues, eigenvectors] = eigh(B);

        std::vector<std::vector<T>> xi_n(n_roots, std::vector<T>(n_dim));
        std::vector<double> theta_n(n_roots);
        std::vector<std::uint8_t> has_converged(n_roots, 0);
        std::vector<double> residue_norm_n_real(n_roots);

        fmt::println("davidson diagonalization iter: {:>2} ", iter + 1);

        // eig_pairs[0..n_roots-1]
        for (int n = 0; n < n_roots; ++n) {
            theta_n[n] = eigenvalues[n]; // the nth smallest eigenvalue

            std::vector<T> s = eigenvectors.col(n);
            std::vector<T> residue_n = Ab_i * s;
            add_(residue_n, orthonormal_subspace * s, -theta_n[n]);
            double residue_norm = norm_(residue_n);
            residue_norm_n += residue_norm;
            residue_norm_n_real[n] = residue_norm;

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

        if (std::all_of(has_converged.begin(), has_converged.end(),
                [](auto c) { return c; })) {
            fmt::println("davidson diagonalization converged in {:>2} iterations",
                iter + 1);
            return theta_n;
        }
        // update the search space
        std::vector<T> xi_n_av(n_dim, 0);
        for (int n = 0; n < n_roots; ++n) {
            add_(xi_n_av, xi_n[n], residue_norm_n_real[n] / residue_norm_n);
        }
        search_space = orthonormal_subspace;
        search_space.conservativeResize(n_dim, M + 1);
        search_space.setCol(M, xi_n_av);
    }
    throw std::runtime_error("Davidson diagonalization failed");
}

} // namespace linalg
#endif