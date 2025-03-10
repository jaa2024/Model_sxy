#pragma once
#ifndef MATRIX_HPP
#define MATRIX_HPP

// C++ standard library
#include "fmt/base.h"
#include <complex>
#include <cstddef>
#include <random>
// Third-party libraries
#include <fmt/core.h>
#include <mkl.h>

#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

namespace linalg {

template <typename T> class Matrix {
private:
  T *data_ = nullptr;                    // 数据指针
  CBLAS_LAYOUT layout_ = CblasColMajor;  // 存储布局
  CBLAS_TRANSPOSE trans_ = CblasNoTrans; // 转置状态
  std::size_t ncol_ = 0;                 // 实际列数
  std::size_t nrow_ = 0;                 // 实际行数
  std::size_t ld_ = 0;                   // leading dimension

public:
  // 构造函数
  Matrix(std::size_t rows, std::size_t cols,
         CBLAS_LAYOUT layout = CblasColMajor)
      : layout_(layout), ncol_(cols), nrow_(rows) {
    // 计算leading dimension
    ld_ = (layout_ == CblasColMajor) ? rows : cols;

    // MKL内存对齐分配
    data_ = static_cast<T *>(mkl_malloc(ld_ * ncol_ * sizeof(T), 64));
    if (!data_)
      throw std::bad_alloc();
  }
  // 析构函数
  ~Matrix() {
    if (data_) {
      mkl_free(data_);
      data_ = nullptr;
    }
  }

  // 拷贝构造函数
  Matrix(const Matrix &other)
      : layout_(other.layout_), trans_(other.trans_), ncol_(other.ncol_),
        nrow_(other.nrow_), ld_(other.ld_) {
    data_ = static_cast<T *>(mkl_malloc(ld_ * ncol_ * sizeof(T), 64));
    if (!data_)
      throw std::bad_alloc();
    // 使用 MKL 进行数据拷贝
    if constexpr (std::is_same_v<T, double>) {
      cblas_dcopy(ld_ * ncol_, other.data_, 1, data_, 1);
    } else if constexpr (std::is_same_v<T, float>) {
      cblas_scopy(ld_ * ncol_, other.data_, 1, data_, 1);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      cblas_zcopy(ld_ * ncol_, other.data_, 1, data_, 1);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      cblas_ccopy(ld_ * ncol_, other.data_, 1, data_, 1);
    }
  }

  // 移动构造函数
  Matrix(Matrix &&other) noexcept
      : data_(other.data_), layout_(other.layout_), trans_(other.trans_),
        ncol_(other.ncol_), nrow_(other.nrow_), ld_(other.ld_) {
    other.data_ = nullptr;
  }

  // 元素访问 (考虑转置和布局)
  T &operator()(std::size_t i, std::size_t j) {
    if (trans_ == CblasTrans)
      std::swap(i, j);
    return layout_ == CblasColMajor ? data_[i + j * ld_] : data_[i * ld_ + j];
  }

  const T &operator()(std::size_t i, std::size_t j) const {
    if (trans_ == CblasTrans)
      std::swap(i, j);
    return layout_ == CblasColMajor ? data_[i + j * ld_] : data_[i * ld_ + j];
  }

  // 获取矩阵维度信息
  std::size_t rows() const { return (trans_ == CblasNoTrans) ? nrow_ : ncol_; }
  std::size_t cols() const { return (trans_ == CblasNoTrans) ? ncol_ : nrow_; }

  // 获取转置矩阵
  Matrix<T> transpose() {
    Matrix view(*this);
    view.trans_ = (trans_ == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    std::swap(view.nrow_, view.ncol_);
    return view;
  }

  // 生成全零矩阵
  static Matrix zero(std::size_t rows, std::size_t cols,
                     CBLAS_LAYOUT layout = CblasColMajor) {
    Matrix mat(rows, cols, layout);
    std::size_t total = mat.ld_ * mat.ncol_;
    if constexpr (std::is_floating_point_v<T>) {
      cblas_sscal(total, 0.0f, mat.data_, 1);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dscal(total, 0.0, mat.data_, 1);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      MKL_Complex8 alpha{0.0f, 0.0f};
      cblas_cscal(total, &alpha, mat.data_, 1);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      MKL_Complex16 alpha{0.0, 0.0};
      cblas_zscal(total, &alpha, mat.data_, 1);
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported element type");
    }
    return mat;
  }

  // 生成单位矩阵
  static Matrix identity(std::size_t n, CBLAS_LAYOUT layout = CblasColMajor) {
    Matrix mat = zero(n, n, layout);
    for (std::size_t i = 0; i < n; ++i) {
      mat(i, i) = T{1};
    }
    return mat;
  }

  // 生成随机矩阵（均匀分布）
  static Matrix random(std::size_t rows, std::size_t cols,
                       CBLAS_LAYOUT layout = CblasColMajor, T min = T{-1},
                       T max = T{1}) {
    Matrix mat(rows, cols, layout);
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, std::random_device{}());

    const std::size_t total = mat.ld_ * mat.ncol_;

    // 实数类型处理
    if constexpr (std::is_same_v<T, float>) {
      vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, total, mat.data_, min,
                   max);
    } else if constexpr (std::is_same_v<T, double>) {
      vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, total, mat.data_, min,
                   max);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * total,
                   reinterpret_cast<float *>(mat.data_), std::real(min),
                   std::real(max));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * total,
                   reinterpret_cast<double *>(mat.data_), std::real(min),
                   std::real(max));
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported element type");
    }

    vslDeleteStream(&stream);
    return mat;
  }

  void print() const {
    // 格式化输出实现
    for (std::size_t i = 0; i < rows(); ++i) {
      fmt::print("["); // 行起始符
      for (std::size_t j = 0; j < cols(); ++j) {
        // 类型分发格式化
        if constexpr (std::is_same_v<T, float>) {
          fmt::print("{:>8.4f}", (*this)(i, j));
        } else if constexpr (std::is_same_v<T, double>) {
          fmt::print("{:>10.4f}", (*this)(i, j));
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
          fmt::print("({:>6.2f}, {:>6.2f})", (*this)(i, j).real(),
                     (*this)(i, j).imag());
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
          fmt::print("({:>8.4f}, {:>8.4f})", (*this)(i, j).real(),
                     (*this)(i, j).imag());
        }
        // 列分隔符
        if (j != cols() - 1)
          fmt::print(", ");
      }
      fmt::print("]\n"); // 行终止符
    }
  }

  // 矩阵加法
  Matrix<T> operator+(const Matrix<T> &other) const {
    if (this->rows() != other.rows() || this->cols() != other.cols()) {
      throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    if (this->layout_ != other.layout_) {
      throw std::invalid_argument("Matrix layouts must match for addition");
    }
    Matrix<T> C(this->rows(), this->cols(), this->layout_);
    std::size_t total = this->ld_ * this->ncol_;

    if constexpr (std::is_same_v<T, double>) {
      vdAdd(total, this->data_, other.data_, C.data_);
    } else if constexpr (std::is_same_v<T, float>) {
      vsAdd(total, this->data_, other.data_, C.data_);
    } else {
      for (std::size_t i = 0; i < total; ++i) {
        C.data_[i] = this->data_[i] + other.data_[i];
      }
    }
    return C;
  }

  // 矩阵减法
  Matrix<T> operator-(const Matrix<T> &other) const {
    if (this->rows() != other.rows() || this->cols() != other.cols()) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction");
    }
    if (this->layout_ != other.layout_) {
      throw std::invalid_argument("Matrix layouts must match for subtraction");
    }
    Matrix<T> C(this->rows(), this->cols(), this->layout_);
    std::size_t total = this->ld_ * this->ncol_;

    if constexpr (std::is_same_v<T, double>) {
      vdSub(total, this->data_, other.data_, C.data_);
    } else if constexpr (std::is_same_v<T, float>) {
      vsSub(total, this->data_, other.data_, C.data_);
    } else {
      for (std::size_t i = 0; i < total; ++i) {
        C.data_[i] = this->data_[i] - other.data_[i];
      }
    }
    return C;
  }

  // 矩阵乘法
  friend Matrix<T> operator*(const Matrix<T> &A, const Matrix<T> &B) {
    if (A.cols() != B.rows()) {
      throw std::invalid_argument(
          "Matrix dimensions must match for multiplication");
    }
    if (A.layout_ != B.layout_) {
      throw std::invalid_argument(
          "Matrix layouts must match for multiplication");
    }
    Matrix<T> C(A.rows(), B.cols(), A.layout_);

    if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(A.layout_, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                  A.cols(), 1.0, A.data_, A.ld_, B.data_, B.ld_, 0.0, C.data_,
                  C.ld_);
    } else if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(A.layout_, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                  A.cols(), 1.0f, A.data_, A.ld_, B.data_, B.ld_, 0.0f, C.data_,
                  C.ld_);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      MKL_Complex16 alpha{0.0f, 0.0f};
      MKL_Complex16 beta{0.0f, 0.0f};
      cblas_zgemm(A.layout_, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                  A.cols(), &alpha, A.data_, A.ld_, B.data_, B.ld_, &beta,
                  C.data_, C.ld_);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      MKL_Complex8 alpha{0.0f, 0.0f};
      MKL_Complex8 beta{0.0f, 0.0f};
      cblas_cgemm(A.layout_, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                  A.cols(), &alpha, A.data_, A.ld_, B.data_, B.ld_, &beta,
                  C.data_, C.ld_);
    } else {
      for (std::size_t i = 0; i < C.rows(); ++i) {
        for (std::size_t j = 0; j < C.cols(); ++j) {
          C(i, j) = 0;
          for (std::size_t k = 0; k < A.cols(); ++k) {
            C(i, j) += A(i, k) * B(k, j);
          }
        }
      }
    }
    return C;
  }
};
} // namespace linalg

#endif // MATRIX_HPP