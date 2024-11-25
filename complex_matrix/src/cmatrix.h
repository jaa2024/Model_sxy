// cmatrix.h
#ifndef CMATRIX_H
#define CMATRIX_H

#ifdef __USE_MKL__
#include <mkl.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <cassert>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace cmatrix {

class CMatrix {
private:
    int rows, cols;
    std::vector<std::complex<double>> data;
    bool isRowMajor;

#ifdef __USE_MKL__
    CBLAS_TRANSPOSE trans_; // MKL
#else
    CBLAS_TRANSPOSE trans_; // OpenBLAS
#endif

public:
    CMatrix(int rows, int cols, bool rowMajor = false);

    // 元素访问
    std::complex<double>& operator()(int i, int j);
    const std::complex<double>& operator()(int i, int j) const;

    // 获取行数和列数
    int getRows() const;
    int getCols() const;
    bool isRowMajorOrder() const;

    // 矩阵乘法
    static CMatrix multiply(const CMatrix& A, const CMatrix& B);

    std::complex<double>& operator()(size_t i, size_t j);
    const std::complex<double>& operator()(size_t i, size_t j) const;

    // 运算符重载
    CMatrix operator*(const CMatrix& other) const;

    // 转置操作
    CMatrix transpose() const;
    CMatrix& transpose_inplace(); // 原地转置
    CMatrix T() const { return transpose(); } // T()作为transpose()的简写

    // 共轭操作
    CMatrix conj() const;
    CMatrix& conj_inplace(); // 原地共轭转换

    // 获取原始数据指针（用于LAPACK）
    std::complex<double>* data_ptr() { return data.data(); }
    const std::complex<double>* data_ptr() const { return data.data(); }
};
}

#endif // CMATRIX_H