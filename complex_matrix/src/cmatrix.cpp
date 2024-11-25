#include "cmatrix.h"
#include <cassert>
#include <iostream>

CMatrix::CMatrix(int rows, int cols, bool rowMajor)
    : rows(rows), cols(cols), isRowMajor(rowMajor), data(rows * cols, std::complex<double>(0.0, 0.0)), trans_(CblasNoTrans) { }

std::complex<double>& CMatrix::operator()(int i, int j)
{
    if (i >= rows || j >= cols)
        throw std::out_of_range("Index out of range");
    return isRowMajor ? data[i * cols + j] : data[j * rows + i];
}

const std::complex<double>& CMatrix::operator()(int i, int j) const
{
    if (i >= rows || j >= cols)
        throw std::out_of_range("Index out of range");
    return isRowMajor ? data[i * cols + j] : data[j * rows + i];
}

int CMatrix::getRows() const { return rows; }
int CMatrix::getCols() const { return cols; }
bool CMatrix::isRowMajorOrder() const { return isRowMajor; }

CMatrix CMatrix::multiply(const CMatrix& A, const CMatrix& B)
{
    size_t m = (A.trans_ == CblasNoTrans) ? A.rows : A.cols;
    size_t k = (A.trans_ == CblasNoTrans) ? A.cols : A.rows;
    size_t n = (B.trans_ == CblasNoTrans) ? B.cols : B.rows;

    if ((A.trans_ == CblasNoTrans ? A.cols : A.rows) != (B.trans_ == CblasNoTrans ? B.rows : B.cols)) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    CMatrix C(m, n, A.isRowMajor);

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

#ifdef __USE_MKL__
    CBLAS_LAYOUT layout = A.isRowMajor ? CblasRowMajor : CblasColMajor;
#else
    CBLAS_ORDER layout = A.isRowMajor ? CblasRowMajor : CblasColMajor;
#endif

    cblas_zgemm(layout, A.trans_, B.trans_,
        m, n, k,
        &alpha,
        A.data.data(), A.isRowMajor ? A.cols : A.rows,
        B.data.data(), B.isRowMajor ? B.cols : B.rows,
        &beta,
        C.data.data(), C.isRowMajor ? C.cols : C.rows);

    return C;
}

std::complex<double>& CMatrix::operator()(size_t i, size_t j)
{
    if (i >= rows || j >= cols)
        throw std::out_of_range("Index out of range");

    if (trans_ == CblasNoTrans) {
        return isRowMajor ? data[i * cols + j] : data[j * rows + i];
    }
    else {
        return isRowMajor ? data[j * rows + i] : data[i * cols + j];
    }
}

const std::complex<double>& CMatrix::operator()(size_t i, size_t j) const
{
    if (i >= rows || j >= cols)
        throw std::out_of_range("Index out of range");

    if (trans_ == CblasNoTrans) {
        return isRowMajor ? data[i * cols + j] : data[j * rows + i];
    }
    else {
        return isRowMajor ? data[j * rows + i] : data[i * cols + j];
    }
}

CMatrix CMatrix::operator*(const CMatrix& other) const
{
    return multiply(*this, other);
}

CMatrix CMatrix::transpose() const
{
    CMatrix tmp(*this);
    tmp.trans_ = (tmp.trans_ == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    std::swap(tmp.rows, tmp.cols);
    return tmp;
}
CMatrix& CMatrix::transpose_inplace()
{
    if (rows != cols) {
        throw std::runtime_error("In-place transpose only works for square matrices");
    }

    this->trans_ = (this->trans_ == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    std::swap(this->rows, this->cols);
    return *this;
}

CMatrix CMatrix::conj() const
{
    CMatrix result(rows, cols, isRowMajor);

#ifdef __USE_MKL__
    // MKL
    vzConj(data.size(),
        reinterpret_cast<const MKL_Complex16*>(data.data()),
        reinterpret_cast<MKL_Complex16*>(result.data.data()));
#else
    // OpenBLAS
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = std::conj(data[i]);
    }
#endif

    return result;
}

CMatrix& CMatrix::conj_inplace()
{
#ifdef __USE_MKL__
    // MKL
    vzConj(data.size(),
        reinterpret_cast<const MKL_Complex16*>(data.data()),
        reinterpret_cast<MKL_Complex16*>(data.data()));
#else
    // OpenBLAS
    for (auto& element : data) {
        element = std::conj(element);
    }
#endif

    return *this;
}
