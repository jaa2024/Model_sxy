#include "cmatrix.h"
#include <cassert>
#include <iostream>

using namespace cmatrix;
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

void CMatrix::print()
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::complex<double> val = (*this)(i, j);
            // 使用 std::format 格式化复数，显示实部和虚部
            std::cout << std::format("{: .8f} {: .8f}i ", val.real(), val.imag());

            // 每两个元素换一行
            if ((j + 1) % 2 == 0) {
                std::cout << std::endl; // 换行
            }
        }
        std::cout << std::endl; // 每行结束后再换行
    }
}

CMatrix CMatrix::Random(int rows, int cols, bool rowMajor)
{
    // 创建一个随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    // 创建矩阵
    CMatrix result(rows, cols, rowMajor);

    // 为矩阵的每个元素生成随机值
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double real_part = dis(gen); // 实部
            double imag_part = dis(gen); // 虚部
            result(i, j) = { real_part, imag_part };
        }
    }
    return result;
}
void CMatrix::Random()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    // 生成一个随机的厄密矩阵
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j <= i; ++j) { // 只填充矩阵的上三角部分
            double real_part = dis(gen); // 实部
            double imag_part = (i == j) ? 0.0 : dis(gen); // 对角线元素的虚部为0
            (*this)(i, j) = { real_part, imag_part };

            // 填充下三角部分，确保矩阵是厄密的
            if (i != j) {
                (*this)(j, i) = { real_part, -imag_part }; // 共轭对称
            }
        }
    }
}

int CMatrix::getRows() const
{
    return rows;
}
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
// 矩阵减法
CMatrix CMatrix::operator-(const CMatrix& other) const
{
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }

    CMatrix result(rows, cols, isRowMajor);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
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

CMatrix CMatrix::getBlock(int rowOffset, int colOffset, int blockRows, int blockCols) const
{
    // 检查边界条件
    if (rowOffset < 0 || colOffset < 0 || rowOffset + blockRows > rows || colOffset + blockCols > cols) {
        throw std::out_of_range("Block dimensions exceed matrix bounds");
    }

    // 创建结果矩阵，使用与源矩阵相同的布局
    CMatrix result(blockRows, blockCols, isRowMajor);

    // 考虑转置标志
    int actualRows = (trans_ == CblasNoTrans) ? blockRows : blockCols;
    int actualCols = (trans_ == CblasNoTrans) ? blockCols : blockRows;

    for (int i = 0; i < actualRows; ++i) {
        for (int j = 0; j < actualCols; ++j) {
            // 根据转置状态确定源矩阵的索引
            int srcI = (trans_ == CblasNoTrans) ? i : j;
            int srcJ = (trans_ == CblasNoTrans) ? j : i;

            if (isRowMajor) {
                result.data[i * result.cols + j] = data[(rowOffset + srcI) * cols + (colOffset + srcJ)];
            }
            else {
                result.data[j * result.rows + i] = data[(colOffset + srcJ) * rows + (rowOffset + srcI)];
            }
        }
    }

    // 设置结果矩阵的转置状态为正常（未转置）
    result.trans_ = CblasNoTrans;
    return result;
}

void CMatrix::setBlock(int rowOffset, int colOffset, const CMatrix& block)
{
    // 检查边界条件
    if (rowOffset < 0 || colOffset < 0 || rowOffset + block.rows > rows || colOffset + block.cols > cols) {
        throw std::out_of_range("Block dimensions exceed matrix bounds");
    }

    // 考虑转置标志
    int actualRows = (trans_ == CblasNoTrans) ? block.rows : block.cols;
    int actualCols = (trans_ == CblasNoTrans) ? block.cols : block.rows;

    // 获取源矩阵的有效行列数
    int srcRows = (block.trans_ == CblasNoTrans) ? block.rows : block.cols;
    int srcCols = (block.trans_ == CblasNoTrans) ? block.cols : block.rows;

    for (int i = 0; i < actualRows; ++i) {
        for (int j = 0; j < actualCols; ++j) {
            // 根据源矩阵的转置状态确定索引
            int srcI = (block.trans_ == CblasNoTrans) ? i : j;
            int srcJ = (block.trans_ == CblasNoTrans) ? j : i;

            // 根据目标矩阵的转置状态确定索引
            int destI = (trans_ == CblasNoTrans) ? i : j;
            int destJ = (trans_ == CblasNoTrans) ? j : i;

            // 设置元素值，考虑行/列主序
            if (isRowMajor) {
                data[(rowOffset + destI) * cols + (colOffset + destJ)] = block.isRowMajor ? block.data[srcI * block.cols + srcJ] : block.data[srcJ * block.rows + srcI];
            }
            else {
                data[(colOffset + destJ) * rows + (rowOffset + destI)] = block.isRowMajor ? block.data[srcI * block.cols + srcJ] : block.data[srcJ * block.rows + srcI];
            }
        }
    }
}

CMatrix CMatrix::convertToLayout(bool targetRowMajor) const
{
    if (isRowMajor == targetRowMajor && trans_ == CblasNoTrans) {
        return *this; // 如果布局已经正确且未转置，直接返回
    }

    CMatrix result(rows, cols, targetRowMajor);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i, j);
        }
    }
    result.trans_ = CblasNoTrans; // 确保结果矩阵未转置
    return result;
}

bool CMatrix::isHermitian() const
{
    if (rows != cols)
        return false;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (std::abs((*this)(i, j) - std::conj((*this)(j, i))) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

std::pair<std::vector<double>, CMatrix> CMatrix::eigh(const CMatrix& matrix)
{
    if (!matrix.isHermitian()) {
        throw std::invalid_argument("Matrix must be Hermitian");
    }

    // Define types based on BLAS implementation

#ifdef __USE_MKL__

#else
    using lapack_int = int;
    using complex_t = double _Complex;
#endif

    lapack_int n = static_cast<lapack_int>(matrix.rows);
    if (n != static_cast<lapack_int>(matrix.cols)) {
        throw std::invalid_argument("Matrix must be square");
    }

    // Prepare output arrays
    std::vector<double> eigenvalues(n);

    // Convert input matrix to column-major format if needed
    CMatrix work_matrix = matrix.convertToLayout(false); // false for column-major

    // Workspace query
    lapack_int lwork = -1;
    std::complex<double> work_size;
    std::vector<double> rwork(std::max(1, 3 * n - 2));
    lapack_int info;

    // Query optimal workspace size
    const char jobz = 'V'; // Compute eigenvalues and eigenvectors
    const char uplo = 'U'; // Upper triangle of matrix is stored

#ifdef __USE_MKL__
    zheev_(&jobz, &uplo, &n,
        reinterpret_cast<MKL_Complex16*>(work_matrix.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<MKL_Complex16*>(&work_size), &lwork,
        rwork.data(), &info);
#else
    zheev_(&jobz, &uplo, &n,
        reinterpret_cast<complex_t*>(work_matrix.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<complex_t*>(&work_size), &lwork,
        rwork.data(), &info,
        1, 1); // 字符串参数的长度
#endif

    if (info != 0) {
        throw std::runtime_error("Workspace query failed with error: " + std::to_string(info));
    }

    // Allocate optimal workspace
    lwork = static_cast<lapack_int>(std::real(work_size));
    std::vector<std::complex<double>> work(lwork);

    // Compute eigenvalues and eigenvectors
#ifdef __USE_MKL__
    zheev_(&jobz, &uplo, &n,
        reinterpret_cast<MKL_Complex16*>(work_matrix.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<MKL_Complex16*>(work.data()), &lwork,
        rwork.data(), &info);
#else
    zheev_(&jobz, &uplo, &n,
        reinterpret_cast<complex_t*>(work_matrix.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<complex_t*>(work.data()), &lwork,
        rwork.data(), &info,
        1, 1); // 字符串参数的长度
#endif

    // Error handling
    if (info < 0) {
        throw std::runtime_error("Invalid argument at position " + std::to_string(-info));
    }
    else if (info > 0) {
        throw std::runtime_error("Algorithm failed to converge. " + std::to_string(info) + " off-diagonal elements did not converge");
    }

    // Convert result back to the same layout as input matrix if needed
    CMatrix eigenvectors = work_matrix.convertToLayout(matrix.isRowMajor);

    return { eigenvalues, eigenvectors };
}

std::pair<std::vector<double>, CMatrix> CMatrix::eigh(const CMatrix& matrix, const CMatrix& overlap)
{
    if (!matrix.isHermitian() || !overlap.isHermitian()) {
        throw std::invalid_argument("Both matrices must be Hermitian");
    }
    if (matrix.rows != overlap.rows || matrix.cols != overlap.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    if (matrix.isRowMajor != overlap.isRowMajor) {
        throw std::invalid_argument("Matrices must have the same layout");
    }

#ifndef __USE_MKL__
    using lapack_int = int;
    using complex_t = double _Complex;
#endif

    lapack_int n = static_cast<lapack_int>(matrix.rows);
    std::vector<double> eigenvalues(n);

    // 将输入矩阵转换为LAPACK所需的列主序格式
    CMatrix work_matrix_a = matrix.convertToLayout(false);
    CMatrix work_matrix_b = overlap.convertToLayout(false);

    // LAPACK工作数组
    lapack_int lwork = -1; // 查询最优工作空间大小
    std::complex<double> work_size;
    std::vector<double> rwork(3 * n);
    lapack_int info;
    lapack_int itype = 1; // 标准特征值问题类型: A*x = lambda*B*x

    // 查询最优工作空间
#ifdef __USE_MKL__
    zhegv_(&itype, "V", "U", &n,
        reinterpret_cast<MKL_Complex16*>(work_matrix_a.data.data()),
        &n,
        reinterpret_cast<MKL_Complex16*>(work_matrix_b.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<MKL_Complex16*>(&work_size), &lwork,
        rwork.data(), &info);
#else
    zhegv_(&itype, "V", "U", &n,
        reinterpret_cast<double _Complex*>(work_matrix_a.data.data()),
        &n,
        reinterpret_cast<double _Complex*>(work_matrix_b.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<double _Complex*>(&work_size), &lwork,
        rwork.data(), &info, 1, 1);
#endif

    // 分配最优大小的工作空间
    lwork = static_cast<lapack_int>(std::real(work_size));
    std::vector<std::complex<double>> work(lwork);

    // 计算特征值和特征向量
#ifdef __USE_MKL__
    zhegv_(&itype, "V", "U", &n,
        reinterpret_cast<MKL_Complex16*>(work_matrix_a.data.data()),
        &n,
        reinterpret_cast<MKL_Complex16*>(work_matrix_b.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<MKL_Complex16*>(work.data()), &lwork,
        rwork.data(), &info);
#else
    zhegv_(&itype, "V", "U", &n,
        reinterpret_cast<double _Complex*>(work_matrix_a.data.data()),
        &n,
        reinterpret_cast<double _Complex*>(work_matrix_b.data.data()),
        &n, eigenvalues.data(),
        reinterpret_cast<double _Complex*>(work.data()), &lwork,
        rwork.data(), &info, 1, 1);
#endif

    if (info != 0) {
        throw std::runtime_error("Failed to compute eigenvalues/eigenvectors: " + std::to_string(info));
    }

    // 将结果转换回与输入矩阵相同的布局
    CMatrix result = work_matrix_a.convertToLayout(matrix.isRowMajor);

    return { eigenvalues, result };
}