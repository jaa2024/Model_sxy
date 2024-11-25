#include "cmatrix.h"

int main()
{
    // 创建两个矩阵
    CMatrix A(2, 3);
    CMatrix B(3, 2);

    // 初始化矩阵 A
    A(0, 0) = { 1.0, 1.0 };
    A(0, 1) = { 2.0, -1.0 };
    A(0, 2) = { 3.0, 0.0 };
    A(1, 0) = { 4.0, -1.0 };
    A(1, 1) = { 5.0, 1.0 };
    A(1, 2) = { 6.0, 0.0 };

    // 初始化矩阵 B
    B(0, 0) = { 7.0, 0.0 };
    B(0, 1) = { 8.0, -1.0 };
    B(1, 0) = { 9.0, 1.0 };
    B(1, 1) = { 10.0, 0.0 };
    B(2, 0) = { 11.0, 0.0 };
    B(2, 1) = { 12.0, 1.0 };

    // 矩阵乘法
    try {
        CMatrix C = CMatrix::multiply(A, B);

        // 输出结果
        for (size_t i = 0; i < C.getRows(); ++i) {
            for (size_t j = 0; j < C.getCols(); ++j) {
                std::cout << C(i, j) << " ";
            }
            std::cout << std::endl;
        }
        C.transpose_inplace();
        // 输出结果
        std::cout << "C.T():" << std::endl;
        for (size_t i = 0; i < C.getRows(); ++i) {
            for (size_t j = 0; j < C.getCols(); ++j) {
                std::cout << C(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "C.conj():" << std::endl;
        C.conj_inplace();
        for (size_t i = 0; i < C.getRows(); ++i) {
            for (size_t j = 0; j < C.getCols(); ++j) {
                std::cout << C(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}