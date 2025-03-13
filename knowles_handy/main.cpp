#include "ci/kh.hpp"
#include "integral/integral.hpp"
#include <chrono>

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

int main() {

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

  return 0;
}