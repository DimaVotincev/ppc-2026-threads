#include "../../omp/include/ops_omp.hpp"

#include <atomic>
#include <numeric>
#include <vector>

#include "omp.h"
#include "shvetsova_k_mult_matrix_complex_col/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_mult_matrix_complex_col {

ShvetsovaKMultMatrixComplexOMP::ShvetsovaKMultMatrixComplexOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = MatrixCCS(0, 0, std::vector<int>{0}, std::vector<int>{}, std::vector<std::complex<double>>{});
}

bool ShvetsovaKMultMatrixComplexOMP::ValidationImpl() {
  return true;
}

bool ShvetsovaKMultMatrixComplexOMP::PreProcessingImpl() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  auto &matrix_c = GetOutput();
  matrix_c.rows = matrix_a.rows;
  matrix_c.cols = matrix_b.cols;
  matrix_c.row_ind.clear();
  matrix_c.values.clear();
  matrix_c.col_ptr.clear();
  matrix_c.col_ptr.push_back(0);
  return true;
}

bool ShvetsovaKMultMatrixComplexOMP::RunImpl() {
  const MatrixCCS &matrix_a = std::get<0>(GetInput());
  const MatrixCCS &matrix_b = std::get<1>(GetInput());
  std::vector<std::vector<std::complex<double>>> columns_c(matrix_b.cols);
  auto &matrix_c = GetOutput();

#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < matrix_b.cols; i++) {
      std::vector<std::complex<double>> column_c(matrix_a.rows);
      for (int j = matrix_b.col_ptr[i]; j < matrix_b.col_ptr[i + 1]; j++) {
        int tmp_ind = matrix_b.row_ind[j];
        std::complex tmp_val = matrix_b.values[j];
        for (int ind = matrix_a.col_ptr[tmp_ind]; ind < matrix_a.col_ptr[tmp_ind + 1]; ind++) {
          int row = matrix_a.row_ind[ind];
          std::complex val_a = matrix_a.values[ind];
          column_c[row] += tmp_val * val_a;
        }
      }
      columns_c[i] = column_c;
    }
  }

  for (size_t i = 0; i < columns_c.size(); i++) {
    int counter = 0;
    for (size_t k = 0; k < columns_c[i].size(); k++) {
      if (columns_c[i][k] != std::complex(0., 0.)) {
        matrix_c.row_ind.push_back(static_cast<int>(k));
        matrix_c.values.push_back(columns_c[i][k]);
        counter++;
      }
    }
    matrix_c.col_ptr.push_back(matrix_c.col_ptr[static_cast<int>(matrix_c.col_ptr.size()) - 1] + counter);
  }
  return true;
}

bool ShvetsovaKMultMatrixComplexOMP::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_mult_matrix_complex_col
