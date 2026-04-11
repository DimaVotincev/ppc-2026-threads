#include "posternak_a_crs_mul_complex_matrix/tbb/include/ops_tbb.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "posternak_a_crs_mul_complex_matrix/common/include/common.hpp"

namespace posternak_a_crs_mul_complex_matrix {

PosternakACRSMulComplexMatrixTBB::PosternakACRSMulComplexMatrixTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRSMatrix{};
}

bool PosternakACRSMulComplexMatrixTBB::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = input.first;
  const auto &b = input.second;
  return a.IsValid() && b.IsValid() && a.cols == b.rows;
}

bool PosternakACRSMulComplexMatrixTBB::PreProcessingImpl() {
  const auto &input = GetInput();
  const auto &a = input.first;
  auto &res = GetOutput();

  res.rows = a.rows;
  res.cols = input.second.cols;
  return true;
}

bool PosternakACRSMulComplexMatrixTBB::RunImpl() {
  const auto &input = GetInput();
  const auto &a = input.first;
  const auto &b = input.second;
  auto &res = GetOutput();

  if (a.values.empty() || b.values.empty()) {
    res.values.clear();
    res.index_col.clear();
    res.index_row.assign(res.rows + 1, 0);
    return true;
  }

  constexpr double kThreshold = 1e-12;

  // === Фаза 1: Подсчёт количества элементов (параллельно) ===
  // Выровненная структура для устранения false sharing
  struct alignas(64) Counter {
    size_t value = 0;
  };
  std::vector<Counter> counts(res.rows);

  // grainsize=64 + auto_partitioner (по умолчанию) = баланс оверхеда и балансировки
  tbb::parallel_for(tbb::blocked_range<int>(0, res.rows, 64),
                    [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      std::unordered_map<int, std::complex<double>> row_sum;
      row_sum.reserve(100);  // bandwidth=40 → ~80 элементов в результате

      for (int idx_a = a.index_row[row]; idx_a < a.index_row[row + 1]; ++idx_a) {
        int col_a = a.index_col[idx_a];
        auto val_a = a.values[idx_a];
        for (int idx_b = b.index_row[col_a]; idx_b < b.index_row[col_a + 1]; ++idx_b) {
          int col_b = b.index_col[idx_b];
          auto val_b = b.values[idx_b];
          row_sum[col_b] += val_a * val_b;
        }
      }

      size_t cnt = 0;
      for (const auto &[col, val] : row_sum) {
        if (std::abs(val) > kThreshold) {
          ++cnt;
        }
      }
      counts[row].value = cnt;
    }
  }
                    // auto_partitioner по умолчанию — лучше балансирует нагрузку
  );

  // === Фаза 2: Построение структуры (последовательно) ===
  res.index_row.resize(res.rows + 1);
  size_t total = 0;
  for (int i = 0; i < res.rows; ++i) {
    res.index_row[i] = static_cast<int>(total);
    total += counts[i].value;
  }
  res.index_row[res.rows] = static_cast<int>(total);
  res.values.resize(total);
  res.index_col.resize(total);

  // === Фаза 3: Запись результатов (параллельно) ===
  tbb::parallel_for(tbb::blocked_range<int>(0, res.rows, 64), [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      std::unordered_map<int, std::complex<double>> row_sum;
      row_sum.reserve(100);

      for (int idx_a = a.index_row[row]; idx_a < a.index_row[row + 1]; ++idx_a) {
        int col_a = a.index_col[idx_a];
        auto val_a = a.values[idx_a];
        for (int idx_b = b.index_row[col_a]; idx_b < b.index_row[col_a + 1]; ++idx_b) {
          int col_b = b.index_col[idx_b];
          auto val_b = b.values[idx_b];
          row_sum[col_b] += val_a * val_b;
        }
      }

      std::vector<std::pair<int, std::complex<double>>> sorted;
      sorted.reserve(row_sum.size());
      for (const auto &[col, val] : row_sum) {
        if (std::abs(val) > kThreshold) {
          sorted.emplace_back(col, val);
        }
      }
      std::ranges::sort(sorted, [](const auto &l, const auto &r) { return l.first < r.first; });

      size_t pos = res.index_row[row];
      for (const auto &[col, val] : sorted) {
        res.values[pos] = val;
        res.index_col[pos] = col;
        ++pos;
      }
    }
  });

  return res.IsValid();
}

bool PosternakACRSMulComplexMatrixTBB::PostProcessingImpl() {
  return GetOutput().IsValid();
}

}  // namespace posternak_a_crs_mul_complex_matrix
