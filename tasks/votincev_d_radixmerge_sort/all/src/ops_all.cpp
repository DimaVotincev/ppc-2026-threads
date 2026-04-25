#include "votincev_d_radixmerge_sort/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

#include "util/include/util.hpp"
#include "votincev_d_radixmerge_sort/common/include/common.hpp"

namespace votincev_d_radixmerge_sort {

VotincevDRadixMergeSortALL::VotincevDRadixMergeSortALL(const InType &in) : input_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool VotincevDRadixMergeSortALL::ValidationImpl() {
  return !input_.empty();
}

bool VotincevDRadixMergeSortALL::PreProcessingImpl() {
  return true;
}

// Вспомогательная функция поразрядной сортировки (локальная)
void VotincevDRadixMergeSortALL::LocalRadixSort(uint32_t *begin, uint32_t *end) {
  auto n = static_cast<int32_t>(end - begin);
  if (n <= 1) {
    return;
  }

  uint32_t max_val = *std::max_element(begin, end);
  std::vector<uint32_t> buffer(n);
  uint32_t *src = begin;
  uint32_t *dst = buffer.data();

  for (int64_t exp = 1; (max_val / exp) > 0; exp *= 10) {
    std::array<int32_t, 10> count{};
    for (int32_t i = 0; i < n; ++i) {
      count[(src[i] / exp) % 10]++;
    }
    for (int32_t i = 1; i < 10; ++i) {
      count[i] += count[i - 1];
    }
    for (int32_t i = n - 1; i >= 0; --i) {
      uint32_t digit = (src[i] / exp) % 10;
      dst[--count[digit]] = src[i];
    }
    std::swap(src, dst);
  }
  if (src != begin) {
    std::copy(src, src + n, begin);
  }
}

// Слияние двух подмассивов
void VotincevDRadixMergeSortALL::Merge(const uint32_t *src, uint32_t *dst, int32_t left, int32_t mid, int32_t right) {
  int32_t i = left, j = mid, k = left;
  while (i < mid && j < right) {
    dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
  }
  while (i < mid) {
    dst[k++] = src[i++];
  }
  while (j < right) {
    dst[k++] = src[j++];
  }
}

bool VotincevDRadixMergeSortALL::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int32_t n = (rank == 0) ? static_cast<int32_t>(input_.size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Рассчитываем распределение данных
  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);
  int32_t items = n / size;
  int32_t rem = n % size;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = items + (i < rem ? 1 : 0);
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + send_counts[i - 1];
  }

  int local_n = send_counts[rank];
  std::vector<uint32_t> local_data(local_n);

  // Рассылаем части массива (используем смещение min_val для корректной сортировки отрицательных)
  int32_t min_val;
  if (rank == 0) {
    min_val = *std::min_element(input_.begin(), input_.end());
  }
  MPI_Bcast(&min_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<uint32_t> unsigned_input;
  if (rank == 0) {
    unsigned_input.resize(n);
    for (int i = 0; i < n; ++i) {
      unsigned_input[i] = static_cast<uint32_t>(input_[i] - min_val);
    }
  }

  MPI_Scatterv(unsigned_input.data(), send_counts.data(), displacements.data(), MPI_UINT32_T, local_data.data(),
               local_n, MPI_UINT32_T, 0, MPI_COMM_WORLD);

  // --- ПАРАЛЛЕЛЬНАЯ ЧАСТЬ (OpenMP внутри MPI процесса) ---
  if (local_n > 0) {
    std::vector<uint32_t> temp_buffer(local_n);
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int n_threads = omp_get_num_threads();

      int t_items = local_n / n_threads;
      int t_rem = local_n % n_threads;
      int l = (tid * t_items) + std::min(tid, t_rem);
      int r = l + t_items + (tid < t_rem ? 1 : 0);

      if (l < r) {
        LocalRadixSort(local_data.data() + l, local_data.data() + r);
      }

      // Простое слияние результатов потоков
      for (int step = 1; step < n_threads; step *= 2) {
#pragma omp barrier
        if (tid % (2 * step) == 0 && tid + step < n_threads) {
          int m = ((tid + step) * t_items) + std::min(tid + step, t_rem);
          int next_tid = std::min(tid + (2 * step), n_threads);
          int next_r = (next_tid * t_items) + std::min(next_tid, t_rem);

          Merge(local_data.data(), temp_buffer.data(), l, m, next_r);
          std::copy(temp_buffer.data() + l, temp_buffer.data() + next_r, local_data.data() + l);
        }
      }
    }
  }

  // Сбор отсортированных кусков на Root
  std::vector<uint32_t> gathered_data;
  if (rank == 0) {
    gathered_data.resize(n);
  }

  MPI_Gatherv(local_data.data(), local_n, MPI_UINT32_T, gathered_data.data(), send_counts.data(), displacements.data(),
              MPI_UINT32_T, 0, MPI_COMM_WORLD);

  // Финальное слияние кусков MPI на процессе 0
  if (rank == 0) {
    // Переменные final_buffer, current_src и current_dst удалены,
    // так как они не используются при std::inplace_merge

    // Последовательное слияние блоков от разных MPI процессов
    for (int i = 1; i < size; ++i) {
      int mid = displacements[i];
      int right = (i == size - 1) ? n : displacements[i + 1];

      std::inplace_merge(gathered_data.begin(), gathered_data.begin() + mid, gathered_data.begin() + right);
    }

    output_.resize(n);
    for (int i = 0; i < n; ++i) {
      output_[i] = static_cast<int32_t>(gathered_data[i] + static_cast<uint32_t>(min_val));
    }
  }

  return true;
}

bool VotincevDRadixMergeSortALL::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput() = std::move(output_);
  }
  return true;
}

}  // namespace votincev_d_radixmerge_sort
