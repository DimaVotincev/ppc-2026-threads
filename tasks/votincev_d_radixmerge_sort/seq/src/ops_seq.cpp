#include "votincev_d_radixmerge_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "votincev_d_radixmerge_sort/common/include/common.hpp"

namespace votincev_d_radixmerge_sort {

VotincevDRadixMergeSortSEQ::VotincevDRadixMergeSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VotincevDRadixMergeSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool VotincevDRadixMergeSortSEQ::PreProcessingImpl() {
  return true;
}

// ОПТИМИЗИРОВАННЫЙ метод: используем Counting Sort вместо векторов в векторах
void VotincevDRadixMergeSortSEQ::SortByDigit(std::vector<int32_t> &array, int32_t exp) {
  size_t n = array.size();
  std::vector<int32_t> output(n);  // Временный буфер для текущего разряда
  int32_t count[10] = {0};         // Массив счетчиков для цифр 0-9

  // 1. Считаем количество вхождений каждой цифры (0-9)
  for (size_t i = 0; i < n; i++) {
    int32_t digit = (array[i] / exp) % 10;
    count[digit]++;
  }

  // 2. Превращаем счетчики в индексы (префиксные суммы)
  // Теперь count[i] содержит позицию, ПЕРЕД которой заканчиваются элементы с цифрой i
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  // 3. Строим выходной массив, проходя с конца (ВАЖНО для стабильности сортировки)
  for (int64_t i = n - 1; i >= 0; i--) {
    int32_t digit = (array[i] / exp) % 10;
    output[count[digit] - 1] = array[i];
    count[digit]--;
  }

  // 4. Копируем результат обратно в основной массив
  array = std::move(output);
}

bool VotincevDRadixMergeSortSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  std::vector<int32_t> working_array = GetInput();

  // Поиск min и max за один проход
  auto [min_it, max_it] = std::minmax_element(working_array.begin(), working_array.end());
  int32_t min_val = *min_it;
  int32_t max_val = *max_it;

  // Сдвиг в положительную область
  if (min_val < 0) {
    for (auto &num : working_array) {
      num -= min_val;
    }
    max_val -= min_val;
  }

  // Цикл по разрядам. Используем int64_t для exp, чтобы избежать переполнения при exp * 10
  for (int64_t exp = 1; max_val / exp > 0; exp *= 10) {
    SortByDigit(working_array, static_cast<int32_t>(exp));
  }

  // Возврат к исходному диапазону
  if (min_val < 0) {
    for (auto &num : working_array) {
      num += min_val;
    }
  }

  GetOutput() = std::move(working_array);
  return true;
}

bool VotincevDRadixMergeSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_radixmerge_sort
