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

// проверка входных данных
bool VotincevDRadixMergeSortSEQ::ValidationImpl() {
  // проверка: входной вектор не должен быть пустым
  return !GetInput().empty();
}

// препроцессинг
bool VotincevDRadixMergeSortSEQ::PreProcessingImpl() {
  return true;
}

// поразрядная сортировка
void RadixSort(std::vector<int32_t> &array) {
  if (array.empty()) {
    return;
  }

  int32_t max_val = *std::ranges::max_element(array);
  std::vector<int32_t> buffer(array.size());

  // цикл по разрядам (единицы, десятки, сотни...)
  for (int32_t exp = 1; max_val / exp > 0; exp *= 10) {
    std::vector<int> count(10, 0);

    // подсчитываем количество цифр
    for (const auto &num : array) {
      int32_t digit = (num / exp) % 10;
      count[digit]++;
    }

    // префиксные суммы для вычисления позиций
    for (int i = 1; i < 10; ++i) {
      count[i] += count[i - 1];
    }

    // заполняем буфер с конца для сохранения стабильности
    for (int i = static_cast<int>(array.size()) - 1; i >= 0; --i) {
      int32_t digit = (array[i] / exp) % 10;
      buffer[--count[digit]] = array[i];
    }

    array = buffer;
  }
}

// простое слияние 2х массивов
std::vector<int32_t> SimpleMerge(const std::vector<int32_t> &left, const std::vector<int32_t> &right) {
  std::vector<int32_t> result;
  result.reserve(left.size() + right.size());

  size_t i = 0, j = 0;
  // слияние элементов из двух массивов
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  // дописываем остатки левой части
  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  // дописываем остатки правой части
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}

// основной метод алгоритма
bool VotincevDRadixMergeSortSEQ::RunImpl() {
  // локальная копия данных для сортировки
  std::vector<int32_t> working_array = GetInput();
  size_t n = working_array.size();

  // обработка отрицательных чисел
  int32_t min_val = *std::ranges::min_element(working_array);
  if (min_val < 0) {
    for (auto &num : working_array) {
      num -= min_val;
    }
  }

  // разобьем массив на 2 части для слияния
  size_t mid = n / 2;
  std::vector<int32_t> left_part(working_array.begin(), working_array.begin() + mid);
  std::vector<int32_t> right_part(working_array.begin() + mid, working_array.end());

  // сортируем каждую часть поразрядно
  RadixSort(left_part);
  RadixSort(right_part);

  // сливаем две отсортированные половины
  working_array = SimpleMerge(left_part, right_part);

  // возвращаем значения к исходному диапазону
  if (min_val < 0) {
    for (auto &num : working_array) {
      num += min_val;
    }
  }

  // запись результата в выходные данные
  GetOutput() = working_array;
  return true;
}

// постпроцессинг
bool VotincevDRadixMergeSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_radixmerge_sort
