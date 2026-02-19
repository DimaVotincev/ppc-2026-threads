#include "votincev_d_radixmerge_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

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

// основной метод алгоритма
bool VotincevDRadixMergeSortSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  // локальная копия данных для сортировки
  std::vector<int32_t> working_array = GetInput();

  // входной массив может иметь отрицательные числа,
  // поэтому нужно сдвинуть значения на минимум (если он отрицательный)
  // ищу минимум
  int32_t min_val = working_array[0];
  for (size_t i = 1; i < working_array.size(); i++) {
    if (working_array[i] < min_val) {
      min_val = working_array[i];
    }
  }

  // если минимум отрицательный - сдвигаю
  // если нет - сортируем массив положительных чисел, сдвиг не нужен
  if (min_val < 0) {
    for (auto &num : working_array) {
      num -= min_val;
    }
  }

  // ищем максимальное число для определения количества разрядов
  int32_t max_val = working_array[0];
  for (size_t i = 1; i < working_array.size(); i++) {
    if (working_array[i] > max_val) {
      max_val = working_array[i];
    }
  }

  // 10 корзин для десятичной системы счисления (0-9)
  std::vector<std::vector<int32_t>> buckets(10);

  // цикл по разрядам (единицы, десятки, сотни...)
  for (int32_t exp = 1; max_val / exp > 0; exp *= 10) {
    // распределение элементов по корзинам
    for (const auto &num : working_array) {
      int32_t digit = (num / exp) % 10;
      buckets[digit].push_back(num);
    }

    // простое слияние корзин обратно в рабочий массив
    size_t index = 0;
    for (int i = 0; i < 10; i++) {
      for (const auto &val : buckets[i]) {
        working_array[index++] = val;
      }
      // очистка корзины для следующего разряда
      buckets[i].clear();
    }
  }

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
