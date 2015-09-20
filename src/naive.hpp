#ifndef NAIVE_H
#define NAIVE_H

#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

#include "Matrix.hpp"

template<typename T>
using fun = std::function<void* (std::vector<row<T>>)>;
//typedef std::function<void* (row<int8_t>)> fun;

template<typename T>
fun<T> get_naive_alg(int cols, int rows);

#endif /* NAIVE_H */
