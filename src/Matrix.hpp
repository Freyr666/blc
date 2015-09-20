#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <algorithm>

template<typename T>
using row = std::vector<T>;

template<typename T>
class Matrix{
private:
  int cols, rows;
  std::vector<std::vector<T>> * t;
public:
  Matrix(int x, int y);
  virtual ~Matrix();
  
  void set_el_m(int x, int y, T val);
  T    get_el(int x, int y) const;
  int  get_rows_num(void) const;
  int  get_cols_num(void) const;
  
  void* apply(void* (*f)(row<T>), int rownum) const;
  void* apply(void* (*f)(std::vector<row<T>>)) const;
  
  void* apply_m(void* (*f)(row<T>*), int rownum);
  void* apply_m(void* (*f)(std::vector<row<T>>*));
};

#endif /* MATRIX_H */
