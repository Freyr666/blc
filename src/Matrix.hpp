#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <algorithm>
#include <functional>


template<typename T>
class Matrix{
private:
  int cols, rows;
  std::vector<T> * t;
public:
  Matrix(int x, int y);
  virtual ~Matrix();
  
  void set_el_m(int x, int y, T val);
  T    get_el(int x, int y) const;
  int  get_rows_num(void) const;
  int  get_cols_num(void) const;
  
  void* apply(std::function<void* (std::vector<T>)> f, int rownum) const;
  void* apply(std::function<void* (std::vector<T>)> f) const;;
  
  void* apply_m(std::function<void* (std::vector<T>*)> f);

  std::vector<T> * val(void) const { return t;} 
  std::vector<T> * val_m(void) { return t;}
};

#endif /* MATRIX_FLAT_H */
