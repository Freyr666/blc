#include "Matrix.hpp"

template<typename T>
Matrix<T>::Matrix(int x, int y){
  t = new std::vector<std::vector<T>> (y, std::vector<T>(x));
  cols = x;
  rows = y;
}

template<typename T>
Matrix<T>::~Matrix(){
  delete t;
}

template<typename T>
void
Matrix<T>::set_el_m(int x, int y, T val){
  t->at(y).at(x) = val;
}

template<typename T>
T
Matrix<T>::get_el(int x, int y) const{
  T rval = t->at(y).at(x);
  return rval;
}

template<typename T>
int
Matrix<T>::get_rows_num(void) const{
  return rows;
}

template<typename T>
int
Matrix<T>::get_cols_num(void) const{
  return cols;
}

template<typename T>
void*
Matrix<T>::apply(void* (*f)(row<T>), int rownum) const{
  return f(t->at(rownum));
}

template<typename T>
void*
Matrix<T>::apply(void *(*f)(std::vector<row<T>>)) const{
  return f(*t);
}

template<typename T>
void*
Matrix<T>::apply_m(void *(*f)(row<T>*), int rownum){
  return f(&(t->at(rownum)));
}

template<typename T>
void*
Matrix<T>::apply_m(void *(*f)(std::vector<row<T>>*)){
  return f(t);
}

template class Matrix<int8_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;
