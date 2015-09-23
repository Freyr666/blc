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
Matrix<T>::apply(std::function<void* (row<T>)> f, int rownum) const{
  return f(t->at(rownum));
}

template<typename T>
void*
Matrix<T>::apply(std::function<void* (std::vector<row<T>>)> f) const{
  return f(*t);
}

template<typename T>
void*
Matrix<T>::apply_m(std::function<void* (row<T>*)> f, int rownum){
  return f(&(t->at(rownum)));
}

template<typename T>
void*
Matrix<T>::apply_m(std::function<void* (std::vector<row<T>>*)> f){
  return f(t);
}
/*
template class Matrix<uint8_t>;
template class Matrix<uint16_t>;
template class Matrix<uint32_t>;
template class Matrix<uint64_t>;
template class Matrix<int8_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;
*/
template class Matrix<int>;
