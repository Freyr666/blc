#include <vector>
#include <iostream>
#include <algorithm>
#include "Matrix.hpp"
#include "Pic2Mat.hpp"

template<typename T>
void*
test_fun(row<T>* r){
  T i = 0;
  for_each(r->begin(), r->end(), [&i](T &k)
	   { k = i++; } );
  int* rv = new int;
  *rv = 13;
  return rv;
}

template<typename T>
void*
sum_row(row<T>* r){
  int i = 0;
  for_each(r->begin(), r->end(), [&i](T k)
	   { i += k; } );
  T* rv = new T;
  *rv = i;
  return rv;
}

int
main(int argc, char **argv){
  if (argc < 3){
    std::cout << "Usage: prog ./blocky_image ./non_blocky image\n";
    return 0;
  }
  Matrix<int8_t>* blc = Pic2Mat<int8_t>(argv[1]);
  Matrix<int8_t>* nblc = Pic2Mat<int8_t>(argv[2]);

  std::cout << blc->get_cols_num() <<"\t" << blc->get_rows_num() <<"\n";
  std::cout << nblc->get_cols_num() <<"\t" << nblc->get_rows_num() <<"\n";
  
  return 0;
}
