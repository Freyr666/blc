#include <vector>
#include <iostream>
#include <algorithm>
#include "Matrix.hpp"
#include "Pic2Mat.hpp"
//#include "naive.hpp"
#include "./Naive/Naive.hpp"


int
main(int argc, char **argv){
  if (argc < 2){
    std::cout << "Usage: prog ./image\n";
    return 0;
  }
  
  Matrix<int>* blc = Pic2Mat<int>(argv[1]);

  Naive<int>* f = new Naive<int> (blc->get_cols_num(), blc->get_rows_num());
  //auto f =  get_naive_alg<int>(blc->get_cols_num(), blc->get_rows_num());
  //double rv = *(double*)blc->apply(Naive<int>(blc->get_cols_num(), blc->get_rows_num()));
  double rv = *(double*)f->eval(*(blc->val()));
  delete f;
  std::cout << "Fun f is ok\n";
  std::cout << rv << "\n";
  
  return 0;
}
