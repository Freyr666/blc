#include <vector>
#include <iostream>
#include <algorithm>
#include "Matrix.hpp"
#include "Pic2Mat.hpp"
#include "./Naive/Naive.hpp"
//#include "./opencl++/Optzd.hpp"

//timestamps for testing
#include <sys/time.h>
//

int
main(int argc, char **argv){
  if (argc < 2){
    std::cout << "Usage: prog ./image\n";
    return 0;
  }
  Matrix<int>* blc = Pic2Mat<int>(argv[1]);
  Naive<int>* f = new Naive<int> (blc->get_cols_num(), blc->get_rows_num());
  //Optzd<int>* fopt = new Optzd<int>  (blc->get_cols_num(), blc->get_rows_num(), 64);
  struct timeval tvb, tva;
  gettimeofday(&tvb,NULL);
  double rv = *(double*)f->eval(blc->val());
  gettimeofday(&tva,NULL);
  //  delete f;
  std::cout << "Fun f is ok\n";
  std::cout << rv << "\n";
  std::cout << "In " << tva.tv_usec - tvb.tv_usec << " us\n";
  return 0;
}
