#include <vector>
#include <iostream>
#include <algorithm>
#include "Matrix.hpp"
#include "Pic2Mat.hpp"
#include "./Naive/Naive.hpp"
#include "./opencl++/Optzd.hpp"
//#include "./Black/IsBlack.hpp"
//timestamps for testing
#include <sys/time.h>
//

int
main(int argc, char **argv){
  if (argc < 2){
    std::cout << "Usage: prog ./image\n";
    return 0;
  }
  Matrix<uint8_t>* blc = Pic2Mat<uint8_t>(argv[1]);
  Naive<uint8_t>* f = new Naive<uint8_t> (blc->get_cols_num(), blc->get_rows_num());
  Optzd* fopt = new Optzd  (blc->get_cols_num(), blc->get_rows_num(), 16, 1);
  //IsBlack* iblck = new IsBlack(blc->get_cols_num(), blc->get_rows_num(), 16, 1, 127);
  struct timeval tvb, tva;
  gettimeofday(&tvb,NULL);
  double rv = *(double*)fopt->eval(blc->val());
  gettimeofday(&tva,NULL);
  //  delete f;
  std::cout << "Opt\n";
  std::cout << rv << "\n";
  std::cout << "In " << tva.tv_usec - tvb.tv_usec << " us\n";
  std::cout << fopt->compilerlog();
  gettimeofday(&tvb,NULL);
  rv = *(double*)f->eval(blc->val());
  gettimeofday(&tva,NULL);
  //  delete f;
  std::cout << "Not opt\n";
  std::cout << rv << "\n";
  std::cout << "In " << tva.tv_usec - tvb.tv_usec << " us\n";
  /*
  for (int i = 0; i < blc->get_rows_num(); i++) {
    for (int j = 0; j < blc->get_cols_num(); j++) {
      std::cout << (int)blc->get_el(j, i) << " ";
    }
    std::cout << "\n";
  }
  */
  return 0;
}
