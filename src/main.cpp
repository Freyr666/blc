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
//multi threading test
#include <thread>

void
in_thread(char* pic, int n){
  Matrix<uint8_t>* blc = Pic2Mat<uint8_t>(pic);
  Naive<uint8_t>* f = new Naive<uint8_t> (blc->get_cols_num(), blc->get_rows_num());
  Optzd* fopt = new Optzd  (blc->get_cols_num(), blc->get_rows_num(), 16, 1);
  struct timeval tvb, tva;
  gettimeofday(&tvb,NULL);
  double rv = *(double*)fopt->eval(blc->val());
  gettimeofday(&tva,NULL);
  std::cout << "Opt in thread " << n << "\n";
  std::cout << rv << "\n";
  std::cout << "In " << tva.tv_usec - tvb.tv_usec << " us\n";
  std::cout << fopt->compilerlog();
  gettimeofday(&tvb,NULL);
  rv = *(double*)f->eval(blc->val());
  gettimeofday(&tva,NULL);
  //  delete f;
  std::cout << "Not opt in thread " << n << "\n";
  std::cout << rv << "\n";
  std::cout << "In " << tva.tv_usec - tvb.tv_usec << " us\n";
  delete f;
  delete fopt;
  delete blc;
}

int
main(int argc, char **argv){
  if (argc < 2){
    std::cout << "Usage: prog ./image\n";
    return 0;
  }/*
  Matrix<uint8_t>* blc = Pic2Mat<uint8_t>(argv[1]);
  Naive<uint8_t>* f = new Naive<uint8_t> (blc->get_cols_num(), blc->get_rows_num());
  Optzd* fopt = new Optzd  (blc->get_cols_num(), blc->get_rows_num(), 16, 1);
  //IsBlack* iblck = new IsBlack(blc->get_cols_num(), blc->get_rows_num(), 16, 1, 127);
  */
  std::thread thread_list[argc-1];
  for (int i = 0; i < argc-1; i++) {
    thread_list[i] = std::thread(in_thread, argv[i+1], i);
  }
  for (int i = 0; i < argc-1; i++) {
    thread_list[i].join();
  }
  return 0;
}
